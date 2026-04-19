import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import  StratifiedKFold

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from models import build_svm_classifier
from models import build_lda_classifier
from models import EEGNet

from feature import make_features_xdawn
from feature import make_features_downscale


# Create results directory if it doesn't exist
RESULTS_DIR = 'results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


def evaluate_model(epochs, feature_name, model_name, T=1.0, N=36):
    """Evaluate model with classification report and ITR calculation.
    
    Parameters:
    T: Trial duration in seconds (default: 1.0)
    N: Number of symbols/targets (default: 36 for EEG Speller)
    """
    if feature_name == 'xdawn':
        X = make_features_xdawn(epochs.copy())
    elif feature_name == 'downscale':
        X = make_features_downscale(epochs.copy())
    else:
        raise ValueError("Invalid feature name. Choose 'xdawn' or 'downscale'.")
    
    y = epochs.events[:,-1]
    # Cross validator
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Classifier
    if model_name == 'svm':
        clf = build_svm_classifier()
    elif model_name == 'lda':
        clf = build_lda_classifier()
    else:
        raise ValueError("Invalid model name. Choose 'svm' or 'lda'.")

    # Do cross-validation
    preds = np.empty(len(y))
    for train, test in cv.split(X, y):
        clf.fit(X[train], y[train])
        preds[test] = clf.predict(X[test])

    # Classification report
    target_names = ['Non-Target', 'Target'] # Corrected target names
    report = classification_report(y, preds, target_names=target_names, output_dict=False)
    print(f"Classification Report for {model_name} with {feature_name} features:\n{report}")

    # Calculate accuracy and ITR
    accuracy = np.mean(preds == y)
    
    # Calculate ITR (Information Transfer Rate)
    if accuracy == 0:
        itr = 0
    elif accuracy == 1:
        itr = (np.log2(N)) * 60 / T
    else:
        itr = (np.log2(N) + accuracy * np.log2(accuracy) + (1 - accuracy) * np.log2((1 - accuracy) / (N - 1))) * 60 / T
    
    print("\n" + "="*70)
    print(f"ACCURACY: {accuracy:.4f}")
    print(f"ITR (bits/min): {itr:.4f}")
    print("="*70)

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    print(cm)
    
    # Plot normalized confusion matrix
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax,
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Percentage'})
    ax.set_title(f'Normalized Confusion Matrix - {model_name} with {feature_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save confusion matrix
    cm_filename = os.path.join(RESULTS_DIR, f'confusion_matrix_{model_name}_{feature_name}.png')
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {cm_filename}")
    
    plt.show()
    
    # Save classification report as CSV
    report_dict = classification_report(y, preds, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_filename = os.path.join(RESULTS_DIR, f'classification_report_{model_name}_{feature_name}.csv')
    report_df.to_csv(report_filename)
    print(f"Classification report saved to: {report_filename}")
    
    return accuracy, itr


def evaluate_eegnet_model(epochs, F1=8, D=2, F2=16, dropoutRate=0.5, T=1.0, N=36):
    """Evaluate EEGNet model with classification report, confusion matrix, and ITR calculation.
    
    Parameters:
    T: Trial duration in seconds (default: 1.0)
    N: Number of symbols/targets (default: 36 for EEG Speller)
    """
    n_classes = len(np.unique(epochs.events[:,-1]))
    n_channels = epochs.info['nchan']
    n_samples = epochs.times.size

    model = EEGNet(nb_classes=n_classes, Chans=n_channels, Samples=n_samples, F1=F1, D=D, F2=F2, dropoutRate=dropoutRate)
    model.summary()
    
    X = np.expand_dims(epochs, axis=-1)
    y = epochs.events[:,-1]-1  # Assuming labels are 1 and 2, convert to 0 and 1 for binary classification

    # Convert labels to one-hot (required for categorical_crossentropy)
    y_cat = to_categorical(y, num_classes=n_classes)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat,
        test_size=0.2,
        random_state=42,
        stratify=y_cat
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=16,
        validation_data=(X_val, y_val),
        verbose=1
    )


    # Generate predictions on validation data
    y_val_pred_probs = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)
    y_val_true = np.argmax(y_val, axis=1)

    # Classification report on validation data
    target_names = ['Non-Target', 'Target']
    print("\n" + "="*70)
    print("VALIDATION DATA - CLASSIFICATION REPORT")
    print("="*70)
    val_report = classification_report(y_val_true, y_val_pred, target_names=target_names, digits=4)
    print(val_report)

    # Confusion matrix on validation data
    cm = confusion_matrix(y_val_true, y_val_pred)
    print("\n" + "="*70)
    print("VALIDATION DATA - CONFUSION MATRIX")
    print("="*70)
    print(cm)
    
    # Plot normalized confusion matrix
    cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax,
                xticklabels=target_names, yticklabels=target_names,
                cbar_kws={'label': 'Percentage'})
    ax.set_title('Normalized Confusion Matrix - EEGNet', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save confusion matrix
    cm_filename = os.path.join(RESULTS_DIR, 'confusion_matrix_eegnet.png')
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {cm_filename}")
    
    plt.show()

    # Calculate accuracy and ITR
    accuracy = np.mean(y_val_pred == y_val_true)
    
    # Calculate ITR (Information Transfer Rate)
    if accuracy == 0:
        itr = 0
    elif accuracy == 1:
        itr = (np.log2(N)) * 60 / T
    else:
        itr = (np.log2(N) + accuracy * np.log2(accuracy) + (1 - accuracy) * np.log2((1 - accuracy) / (N - 1))) * 60 / T
    
    print("\n" + "="*70)
    print(f"VALIDATION ACCURACY: {accuracy:.4f}")
    print(f"VALIDATION ITR (bits/min): {itr:.4f}")
    print("="*70)
    
    # Save classification report as CSV
    report_dict = classification_report(y_val_true, y_val_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_filename = os.path.join(RESULTS_DIR, 'classification_report_eegnet.csv')
    report_df.to_csv(report_filename)
    print(f"Classification report saved to: {report_filename}")
    
    return accuracy, itr





