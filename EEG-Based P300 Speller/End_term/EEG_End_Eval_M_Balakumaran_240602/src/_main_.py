import mne
import matplotlib.pyplot as plt
import preprocessing as pre
import scipy
import numpy as np
import pandas as pd

from moabb.datasets import BNCI2014_009
dataset = BNCI2014_009()   # P300 speller, 10 subjects
dataset.download()

file_path = 'E:\\EEG_Speller\\C-\\Users\\thend\\mne_data\\MNE-bnci-data\\~bci\\database\\009-2014\\A10S.mat'
mat_data = scipy.io.loadmat(file_path)
print(mat_data.keys())

raw_data = mat_data['data']
eeg_data = np.array(raw_data[0,0][0][0][1])


event_id = np.array(raw_data[0,0][0][0][3])
labels = np.array(raw_data[0,0][0][0][2])

event_arr = np.array([],dtype='int32').reshape(0,3)
for j in range(labels.shape[0]):
    if j==0:
        if labels[j,0]!=0:
            event_arr=np.vstack([event_arr,[j,0,labels[j,0]]])

    else:
        if labels[j,0]!=0 and labels[j-1,0]==0:
            event_arr=np.vstack([event_arr,[j,0,labels[j,0]]])

final_raw = pre.preprocessing(eeg_data)
reconstructed_raw = pre.ica_analysis(final_raw)
reconstructed_raw = pre.bad_channel_rej(reconstructed_raw,1.2,10)
epochs = pre.make_epochs(event_arr,reconstructed_raw)
print(epochs)


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from mne import Epochs, io, pick_types, read_events
from mne.datasets import sample
from mne.decoding import Vectorizer, XdawnTransformer, get_spatial_filter_from_estimator
from mne.utils import check_version

import os
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.svm import SVC
from train_and_test import evaluate_eegnet_model, evaluate_model

svm_acc, svm_itr = evaluate_model(epochs, feature_name='xdawn', model_name='svm')
lda_acc, lda_itr = evaluate_model(epochs, feature_name='xdawn', model_name='lda')
evaluate_model(epochs, feature_name='downscale', model_name='svm')
evaluate_model(epochs, feature_name='downscale', model_name='lda')
eegnet_acc, eegnet_itr = evaluate_eegnet_model(epochs)

def plot_max_performance_comparison(lda_acc, lda_itr, svm_acc, svm_itr, eegnet_acc, eegnet_itr):
    """Create and save comparison bar charts for max accuracy and ITR across models.
    
    Parameters:
    lda_acc, lda_itr: LDA accuracy and ITR values
    svm_acc, svm_itr: SVM accuracy and ITR values
    eegnet_acc, eegnet_itr: EEGNet accuracy and ITR values
    """
    
    # Model names and values
    models = ['LDA', 'SVM', 'EEGNet']
    accuracies = [lda_acc, svm_acc, eegnet_acc]
    itrs = [lda_itr, svm_itr, eegnet_itr]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Classification Accuracy
    colors_acc = ['#1f77b4', '#1f77b4', '#1f77b4']  # Blue for all
    bars1 = axes[0].bar(models, accuracies, color=colors_acc, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Classification Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Information Transfer Rate
    colors_itr = ['#ff7f0e', '#ff7f0e', '#ff7f0e']  # Orange for all
    bars2 = axes[1].bar(models, itrs, color=colors_itr, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('ITR (bits/min)', fontsize=12, fontweight='bold')
    axes[1].set_title('Information Transfer Rate', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, itr in zip(bars2, itrs):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{itr:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_filename = os.path.join(results_dir, 'max_performance_comparison.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_filename}")
    
    # Print summary
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Model':<15} {'Accuracy':<15} {'ITR (bits/min)':<15}")
    print("-"*70)
    for model, acc, itr in zip(models, accuracies, itrs):
        print(f"{model:<15} {acc:<15.4f} {itr:<15.2f}")
    print("="*70)
    
    plt.show()

plot_max_performance_comparison(lda_acc, lda_itr, svm_acc, svm_itr, eegnet_acc, eegnet_itr)