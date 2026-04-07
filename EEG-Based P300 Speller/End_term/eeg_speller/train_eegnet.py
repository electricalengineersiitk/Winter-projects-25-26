
"""
train_eegnet.py — Train EEGNet on BNCI2014_009
"""
import os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.makedirs('results', exist_ok=True)
os.makedirs('models',  exist_ok=True)
from src.preprocess import load_subject, apply_notch_to_epochs
from src.models     import build_eegnet
from src.evaluate   import information_transfer_rate
import tensorflow as tf
from tensorflow.keras.utils    import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection    import train_test_split
from sklearn.metrics            import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('subject', type=int, default=1)
parser.add_argument('epochs',  type=int, default=100)
args = parser.parse_args()

SID = args.subject

# 1.Load data 
print(f"Loading Subject {SID:02d}")
X, y, _ = load_subject(SID)
X        = apply_notch_to_epochs(X)
X_4d = X[:, :, :, np.newaxis]
y_oh = to_categorical(y, num_classes=2)

X_train, X_val, y_train, y_val, y_tr_raw, y_va_raw = train_test_split(
    X_4d, y_oh, y,
    test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {X_train.shape}  |  Val: {X_val.shape}")
print(f"Target ratio — train: {y_tr_raw.mean():.2f}  val: {y_va_raw.mean():.2f}")

_, n_channels, n_times, _ = X_train.shape

# 2.Build model
model = build_eegnet(n_channels=n_channels, n_times=n_times)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3.Class weights (handle target/non-target imbalance)
n_target     = y_tr_raw.sum()
n_non_target = len(y_tr_raw) - n_target
class_weight = {0: 1.0,
                1: n_non_target / max(n_target, 1)}
print(f"\nClass weights: NonTarget=1.0, Target={class_weight[1]:.2f}")

# 4.Callbacks 
ckpt_path = f'models/eegnet_subject_{SID:02d}.keras'
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=7, verbose=1),
    ModelCheckpoint(ckpt_path, save_best_only=True, verbose=0)
]

# 5.Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=args.epochs,
    batch_size=64,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# 6.Evaluate
y_pred_prob  = model.predict(X_val)
y_pred_class = np.argmax(y_pred_prob, axis=1)
acc = accuracy_score(y_va_raw, y_pred_class)
T_trial = 10 * 12 * 0.125
itr_val = information_transfer_rate(N=36, P=acc, T=T_trial)

print(f"\n{'='*45}")
print(f"  EEGNet — Subject {SID:02d}")
print(f"  Val Accuracy : {acc*100:.2f}%")
print(f"  ITR estimate : {itr_val:.1f} bits/min")
print(f"  Model saved  : {ckpt_path}")
print(f"{'='*45}")

# 7.Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['accuracy'],     label='Train', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val',   linewidth=2)
axes[0].set_title(f'EEGNet Accuracy — S{SID:02d}')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'],         label='Train', linewidth=2)
axes[1].plot(history.history['val_loss'],     label='Val',   linewidth=2)
axes[1].set_title(f'EEGNet Loss — S{SID:02d}')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plot_path = f'results/eegnet_training_S{SID:02d}.png'
plt.savefig(plot_path, dpi=100); plt.close()
print(f"  Training plot: {plot_path}")