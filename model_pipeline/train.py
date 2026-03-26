"""
Training pipeline with class weights, data augmentation, and focal loss.
Fixes Maggi false negative bias through proper class weighting and augmentation.
"""
import os
import tensorflow as tf
from datetime import datetime
import numpy as np
from collections import Counter
from pathlib import Path

from model import build_model, compile_model, FocalLoss
from config import (TRAIN_DIR, VAL_DIR, BATCH_SIZE, EPOCHS, 
                   LEARNING_RATE, MODEL_PATH, LOGS_DIR, CLASS_NAMES)

# ============================================================
# PHASE 1: Data Inspection & Class Weight Calculation
# ============================================================

def inspect_class_distribution(directory):
    """Count real vs fake samples in directory."""
    counts = {}
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(directory, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            counts[class_name] = len(files)
        else:
            counts[class_name] = 0
    return counts

def calculate_dynamic_class_weights(train_dir):
    """
    Calculate class weights from training data.
    Handles imbalance between FAKE and REAL samples.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: DATA INSPECTION & CLASS WEIGHTING")
    print("=" * 60)
    
    # Count samples
    fake_count = len([f for f in os.listdir(os.path.join(train_dir, 'fake'))
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    real_count = len([f for f in os.listdir(os.path.join(train_dir, 'real'))
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    total = fake_count + real_count
    
    # Print distribution
    print("\nTRAINING DATA DISTRIBUTION:")
    print("-" * 60)
    print(f"FAKE samples: {fake_count}")
    print(f"REAL samples: {real_count}")
    print(f"Total: {total}")
    print(f"Imbalance ratio (FAKE:REAL): {fake_count/real_count:.2f}:1")
    
    if fake_count / real_count > 1.5:
        print("[WARNING] Significant class imbalance detected!")
    
    # Calculate weights: weight = total_samples / (num_classes * class_count)
    num_classes = 2
    fake_weight = total / (num_classes * fake_count) if fake_count > 0 else 1.0
    real_weight = total / (num_classes * real_count) if real_count > 0 else 1.0
    
    class_weights = {
        0: fake_weight,  # FAKE
        1: real_weight   # REAL
    }
    
    print("\nCOMPUTED CLASS WEIGHTS:")
    print("-" * 60)
    print(f"FAKE (class 0): {fake_weight:.4f} (weight up minority)")
    print(f"REAL (class 1): {real_weight:.4f}")
    print(f"Weight ratio: {fake_weight/real_weight:.4f}")
    print(f"\nFor model.fit(): class_weight={class_weights}")
    
    return class_weights

# ============================================================
# PHASE 3: Strong Data Augmentation
# ============================================================

def create_augmented_dataset(directory, batch_size, augment=True):
    """
    Create tf.data.Dataset with strong data augmentation.
    Applied ONLY to training data to prevent Maggi memorization.
    """
    # Base dataset loading
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='binary',
        class_names=CLASS_NAMES,
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=True if augment else False
    )
    
    # Apply MobileNetV2 preprocessing
    normalization = tf.keras.applications.mobilenet_v2.preprocess_input
    dataset = dataset.map(lambda x, y: (normalization(x), y))
    
    # Apply data augmentation (if requested and not validation)
    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ])
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)
    
    # Performance optimization
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# ============================================================
# PHASE 2 & 7: Train with Focal Loss & Class Weights
# ============================================================

def train_model(class_weights, epochs=EPOCHS, fine_tune=False):
    """
    Train model with:
    - Dynamic class weights (handle imbalance)
    - Strong data augmentation (prevent Maggi memorization)
    - Focal loss (focus on hard negatives)
    - Proper preprocessing consistency
    """
    print("\n" + "=" * 60)
    print("PHASE 2: TRAINING WITH IMPROVEMENTS")
    print("=" * 60)
    
    # Create datasets
    print("\nLoading and augmenting training data...")
    train_ds = create_augmented_dataset(TRAIN_DIR, BATCH_SIZE, augment=True)
    
    print("Loading validation data (no augmentation)...")
    val_ds = create_augmented_dataset(VAL_DIR, BATCH_SIZE, augment=False)
    
    # Build model
    print("\nBuilding model with focal loss...")
    model = build_model(trainable_base=fine_tune)
    model = compile_model(model, learning_rate=LEARNING_RATE, use_focal_loss=True)
    
    print(f"\nModel architecture:")
    print(f"  Base: MobileNetV2 (trainable={fine_tune})")
    print(f"  Loss: Focal Loss (alpha=0.25, gamma=2.0)")
    print(f"  Head: GlobalAvgPool -> Dropout(0.5) -> Dense(1, sigmoid)")
    print(f"  Preprocessing: MobileNetV2 [-1, 1] normalization (consistent)")
    print(f"  Data Augmentation: Rotation, Zoom, Flip, Brightness, Translation")
    print(f"  Total params: {model.count_params():,}")
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(LOGS_DIR, f"training_{timestamp}")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ]
    
    # Training
    print(f"\nTraining configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Class weights: {class_weights}")
    print("-" * 60)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved: {MODEL_PATH}")
    print(f"Logs saved: {log_dir}")
    
    final_acc = history.history['val_accuracy'][-1]
    final_loss = history.history['val_loss'][-1]
    print(f"\nFinal validation accuracy: {final_acc:.4f}")
    print(f"Final validation loss: {final_loss:.4f}")
    
    return model, history

if __name__ == '__main__':
    # Calculate class weights dynamically from training data
    class_weights = calculate_dynamic_class_weights(TRAIN_DIR)
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING PIPELINE")
    print("=" * 60)
    print("\nImprovements:")
    print("  ✓ Dynamic class weights (address FAKE/REAL imbalance)")
    print("  ✓ Strong data augmentation (prevent Maggi memorization)")
    print("  ✓ Focal loss (focus on hard negatives)")
    print("  ✓ Consistent preprocessing (MobileNetV2 [-1, 1])")
    print("  ✓ Next: Run evaluate.py to optimize threshold")
    
    model, history = train_model(
        class_weights=class_weights,
        epochs=EPOCHS,
        fine_tune=False
    )
    
    print("\n" + "=" * 60)
    print("Next step: Run evaluate.py to check performance and optimize threshold")
    print("=" * 60)
