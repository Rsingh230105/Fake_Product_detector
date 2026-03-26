"""
Enhanced training pipeline with data augmentation and fine-tuning.
Improves model accuracy through advanced techniques.
"""
import os
import tensorflow as tf
from datetime import datetime
from model import build_model, compile_model
from config import (TRAIN_DIR, VAL_DIR, BATCH_SIZE, EPOCHS, 
                   LEARNING_RATE, MODEL_PATH, LOGS_DIR, CLASS_NAMES)

def create_augmented_dataset(directory, batch_size, shuffle=True, augment=True):
    """Create dataset with data augmentation."""
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='binary',
        class_names=CLASS_NAMES,
        batch_size=batch_size,
        image_size=(224, 224),
        shuffle=shuffle
    )
    
    # Data augmentation layer
    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1),
        ])
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    
    # MobileNetV2 preprocessing
    normalization = tf.keras.applications.mobilenet_v2.preprocess_input
    dataset = dataset.map(lambda x, y: (normalization(x), y))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def build_enhanced_model(trainable_base=False):
    """Build enhanced model with better architecture."""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_base
    
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.models.Model(inputs, outputs)
    return model

def train_enhanced(class_weights=None, use_augmentation=True):
    """Enhanced training with two-stage approach."""
    print("=" * 70)
    print("ENHANCED TRAINING PIPELINE - TWO-STAGE APPROACH")
    print("=" * 70)
    
    # Stage 1: Train with frozen base
    print("\n" + "=" * 70)
    print("STAGE 1: Training with frozen MobileNetV2 base")
    print("=" * 70)
    
    train_ds = create_augmented_dataset(TRAIN_DIR, BATCH_SIZE, shuffle=True, augment=use_augmentation)
    val_ds = create_augmented_dataset(VAL_DIR, BATCH_SIZE, shuffle=False, augment=False)
    
    model = build_enhanced_model(trainable_base=False)
    model = compile_model(model, learning_rate=LEARNING_RATE)
    
    print(f"\nModel: Enhanced MobileNetV2")
    print(f"Total params: {model.count_params():,}")
    print(f"Data augmentation: {use_augmentation}")
    print(f"Class weights: {class_weights}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(LOGS_DIR, f"enhanced_training_{timestamp}")
    
    callbacks_stage1 = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH.replace('.keras', '_stage1.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
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
        tf.keras.callbacks.TensorBoard(log_dir=log_dir + "/stage1")
    ]
    
    print("\nTraining Stage 1...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=min(EPOCHS, 30),
        class_weight=class_weights,
        callbacks=callbacks_stage1,
        verbose=1
    )
    
    stage1_acc = max(history1.history['val_accuracy'])
    print(f"\nStage 1 Best Validation Accuracy: {stage1_acc:.4f}")
    
    # Stage 2: Fine-tune top layers
    print("\n" + "=" * 70)
    print("STAGE 2: Fine-tuning top layers of MobileNetV2")
    print("=" * 70)
    
    # Unfreeze top layers
    base_model = model.layers[1]
    base_model.trainable = True
    
    # Freeze bottom layers, unfreeze top 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model = compile_model(model, learning_rate=LEARNING_RATE / 10)
    
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    print(f"\nTrainable layers: {trainable_count}")
    print(f"Learning rate: {LEARNING_RATE / 10}")
    
    callbacks_stage2 = [
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
            patience=2,
            min_lr=1e-8,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir + "/stage2")
    ]
    
    print("\nTraining Stage 2...")
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=min(EPOCHS, 20),
        class_weight=class_weights,
        callbacks=callbacks_stage2,
        verbose=1
    )
    
    stage2_acc = max(history2.history['val_accuracy'])
    print(f"\nStage 2 Best Validation Accuracy: {stage2_acc:.4f}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Stage 1 Best Accuracy: {stage1_acc:.4f}")
    print(f"Stage 2 Best Accuracy: {stage2_acc:.4f}")
    print(f"Improvement: {(stage2_acc - stage1_acc):.4f}")
    print(f"\nFinal model saved: {MODEL_PATH}")
    print(f"Logs saved: {log_dir}")
    
    return model, history1, history2

if __name__ == '__main__':
    # Class weights from dataset analysis
    class_weights = {0: 0.68, 1: 1.90}
    
    print("\nStarting enhanced training pipeline...")
    print("This will train in two stages:")
    print("  1. Frozen base (faster, learns task-specific features)")
    print("  2. Fine-tuning (slower, adapts pre-trained features)")
    print()
    
    model, hist1, hist2 = train_enhanced(
        class_weights=class_weights,
        use_augmentation=True
    )
    
    print("\n" + "=" * 70)
    print("Next: Run evaluate.py to check final performance")
    print("=" * 70)
