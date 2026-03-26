"""
MobileNetV2 model architecture for binary classification.
With focal loss and improved training capabilities.
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_SIZE, NUM_CLASSES

# Focal loss for handling hard examples (esp. fake Maggi)
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss: Focus learning on hard negatives."""
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, reduction='auto'):
        super(FocalLoss, self).__init__(reduction=reduction)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits
    
    def call(self, y_true, y_pred):
        """Compute focal loss."""
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        
        y_true = tf.cast(y_true, tf.float32)
        
        # Binary crossentropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        
        # Focal term: (1 - p_t)^gamma
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        
        # Weighted focal loss
        focal_loss = self.alpha * focal_weight * bce
        
        return focal_loss

def build_model(trainable_base=False, enhanced=False):
    """
    Build MobileNetV2 model for fake vs real classification.
    
    Args:
        trainable_base: If True, allows fine-tuning of base model
        enhanced: If True, uses deeper classification head
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV2 (without top classification layer)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = trainable_base
    
    # Build classification head
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    
    if enhanced:
        # Enhanced head with more layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
    else:
        # Simple head
        x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    
    return model

def compile_model(model, learning_rate=1e-4, use_focal_loss=True):
    """
    Compile model with optimizer and loss.
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
        use_focal_loss: If True, uses focal loss for hard examples
    """
    if use_focal_loss:
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        loss_fn = 'binary_crossentropy'
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    return model
