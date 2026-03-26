"""
Single preprocessing pipeline for ALL use cases.
CRITICAL: This exact preprocessing is used for:
- Training
- Evaluation  
- Prediction
- Web inference
"""
import tensorflow as tf
import numpy as np
from config import IMG_SIZE

def preprocess_image(image_input):
    """
    Load and preprocess image.
    
    Args:
        image_input: Path to image file, numpy array, or PIL Image
        
    Returns:
        Preprocessed image tensor (224, 224, 3) normalized to [-1, 1]
    """
    # Handle different input types
    if isinstance(image_input, str):
        # File path
        img = tf.io.read_file(image_input)
        img = tf.image.decode_jpeg(img, channels=3)
    elif isinstance(image_input, np.ndarray):
        # Numpy array (from web upload)
        img = tf.convert_to_tensor(image_input)
        if len(img.shape) == 2:
            img = tf.image.grayscale_to_rgb(tf.expand_dims(img, -1))
    else:
        # PIL Image or tensor
        img = tf.convert_to_tensor(np.array(image_input))
    
    # Ensure RGB (3 channels)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    
    # Resize to target size
    img = tf.image.resize(img, IMG_SIZE)
    
    # Normalize to [-1, 1] (MobileNetV2 preprocessing)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    
    return img

def preprocess_for_prediction(image_input):
    """
    Preprocess single image for prediction.
    Adds batch dimension.
    
    Args:
        image_input: Path to image file, numpy array, or PIL Image
        
    Returns:
        Preprocessed image tensor (1, 224, 224, 3)
    """
    img = preprocess_image(image_input)
    img = tf.expand_dims(img, 0)
    return img

def preprocess_from_pil(pil_image):
    """
    Preprocess PIL Image (for Django web uploads).
    
    Args:
        pil_image: PIL Image object
        
    Returns:
        Preprocessed image tensor (1, 224, 224, 3)
    """
    img_array = np.array(pil_image.convert('RGB'))
    return preprocess_for_prediction(img_array)
