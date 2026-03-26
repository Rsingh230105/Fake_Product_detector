"""
Single image prediction interface.
Production-ready inference with optimal threshold.
Implements PHASE 10: Inference Alignment
"""
import tensorflow as tf
import numpy as np
import os
import json
from config import MODEL_PATH, CLASS_NAMES


class FoodDetector:
    """Production-ready food detector with optimal threshold."""
    
    def __init__(self, model_path=MODEL_PATH):
        """Load trained model and optimal threshold."""
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = CLASS_NAMES
        self.threshold = self._load_optimal_threshold(model_path)
    
    def _load_optimal_threshold(self, model_path):
        """
        Load optimal threshold from evaluation results.
        Falls back to 0.5 if not found, but warns.
        """
        threshold_file = os.path.join(os.path.dirname(model_path), 'threshold_config.json')
        
        if os.path.exists(threshold_file):
            try:
                with open(threshold_file, 'r') as f:
                    config = json.load(f)
                    threshold = config.get('OPTIMAL_THRESHOLD', 0.5)
                    print(f"[THRESHOLD] Loaded optimal threshold from evaluation: {threshold:.4f}")
                    return threshold
            except Exception as e:
                print(f"[WARNING] Could not load threshold config: {e}")
                print(f"[FALLBACK] Using default threshold: 0.50")
                return 0.5
        else:
            print(f"[INFO] Threshold config not found at: {threshold_file}")
            print(f"[FALLBACK] Using default threshold: 0.50")
            print(f"[RECOMMENDED] Run evaluate.py to find optimal threshold")
            return 0.5
    
    def predict(self, image_path):
        """
        Predict if food product is fake or real.
        Uses optimal threshold from evaluation.
        Applies consistent preprocessing.
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict with prediction, confidence, and class
        """
        # Load and preprocess image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        
        # CRITICAL: Use same preprocessing as training  - MobileNetV2 [-1, 1]
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        img = tf.expand_dims(img, 0)
        
        # Predict
        raw_prediction = self.model.predict(img, verbose=0)[0][0]
        
        # Apply optimal threshold (not hardcoded 0.5)
        predicted_class = 'real' if raw_prediction > self.threshold else 'fake'
        confidence = raw_prediction if raw_prediction > self.threshold else 1 - raw_prediction
        
        return {
            'prediction': predicted_class,
            'confidence': float(confidence),
            'raw_score': float(raw_prediction),
            'threshold_used': float(self.threshold),
            'is_fake': predicted_class == 'fake',
            'is_real': predicted_class == 'real'
        }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print(f"Loading model: {MODEL_PATH}")
    detector = FoodDetector()
    
    print(f"Analyzing: {image_path}")
    result = detector.predict(image_path)
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Raw Score: {result['raw_score']:.4f}")
    print(f"Threshold: {result['threshold_used']:.4f}")
    print("=" * 60)
