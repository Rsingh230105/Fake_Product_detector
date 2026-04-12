#!/usr/bin/env python3
"""
Health check script for ML model loading.
Run this to test if the model loads correctly in production.
"""

import os
import sys
import django
from pathlib import Path

# Add the webapp directory to Python path
webapp_dir = Path(__file__).parent / "webapp"
sys.path.insert(0, str(webapp_dir))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'food_detection.settings')
django.setup()

def test_ml_model():
    """Test ML model loading and basic functionality."""
    print("=" * 50)
    print("ML Model Health Check")
    print("=" * 50)
    
    try:
        # Test TensorFlow import
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Test model loading
        from detector.utils.ml_utils import get_ml_predictor
        print("🔄 Loading ML predictor...")
        
        predictor = get_ml_predictor()
        print("✅ ML predictor loaded successfully")
        print(f"   Model path: {predictor.model_path}")
        print(f"   Threshold: {predictor.threshold}")
        
        # Test with dummy data
        import numpy as np
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print("🔄 Testing prediction with dummy image...")
        
        label, confidence = predictor.predict_single(dummy_image)
        print(f"✅ Prediction successful: {label} (confidence: {confidence:.4f})")
        
        print("=" * 50)
        print("✅ All tests passed! ML model is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print("=" * 50)
        print("❌ Health check failed!")
        return False

if __name__ == "__main__":
    success = test_ml_model()
    sys.exit(0 if success else 1)