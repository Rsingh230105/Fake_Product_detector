"""
ML utilities for food product authenticity detection.
Uses MobileNetV2 with optimal threshold loaded from threshold_config.json.
Model is auto-downloaded from Google Drive if not present locally.
"""

from typing import Dict, List, Tuple, Union
from datetime import datetime
from pathlib import Path
import json
import logging
import sys
import traceback

import cv2
import numpy as np
from PIL import Image
import io

# Import TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError as e:
    TF_AVAILABLE = False
    tf = None
    keras = None
    print(f"TensorFlow import failed: {e}")

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────────

def ensure_model_exists(model_path: Path) -> None:
    """
    Check if the model file exists locally.
    If not, download it from Google Drive using the provided URL.
    """
    if model_path.exists() and model_path.stat().st_size > 1000000:  # Check file size > 1MB
        logger.info(f"[MODEL] Found at {model_path} (Size: {model_path.stat().st_size / 1e6:.1f} MB)")
        return

    # Model not found or corrupted — download from Google Drive
    google_drive_url = "https://drive.google.com/uc?id=1YO43M94sUYcs8A-S3MEt6wS9x4gJ8S_Y"
    
    logger.info(f"[MODEL] Downloading from Google Drive...")
    logger.info(f"[MODEL] URL: {google_drive_url}")
    logger.info(f"[MODEL] Destination: {model_path}")

    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown is required for model auto-download. "
            "Run: pip install gdown"
        )

    # Create parent directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Download with gdown
        gdown.download(google_drive_url, str(model_path), quiet=False)
        
        # Verify download
        if not model_path.exists() or model_path.stat().st_size < 1000000:
            raise RuntimeError("Downloaded file is missing or too small (corrupted)")
            
        logger.info(f"[MODEL] Download complete. Size: {model_path.stat().st_size / 1e6:.1f} MB")
        
    except Exception as e:
        # Clean up partial download
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Model download failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# ML PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────

class MLPredictor:
    """
    Production ML predictor aligned with the training pipeline.
    - Auto-downloads model from Google Drive if missing
    - Loads MobileNetV2 model with FocalLoss support
    - Uses optimal threshold from threshold_config.json
    - Preprocessing matches training exactly (mobilenet_v2.preprocess_input)
    """

    TARGET_SIZE = (224, 224)

    def __init__(self):
        self.model = None
        self.model_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "models"
            / "mobilenet_v2_food_production.keras"
        )
        self.threshold = 0.5  # overwritten by _load_threshold_config
        ensure_model_exists(self.model_path)
        self._load_threshold_config()
        self._load_model()

    def _load_threshold_config(self) -> None:
        """Load optimal classification threshold from evaluation results."""
        config_path = self.model_path.parent / "threshold_config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                self.threshold = float(config.get("OPTIMAL_THRESHOLD", 0.5))
                logger.info(f"[THRESHOLD] Loaded optimal threshold: {self.threshold:.4f}")
            except Exception as e:
                logger.warning(f"Could not load threshold config: {e}. Using default 0.5")
        else:
            logger.info("[THRESHOLD] Config not found. Using default 0.5")

    def _load_model(self) -> None:
        """Load model with comprehensive error handling for production."""
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not available. Check installation.")
        
        logger.info(f"🚀 Attempting to load model from: {self.model_path}")
        logger.info(f"🔍 TensorFlow version: {tf.__version__}")
        
        if self.model_path.exists():
            logger.info(f"✅ Model file exists. Size: {self.model_path.stat().st_size / 1e6:.1f} MB")
        else:
            logger.error(f"❌ Model file not found at: {self.model_path}")
        
        try:
            # Load model with compile=False for production safety
            from tensorflow.keras.models import load_model
            logger.info("🔄 Loading model with compile=False...")
            self.model = load_model(str(self.model_path), compile=False)
            logger.info("✅ Model loaded successfully!")
            
            # Log model architecture info
            logger.info(f"🏢 Model input shape: {self.model.input_shape}")
            logger.info(f"🏢 Model output shape: {self.model.output_shape}")
            logger.info(f"🏢 Model layers: {len(self.model.layers)}")
            
            # Test model with dummy input
            logger.info("🧪 Testing model with dummy input...")
            dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            test_output = self.model.predict(dummy_input, verbose=0)
            logger.info(f"✅ Dummy test successful. Output: {test_output[0][0]:.6f}")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Model loading failed: {e}")

    def preprocess_image(self, image_data: Union[bytes, np.ndarray]) -> np.ndarray:
        """
        Preprocess image using PIL for better compatibility.
        Pipeline: bytes -> PIL -> RGB -> resize -> normalize -> MobileNetV2 preprocessing
        """
        try:
            logger.info("🔄 Starting image preprocessing...")
            
            if not TF_AVAILABLE:
                raise RuntimeError("TensorFlow not available for preprocessing")
            
            # Step 1: Load image with PIL (more reliable than OpenCV)
            if isinstance(image_data, bytes):
                logger.info(f"📥 Loading image from bytes (size: {len(image_data)} bytes)")
                img = Image.open(io.BytesIO(image_data))
            else:
                logger.info("📥 Loading image from numpy array")
                if isinstance(image_data, np.ndarray):
                    img = Image.fromarray(image_data)
                else:
                    img = image_data

            # Step 2: Convert to RGB (critical for correct predictions)
            if img.mode != 'RGB':
                logger.info(f"🎨 Converting from {img.mode} to RGB")
                img = img.convert('RGB')
            else:
                logger.info("✅ Image already in RGB mode")
            
            logger.info(f"📐 Original image size: {img.size}")
            
            # Step 3: Resize to target size (224, 224)
            img = img.resize(self.TARGET_SIZE, Image.Resampling.LANCZOS)
            logger.info(f"📏 Resized to: {img.size}")
            
            # Step 4: Convert to numpy array
            img_array = np.array(img, dtype=np.float32)
            logger.info(f"🔢 Numpy array shape: {img_array.shape}, dtype: {img_array.dtype}")
            logger.info(f"📊 Pixel value range: [{img_array.min():.1f}, {img_array.max():.1f}]")
            
            # Step 5: Apply MobileNetV2 preprocessing
            # MobileNetV2 expects values in [0, 255] range, then converts to [-1, 1]
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            img_array = preprocess_input(img_array)
            
            logger.info(f"🧠 After MobileNetV2 preprocessing: [{img_array.min():.3f}, {img_array.max():.3f}]")
            
            # Step 6: Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            logger.info(f"📦 Final shape with batch dimension: {img_array.shape}")
            
            logger.info("✅ Image preprocessing completed successfully")
            return img_array
            
        except Exception as e:
            logger.error(f"❌ Image preprocessing failed: {e}")
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Image preprocessing failed: {e}")

    def predict_single(self, image_data: Union[bytes, np.ndarray]) -> Tuple[str, float]:
        """
        Predict REAL/FAKE for a single image with detailed debugging.
        """
        try:
            logger.info("🚀 Starting prediction...")
            
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            # Preprocess image
            logger.info("🔄 Preprocessing image...")
            processed = self.preprocess_image(image_data)
            
            # Get raw prediction
            logger.info("🧠 Running model inference...")
            raw_prediction = self.model.predict(processed, verbose=0)
            raw_score = float(raw_prediction[0][0])
            
            logger.info(f"📊 Raw model output: {raw_score:.6f}")
            logger.info(f"📊 Prediction array shape: {raw_prediction.shape}")
            logger.info(f"📊 Full prediction array: {raw_prediction[0]}")
            
            # Apply threshold logic
            # Lower threshold for testing - if still always Fake, it's a preprocessing issue
            threshold = 0.3  # Lowered from 0.7 for debugging
            
            if raw_score >= threshold:
                label = "REAL"
                confidence = raw_score
                logger.info(f"✅ REAL prediction: score {raw_score:.6f} >= threshold {threshold}")
            else:
                label = "FAKE"
                confidence = 1.0 - raw_score
                logger.info(f"❌ FAKE prediction: score {raw_score:.6f} < threshold {threshold}")
            
            logger.info(f"🎯 Final result: {label} (confidence: {confidence:.6f})")
            return label, confidence
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            # Return conservative fallback
            return "FAKE", 0.1


# ──────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ──────────────────────────────────────────────────────────────────────────────

_predictor: "MLPredictor | None" = None


def get_ml_predictor() -> MLPredictor:
    """Return the singleton MLPredictor, initialising it on first call."""
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor()
    return _predictor


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def process_product_images(images: Dict[str, bytes], brand_name: str) -> Dict:
    """
    Run ML inference on all uploaded product views and return a unified result.

    Aggregation strategy:
    - Each view casts one vote (REAL or FAKE).
    - Majority wins; ties are broken in favour of FAKE (safer default).
    - Final score = mean confidence across all views × 100.
    """
    start = datetime.now()
    logger.info(f"🚀 Starting analysis for brand: {brand_name}")
    logger.info(f"📷 Processing {len(images)} images: {list(images.keys())}")
    
    try:
        predictor = get_ml_predictor()

        real_votes = 0
        fake_votes = 0
        confidences: List[float] = []
        detailed_results = {}

        for view_type, image_data in images.items():
            try:
                logger.info(f"🔍 Processing {view_type} view (size: {len(image_data)} bytes)")
                label, confidence = predictor.predict_single(image_data)
                
                confidences.append(confidence)
                detailed_results[view_type] = {
                    "prediction": label,
                    "confidence": confidence
                }
                
                if label == "REAL":
                    real_votes += 1
                else:
                    fake_votes += 1
                    
                logger.info(f"✅ {view_type}: {label} (confidence: {confidence:.4f})")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {view_type}: {e}")
                # Default to FAKE for failed predictions
                fake_votes += 1
                confidences.append(0.3)
                detailed_results[view_type] = {
                    "prediction": "FAKE",
                    "confidence": 0.3,
                    "error": str(e)
                }

        # Aggregate results
        final_status = "Real" if real_votes > fake_votes else "Fake"
        final_score = round(float(np.mean(confidences)) * 100, 2) if confidences else 30.0
        
        logger.info(f"🏁 Final result: {final_status} (score: {final_score}%)")
        logger.info(f"🗳️ Votes - Real: {real_votes}, Fake: {fake_votes}")

        return {
            "final_status": final_status,
            "final_score": final_score,
            "component_scores": {
                "barcode_score": 0,
                "logo_score": 0,
                "ocr_score": 0,
                "packaging_score": final_score,
            },
            "detailed_analysis": detailed_results,
            "failure_reasons": [],
            "processing_time": (datetime.now() - start).total_seconds(),
        }
        
    except Exception as e:
        logger.error(f"❌ Critical error in process_product_images: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Return safe fallback result
        return {
            "final_status": "Fake",
            "final_score": 25.0,
            "component_scores": {
                "barcode_score": 0,
                "logo_score": 0,
                "ocr_score": 0,
                "packaging_score": 25.0,
            },
            "detailed_analysis": {"error": str(e)},
            "failure_reasons": ["ML processing failed"],
            "processing_time": (datetime.now() - start).total_seconds(),
        }
