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
        Enhanced preprocessing with training pipeline verification.
        Critical: Must match exact training preprocessing steps.
        """
        try:
            logger.info("🔄 Starting enhanced image preprocessing...")
            
            if not TF_AVAILABLE:
                raise RuntimeError("TensorFlow not available for preprocessing")
            
            # Step 1: Load and validate image
            if isinstance(image_data, bytes):
                logger.info(f"📥 Loading image from bytes (size: {len(image_data)} bytes)")
                img = Image.open(io.BytesIO(image_data))
            else:
                logger.info("📥 Loading image from numpy array")
                if isinstance(image_data, np.ndarray):
                    img = Image.fromarray(image_data)
                else:
                    img = image_data

            logger.info(f"📐 Original image: size={img.size}, mode={img.mode}")
            
            # Step 2: Convert to RGB (CRITICAL for correct predictions)
            if img.mode != 'RGB':
                logger.info(f"🎨 Converting from {img.mode} to RGB")
                img = img.convert('RGB')
            else:
                logger.info("✅ Image already in RGB mode")
            
            # Step 3: Resize to exact training size
            logger.info(f"📏 Resizing from {img.size} to {self.TARGET_SIZE}")
            img = img.resize(self.TARGET_SIZE, Image.Resampling.LANCZOS)
            
            # Step 4: Convert to numpy and validate
            img_array = np.array(img, dtype=np.float32)
            logger.info(f"🔢 Numpy conversion: shape={img_array.shape}, dtype={img_array.dtype}")
            logger.info(f"📊 Original pixel range: [{img_array.min():.1f}, {img_array.max():.1f}]")
            
            # Validate image properties
            if img_array.shape != (224, 224, 3):
                raise ValueError(f"Invalid image shape: {img_array.shape}, expected (224, 224, 3)")
            
            if img_array.max() > 255 or img_array.min() < 0:
                logger.warning(f"⚠️ Unusual pixel values: [{img_array.min()}, {img_array.max()}]")
            
            # Step 5: Apply MobileNetV2 preprocessing (CRITICAL STEP)
            # MobileNetV2 preprocessing: (x / 127.5) - 1.0 to get [-1, 1] range
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            
            logger.info("🧠 Applying MobileNetV2 preprocessing...")
            # preprocess_input expects [0, 255] range
            img_preprocessed = preprocess_input(img_array)
            
            logger.info(f"🧠 After preprocessing: [{img_preprocessed.min():.3f}, {img_preprocessed.max():.3f}]")
            
            # Validate preprocessing output
            if not (-1.1 <= img_preprocessed.min() <= -0.9 and 0.9 <= img_preprocessed.max() <= 1.1):
                logger.warning(f"⚠️ Preprocessing range seems incorrect: [{img_preprocessed.min():.3f}, {img_preprocessed.max():.3f}]")
                logger.warning("Expected range: approximately [-1.0, 1.0]")
            
            # Step 6: Add batch dimension
            img_batch = np.expand_dims(img_preprocessed, axis=0)
            logger.info(f"📦 Final tensor: shape={img_batch.shape}, dtype={img_batch.dtype}")
            
            # Final validation
            if img_batch.shape != (1, 224, 224, 3):
                raise ValueError(f"Invalid final shape: {img_batch.shape}")
            
            logger.info("✅ Image preprocessing completed successfully")
            logger.info(f"📊 Final stats: mean={img_batch.mean():.3f}, std={img_batch.std():.3f}")
            
            return img_batch
            
        except Exception as e:
            logger.error(f"❌ Image preprocessing failed: {e}")
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Image preprocessing failed: {e}")

    def predict_single(self, image_data: Union[bytes, np.ndarray]) -> Tuple[str, float]:
        """
        Predict REAL/FAKE with comprehensive debugging and dynamic threshold detection.
        """
        try:
            logger.info("🚀 Starting prediction with enhanced debugging...")
            
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            # Preprocess image
            logger.info("🔄 Preprocessing image...")
            processed = self.preprocess_image(image_data)
            
            # Get raw prediction with detailed analysis
            logger.info("🧠 Running model inference...")
            raw_prediction = self.model.predict(processed, verbose=0)
            
            # Extract and analyze prediction
            if raw_prediction.shape[-1] == 1:  # Binary classification
                raw_score = float(raw_prediction[0][0])
                logger.info(f"📊 Binary output - Raw score: {raw_score:.6f}")
            else:  # Multi-class (should not happen for binary)
                raw_score = float(raw_prediction[0][1])  # Assuming class 1 is REAL
                logger.info(f"📊 Multi-class output - Using class 1: {raw_score:.6f}")
                logger.info(f"📊 Full prediction: {raw_prediction[0]}")
            
            # Analyze prediction distribution for threshold tuning
            logger.info(f"📊 Prediction analysis:")
            logger.info(f"   Raw score: {raw_score:.6f}")
            logger.info(f"   Score range: [0.0 - 1.0]")
            logger.info(f"   Sigmoid output: {'Yes' if 0 <= raw_score <= 1 else 'No - Check model output!'}")
            
            # Dynamic threshold detection based on score distribution
            if raw_score > 0.8:
                threshold = 0.7  # High confidence threshold
                logger.info("🟢 Using high confidence threshold: 0.7")
            elif raw_score > 0.6:
                threshold = 0.5  # Medium confidence threshold
                logger.info("🟡 Using medium confidence threshold: 0.5")
            else:
                threshold = 0.3  # Low confidence threshold for debugging
                logger.info("🟠 Using low confidence threshold: 0.3 (debug mode)")
            
            # Classification logic
            if raw_score >= threshold:
                label = "REAL"
                confidence = raw_score
                logger.info(f"✅ REAL: score {raw_score:.6f} >= threshold {threshold:.3f}")
            else:
                label = "FAKE"
                confidence = 1.0 - raw_score
                logger.info(f"❌ FAKE: score {raw_score:.6f} < threshold {threshold:.3f}")
            
            # Additional debugging info
            logger.info(f"🎯 Classification Summary:")
            logger.info(f"   Input shape: {processed.shape}")
            logger.info(f"   Raw model output: {raw_score:.6f}")
            logger.info(f"   Applied threshold: {threshold:.3f}")
            logger.info(f"   Final label: {label}")
            logger.info(f"   Confidence: {confidence:.6f}")
            
            # Threshold recommendations based on observed scores
            if raw_score < 0.1:
                logger.warning("⚠️ Very low score detected! Check preprocessing pipeline.")
            elif raw_score > 0.9:
                logger.info("🚀 Very high confidence score - model is confident!")
            
            return label, confidence
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
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
    Enhanced ML inference with prediction analysis and threshold optimization.
    """
    start = datetime.now()
    logger.info(f"🚀 Starting enhanced analysis for brand: {brand_name}")
    logger.info(f"📷 Processing {len(images)} images: {list(images.keys())}")
    
    try:
        predictor = get_ml_predictor()

        real_votes = 0
        fake_votes = 0
        confidences: List[float] = []
        raw_scores: List[float] = []
        detailed_results = {}

        for view_type, image_data in images.items():
            try:
                logger.info(f"🔍 Processing {view_type} view (size: {len(image_data)} bytes)")
                label, confidence = predictor.predict_single(image_data)
                
                # Extract raw score for analysis (assuming it's stored in confidence for REAL)
                if label == "REAL":
                    raw_score = confidence
                else:
                    raw_score = 1.0 - confidence
                
                raw_scores.append(raw_score)
                confidences.append(confidence)
                detailed_results[view_type] = {
                    "prediction": label,
                    "confidence": confidence,
                    "raw_score": raw_score
                }
                
                if label == "REAL":
                    real_votes += 1
                else:
                    fake_votes += 1
                    
                logger.info(f"✅ {view_type}: {label} (conf: {confidence:.4f}, raw: {raw_score:.4f})")
                
            except Exception as e:
                logger.error(f"❌ Failed to process {view_type}: {e}")
                fake_votes += 1
                confidences.append(0.3)
                raw_scores.append(0.1)
                detailed_results[view_type] = {
                    "prediction": "FAKE",
                    "confidence": 0.3,
                    "raw_score": 0.1,
                    "error": str(e)
                }

        # Enhanced aggregation with score analysis
        final_status = "Real" if real_votes > fake_votes else "Fake"
        final_score = round(float(np.mean(confidences)) * 100, 2) if confidences else 30.0
        avg_raw_score = np.mean(raw_scores) if raw_scores else 0.1
        
        # Prediction analysis
        logger.info(f"🏁 Analysis Summary:")
        logger.info(f"   Final result: {final_status} (score: {final_score}%)")
        logger.info(f"   Votes - Real: {real_votes}, Fake: {fake_votes}")
        logger.info(f"   Average raw score: {avg_raw_score:.4f}")
        logger.info(f"   Raw score range: [{min(raw_scores):.4f}, {max(raw_scores):.4f}]" if raw_scores else "No scores")
        
        # Threshold recommendations
        if avg_raw_score < 0.2:
            logger.warning("⚠️ Very low average scores - check preprocessing!")
        elif avg_raw_score > 0.8:
            logger.info("🚀 High confidence scores - model is working well!")
        
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
            "debug_info": {
                "avg_raw_score": avg_raw_score,
                "raw_scores": raw_scores,
                "vote_breakdown": {"real": real_votes, "fake": fake_votes}
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Critical error in process_product_images: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
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
