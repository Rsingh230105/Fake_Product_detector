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
    If not, download it from the URL defined in ML_MODEL_URL (settings/env).

    Raises:
        RuntimeError: if ML_MODEL_URL is not set and model is missing.
        RuntimeError: if download fails.
    """
    if model_path.exists():
        logger.info(f"[MODEL] Found at {model_path}")
        return

    # Model not found — attempt download
    from django.conf import settings  # imported here to avoid circular import

    url = getattr(settings, "ML_MODEL_URL", None)
    if not url:
        raise RuntimeError(
            f"Model not found at '{model_path}' and ML_MODEL_URL is not set. "
            "Set ML_MODEL_URL in your .env file to enable auto-download."
        )

    logger.info(f"[MODEL] Not found locally. Downloading from Google Drive...")
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
        gdown.download(url, str(model_path), quiet=False, fuzzy=True)
    except Exception as e:
        # Clean up partial download
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Model download failed: {e}")

    if not model_path.exists():
        raise RuntimeError(
            "Download appeared to succeed but model file was not created. "
            "Check that the Google Drive link is publicly accessible."
        )

    logger.info(f"[MODEL] Download complete. Size: {model_path.stat().st_size / 1e6:.1f} MB")


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
        
        logger.info(f"Attempting to load model from: {self.model_path}")
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        errors = []
        
        # Method 1: Try loading with compile=False (safest for production)
        try:
            self.model = keras.models.load_model(
                str(self.model_path),
                compile=False
            )
            logger.info("Model loaded successfully with compile=False")
            return
        except Exception as e:
            errors.append(f"Method 1 (compile=False): {e}")
            logger.warning(f"Method 1 failed: {e}")
        
        # Method 2: Try with custom_objects=None
        try:
            self.model = keras.models.load_model(
                str(self.model_path),
                custom_objects=None
            )
            logger.info("Model loaded successfully with custom_objects=None")
            return
        except Exception as e:
            errors.append(f"Method 2 (custom_objects=None): {e}")
            logger.warning(f"Method 2 failed: {e}")
        
        # Method 3: Try with tf.keras directly
        try:
            self.model = tf.keras.models.load_model(
                str(self.model_path),
                compile=False
            )
            logger.info("Model loaded successfully with tf.keras")
            return
        except Exception as e:
            errors.append(f"Method 3 (tf.keras): {e}")
            logger.warning(f"Method 3 failed: {e}")
        
        # All methods failed
        error_msg = f"All model loading methods failed. Errors: {'; '.join(errors)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(error_msg)

    def preprocess_image(self, image_data: Union[bytes, np.ndarray]) -> np.ndarray:
        """
        Preprocess image exactly as during training:
        BGR→RGB, resize to 224×224, mobilenet_v2.preprocess_input.
        """
        try:
            # Import with error handling
            if not TF_AVAILABLE:
                raise RuntimeError("TensorFlow not available for preprocessing")
            
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

            if isinstance(image_data, bytes):
                arr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                img = image_data

            if img is None:
                raise ValueError("Failed to decode image")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.TARGET_SIZE)
            img = img.astype(np.float32)
            img = preprocess_input(img)
            return np.expand_dims(img, axis=0)
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise RuntimeError(f"Image preprocessing failed: {e}")

    def predict_single(self, image_data: Union[bytes, np.ndarray]) -> Tuple[str, float]:
        """
        Predict REAL/FAKE for a single image using the optimal threshold.

        Returns:
            (label, confidence) where confidence is the probability of the
            predicted class (always in [0, 1]).
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            processed = self.preprocess_image(image_data)
            raw_score = float(self.model.predict(processed, verbose=0)[0][0])

            if raw_score >= self.threshold:
                label = "REAL"
                confidence = raw_score
            else:
                label = "FAKE"
                confidence = 1.0 - raw_score

            logger.debug(
                f"raw={raw_score:.4f} threshold={self.threshold:.4f} "
                f"→ {label} (conf={confidence:.4f})"
            )
            return label, confidence
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return fallback prediction
            return "FAKE", 0.5


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
    
    try:
        predictor = get_ml_predictor()

        real_votes = 0
        fake_votes = 0
        confidences: List[float] = []

        for view_type, image_data in images.items():
            try:
                label, confidence = predictor.predict_single(image_data)
                confidences.append(confidence)
                if label == "REAL":
                    real_votes += 1
                else:
                    fake_votes += 1
            except Exception as e:
                logger.error(f"Failed to process {view_type}: {e}")
                # Default to FAKE for failed predictions
                fake_votes += 1
                confidences.append(0.3)

        # Ties go to FAKE — safer for a counterfeit-detection system
        final_status = "Real" if real_votes > fake_votes else "Fake"
        final_score = round(float(np.mean(confidences)) * 100, 2) if confidences else 30.0

        return {
            "final_status": final_status,
            "final_score": final_score,
            "component_scores": {
                "barcode_score": 0,
                "logo_score": 0,
                "ocr_score": 0,
                "packaging_score": final_score,
            },
            "detailed_analysis": {},
            "failure_reasons": [],
            "processing_time": (datetime.now() - start).total_seconds(),
        }
    except Exception as e:
        logger.error(f"Critical error in process_product_images: {e}")
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
