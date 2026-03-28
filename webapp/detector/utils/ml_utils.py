"""
ML utilities for food product authenticity detection.
Uses MobileNetV2 with optimal threshold loaded from threshold_config.json.
"""

from typing import Dict, Tuple, Union
from datetime import datetime
from pathlib import Path
import json
import logging
import sys

import cv2
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# ML PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────

class MLPredictor:
    """
    Production ML predictor aligned with the training pipeline.
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
        """Load model with FocalLoss custom object support."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        custom_objects = {}
        sys_path_backup = sys.path.copy()
        try:
            model_pipeline_path = self.model_path.parent.parent / "model_pipeline"
            if model_pipeline_path.exists():
                sys.path.insert(0, str(model_pipeline_path))
                from model import FocalLoss  # noqa: PLC0415
                custom_objects = {"FocalLoss": FocalLoss}
                logger.info("FocalLoss loaded from model_pipeline")
            else:
                logger.warning("model_pipeline not found — loading without FocalLoss")
        except ImportError:
            logger.warning("Could not import FocalLoss — loading without custom objects")
        finally:
            sys.path = sys_path_backup

        self.model = tf.keras.models.load_model(
            str(self.model_path),
            custom_objects=custom_objects or None,
        )
        logger.info("ML model loaded successfully")

    def preprocess_image(self, image_data: Union[bytes, np.ndarray]) -> np.ndarray:
        """
        Preprocess image exactly as during training:
        BGR→RGB, resize to 224×224, mobilenet_v2.preprocess_input.
        """
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        if isinstance(image_data, bytes):
            arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            img = image_data

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.TARGET_SIZE)
        img = img.astype(np.float32)
        img = preprocess_input(img)
        return np.expand_dims(img, axis=0)

    def predict_single(self, image_data: Union[bytes, np.ndarray]) -> Tuple[str, float]:
        """
        Predict REAL/FAKE for a single image using the optimal threshold.

        Returns:
            (label, confidence) where confidence is the probability of the
            predicted class (always in [0, 1]).
        """
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


# ──────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ──────────────────────────────────────────────────────────────────────────────

_predictor: MLPredictor | None = None


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

    Args:
        images:     {view_type: raw_image_bytes}
        brand_name: User-supplied brand name (reserved for future OCR matching)

    Returns:
        Dict with keys: final_status, final_score, component_scores,
                        detailed_analysis, failure_reasons, processing_time
    """
    start = datetime.now()
    predictor = get_ml_predictor()

    real_votes = 0
    fake_votes = 0
    confidences: list[float] = []

    for view_type, image_data in images.items():
        label, confidence = predictor.predict_single(image_data)
        confidences.append(confidence)
        if label == "REAL":
            real_votes += 1
        else:
            fake_votes += 1

    # Ties go to FAKE — safer for a counterfeit-detection system
    final_status = "Real" if real_votes > fake_votes else "Fake"
    final_score = round(float(np.mean(confidences)) * 100, 2)

    return {
        "final_status": final_status,
        "final_score": final_score,
        "component_scores": {
            "barcode_score": 0,
            "logo_score": 0,
            "ocr_score": 0,
            "packaging_score": final_score,  # ML score mapped to packaging
        },
        "detailed_analysis": {},
        "failure_reasons": [],
        "processing_time": (datetime.now() - start).total_seconds(),
    }
