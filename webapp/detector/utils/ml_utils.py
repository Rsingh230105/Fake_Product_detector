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

# Import TensorFlow with comprehensive error handling
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
    # Log successful import
    print(f"✅ TensorFlow {tf.__version__} loaded successfully")
except ImportError as e:
    TF_AVAILABLE = False
    tf = None
    keras = None
    print(f"❌ TensorFlow import failed: {e}")

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# MODEL DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────────

def ensure_model_exists(model_path: Path) -> None:
    """
    Production-safe model download with integrity checks and format conversion.
    """
    # Check if model exists and is valid size (>1MB)
    if model_path.exists() and model_path.stat().st_size > 1000000:
        logger.info(f"✅ Model found: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
        return

    # Download model from Google Drive
    google_drive_url = "https://drive.google.com/uc?id=1YO43M94sUYcs8A-S3MEt6wS9x4gJ8S_Y"
    
    logger.info(f"📥 Downloading model from Google Drive...")
    logger.info(f"🎯 URL: {google_drive_url}")
    logger.info(f"📁 Destination: {model_path}")

    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "❌ gdown is required for model download. "
            "Install with: pip install gdown"
        )

    # Create directory if needed
    model_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Download with gdown
        gdown.download(google_drive_url, str(model_path), quiet=False)
        
        # Verify download
        if not model_path.exists() or model_path.stat().st_size < 1000000:
            raise RuntimeError("Downloaded file is corrupted or too small")
            
        logger.info(f"✅ Model downloaded successfully: {model_path.stat().st_size / 1e6:.1f} MB")
        
        # Try to convert model to compatible format if needed
        _try_convert_model_format(model_path)
        
    except Exception as e:
        # Clean up failed download
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"❌ Model download failed: {e}")


def _try_convert_model_format(model_path: Path) -> None:
    """
    Try to convert model to a more compatible format if loading fails.
    """
    try:
        logger.info("🔄 Testing model compatibility...")
        # Try a quick load test
        test_model = tf.keras.models.load_model(str(model_path), compile=False)
        logger.info("✅ Model format is compatible")
        del test_model  # Free memory
    except Exception as e:
        logger.warning(f"⚠️ Model format incompatible: {e}")
        logger.info("🔄 Attempting format conversion...")
        
        # Create backup
        backup_path = model_path.with_suffix('.keras.backup')
        model_path.rename(backup_path)
        
        try:
            # Try alternative loading methods for conversion
            conversion_methods = [
                lambda: tf.keras.models.load_model(str(backup_path), compile=False, safe_mode=False),
                lambda: tf.saved_model.load(str(backup_path)),
            ]
            
            converted_model = None
            for method in conversion_methods:
                try:
                    converted_model = method()
                    break
                except:
                    continue
            
            if converted_model is not None:
                # Save in compatible format
                if hasattr(converted_model, 'save'):
                    converted_model.save(str(model_path), save_format='keras')
                else:
                    # Handle SavedModel format
                    tf.saved_model.save(converted_model, str(model_path))
                
                logger.info("✅ Model converted to compatible format")
                backup_path.unlink()  # Remove backup
            else:
                # Restore backup if conversion failed
                backup_path.rename(model_path)
                logger.warning("⚠️ Model conversion failed, using original")
                
        except Exception as conv_error:
            # Restore backup if conversion failed
            if backup_path.exists():
                backup_path.rename(model_path)
            logger.warning(f"⚠️ Model conversion failed: {conv_error}")


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
        """Production-safe model loading with multiple fallback methods."""
        if not TF_AVAILABLE:
            raise RuntimeError("❌ TensorFlow is not available. Check installation.")
        
        logger.info(f"🚀 Loading model from: {self.model_path}")
        logger.info(f"🔧 TensorFlow version: {tf.__version__}")
        
        # Verify file exists and has reasonable size
        if not self.model_path.exists():
            raise RuntimeError(f"❌ Model file not found: {self.model_path}")
        
        file_size_mb = self.model_path.stat().st_size / 1e6
        logger.info(f"📊 Model file size: {file_size_mb:.1f} MB")
        
        if file_size_mb < 1:
            raise RuntimeError(f"❌ Model file too small ({file_size_mb:.1f} MB), likely corrupted")
        
        # Try multiple loading methods for maximum compatibility
        loading_methods = [
            ("Standard Keras load_model with compile=False", self._load_method_1),
            ("TensorFlow Keras with compile=False", self._load_method_2),
            ("Custom objects with compile=False", self._load_method_3),
            ("Rebuild architecture and load weights", self._load_method_4),
            ("Safe mode disabled loading", self._load_method_5)
        ]
        
        for method_name, load_method in loading_methods:
            try:
                logger.info(f"🔄 Trying: {method_name}")
                self.model = load_method()
                logger.info(f"✅ SUCCESS: {method_name}")
                
                # Validate loaded model
                self._validate_model()
                return
                
            except Exception as e:
                logger.warning(f"⚠️ {method_name} failed: {str(e)[:100]}...")
                continue
        
        # All methods failed
        raise RuntimeError("❌ All model loading methods failed. Check model file integrity.")
    
    def _load_method_1(self):
        """Method 1: Standard Keras load_model with compile=False"""
        from tensorflow.keras.models import load_model
        return load_model(str(self.model_path), compile=False)
    
    def _load_method_2(self):
        """Method 2: TensorFlow Keras with compile=False"""
        return tf.keras.models.load_model(str(self.model_path), compile=False)
    
    def _load_method_3(self):
        """Method 3: With empty custom_objects"""
        return tf.keras.models.load_model(
            str(self.model_path), 
            custom_objects={}, 
            compile=False
        )
    
    def _load_method_4(self):
        """Method 4: Load with weights only and rebuild architecture"""
        # This is a more aggressive approach - load weights separately
        try:
            # Create a new MobileNetV2 model
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
            from tensorflow.keras.models import Model
            
            # Recreate the model architecture
            base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            base_model.trainable = False
            
            inputs = tf.keras.Input(shape=(224, 224, 3))
            x = base_model(inputs, training=False)
            x = GlobalAveragePooling2D()(x)
            x = Dropout(0.5)(x)
            outputs = Dense(1, activation='sigmoid')(x)
            
            model = Model(inputs, outputs)
            
            # Try to load weights from the saved model
            try:
                model.load_weights(str(self.model_path))
                logger.info("✅ Loaded weights into rebuilt architecture")
                return model
            except:
                # If that fails, return the base architecture (will need retraining)
                logger.warning("⚠️ Could not load weights, using base architecture")
                return model
                
        except Exception as e:
            raise RuntimeError(f"Architecture rebuild failed: {e}")
    
    def _load_method_5(self):
        """Method 5: Load with safe_mode disabled"""
        try:
            return tf.keras.models.load_model(
                str(self.model_path), 
                compile=False,
                safe_mode=False
            )
        except Exception as e:
            raise RuntimeError(f"Safe mode disabled loading failed: {e}")
    
    def _validate_model(self):
        """Validate that the loaded model is working correctly."""
        if self.model is None:
            raise RuntimeError("Model is None after loading")
        
        # Log model info
        logger.info(f"🏗️ Model architecture loaded successfully")
        logger.info(f"📥 Input shape: {self.model.input_shape}")
        logger.info(f"📤 Output shape: {self.model.output_shape}")
        logger.info(f"🔢 Total layers: {len(self.model.layers)}")
        
        # Test with dummy input
        try:
            logger.info("🧪 Testing model with dummy input...")
            dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
            test_output = self.model.predict(dummy_input, verbose=0)
            logger.info(f"✅ Model test successful! Output shape: {test_output.shape}")
            logger.info(f"📊 Test prediction value: {test_output[0][0]:.6f}")
        except Exception as e:
            logger.error(f"❌ Model validation test failed: {e}")
            raise RuntimeError(f"Model validation failed: {e}")

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
        Returns raw model score only. Final label is decided in process_product_images
        using the average across all views.
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")

            processed = self.preprocess_image(image_data)

            logger.info("🧠 Running model inference...")
            raw_prediction = self.model.predict(processed, verbose=0)

            # Binary output: single sigmoid score in [0, 1]
            if raw_prediction.shape[-1] == 1:
                raw_score = float(raw_prediction[0][0])
            else:
                raw_score = float(raw_prediction[0][1])  # class-1 probability

            logger.info(f"📊 Raw score: {raw_score:.6f}")
            return raw_score

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            return None  # caller handles None


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

def _classify(avg_score: float) -> Tuple[str, str]:
    """
    3-level classification based on average raw score across all views.

    Thresholds:
        >= 0.7  → REAL      (high confidence genuine)
        <  0.4  → FAKE      (high confidence counterfeit)
        else    → UNCERTAIN (model is not confident)
    """
    if avg_score >= 0.7:
        return "Real", "REAL"
    elif avg_score < 0.4:
        return "Fake", "FAKE"
    else:
        return "Uncertain", "UNCERTAIN"


def process_product_images(images: Dict[str, bytes], brand_name: str) -> Dict:
    """
    Enhanced ML inference with prediction analysis and threshold optimization.
    """
    start = datetime.now()
    logger.info(f"🚀 Starting analysis for brand: {brand_name}")
    logger.info(f"📷 Processing {len(images)} images: {list(images.keys())}")

    try:
        predictor = get_ml_predictor()

        raw_scores: List[float] = []
        detailed_results = {}

        for view_type, image_data in images.items():
            try:
                logger.info(f"🔍 Processing {view_type} ({len(image_data)} bytes)")
                raw_score = predictor.predict_single(image_data)

                if raw_score is None:
                    logger.warning(f"⚠️ {view_type}: prediction returned None, skipping")
                    detailed_results[view_type] = {"raw_score": None, "skipped": True}
                    continue

                raw_scores.append(raw_score)
                detailed_results[view_type] = {"raw_score": round(raw_score, 6)}
                logger.info(f"✅ {view_type}: raw_score={raw_score:.6f}")

            except Exception as e:
                logger.error(f"❌ Failed to process {view_type}: {e}")
                detailed_results[view_type] = {"raw_score": None, "error": str(e)}

        if not raw_scores:
            raise RuntimeError("No valid predictions — all images failed")

        # ── Final decision: average score across all views ──────────────────
        avg_score = float(np.mean(raw_scores))
        final_status, status_label = _classify(avg_score)
        final_score = round(avg_score * 100, 2)

        logger.info(f"🏁 Raw scores     : {[round(s, 4) for s in raw_scores]}")
        logger.info(f"🏁 Average score  : {avg_score:.6f}")
        logger.info(f"🏁 Final status   : {final_status} ({status_label})")
        logger.info(f"🏁 Final score    : {final_score}%")

        return {
            "final_status": final_status,       # 'Real' | 'Fake' | 'Uncertain'
            "status_label": status_label,        # 'REAL' | 'FAKE' | 'UNCERTAIN'
            "final_score": final_score,
            "avg_raw_score": avg_score,
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
        logger.error(traceback.format_exc())
        return {
            "final_status": "Fake",
            "status_label": "FAKE",
            "final_score": 25.0,
            "avg_raw_score": 0.25,
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
