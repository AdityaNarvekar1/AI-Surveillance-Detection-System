"""
model_loader.py
---------------
Loads and caches all AI models used by the surveillance system.

Models:
    - violence_detection.h5  -> Keras 2.15.0 / TF backend (saved by friend)
    - guns_knives.pt         -> YOLOv8  (weapon detection)
    - Fire_smoke.pt          -> YOLOv8  (fire/smoke detection)

Compatibility:
    Python 3.12, Windows 10/11, CPU only.
    tensorflow-cpu 2.15/2.16 + TF_USE_LEGACY_KERAS=1 to load Keras 2.x .h5 files.
"""

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---- Force TF to use its built-in legacy Keras 2 instead of Keras 3 ----------
# MUST be set before ANY tensorflow or keras import anywhere in the process.
# This is the key fix for: "model saved with Keras 2.15.0" failing to load
# under TF 2.15+ which ships Keras 3 by default.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ---- Paths -------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

VIOLENCE_MODEL_PATH    = BASE_DIR / "violence_detection.h5"
GUNS_KNIVES_MODEL_PATH = BASE_DIR / "guns_knives.pt"
FIRE_SMOKE_MODEL_PATH  = BASE_DIR / "Fire_smoke.pt"


# ---- Helper: detect input shape ----------------------------------------------

def _probe_input_shape(model) -> tuple:
    """Best-effort detection of model input shape."""
    try:
        shape = model.input_shape
        logger.info("Input shape from model.input_shape: %s", shape)
        return tuple(shape)
    except Exception:
        pass

    try:
        shape = tuple(model.inputs[0].shape.as_list())
        logger.info("Input shape from model.inputs: %s", shape)
        return shape
    except Exception:
        pass

    for h, w in [(224, 224), (128, 128), (64, 64)]:
        try:
            dummy = np.zeros((1, h, w, 3), dtype=np.float32)
            model.predict(dummy, verbose=0)
            logger.info("Input shape probed as (None, %d, %d, 3)", h, w)
            return (None, h, w, 3)
        except Exception:
            pass

    logger.warning("Could not probe input shape; defaulting to (None, 224, 224, 3)")
    return (None, 224, 224, 3)


# ---- Violence model loader ---------------------------------------------------

def load_violence_model():
    """
    Load violence_detection.h5 (saved with Keras 2.15.0 / TF backend).

    Strategy order:
      1. tf.keras with TF_USE_LEGACY_KERAS=1  <- main fix
      2. tf_keras standalone package          <- pip install tf_keras
      3. Reconstruct from JSON config         <- deep fallback
    """
    path = str(VIOLENCE_MODEL_PATH)

    if not VIOLENCE_MODEL_PATH.exists():
        logger.error("violence_detection.h5 not found at: %s", path)
        return None, None

    # Strategy 1: tf.keras with legacy mode forced
    # TF_USE_LEGACY_KERAS=1 (set at module top) makes tf.keras point to the
    # bundled Keras 2 inside TF, bypassing the standalone Keras 3 package.
    try:
        import tensorflow as tf
        logger.info("TensorFlow version: %s", tf.__version__)
        logger.info("Keras version: %s", tf.keras.__version__)
        tf.get_logger().setLevel("ERROR")

        model = tf.keras.models.load_model(path, compile=False)
        input_shape = _probe_input_shape(model)
        logger.info("Strategy 1 success (tf.keras legacy). Shape: %s", input_shape)
        return model, input_shape

    except ImportError:
        logger.error("TensorFlow not installed. Run: pip install tensorflow-cpu>=2.15,<2.17")
        return None, None
    except Exception as e:
        logger.warning("Strategy 1 (tf.keras legacy) failed: %s", e)

    # Strategy 2: tf_keras standalone (pip install tf_keras)
    try:
        import tf_keras
        logger.info("tf_keras version: %s", tf_keras.__version__)
        model = tf_keras.models.load_model(path, compile=False)
        input_shape = _probe_input_shape(model)
        logger.info("Strategy 2 success (tf_keras). Shape: %s", input_shape)
        return model, input_shape
    except ImportError:
        logger.info("tf_keras not installed, skipping. (Run: pip install tf_keras)")
    except Exception as e:
        logger.warning("Strategy 2 (tf_keras) failed: %s", e)

    # Strategy 3: Reconstruct from JSON config + load_weights
    try:
        import tensorflow as tf
        import h5py

        with h5py.File(path, "r") as f:
            model_config = f.attrs.get("model_config", None)
            if model_config is None:
                raise ValueError("No model_config in .h5 file")
            if isinstance(model_config, bytes):
                model_config = model_config.decode("utf-8")

        logger.info("Strategy 3: Reconstructing from JSON config (len=%d)", len(model_config))
        model = tf.keras.models.model_from_json(model_config)
        model.load_weights(path)
        input_shape = _probe_input_shape(model)
        logger.info("Strategy 3 success (config+weights). Shape: %s", input_shape)
        return model, input_shape

    except Exception as e:
        logger.warning("Strategy 3 (reconstruct from config) failed: %s", e)

    logger.error(
        "All strategies failed for violence_detection.h5.\n"
        "Quick fix: pip install tf_keras  then restart the app."
    )
    return None, None


# ---- YOLO loader -------------------------------------------------------------

def load_yolo_model(model_path: Path, device: str = ""):
    """Load a YOLOv8 model. device='' means CPU."""
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        if device:
            model.to(device)
        logger.info("YOLO loaded: %s  device=%s", model_path.name, device or "cpu")
        return model
    except Exception as e:
        logger.error("YOLO load failed (%s): %s", model_path.name, e)
        return None


# ---- Device detection --------------------------------------------------------

def get_device() -> str:
    """Return 'cuda' if a GPU is available, else '' (CPU)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else ""
    except ImportError:
        return ""


# ---- Main factory ------------------------------------------------------------

def load_all_models() -> dict:
    """
    Load all three models. Safe to call even if some models fail.
    YOLO models work independently of the violence model.
    """
    device = get_device()
    logger.info("Device: %s", device or "cpu")

    violence_model, violence_input_shape = load_violence_model()
    guns_knives_model = load_yolo_model(GUNS_KNIVES_MODEL_PATH, device)
    fire_smoke_model  = load_yolo_model(FIRE_SMOKE_MODEL_PATH, device)

    loaded_count = sum([
        violence_model is not None,
        guns_knives_model is not None,
        fire_smoke_model is not None,
    ])

    logger.info("Models loaded: %d / 3", loaded_count)

    return {
        "violence_model":       violence_model,
        "violence_input_shape": violence_input_shape,
        "guns_knives_model":    guns_knives_model,
        "fire_smoke_model":     fire_smoke_model,
        "device":               device,
        "loaded_count":         loaded_count,
    }