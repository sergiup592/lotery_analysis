import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

_TF_CONFIGURED = False
_XGB_DEVICE_PARAMS = None


def configure_tensorflow() -> bool:
    """Configure TensorFlow to use GPU if available."""
    global _TF_CONFIGURED
    if _TF_CONFIGURED:
        return False

    try:
        import tensorflow as tf
    except Exception as exc:
        logger.info("TensorFlow not available; GPU config skipped: %s", exc)
        _TF_CONFIGURED = True
        return False

    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
            logger.info("TensorFlow GPU(s) available: %s", [gpu.name for gpu in gpus])
            _TF_CONFIGURED = True
            return True
        logger.info("TensorFlow GPU not available; using CPU.")
    except Exception as exc:
        logger.info("TensorFlow GPU detection failed; using CPU: %s", exc)

    _TF_CONFIGURED = True
    return False


def get_xgboost_device_params() -> Dict[str, str]:
    """Return XGBoost device parameters, preferring GPU if available."""
    global _XGB_DEVICE_PARAMS
    if _XGB_DEVICE_PARAMS is not None:
        return dict(_XGB_DEVICE_PARAMS)

    if os.environ.get("XGBOOST_FORCE_CPU") == "1":
        _XGB_DEVICE_PARAMS = {"tree_method": "hist"}
        logger.info("XGBoost forced to CPU via XGBOOST_FORCE_CPU.")
        return dict(_XGB_DEVICE_PARAMS)

    try:
        import numpy as np
        import xgboost as xgb
    except Exception as exc:
        _XGB_DEVICE_PARAMS = {"tree_method": "hist"}
        logger.info("XGBoost import failed; using CPU: %s", exc)
        return dict(_XGB_DEVICE_PARAMS)

    if os.environ.get("XGBOOST_FORCE_GPU") == "1":
        _XGB_DEVICE_PARAMS = {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}
        logger.info("XGBoost forced to GPU via XGBOOST_FORCE_GPU.")
        return dict(_XGB_DEVICE_PARAMS)

    use_gpu = True
    has_cuda = getattr(xgb.core, "_has_cuda_support", None)
    try:
        if callable(has_cuda) and not has_cuda():
            use_gpu = False
    except Exception:
        pass

    if use_gpu:
        try:
            rng = np.random.default_rng(42)
            X = rng.normal(size=(8, 4))
            y = rng.integers(0, 2, size=8)
            probe = xgb.XGBClassifier(
                n_estimators=1,
                max_depth=2,
                tree_method="gpu_hist",
                predictor="gpu_predictor",
                verbosity=0,
            )
            probe.fit(X, y)
            _XGB_DEVICE_PARAMS = {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}
            logger.info("XGBoost GPU enabled.")
            return dict(_XGB_DEVICE_PARAMS)
        except Exception as exc:
            logger.info("XGBoost GPU not available; using CPU: %s", exc)

    _XGB_DEVICE_PARAMS = {"tree_method": "hist"}
    return dict(_XGB_DEVICE_PARAMS)
