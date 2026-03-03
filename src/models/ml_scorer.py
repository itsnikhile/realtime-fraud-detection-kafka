"""
ML Scorer
XGBoost-based fraud scoring model loaded via ONNX Runtime for low-latency inference.
Falls back to a lightweight heuristic model when ONNX model is unavailable.
"""

import logging
import numpy as np
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts and normalizes features from transaction + context."""

    FEATURE_NAMES = [
        "amount_normalized",
        "txn_count_1h_normalized",
        "total_amount_1h_normalized",
        "geo_distance_normalized",
        "amount_vs_avg_ratio_normalized",
        "is_international",
        "is_online",
        "is_atm",
        "is_night",
        "hour_sin",
        "hour_cos",
        "is_high_risk_category",
    ]

    HIGH_RISK_CATEGORIES = {"atm", "online", "travel"}

    def extract(self, txn: Dict, features: Dict) -> np.ndarray:
        from datetime import datetime

        hour = 12
        try:
            ts = txn.get("timestamp", "2000-01-01T12:00:00").rstrip("Z")
            hour = datetime.fromisoformat(ts).hour
        except Exception:
            pass

        return np.array([
            min(txn.get("amount", 0) / 10000.0, 1.0),
            min(features.get("txn_count_1h", 0) / 20.0, 1.0),
            min(features.get("total_amount_1h", 0) / 20000.0, 1.0),
            min(features.get("geo_distance_km", 0) / 2000.0, 1.0),
            min(features.get("amount_vs_avg_ratio", 1.0) / 15.0, 1.0),
            float(txn.get("is_international", False)),
            float(txn.get("channel") == "online"),
            float(txn.get("channel") == "atm"),
            float(hour >= 23 or hour <= 5),
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            float(txn.get("merchant_category", "") in self.HIGH_RISK_CATEGORIES),
        ], dtype=np.float32)


class ONNXModel:
    """Loads and runs an ONNX model for inference."""

    def __init__(self, model_path: str):
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"ONNX model loaded from {model_path}")
            self._available = True
        except Exception as e:
            logger.warning(f"ONNX model not available: {e}. Using heuristic fallback.")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def predict(self, features: np.ndarray) -> float:
        result = self.session.run(None, {self.input_name: features.reshape(1, -1)})
        return float(result[1][0][1])  # probability of fraud class


class HeuristicModel:
    """
    Lightweight heuristic model used as fallback.
    Simulates XGBoost output with weighted sigmoid.
    """

    WEIGHTS = np.array([0.28, 0.22, 0.15, 0.18, 0.08, 0.04, 0.02, 0.05, 0.06, 0.0, 0.0, 0.07], dtype=np.float32)

    def predict(self, features: np.ndarray) -> float:
        raw = float(np.dot(features, self.WEIGHTS))
        return float(1.0 / (1.0 + np.exp(-8.0 * (raw - 0.45))))


class MLScorer:
    """
    Main ML scoring interface.
    Uses ONNX model if available, otherwise falls back to heuristic model.
    """

    MODEL_PATH = "models/fraud_model.onnx"

    def __init__(self, model_path: Optional[str] = None):
        path = model_path or self.MODEL_PATH
        self.extractor = FeatureExtractor()
        self.onnx_model = ONNXModel(path) if Path(path).exists() else None
        self.fallback = HeuristicModel()
        self._inference_count = 0
        self._total_latency_ms = 0.0

    def score(self, txn: Dict, features: Dict) -> float:
        """
        Returns fraud probability [0.0, 1.0].
        Higher = more likely fraud.
        """
        import time
        t = time.perf_counter()

        feature_vec = self.extractor.extract(txn, features)

        if self.onnx_model and self.onnx_model.available:
            score = self.onnx_model.predict(feature_vec)
        else:
            score = self.fallback.predict(feature_vec)

        latency = (time.perf_counter() - t) * 1000
        self._inference_count += 1
        self._total_latency_ms += latency

        return score

    @property
    def avg_inference_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return self._total_latency_ms / self._inference_count
