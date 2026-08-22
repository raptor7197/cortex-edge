"""Learned router (PLAN 15.7): lightweight classifier trained on
datasets/routing_training.csv by scripts/train_router.py.

Inputs: prompt features + device state; output: route label
(small | medium | cloud | local_rag), matching PLAN section 7.
"""

from pathlib import Path

import numpy as np

from app.router.features import QueryFeatures
from app.schemas import SystemState

FEATURE_ORDER = [
    "token_count", "sentence_count", "has_code", "has_math",
    "reasoning_score", "multi_part_score", "requires_documents",
    "complexity_score", "free_ram_mb", "cpu_percent",
    "cpu_temperature_c", "network_available",
]

_TRAINING_CSV = (
    Path(__file__).resolve().parents[2] / "datasets" / "routing_training.csv"
)


def _median_temperature_from_csv() -> float:
    """Read training median for cpu_temperature_c so machines without a
    thermal sensor get a reasonable imputed value instead of 0.0."""
    try:
        import pandas as pd

        df = pd.read_csv(_TRAINING_CSV)
        if "cpu_temperature_c" in df.columns:
            med = df["cpu_temperature_c"].median()
            if med > 0:
                return float(med)
    except Exception:
        pass
    return 55.0  # safe fallback: typical Pi load temperature


class LearnedRouter:
    def __init__(self, model_path: str | Path | None = None):
        self.model = None
        self.temp_median = _median_temperature_from_csv()
        if model_path is not None:
            import joblib

            self.model = joblib.load(model_path)

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, q: QueryFeatures, s: SystemState) -> str:
        if self.model is None:
            raise RuntimeError("No trained router model loaded")
        row = q.as_dict() | {
            "free_ram_mb": s.free_ram_mb,
            "cpu_percent": s.cpu_percent,
            "cpu_temperature_c": s.cpu_temperature_c
            if s.cpu_temperature_c is not None
            else self.temp_median,
            "network_available": int(s.network_available),
        }
        x = np.array([[row[name] for name in FEATURE_ORDER]], dtype=float)
        return str(self.model.predict(x)[0])
