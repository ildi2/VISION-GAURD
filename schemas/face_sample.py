
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class FaceSample:

    bbox: Optional[Tuple[float, float, float, float]] = None
    embedding: Optional[np.ndarray] = None

    det_score: float = 0.0
    quality: float = 0.0

    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

    ts: Optional[float] = None
    pose_bin: Optional[str] = None

    source: str = "runtime"
    extra: Optional[Dict[str, Any]] = None

    def as_embedding_1d(self) -> Optional[np.ndarray]:
        if self.embedding is None:
            return None
        e = np.asarray(self.embedding, dtype=np.float32).reshape(-1)
        return e

    def clamped_quality(self) -> float:
        q = float(self.quality)
        if q < 0.0:
            return 0.0
        if q > 1.0:
            return 1.0
        return q
