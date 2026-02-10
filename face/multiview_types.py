
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PoseBin(str, Enum):

    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

    OCCLUDED = "occluded"
    UNKNOWN = "unknown"


ALL_PRIMARY_BINS: List[PoseBin] = [
    PoseBin.FRONT,
    PoseBin.LEFT,
    PoseBin.RIGHT,
    PoseBin.UP,
    PoseBin.DOWN,
]


def posebin_from_hint(hint: Any) -> PoseBin:
    if isinstance(hint, PoseBin):
        return hint

    if not isinstance(hint, str):
        return PoseBin.UNKNOWN

    s = hint.strip()
    if not s:
        return PoseBin.UNKNOWN

    low = s.lower()

    for b in PoseBin:
        if low == b.value:
            return b

    up = s.upper()
    for b in PoseBin:
        if up == b.name:
            return b

    return PoseBin.UNKNOWN


@dataclass(frozen=True)
class MultiViewConfig:


    front_yaw_max_deg: float = 15.0
    front_pitch_max_deg: float = 12.0

    side_yaw_min_deg: float = 20.0
    side_yaw_max_deg: float = 70.0

    up_pitch_min_deg: float = 17.0
    down_pitch_min_deg: float = 25.0


    max_samples_per_bin: int = 16

    min_quality_for_model: float = 0.40

    ema_alpha: float = 0.10


    def validate(self) -> None:
        if self.front_yaw_max_deg <= 0:
            raise ValueError("front_yaw_max_deg must be positive")

        if self.front_pitch_max_deg <= 0:
            raise ValueError("front_pitch_max_deg must be positive")

        if self.side_yaw_min_deg <= 0 or self.side_yaw_max_deg <= 0:
            raise ValueError("side_yaw_* thresholds must be positive")

        if self.side_yaw_min_deg >= self.side_yaw_max_deg:
            raise ValueError("side_yaw_min_deg must be < side_yaw_max_deg")

        if self.up_pitch_min_deg <= 0 or self.down_pitch_min_deg <= 0:
            raise ValueError("up/down pitch thresholds must be positive")

        if self.max_samples_per_bin <= 0:
            raise ValueError("max_samples_per_bin must be positive")

        if not (0.0 <= self.min_quality_for_model <= 1.0):
            raise ValueError("min_quality_for_model must be in [0, 1]")

        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1].")


@dataclass
class MultiViewSample:

    embedding: np.ndarray

    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None

    quality: float = 0.0

    pose_bin: PoseBin = PoseBin.UNKNOWN

    source: str = "unknown"
    ts: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def clone_with_bin(self, pose_bin: PoseBin) -> "MultiViewSample":
        return MultiViewSample(
            embedding=self.embedding,
            yaw_deg=self.yaw_deg,
            pitch_deg=self.pitch_deg,
            roll_deg=self.roll_deg,
            quality=self.quality,
            pose_bin=pose_bin,
            source=self.source,
            ts=self.ts,
            metadata=dict(self.metadata),
        )

    def as_report_dict(self) -> Dict[str, Any]:
        return {
            "pose_bin": self.pose_bin.value,
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "roll_deg": self.roll_deg,
            "quality": float(self.quality),
            "source": self.source,
            "ts": float(self.ts),
        }


@dataclass
class MultiViewBin:

    pose_bin: PoseBin
    samples: List[MultiViewSample] = field(default_factory=list)

    centroid: Optional[np.ndarray] = None

    avg_quality: float = 0.0
    last_updated_ts: float = field(default_factory=lambda: time.time())

    def is_populated(self) -> bool:
        return bool(self.samples) and self.centroid is not None

    def num_samples(self) -> int:
        return len(self.samples)

    def to_report_dict(self) -> Dict[str, Any]:
        return {
            "pose_bin": self.pose_bin.value,
            "num_samples": self.num_samples(),
            "avg_quality": float(self.avg_quality),
            "has_centroid": self.centroid is not None,
            "last_updated_ts": float(self.last_updated_ts),
        }


@dataclass
class MultiViewPersonModel:

    person_id: str
    bins: Dict[PoseBin, MultiViewBin] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    notes: Dict[str, Any] = field(default_factory=dict)

    def primary_bins(self) -> List[PoseBin]:
        return [
            b for b in ALL_PRIMARY_BINS
            if b in self.bins and self.bins[b].is_populated()
        ]

    def num_populated_bins(self) -> int:
        return len(self.primary_bins())

    def coverage_score(self, expected_bins: Iterable[PoseBin] = ALL_PRIMARY_BINS) -> float:
        expected = list(expected_bins)
        if not expected:
            return 0.0

        filled = sum(
            1 for b in expected
            if b in self.bins and self.bins[b].is_populated()
        )
        return filled / float(len(expected))

    def bin_or_none(self, pose_bin: PoseBin) -> Optional[MultiViewBin]:
        return self.bins.get(pose_bin)

    def to_report_dict(self) -> Dict[str, Any]:
        bin_reports = {
            b.value: self.bins[b].to_report_dict()
            for b in self.bins.keys()
        }
        return {
            "person_id": self.person_id,
            "bins": bin_reports,
            "coverage": self.coverage_score(),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "notes": dict(self.notes),
        }


def _abs_or_none(x: Optional[float]) -> Optional[float]:
    return None if x is None else abs(float(x))


def classify_pose_bin(
    yaw_deg: Optional[float],
    pitch_deg: Optional[float],
    *,
    cfg: MultiViewConfig,
    is_occluded: bool = False,
) -> PoseBin:
    if yaw_deg is None or pitch_deg is None:
       return PoseBin.UNKNOWN

    if is_occluded:
        return PoseBin.OCCLUDED

    y = _abs_or_none(yaw_deg)
    p = pitch_deg

    if y is None or p is None:
        return PoseBin.UNKNOWN

    ay = float(y)
    ap = abs(float(p))

    if ay <= cfg.front_yaw_max_deg and ap <= cfg.front_pitch_max_deg:
        return PoseBin.FRONT

    if cfg.side_yaw_min_deg <= ay <= cfg.side_yaw_max_deg:
        if yaw_deg is not None and yaw_deg < 0:
            return PoseBin.LEFT
        elif yaw_deg is not None and yaw_deg > 0:
            return PoseBin.RIGHT

    if p >= cfg.up_pitch_min_deg:
        return PoseBin.UP
    if p <= -cfg.down_pitch_min_deg:
        return PoseBin.DOWN

    return PoseBin.UNKNOWN


def compute_coverage_score(
    models: Mapping[str, MultiViewPersonModel],
    *,
    expected_bins: Iterable[PoseBin] = ALL_PRIMARY_BINS,
) -> float:
    if not models:
        return 0.0

    expected = list(expected_bins)
    if not expected:
        return 0.0

    scores = [m.coverage_score(expected) for m in models.values()]
    return float(sum(scores) / len(scores))
