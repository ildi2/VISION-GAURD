
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np


SourceAuthState = Literal[
    "REAL",
    "LIKELY_REAL",
    "UNCERTAIN",
    "LIKELY_SPOOF",
    "SPOOF",
]


BBoxTuple = Tuple[float, float, float, float]


SourceAuthDebug = Dict[str, Union[float, int, bool, str]]


@dataclass
class SourceAuthComponentScores:

    planar_3d: float = 0.5
    screen_artifacts: float = 0.5
    background_consistency: float = 0.5


@dataclass
class SourceAuthReliabilityFlags:

    enough_motion: bool = False

    enough_landmarks: bool = False

    enough_background: bool = False


@dataclass
class SourceAuthScores:

    track_id: int

    source_auth_score: float = 0.5

    state: SourceAuthState = "UNCERTAIN"

    components: SourceAuthComponentScores = field(
        default_factory=SourceAuthComponentScores
    )
    reliability: SourceAuthReliabilityFlags = field(
        default_factory=SourceAuthReliabilityFlags
    )

    debug: SourceAuthDebug = field(default_factory=dict)


    @property
    def score_3d_motion(self) -> float:
        return float(self.components.planar_3d)

    @score_3d_motion.setter
    def score_3d_motion(self, value: float) -> None:
        self.components.planar_3d = float(value)


    @property
    def score_screen_artifacts(self) -> float:
        return float(self.components.screen_artifacts)

    @score_screen_artifacts.setter
    def score_screen_artifacts(self, value: float) -> None:
        self.components.screen_artifacts = float(value)


    @property
    def score_background_consistency(self) -> float:
        return float(self.components.background_consistency)

    @score_background_consistency.setter
    def score_background_consistency(self, value: float) -> None:
        self.components.background_consistency = float(value)


    @property
    def motion_reliable(self) -> bool:
        return bool(self.reliability.enough_motion)

    @motion_reliable.setter
    def motion_reliable(self, value: bool) -> None:
        self.reliability.enough_motion = bool(value)
        if value:
            self.reliability.enough_landmarks = True


    @property
    def screen_reliable(self) -> bool:
        return bool(self.reliability.enough_background)

    @screen_reliable.setter
    def screen_reliable(self, value: bool) -> None:
        self.reliability.enough_background = bool(value)

    @property
    def background_reliable(self) -> bool:
        return bool(self.reliability.enough_background)

    @background_reliable.setter
    def background_reliable(self, value: bool) -> None:
        self.reliability.enough_background = bool(value)


    def update_debug(self, prefix: str, metrics: SourceAuthDebug) -> None:
        if not metrics:
            return
        for key, value in metrics.items():
            self.debug[f"{prefix}{str(key)}"] = value


    def as_dict(self) -> Dict[str, Union[int, float, str, bool]]:
        data: Dict[str, Union[int, float, str, bool]] = {
            "track_id": int(self.track_id),
            "source_auth_score": float(self.source_auth_score),
            "state": str(self.state),
            "score_3d_motion": float(self.score_3d_motion),
            "score_screen_artifacts": float(self.score_screen_artifacts),
            "score_background_consistency": float(
                self.score_background_consistency
            ),
            "motion_reliable": bool(self.motion_reliable),
            "screen_reliable": bool(self.screen_reliable),
            "background_reliable": bool(self.background_reliable),
        }
        return data


@dataclass
class LandmarkFrame:

    ts: float

    bbox: Optional[BBoxTuple]

    landmarks_2d: np.ndarray

    quality: float = 0.0

    yaw_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None

    det_score: Optional[float] = None

    def num_landmarks(self) -> int:
        try:
            return int(self.landmarks_2d.shape[0])
        except Exception:
            return 0

    def as_float_array(self) -> np.ndarray:
        arr = np.asarray(self.landmarks_2d, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            size = arr.size
            if size % 2 != 0:
                return np.zeros((0, 2), dtype=np.float32)
            arr = arr.reshape(-1, 2)
        return arr
