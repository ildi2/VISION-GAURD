
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class IdSignals:
    track_id: int

    face_embedding: Optional[np.ndarray] = None
    face_quality: float = 0.0

    gait_embedding: Optional[np.ndarray] = None
    gait_quality: float = 0.0

    appearance_embedding: Optional[np.ndarray] = None
    appearance_quality: float = 0.0

@dataclass
class IdSignal:
    track_id:int
    identity_id: Optional[str]
    confidence:float
    method:str