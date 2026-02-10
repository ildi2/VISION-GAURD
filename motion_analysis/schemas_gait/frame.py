from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Frame:
    frame_id: int
    ts: float
    camera_id: str = "cam0"
    size: Optional[Tuple[int, int]] = None
    image: Optional[np.ndarray] = None
