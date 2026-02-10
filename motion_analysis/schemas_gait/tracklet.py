from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class Tracklet:
    track_id: int
    camera_id: str
    last_frame_id: int
    last_box: Tuple[float, float, float, float]
    confidence: float

    age_frames: int = 0
    lost_frames: int = 0
    history_boxes: List[Tuple[float, float, float, float]] = field(
        default_factory=list
    )

    gait_sequence_data: List[np.ndarray] = field(default_factory=list)

    gait_embedding: Optional[np.ndarray] = None
    
    gait_quality: float=0.0
    
    gait_identity_id: Optional[str] = None

    gait_confidence: Optional[float]=None