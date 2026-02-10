from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Alert:
    alert_id: int
    created_at: float
    camera_id: str
    track_ids: List[int] = field(default_factory=list)

    type: str = "generic"
    severity: int = 1
    message: str = ""

    evidence_clip_path: Optional[str] = None
    snapshot_path: Optional[str] = None

    resolved: bool = False
