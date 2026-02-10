
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

Category = Literal["resident", "visitor", "watchlist", "unknown"]


@dataclass
class IdentityDecision:

    track_id: int
    identity_id: Optional[str] = None
    category: Category = "unknown"
    confidence: float = 0.0
    reason: Optional[str] = None

    canonical_id: Optional[int] = None

    binding_state: Optional[str] = None

    pose_bin: Optional[str] = None
    engine: Optional[str] = None
    distance: Optional[float] = None
    score: Optional[float] = None

    extra: Optional[Dict[str, Any]] = None

    quality: float = 0.0

    source_auth_score: Optional[float] = None 
    source_auth_state: Optional[str] = None
    source_auth_reason: Optional[str] = None

    id_source: Optional[str] = None
