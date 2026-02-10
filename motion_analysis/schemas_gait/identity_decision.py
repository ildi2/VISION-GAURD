from dataclasses import dataclass
from typing import Optional, Literal


Category = Literal["resident", "visitor", "watchlist", "unknown"]


@dataclass
class IdentityDecision:
    track_id: int
    identity_id: Optional[str] = None
    category: Category = "unknown"
    confidence: float = 0.0
    reason: Optional[str] = None
