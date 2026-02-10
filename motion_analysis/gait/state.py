from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple
import time

class GaitState(str, Enum):
    COLLECTING = "COLLECTING"
    EVALUATING = "EVALUATING"
    CONFIRMED = "CONFIRMED"
    UNSURE = "UNSURE"

class GaitReason(str, Enum):
    NONE = "NONE"
    
    SKIP_LOW_QUALITY = "SKIP_LOW_QUALITY"
    SKIP_STILL = "SKIP_STILL"
    SKIP_SHORT_SEQ = "SKIP_SHORT_SEQ"
    
    HOLD_NEUTRAL = "HOLD_NEUTRAL"
    HOLD_LOW_SIM = "HOLD_LOW_SIM"
    HOLD_BORDERLINE = "HOLD_BORDERLINE"
    HOLD_LOW_MARGIN = "HOLD_LOW_MARGIN"
    HOLD_STREAK = "HOLD_STREAK"
    
    CONFIRM_STRONG = "CONFIRM_STRONG"
    REJECT_LOW_SIM = "REJECT_LOW_SIM"
    
    UNSURE_QUALITY_DROP = "UNSURE_QUALITY_DROP"
    UNSURE_CONFLICT = "UNSURE_CONFLICT"

@dataclass
class GaitTrackState:
    track_id: int
    state: GaitState = GaitState.COLLECTING
    
    created_ts: float = field(default_factory=time.perf_counter)
    last_eval_ts: float = 0.0
    
    q_seq: float = 0.0
    
    best_id: Optional[str] = None
    best_sim: float = 0.0
    best_dist: float = 1.0
    
    confirm_streak: int = 0
    bad_eval_streak: int = 0
    
    reason: GaitReason = GaitReason.NONE
    
    def can_evaluate(self, now: float, eval_period: float) -> bool:
        return (now - self.last_eval_ts) >= eval_period

    def update_match(self, best_id: Optional[str], best_sim: float, best_dist: float):
        self.best_id = best_id
        self.best_sim = best_sim
        self.best_dist = best_dist

    def reset_streak(self):
        self.confirm_streak = 0
        
    def increment_streak(self):
        self.confirm_streak += 1

    def set_reason(self, reason: GaitReason):
        self.reason = reason
