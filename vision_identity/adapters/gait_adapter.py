
from __future__ import annotations

import logging
import time
from typing import Optional, Dict, Any

from schemas.identity_decision import IdentityDecision
from schemas.tracklet import Tracklet

from motion_analysis.gait.state import GaitTrackState, GaitState, GaitReason

from vision_identity.types import (
    GaitEvidence,
    EvidenceStatus,
)

logger = logging.getLogger(__name__)


class GaitAdapter:
    
    def __init__(self):
        logger.debug("[GAIT-ADAPTER] Initialized (stateless)")
    
    @staticmethod
    def adapt_decision(
        tracklet: Tracklet,
        gait_identity_decision: Optional[IdentityDecision],
        gait_track_state: Optional[GaitTrackState],
        now: Optional[float] = None
    ) -> Optional[GaitEvidence]:
        if now is None:
            now = time.time()
        
        seq_len = len(tracklet.gait_sequence_data) if tracklet.gait_sequence_data else 0
        if seq_len < 30:
            logger.debug(
                f"[GAIT-ADAPTER] track_id={tracklet.track_id} → "
                f"sequence too short ({seq_len} < 30), returning None"
            )
            return None
        
        if gait_track_state is None:
            return None
        
        gait_state = gait_track_state.state
        gait_reason = gait_track_state.reason
        q_seq = gait_track_state.q_seq
        
        if gait_identity_decision is None:
            identity_id = gait_track_state.best_id
            similarity = gait_track_state.best_sim
            distance = gait_track_state.best_dist
        else:
            identity_id = gait_identity_decision.identity_id
            if hasattr(gait_identity_decision, 'score'):
                similarity = gait_identity_decision.score or 0.0
            elif hasattr(gait_identity_decision, 'distance'):
                distance = gait_identity_decision.distance or 1.0
                similarity = max(0.0, 1.0 - distance)
            else:
                similarity = gait_track_state.best_sim
            distance = 1.0 - similarity
        
        margin = 0.0
        second_best_similarity = 0.0
        second_best_id = None
        
        if gait_identity_decision and gait_identity_decision.extra:
            margin = gait_identity_decision.extra.get("margin", 0.0)
            second_best_sim = gait_identity_decision.extra.get("second_best_sim", 0.0)
            second_best_similarity = second_best_sim
            second_best_id = gait_identity_decision.extra.get("second_best_id")
        
        evidence_status = GaitAdapter._map_gait_state(gait_state, gait_reason)
        
        confidence = GaitAdapter._extract_gait_confidence(
            similarity=similarity,
            margin=margin,
            sequence_length=seq_len,
            quality=q_seq,
            confirm_streak=gait_track_state.confirm_streak
        )
        
        gait_ev = GaitEvidence(
            identity_id=identity_id,
            similarity=similarity,
            margin=margin,
            quality=q_seq,
            status=evidence_status,
            state_machine_state=gait_state.value,
            confidence=confidence,
            sequence_length=seq_len,
            confirm_streak=gait_track_state.confirm_streak,
            second_best_id=second_best_id,
            second_best_similarity=second_best_similarity,
            timestamp=now,
            freshness_window_sec=3.0,
            extra={
                "gait_state": gait_state.value,
                "gait_reason": gait_reason.value if gait_reason else "NONE",
                "bad_eval_streak": gait_track_state.bad_eval_streak,
            }
        )
        
        logger.debug(
            f"[GAIT-ADAPTER] track_id={tracklet.track_id} adapted: "
            f"id={identity_id}, sim={similarity:.3f}, margin={margin:.3f}, "
            f"quality={q_seq:.2f}, seq_len={seq_len}, status={evidence_status.value}, "
            f"gait_state={gait_state.value}, confidence={confidence:.3f}"
        )
        
        return gait_ev
    
    @staticmethod
    def _map_gait_state(
        gait_state: GaitState,
        gait_reason: Optional[GaitReason]
    ) -> EvidenceStatus:
        state_upper = gait_state.value.upper()
        
        if state_upper == "COLLECTING":
            return EvidenceStatus.COLLECTING
        elif state_upper == "EVALUATING":
            if gait_reason and "CONFIRM" in gait_reason.value:
                return EvidenceStatus.CONFIRMED_WEAK
            else:
                return EvidenceStatus.EVALUATING
        elif state_upper == "CONFIRMED":
            return EvidenceStatus.CONFIRMED_STRONG
        elif state_upper == "UNSURE":
            return EvidenceStatus.UNKNOWN
        else:
            return EvidenceStatus.UNKNOWN
    
    @staticmethod
    def _extract_gait_confidence(
        similarity: float,
        margin: float,
        sequence_length: int,
        quality: float,
        confirm_streak: int = 0
    ) -> float:
        if margin >= 0.10:
            margin_weight = 1.0
        elif margin >= 0.05:
            margin_weight = 0.5 + (margin / 0.10)
        else:
            margin_weight = 0.2
        
        quality_weight = (quality ** 1.0)
        
        seq_bonus = min(1.10, 1.0 + (sequence_length - 30) / 300)
        
        streak_boost = min(1.20, 1.0 + 0.10 * confirm_streak)
        
        base_conf = similarity * margin_weight * quality_weight
        
        confidence = base_conf * seq_bonus * streak_boost
        
        return min(1.0, max(0.0, confidence))


def get_gait_evidence_from_engine(
    tracklet: Tracklet,
    gait_identity_decision: Optional[IdentityDecision],
    gait_track_state: Optional[GaitTrackState],
    now: Optional[float] = None
) -> Optional[GaitEvidence]:
    return GaitAdapter.adapt_decision(
        tracklet,
        gait_identity_decision,
        gait_track_state,
        now
    )


