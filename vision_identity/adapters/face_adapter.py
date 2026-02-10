
from __future__ import annotations

import logging
import time
from typing import Optional

from schemas.identity_decision import IdentityDecision
from schemas.tracklet import Tracklet

from vision_identity.types import (
    FaceEvidence,
    EvidenceStatus,
    SourceAuthState,
)

logger = logging.getLogger(__name__)


class FaceAdapter:
    
    def __init__(self):
        logger.debug("[FACE-ADAPTER] Initialized (stateless)")
    
    @staticmethod
    def adapt_decision(
        tracklet: Tracklet,
        identity_decision: Optional[IdentityDecision],
        now: Optional[float] = None
    ) -> Optional[FaceEvidence]:
        if now is None:
            now = time.time()
        
        if identity_decision is None:
            logger.debug(
                f"[FACE-ADAPTER] track_id={tracklet.track_id} → "
                f"identity_decision=None, returning None"
            )
            return None
        
        track_id = identity_decision.track_id
        identity_id = identity_decision.identity_id
        confidence = identity_decision.confidence
        quality = identity_decision.quality
        binding_state = identity_decision.binding_state
        
        distance = identity_decision.distance or 0.0
        similarity = max(0.0, 1.0 - distance)
        
        margin = 0.0
        second_best_id = None
        second_best_similarity = 0.0
        
        if identity_decision.extra:
            margin = identity_decision.extra.get("margin", 0.0)
            second_best_id = identity_decision.extra.get("second_best_id")
            second_best_similarity = identity_decision.extra.get("second_best_sim", 0.0)
        
        evidence_status = FaceAdapter._map_binding_state(binding_state)
        
        if quality < 0.45:
            evidence_status = EvidenceStatus.UNKNOWN
        
        source_auth_score = identity_decision.source_auth_score
        source_auth_state = None
        if identity_decision.source_auth_state:
            try:
                source_auth_state = SourceAuthState[
                    identity_decision.source_auth_state.upper()
                ]
            except (KeyError, AttributeError):
                source_auth_state = None
        
        face_ev = FaceEvidence(
            identity_id=identity_id,
            similarity=similarity,
            quality=quality,
            status=evidence_status,
            binding_state=binding_state,
            margin=margin,
            second_best_id=second_best_id,
            second_best_similarity=second_best_similarity,
            timestamp=now,
            freshness_window_sec=2.0,
            source_auth_score=source_auth_score,
            source_auth_state=source_auth_state,
            extra={
                "original_confidence": confidence,
                "original_distance": distance,
                "original_reason": identity_decision.reason,
            }
        )
        
        logger.debug(
            f"[FACE-ADAPTER] track_id={track_id} adapted: "
            f"id={identity_id}, sim={similarity:.3f}, quality={quality:.2f}, "
            f"status={evidence_status.value}, binding={binding_state}"
        )
        
        return face_ev
    
    @staticmethod
    def _map_binding_state(binding_state: Optional[str]) -> EvidenceStatus:
        if binding_state is None:
            return EvidenceStatus.UNKNOWN
        
        state_upper = binding_state.upper()
        
        mapping = {
            "UNKNOWN": EvidenceStatus.UNKNOWN,
            "PENDING": EvidenceStatus.TENTATIVE,
            "CONFIRMED_WEAK": EvidenceStatus.CONFIRMED_WEAK,
            "CONFIRMED_STRONG": EvidenceStatus.CONFIRMED_STRONG,
            "SWITCH_PENDING": EvidenceStatus.TENTATIVE,
            "STALE": EvidenceStatus.STALE,
        }
        
        return mapping.get(state_upper, EvidenceStatus.UNKNOWN)
    
    @staticmethod
    def extract_confidence_from_similarity(
        similarity: float,
        quality: float,
        margin: float
    ) -> float:
        quality_weight = (quality ** 0.5)
        
        if margin >= 0.10:
            margin_weight = 1.0
        else:
            margin_weight = margin / 0.10
            margin_weight = max(0.1, margin_weight)
        
        confidence = similarity * quality_weight * margin_weight
        
        return min(1.0, max(0.0, confidence))


def get_face_evidence_from_engine(
    tracklet: Tracklet,
    identity_decision: Optional[IdentityDecision],
    now: Optional[float] = None
) -> Optional[FaceEvidence]:
    return FaceAdapter.adapt_decision(tracklet, identity_decision, now)


