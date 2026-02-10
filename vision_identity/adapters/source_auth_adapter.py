
from __future__ import annotations

import logging
import time
from typing import Optional, Any, Dict

from source_auth.types import SourceAuthScores, SourceAuthState as SAState

from vision_identity.types import (
    SourceAuthEvidence,
    SourceAuthState,
)

logger = logging.getLogger(__name__)


class SourceAuthAdapter:
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            logger.info("[SOURCE-AUTH-ADAPTER] Initialized (enabled)")
        else:
            logger.info("[SOURCE-AUTH-ADAPTER] Initialized (disabled)")
    
    def adapt_scores(
        self,
        source_auth_scores: Optional[Any],
        now: Optional[float] = None
    ) -> Optional[SourceAuthEvidence]:
        if now is None:
            now = time.time()
        
        if not self.enabled or source_auth_scores is None:
            return None
        
        realness_score = None
        
        for attr_name in ["realness_score", "score", "probability"]:
            if hasattr(source_auth_scores, attr_name):
                try:
                    realness_score = float(getattr(source_auth_scores, attr_name))
                    break
                except (ValueError, TypeError):
                    continue
        
        if realness_score is None:
            logger.warning(
                "[SOURCE-AUTH-ADAPTER] Could not extract realness_score, "
                "returning UNCERTAIN"
            )
            return SourceAuthEvidence(
                realness_score=0.5,
                state=SourceAuthState.UNCERTAIN,
                timestamp=now,
                reason="score_extraction_failed"
            )
        
        realness_score = max(0.0, min(1.0, realness_score))
        
        state = SourceAuthAdapter._extract_state(realness_score)
        
        source_auth_ev = SourceAuthEvidence(
            realness_score=realness_score,
            state=state,
            timestamp=now,
            reason=None,
            extra={
                "original_scores": str(source_auth_scores)[:100]
            }
        )
        
        logger.debug(
            f"[SOURCE-AUTH-ADAPTER] Adapted: "
            f"realness={realness_score:.3f}, state={state.value}"
        )
        
        return source_auth_ev
    
    @staticmethod
    def _extract_state(realness_score: float) -> SourceAuthState:
        if realness_score < 0.2:
            return SourceAuthState.SPOOF
        elif realness_score < 0.4:
            return SourceAuthState.LIKELY_SPOOF
        elif realness_score < 0.6:
            return SourceAuthState.UNCERTAIN
        elif realness_score < 0.8:
            return SourceAuthState.LIKELY_REAL
        else:
            return SourceAuthState.REAL
    
    @staticmethod
    def should_block_decision(
        source_auth_ev: Optional[SourceAuthEvidence]
    ) -> bool:
        if source_auth_ev is None:
            return False
        
        if source_auth_ev.is_spoof():
            return True
        
        if source_auth_ev.realness_score < 0.25:
            return True
        
        return False
    
    @staticmethod
    def should_block_learning(
        source_auth_ev: Optional[SourceAuthEvidence]
    ) -> bool:
        if source_auth_ev is None:
            return False
        
        return not source_auth_ev.is_real()


def get_source_auth_evidence(
    source_auth_scores: Optional[Any],
    enabled: bool = True,
    now: Optional[float] = None
) -> Optional[SourceAuthEvidence]:
    adapter = SourceAuthAdapter(enabled=enabled)
    return adapter.adapt_scores(source_auth_scores, now)


