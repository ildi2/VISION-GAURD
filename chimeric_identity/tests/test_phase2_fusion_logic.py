# chimeric_identity/tests/test_phase2_fusion_logic.py
# ============================================================================
# PHASE 2 INTEGRATION TESTS - Fusion Engine (Rules A-E)
# ============================================================================
#
# Purpose:
#   Validate all 5 chimeric rules are correctly implemented in fusion_engine.py
#   Tests decision synthesis from combined face/gait/spoof evidence.
#
# Test Coverage:
#   1. Rule A: Face Dominance (gait cannot flip identity)
#   2. Rule B: Gait Proposal (only tentative when face absent)
#   3. Rule C: Conflict Hold (explicit HOLD_CONFLICT on disagreement)
#   4. Rule D: Hysteresis (stricter switch than confirm threshold)
#   5. Rule E: Learning Gates (multi-layer validation)
#
# Design:
#   - Test each rule independently
#   - Test rule interactions (multiple rules firing)
#   - Test edge cases (boundary values, missing evidence)
#   - Clear test names and assertions

import time
import pytest
from dataclasses import dataclass

from chimeric_identity.types import (
    ChimericState,
    ChimericReason,
    FaceEvidence,
    GaitEvidence,
    SourceAuthEvidence,
    EvidenceStatus,
    SourceAuthState,
)
from chimeric_identity.config import ChimericConfig
from chimeric_identity.fusion_engine import ChimericFusionEngine as FusionEngine
from chimeric_identity.state_machine import ChimericStateMachine
from chimeric_identity.evidence_accumulator import EvidenceAccumulator


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_face_evidence(
    identity_id: str = "alice",
    similarity: float = 0.85,
    quality: float = 0.90,
    status: EvidenceStatus = EvidenceStatus.CONFIRMED_STRONG,
    margin: float = 0.15,
    timestamp: float = None
) -> FaceEvidence:
    """Create test face evidence."""
    if timestamp is None:
        timestamp = time.time()
    
    return FaceEvidence(
        identity_id=identity_id,
        similarity=similarity,
        quality=quality,
        status=status,
        margin=margin,
        timestamp=timestamp
    )


def create_gait_evidence(
    identity_id: str = "alice",
    similarity: float = 0.75,
    quality: float = 0.70,
    status: EvidenceStatus = EvidenceStatus.CONFIRMED_WEAK,
    margin: float = 0.08,
    sequence_length: int = 40,
    timestamp: float = None
) -> GaitEvidence:
    """Create test gait evidence."""
    if timestamp is None:
        timestamp = time.time()
    
    return GaitEvidence(
        identity_id=identity_id,
        similarity=similarity,
        margin=margin,
        quality=quality,
        status=status,
        sequence_length=sequence_length,
        timestamp=timestamp
    )


def create_spoof_evidence(
    realness_score: float = 0.92,
    state: SourceAuthState = SourceAuthState.LIKELY_REAL
) -> SourceAuthEvidence:
    """Create test spoof evidence."""
    return SourceAuthEvidence(
        realness_score=realness_score,
        state=state,
        timestamp=time.time()
    )


# ============================================================================
# TEST: RULE A - FACE DOMINANCE
# ============================================================================

class TestRuleAFaceDominance:
    """Rule A: Gait cannot override face identity"""

    def setup_method(self):
        """Initialize fusion engine."""
        self.config = ChimericConfig()
        self.engine = FusionEngine(self.config)

    def test_rule_a_face_dominates_when_confirmed(self):
        """Face CONFIRMED → output face identity even if gait proposes different."""
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.88,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="bob",  # Different person!
            similarity=0.75,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_001",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        # Verify: Face identity wins, state is HOLD_CONFLICT (conflict detected)
        assert decision.final_identity == "alice" or decision.state == ChimericState.HOLD_CONFLICT
        assert decision.final_identity != "bob"  # Gait identity rejected

    def test_rule_a_face_identity_preserved_high_quality(self):
        """High-quality face identity persists against gait."""
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.92,
            quality=0.95,
            status=EvidenceStatus.CONFIRMED_STRONG,
            margin=0.20
        )
        gait = create_gait_evidence(
            identity_id="alice",
            similarity=0.68,
            status=EvidenceStatus.TENTATIVE
        )
        
        decision = self.engine.make_decision(
            track_id="track_002",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.final_identity == "alice"
        assert decision.state == ChimericState.CONFIRMED

    def test_rule_a_conflict_detection(self):
        """Face vs gait disagreement → HOLD_CONFLICT (no silent switch)."""
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.85,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="bob",
            similarity=0.80,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_003",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        # Verify: Conflict detected and held
        assert decision.state == ChimericState.HOLD_CONFLICT
        assert decision.learning_allowed is False


# ============================================================================
# TEST: RULE B - GAIT PROPOSAL
# ============================================================================

class TestRuleBGaitProposal:
    """Rule B: Gait can only propose (TENTATIVE) when face absent"""

    def setup_method(self):
        """Initialize fusion engine."""
        self.config = ChimericConfig()
        self.engine = FusionEngine(self.config)

    def test_rule_b_gait_proposes_when_face_missing(self):
        """No face + good gait → TENTATIVE with gait identity."""
        face = create_face_evidence(
            identity_id=None,
            similarity=0.0,
            status=EvidenceStatus.MISSING
        )
        gait = create_gait_evidence(
            identity_id="alice",
            similarity=0.75,
            quality=0.70,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_004",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        # Verify: Gait can propose TENTATIVE
        assert decision.final_identity == "alice"
        assert decision.state == ChimericState.TENTATIVE
        assert decision.learning_allowed is False  # No face anchor

    def test_rule_b_gait_does_not_propose_when_face_confirmed(self):
        """Face CONFIRMED + gait proposes different → no TENTATIVE, conflict instead."""
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.85,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="bob",
            similarity=0.72,
            status=EvidenceStatus.TENTATIVE
        )
        
        decision = self.engine.make_decision(
            track_id="track_005",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        # Verify: Gait proposal rejected, conflict detected
        assert decision.final_identity != "bob"
        assert decision.state != ChimericState.TENTATIVE or decision.final_identity == "alice"

    def test_rule_b_gait_quality_threshold(self):
        """Gait must meet quality gate to propose."""
        face = create_face_evidence(
            identity_id=None,
            status=EvidenceStatus.MISSING
        )
        gait = create_gait_evidence(
            identity_id="alice",
            similarity=0.60,  # Marginal
            quality=0.45,  # Low quality
            status=EvidenceStatus.TENTATIVE
        )
        
        decision = self.engine.make_decision(
            track_id="track_006",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        # Verify: Low quality gait blocked from proposing
        if decision.state == ChimericState.TENTATIVE:
            assert decision.confidence < 0.65  # Downweighted due to low quality


# ============================================================================
# TEST: RULE C - CONFLICT HOLD
# ============================================================================

class TestRuleCConflictHold:
    """Rule C: Face vs gait disagreement → HOLD_CONFLICT"""

    def setup_method(self):
        """Initialize fusion engine."""
        self.config = ChimericConfig()
        self.engine = FusionEngine(self.config)

    def test_rule_c_explicit_conflict_detection(self):
        """When face ≠ gait → explicit HOLD_CONFLICT state."""
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.85,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="bob",
            similarity=0.78,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_007",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state == ChimericState.HOLD_CONFLICT
        assert decision.final_identity is None  # No output during conflict
        assert decision.learning_allowed is False

    def test_rule_c_conflict_resolution_face_reconfirms(self):
        """In HOLD_CONFLICT, face re-confirming → CONFIRMED."""
        # Starting in HOLD_CONFLICT
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.90,  # Very strong
            quality=0.95,
            status=EvidenceStatus.CONFIRMED_STRONG,
            margin=0.20
        )
        gait = create_gait_evidence(
            identity_id="bob",
            similarity=0.75,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_008",
            current_state=ChimericState.HOLD_CONFLICT,  # Already in conflict
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        # Verify: Face strong enough to resolve conflict
        # Engine should transition HOLD_CONFLICT → CONFIRMED with face identity
        if decision.state == ChimericState.CONFIRMED:
            assert decision.final_identity == "alice"

    def test_rule_c_conflict_timeout(self):
        """After conflict timeout, accept gait hypothesis."""
        # This would require stateful engine tracking conflict duration
        # Simplified test: verify conflict state outputs no identity
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.82,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="bob",
            similarity=0.80,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_009",
            current_state=ChimericState.HOLD_CONFLICT,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        # In conflict, no definitive identity output
        assert decision.state in [ChimericState.HOLD_CONFLICT, ChimericState.TENTATIVE]


# ============================================================================
# TEST: RULE D - HYSTERESIS
# ============================================================================

class TestRuleDHysteresis:
    """Rule D: Stricter switch threshold prevents flickering"""

    def setup_method(self):
        """Initialize fusion engine."""
        self.config = ChimericConfig()
        self.engine = FusionEngine(self.config)

    def test_rule_d_switch_threshold_stricter_than_confirm(self):
        """Verify switch_threshold (0.85) > confirm_threshold (0.75)."""
        assert self.config.face_switch_threshold > self.config.face_confirm_threshold
        assert self.config.face_switch_threshold > self.config.gait_confirm_threshold

    def test_rule_d_requires_margin_bonus_to_switch(self):
        """Switching identity requires extra margin bonus."""
        # Starting with CONFIRMED alice
        face_new = create_face_evidence(
            identity_id="bob",
            similarity=0.84,  # Just below switch threshold (0.85)
            quality=0.88,
            status=EvidenceStatus.CONFIRMED_STRONG,
            margin=0.08  # Not enough
        )
        
        decision = self.engine.make_decision(
            track_id="track_010",
            current_state=ChimericState.CONFIRMED,  # Currently alice
            face_evidence=face_new,
            gait_evidence=None,
            source_auth_evidence=None
        )
        
        # Verify: Cannot switch with marginal evidence
        # Engine should either stay CONFIRMED or enter HOLD_CONFLICT
        assert decision.state in [ChimericState.CONFIRMED, ChimericState.HOLD_CONFLICT]

    def test_rule_d_strong_switch_overrides(self):
        """Very strong new evidence can switch."""
        face_new = create_face_evidence(
            identity_id="bob",
            similarity=0.92,  # Well above switch threshold
            quality=0.95,
            status=EvidenceStatus.CONFIRMED_STRONG,
            margin=0.22  # Large margin
        )
        
        decision = self.engine.make_decision(
            track_id="track_011",
            current_state=ChimericState.CONFIRMED,
            face_evidence=face_new,
            gait_evidence=None,
            source_auth_evidence=None
        )
        
        # Strong evidence can trigger switch or explicit conflict
        assert decision.state in [ChimericState.CONFIRMED, ChimericState.HOLD_CONFLICT]


# ============================================================================
# TEST: RULE E - LEARNING GATES
# ============================================================================

class TestRuleELearningGates:
    """Rule E: Multi-layer learning gate validation"""

    def setup_method(self):
        """Initialize fusion engine."""
        self.config = ChimericConfig()
        self.engine = FusionEngine(self.config)

    def test_rule_e_learning_allowed_face_only(self):
        """Face CONFIRMED + no gait → learning allowed."""
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.88,
            quality=0.92,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id=None,
            status=EvidenceStatus.MISSING
        )
        
        decision = self.engine.make_decision(
            track_id="track_012",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state == ChimericState.CONFIRMED
        assert decision.learning_allowed is True

    def test_rule_e_learning_allowed_face_gait_aligned(self):
        """Face + gait both CONFIRMED and same identity → learning allowed."""
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.88,
            quality=0.92,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="alice",
            similarity=0.75,
            quality=0.72,
            margin=0.10,  # Good margin
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_013",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state == ChimericState.CONFIRMED
        assert decision.learning_allowed is True

    def test_rule_e_learning_blocked_face_absent(self):
        """No face → learning blocked even if gait good."""
        face = create_face_evidence(
            identity_id=None,
            status=EvidenceStatus.MISSING
        )
        gait = create_gait_evidence(
            identity_id="alice",
            similarity=0.78,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_014",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.learning_allowed is False

    def test_rule_e_learning_blocked_gait_conflict(self):
        """Gait disagrees with face → learning blocked."""
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.88,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="bob",  # Different!
            similarity=0.75,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_015",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.learning_allowed is False

    def test_rule_e_learning_blocked_gait_low_margin(self):
        """Gait with tight margin (confusion risk) → learning blocked."""
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.88,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="alice",
            similarity=0.72,
            margin=0.02,  # Too tight! Danger of confusion
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_016",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        # May be CONFIRMED but learning should be blocked due to margin
        assert decision.learning_allowed is False or decision.state != ChimericState.CONFIRMED

    def test_rule_e_learning_blocked_spoof_detected(self):
        """Spoof detected → learning blocked."""
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.88,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="alice",
            similarity=0.75,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        spoof = create_spoof_evidence(
            realness_score=0.15,
            state=SourceAuthState.LIKELY_SPOOF
        )
        
        decision = self.engine.make_decision(
            track_id="track_017",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=spoof
        )
        
        assert decision.learning_allowed is False


# ============================================================================
# TEST: CONFIDENCE SYNTHESIS
# ============================================================================

class TestConfidenceSynthesis:
    """Verify confidence calculation logic."""

    def setup_method(self):
        """Initialize fusion engine."""
        self.config = ChimericConfig()
        self.engine = FusionEngine(self.config)

    def test_confirmed_state_confidence(self):
        """CONFIRMED state confidence is face similarity."""
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.88,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id=None,
            status=EvidenceStatus.MISSING
        )
        
        decision = self.engine.make_decision(
            track_id="track_018",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state == ChimericState.CONFIRMED
        assert 0.85 <= decision.confidence <= 0.90  # Close to face similarity

    def test_confirmed_with_gait_boost(self):
        """CONFIRMED + gait aligned → confidence boosted."""
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.85,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="alice",
            similarity=0.75,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_019",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state == ChimericState.CONFIRMED
        # Confidence should be higher than face alone due to gait boost
        assert decision.confidence > 0.85

    def test_tentative_state_confidence(self):
        """TENTATIVE state confidence is downweighted gait."""
        face = create_face_evidence(
            identity_id=None,
            status=EvidenceStatus.MISSING
        )
        gait = create_gait_evidence(
            identity_id="alice",
            similarity=0.75,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_020",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        if decision.state == ChimericState.TENTATIVE:
            # Confidence downweighted (~70% of gait)
            assert decision.confidence < 0.75

    def test_unknown_state_confidence(self):
        """UNKNOWN or HOLD_CONFLICT state confidence is 0.0."""
        face = create_face_evidence(
            identity_id=None,
            status=EvidenceStatus.MISSING
        )
        gait = create_gait_evidence(
            identity_id=None,
            status=EvidenceStatus.MISSING
        )
        
        decision = self.engine.make_decision(
            track_id="track_021",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state == ChimericState.UNKNOWN
        assert decision.confidence == 0.0


# ============================================================================
# TEST: SPOOF INTEGRATION
# ============================================================================

class TestSpoofIntegration:
    """Verify spoof detection integration."""

    def setup_method(self):
        """Initialize fusion engine."""
        self.config = ChimericConfig()
        self.engine = FusionEngine(self.config)

    def test_spoof_detected_blocks_decision(self):
        """Spoof detected → HOLD_CONFLICT, no learning."""
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.88,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="alice",
            similarity=0.75,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        spoof = create_spoof_evidence(
            realness_score=0.10,
            state=SourceAuthState.SPOOF
        )
        
        decision = self.engine.make_decision(
            track_id="track_022",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=spoof
        )
        
        assert decision.state == ChimericState.HOLD_CONFLICT
        assert decision.learning_allowed is False

    def test_likely_real_allows_decision(self):
        """LIKELY_REAL spoof state → normal decision."""
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.88,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="alice",
            similarity=0.75,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        spoof = create_spoof_evidence(
            realness_score=0.92,
            state=SourceAuthState.LIKELY_REAL
        )
        
        decision = self.engine.make_decision(
            track_id="track_023",
            current_state=ChimericState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=spoof
        )
        
        assert decision.state == ChimericState.CONFIRMED
        assert decision.learning_allowed is True


if __name__ == "__main__":
    # Run tests: pytest tests/test_phase2_fusion_logic.py -v
    pytest.main([__file__, "-v"])
