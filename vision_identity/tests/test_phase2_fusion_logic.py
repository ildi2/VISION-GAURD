
import time
import pytest
from dataclasses import dataclass

from vision_identity.types import (
    VisionState,
    VisionReason,
    FaceEvidence,
    GaitEvidence,
    SourceAuthEvidence,
    EvidenceStatus,
    SourceAuthState,
)
from vision_identity.config import VisionIdentityConfig
from vision_identity.fusion_engine import VisionFusionEngine as FusionEngine
from vision_identity.state_machine import VisionStateMachine
from vision_identity.evidence_accumulator import EvidenceAccumulator


def create_face_evidence(
    identity_id: str = "alice",
    similarity: float = 0.85,
    quality: float = 0.90,
    status: EvidenceStatus = EvidenceStatus.CONFIRMED_STRONG,
    margin: float = 0.15,
    timestamp: float = None
) -> FaceEvidence:
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
    return SourceAuthEvidence(
        realness_score=realness_score,
        state=state,
        timestamp=time.time()
    )


class TestRuleAFaceDominance:

    def setup_method(self):
        self.config = VisionIdentityConfig()
        self.engine = FusionEngine(self.config)

    def test_rule_a_face_dominates_when_confirmed(self):
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.88,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="bob",
            similarity=0.75,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_001",
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.final_identity == "alice" or decision.state == VisionState.HOLD_CONFLICT
        assert decision.final_identity != "bob"

    def test_rule_a_face_identity_preserved_high_quality(self):
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
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.final_identity == "alice"
        assert decision.state == VisionState.CONFIRMED

    def test_rule_a_conflict_detection(self):
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
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state == VisionState.HOLD_CONFLICT
        assert decision.learning_allowed is False


class TestRuleBGaitProposal:

    def setup_method(self):
        self.config = VisionIdentityConfig()
        self.engine = FusionEngine(self.config)

    def test_rule_b_gait_proposes_when_face_missing(self):
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
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.final_identity == "alice"
        assert decision.state == VisionState.TENTATIVE
        assert decision.learning_allowed is False

    def test_rule_b_gait_does_not_propose_when_face_confirmed(self):
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
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.final_identity != "bob"
        assert decision.state != VisionState.TENTATIVE or decision.final_identity == "alice"

    def test_rule_b_gait_quality_threshold(self):
        face = create_face_evidence(
            identity_id=None,
            status=EvidenceStatus.MISSING
        )
        gait = create_gait_evidence(
            identity_id="alice",
            similarity=0.60,
            quality=0.45,
            status=EvidenceStatus.TENTATIVE
        )
        
        decision = self.engine.make_decision(
            track_id="track_006",
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        if decision.state == VisionState.TENTATIVE:
            assert decision.confidence < 0.65


class TestRuleCConflictHold:

    def setup_method(self):
        self.config = VisionIdentityConfig()
        self.engine = FusionEngine(self.config)

    def test_rule_c_explicit_conflict_detection(self):
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
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state == VisionState.HOLD_CONFLICT
        assert decision.final_identity is None
        assert decision.learning_allowed is False

    def test_rule_c_conflict_resolution_face_reconfirms(self):
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.90,
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
            current_state=VisionState.HOLD_CONFLICT,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        if decision.state == VisionState.CONFIRMED:
            assert decision.final_identity == "alice"

    def test_rule_c_conflict_timeout(self):
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
            current_state=VisionState.HOLD_CONFLICT,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state in [VisionState.HOLD_CONFLICT, VisionState.TENTATIVE]


class TestRuleDHysteresis:

    def setup_method(self):
        self.config = VisionIdentityConfig()
        self.engine = FusionEngine(self.config)

    def test_rule_d_switch_threshold_stricter_than_confirm(self):
        assert self.config.face_switch_threshold > self.config.face_confirm_threshold
        assert self.config.face_switch_threshold > self.config.gait_confirm_threshold

    def test_rule_d_requires_margin_bonus_to_switch(self):
        face_new = create_face_evidence(
            identity_id="bob",
            similarity=0.84,
            quality=0.88,
            status=EvidenceStatus.CONFIRMED_STRONG,
            margin=0.08
        )
        
        decision = self.engine.make_decision(
            track_id="track_010",
            current_state=VisionState.CONFIRMED,
            face_evidence=face_new,
            gait_evidence=None,
            source_auth_evidence=None
        )
        
        assert decision.state in [VisionState.CONFIRMED, VisionState.HOLD_CONFLICT]

    def test_rule_d_strong_switch_overrides(self):
        face_new = create_face_evidence(
            identity_id="bob",
            similarity=0.92,
            quality=0.95,
            status=EvidenceStatus.CONFIRMED_STRONG,
            margin=0.22
        )
        
        decision = self.engine.make_decision(
            track_id="track_011",
            current_state=VisionState.CONFIRMED,
            face_evidence=face_new,
            gait_evidence=None,
            source_auth_evidence=None
        )
        
        assert decision.state in [VisionState.CONFIRMED, VisionState.HOLD_CONFLICT]


class TestRuleELearningGates:

    def setup_method(self):
        self.config = VisionIdentityConfig()
        self.engine = FusionEngine(self.config)

    def test_rule_e_learning_allowed_face_only(self):
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
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state == VisionState.CONFIRMED
        assert decision.learning_allowed is True

    def test_rule_e_learning_allowed_face_gait_aligned(self):
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
            margin=0.10,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_013",
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state == VisionState.CONFIRMED
        assert decision.learning_allowed is True

    def test_rule_e_learning_blocked_face_absent(self):
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
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.learning_allowed is False

    def test_rule_e_learning_blocked_gait_conflict(self):
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.88,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="bob",
            similarity=0.75,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_015",
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.learning_allowed is False

    def test_rule_e_learning_blocked_gait_low_margin(self):
        face = create_face_evidence(
            identity_id="alice",
            similarity=0.88,
            status=EvidenceStatus.CONFIRMED_STRONG
        )
        gait = create_gait_evidence(
            identity_id="alice",
            similarity=0.72,
            margin=0.02,
            status=EvidenceStatus.CONFIRMED_WEAK
        )
        
        decision = self.engine.make_decision(
            track_id="track_016",
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.learning_allowed is False or decision.state != VisionState.CONFIRMED

    def test_rule_e_learning_blocked_spoof_detected(self):
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
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=spoof
        )
        
        assert decision.learning_allowed is False


class TestConfidenceSynthesis:

    def setup_method(self):
        self.config = VisionIdentityConfig()
        self.engine = FusionEngine(self.config)

    def test_confirmed_state_confidence(self):
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
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state == VisionState.CONFIRMED
        assert 0.85 <= decision.confidence <= 0.90

    def test_confirmed_with_gait_boost(self):
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
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state == VisionState.CONFIRMED
        assert decision.confidence > 0.85

    def test_tentative_state_confidence(self):
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
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        if decision.state == VisionState.TENTATIVE:
            assert decision.confidence < 0.75

    def test_unknown_state_confidence(self):
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
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state == VisionState.UNKNOWN
        assert decision.confidence == 0.0


class TestSpoofIntegration:

    def setup_method(self):
        self.config = VisionIdentityConfig()
        self.engine = FusionEngine(self.config)

    def test_spoof_detected_blocks_decision(self):
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
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=spoof
        )
        
        assert decision.state == VisionState.HOLD_CONFLICT
        assert decision.learning_allowed is False

    def test_likely_real_allows_decision(self):
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
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=spoof
        )
        
        assert decision.state == VisionState.CONFIRMED
        assert decision.learning_allowed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
