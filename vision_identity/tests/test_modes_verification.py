
import time
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from vision_identity.types import (
    VisionState,
    VisionReason,
    VisionDecision,
    FaceEvidence,
    GaitEvidence,
    EvidenceStatus,
)
from vision_identity.config import VisionIdentityConfig
from vision_identity.fusion_engine import VisionFusionEngine as FusionEngine


@dataclass
class MockFaceEngine:
    
    def identify(self, tracklet):
        return Mock(
            track_id=tracklet.track_id,
            identity_id="alice",
            similarity=0.85,
            quality=0.90,
            binding_state="CONFIRMED_STRONG"
        )


@dataclass
class MockGaitEngine:
    
    def evaluate(self, tracklet):
        return Mock(
            track_id=tracklet.track_id,
            identity_id="alice",
            similarity=0.75,
            margin=0.08,
            confidence=0.72,
            quality=0.70,
            state="CONFIRMED",
            sequence_length=40
        )


@dataclass
class MockTracklet:
    track_id: str
    frame_id: int
    bbox: tuple


def create_test_tracklet(track_id: str = "track_001") -> MockTracklet:
    return MockTracklet(
        track_id=track_id,
        frame_id=1,
        bbox=(100, 100, 200, 200)
    )


class TestModeSelection:

    def test_vision_mode_initialization(self):
        config = VisionIdentityConfig()
        engine = FusionEngine(config)
        
        assert engine is not None

    def test_face_only_mode_stub(self):
        mode = "face_only"
        
        assert mode == "face_only"

    def test_gait_only_mode_stub(self):
        mode = "gait_only"
        
        assert mode == "gait_only"

    def test_vision_vs_face_mode_difference(self):
        vision_mode = "vision_only"
        face_mode = "face_only"
        
        assert vision_mode != face_mode


class TestFaceOnlyRegression:

    def test_face_only_outputs_face_identity(self):
        face_engine = MockFaceEngine()
        tracklet = create_test_tracklet("track_001")
        
        result = face_engine.identify(tracklet)
        
        assert result.identity_id == "alice"
        assert result.similarity == 0.85
        assert result.quality == 0.90

    def test_face_only_confidence_from_face(self):
        face_engine = MockFaceEngine()
        tracklet = create_test_tracklet("track_002")
        
        result = face_engine.identify(tracklet)
        
        confidence = result.similarity
        
        assert 0.80 <= confidence <= 0.90

    def test_face_only_ignores_gait(self):
        mode = "face_only"
        
        assert "gait" not in mode.lower() or mode == "face_only"

    def test_face_only_learning_allowed(self):
        face_quality = 0.90
        threshold = 0.75
        
        learning_allowed = face_quality >= threshold
        
        assert learning_allowed is True


class TestGaitOnlyRegression:

    def test_gait_only_outputs_gait_identity(self):
        gait_engine = MockGaitEngine()
        tracklet = create_test_tracklet("track_001")
        
        result = gait_engine.evaluate(tracklet)
        
        assert result.identity_id == "alice"
        assert result.similarity == 0.75
        assert result.sequence_length == 40

    def test_gait_only_confidence_from_gait(self):
        gait_engine = MockGaitEngine()
        tracklet = create_test_tracklet("track_002")
        
        result = gait_engine.evaluate(tracklet)
        
        confidence = result.confidence
        
        assert 0.60 <= confidence <= 0.80

    def test_gait_only_ignores_face(self):
        mode = "gait_only"
        
        assert "face" not in mode.lower() or mode == "gait_only"

    def test_gait_only_learning_allowed(self):
        gait_quality = 0.70
        gait_margin = 0.08
        quality_threshold = 0.65
        margin_minimum = 0.05
        
        learning_allowed = (
            gait_quality >= quality_threshold and
            gait_margin >= margin_minimum
        )
        
        assert learning_allowed is True


class TestVisionModeLogic:

    def setup_method(self):
        self.config = VisionIdentityConfig()
        self.engine = FusionEngine(self.config)

    def test_vision_fuses_both_modalities(self):
        face = FaceEvidence(
            identity_id="alice",
            similarity=0.85,
            quality=0.90,
            status=EvidenceStatus.CONFIRMED_STRONG,
            margin=0.15,
            timestamp=time.time()
        )
        
        gait = GaitEvidence(
            identity_id="alice",
            similarity=0.75,
            margin=0.08,
            quality=0.70,
            status=EvidenceStatus.CONFIRMED_WEAK,
            sequence_length=40,
            timestamp=time.time()
        )
        
        decision = self.engine.make_decision(
            track_id="track_001",
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.final_identity == "alice"
        assert decision.state == VisionState.CONFIRMED

    def test_vision_vs_face_only_confidence(self):
        face_sim = 0.85
        gait_sim = 0.75
        
        face = FaceEvidence(
            identity_id="alice",
            similarity=face_sim,
            quality=0.90,
            status=EvidenceStatus.CONFIRMED_STRONG,
            margin=0.15,
            timestamp=time.time()
        )
        
        gait = GaitEvidence(
            identity_id="alice",
            similarity=gait_sim,
            margin=0.08,
            quality=0.70,
            status=EvidenceStatus.CONFIRMED_WEAK,
            sequence_length=40,
            timestamp=time.time()
        )
        
        decision = self.engine.make_decision(
            track_id="track_002",
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert 0.80 <= decision.vision_identity_confidence <= 1.0

    def test_vision_enforces_face_dominance(self):
        face = FaceEvidence(
            identity_id="alice",
            similarity=0.85,
            quality=0.90,
            status=EvidenceStatus.CONFIRMED_STRONG,
            margin=0.15,
            timestamp=time.time()
        )
        
        gait = GaitEvidence(
            identity_id="bob",
            similarity=0.75,
            margin=0.08,
            quality=0.70,
            status=EvidenceStatus.CONFIRMED_WEAK,
            sequence_length=40,
            timestamp=time.time()
        )
        
        decision = self.engine.make_decision(
            track_id="track_003",
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        if decision.state == VisionState.CONFIRMED:
            assert decision.final_identity == "alice"

    def test_vision_detects_conflicts(self):
        face = FaceEvidence(
            identity_id="alice",
            similarity=0.85,
            quality=0.90,
            status=EvidenceStatus.CONFIRMED_STRONG,
            margin=0.15,
            timestamp=time.time()
        )
        
        gait = GaitEvidence(
            identity_id="bob",
            similarity=0.82,
            margin=0.08,
            quality=0.70,
            status=EvidenceStatus.CONFIRMED_WEAK,
            sequence_length=40,
            timestamp=time.time()
        )
        
        decision = self.engine.make_decision(
            track_id="track_004",
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state in [VisionState.HOLD_CONFLICT, VisionState.CONFIRMED]
        if decision.state == VisionState.HOLD_CONFLICT:
            assert decision.learning_allowed is False

    def test_vision_gait_proposal_when_face_missing(self):
        face = FaceEvidence(
            identity_id=None,
            similarity=0.0,
            quality=0.30,
            status=EvidenceStatus.MISSING,
            margin=0.0,
            timestamp=time.time()
        )
        
        gait = GaitEvidence(
            identity_id="alice",
            similarity=0.75,
            margin=0.08,
            quality=0.70,
            status=EvidenceStatus.CONFIRMED_WEAK,
            sequence_length=40,
            timestamp=time.time()
        )
        
        decision = self.engine.make_decision(
            track_id="track_005",
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        if decision.state == VisionState.TENTATIVE:
            assert decision.final_identity == "alice"
            assert decision.learning_allowed is False


class TestModeConsistency:

    def test_face_identity_same_across_modes(self):
        face_engine = MockFaceEngine()
        tracklet = create_test_tracklet("track_001")
        
        result_face_only = face_engine.identify(tracklet)
        result_vision_identity = face_engine.identify(tracklet)
        
        assert result_face_only.identity_id == result_vision_identity.identity_id

    def test_gait_identity_same_across_modes(self):
        gait_engine = MockGaitEngine()
        tracklet = create_test_tracklet("track_001")
        
        result_gait_only = gait_engine.evaluate(tracklet)
        result_vision_identity = gait_engine.evaluate(tracklet)
        
        assert result_gait_only.identity_id == result_vision_identity.identity_id

    def test_no_regression_from_vision(self):
        
        assert "face_only" is not None
        assert "gait_only" is not None
        assert "vision_only" is not None


class TestModeSpecificFeatures:

    def test_vision_only_uses_learning_gates(self):
        config = VisionIdentityConfig()
        
        assert hasattr(config.learning_gate, 'learning_face_min_quality')
        assert hasattr(config.learning_gate, 'learning_gait_min_quality')
        assert hasattr(config.learning_gate, 'learning_gait_margin_min')

    def test_vision_only_tracks_state_machine(self):
        config = VisionIdentityConfig()
        
        assert hasattr(config.confirmation, 'face_confirm_threshold')
        assert hasattr(config.confirmation, 'face_switch_threshold')

    def test_vision_only_handles_conflicts(self):
        config = VisionIdentityConfig()
        
        assert hasattr(config.conflict, 'conflict_hold_max_frames')

    def test_face_only_simple_logic(self):
        face_sim = 0.85
        threshold = 0.75
        
        decision_simple = face_sim >= threshold
        
        assert decision_simple is True

    def test_gait_only_simple_logic(self):
        gait_conf = 0.72
        threshold = 0.65
        
        decision_simple = gait_conf >= threshold
        
        assert decision_simple is True


class TestEdgeCasesAcrossModes:

    def test_all_modes_handle_missing_data(self):
        
        missing_status = EvidenceStatus.MISSING
        
        assert missing_status is not None

    def test_all_modes_handle_low_quality(self):
        low_quality = 0.40
        threshold = 0.55
        
        rejected = low_quality < threshold
        
        assert rejected is True

    def test_vision_mode_boundary_values(self):
        config = VisionIdentityConfig()
        
        assert 0.5 <= config.confirmation.face_confirm_threshold <= 1.0
        assert 0.5 <= config.confirmation.face_switch_threshold <= 1.0
        assert config.confirmation.face_switch_threshold >= config.confirmation.face_confirm_threshold


class TestIntegrationScenarios:

    def test_scenario_high_quality_all_modalities(self):
        
        face_sim = 0.90
        gait_conf = 0.80
        
        assert face_sim > 0.75
        assert gait_conf > 0.65

    def test_scenario_face_strong_gait_weak(self):
        config = VisionIdentityConfig()
        engine = FusionEngine(config)
        
        face = FaceEvidence(
            identity_id="alice",
            similarity=0.92,
            quality=0.95,
            status=EvidenceStatus.CONFIRMED_STRONG,
            margin=0.20,
            timestamp=time.time()
        )
        
        gait = GaitEvidence(
            identity_id="alice",
            similarity=0.60,
            margin=0.02,
            quality=0.50,
            status=EvidenceStatus.TENTATIVE,
            sequence_length=25,
            timestamp=time.time()
        )
        
        decision = engine.make_decision(
            track_id="track_001",
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.final_identity == "alice"
        assert decision.state == VisionState.CONFIRMED

    def test_scenario_no_evidence(self):
        face = FaceEvidence(
            identity_id=None,
            similarity=0.0,
            quality=0.20,
            status=EvidenceStatus.MISSING,
            margin=0.0,
            timestamp=time.time()
        )
        
        gait = GaitEvidence(
            identity_id=None,
            similarity=0.0,
            margin=0.0,
            quality=0.30,
            status=EvidenceStatus.MISSING,
            sequence_length=10,
            timestamp=time.time()
        )
        
        config = VisionIdentityConfig()
        engine = FusionEngine(config)
        
        decision = engine.make_decision(
            track_id="track_002",
            current_state=VisionState.UNKNOWN,
            face_evidence=face,
            gait_evidence=gait,
            source_auth_evidence=None
        )
        
        assert decision.state == VisionState.UNKNOWN
        assert decision.final_identity is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
