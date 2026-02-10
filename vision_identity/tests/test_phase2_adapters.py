
import time
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from vision_identity.types import (
    EvidenceStatus,
    FaceEvidence,
    GaitEvidence,
    SourceAuthEvidence,
    SourceAuthState,
)
from vision_identity.adapters.face_adapter import FaceAdapter
from vision_identity.adapters.gait_adapter import GaitAdapter
from vision_identity.adapters.source_auth_adapter import SourceAuthAdapter


@dataclass
class MockIdentityDecision:
    track_id: str
    identity_id: str
    similarity: float
    quality: float
    binding_state: str
    margin: float


@dataclass
class MockGaitTrackState:
    track_id: str
    identity_id: str
    similarity: float
    margin: float
    confidence: float
    quality: float
    state: str
    sequence_length: int


@dataclass
class MockSourceAuthScores:
    track_id: str
    realness_score: float
    spoof_confidence: float


@dataclass
class MockTracklet:
    track_id: str
    frame_id: int
    bbox: tuple
    face_crops: list = None
    gait_sequence_data: dict = None


class TestFaceAdapter:

    def setup_method(self):
        self.adapter = FaceAdapter()

    def test_adapter_initialization(self):
        assert self.adapter is not None
        assert hasattr(self.adapter, 'adapt')

    def test_adapt_confirmed_strong_face(self):
        decision = MockIdentityDecision(
            track_id="track_001",
            identity_id="alice",
            similarity=0.88,
            quality=0.92,
            binding_state="CONFIRMED_STRONG",
            margin=0.15
        )
        
        evidence = self.adapter.adapt(decision)
        
        assert evidence.track_id == "track_001"
        assert evidence.identity_id == "alice"
        assert evidence.similarity == 0.88
        assert evidence.quality == 0.92
        assert evidence.margin == 0.15
        assert evidence.status == EvidenceStatus.CONFIRMED_STRONG

    def test_adapt_confirmed_weak_face(self):
        decision = MockIdentityDecision(
            track_id="track_002",
            identity_id="bob",
            similarity=0.72,
            quality=0.65,
            binding_state="CONFIRMED_WEAK",
            margin=0.08
        )
        
        evidence = self.adapter.adapt(decision)
        
        assert evidence.status == EvidenceStatus.CONFIRMED_WEAK
        assert evidence.similarity == 0.72
        assert evidence.quality == 0.65

    def test_adapt_pending_face(self):
        decision = MockIdentityDecision(
            track_id="track_003",
            identity_id="charlie",
            similarity=0.68,
            quality=0.70,
            binding_state="PENDING",
            margin=0.05
        )
        
        evidence = self.adapter.adapt(decision)
        
        assert evidence.status == EvidenceStatus.TENTATIVE
        assert evidence.identity_id == "charlie"

    def test_adapt_unknown_face(self):
        decision = MockIdentityDecision(
            track_id="track_004",
            identity_id=None,
            similarity=0.0,
            quality=0.40,
            binding_state="UNKNOWN",
            margin=0.0
        )
        
        evidence = self.adapter.adapt(decision)
        
        assert evidence.status == EvidenceStatus.MISSING
        assert evidence.identity_id is None
        assert evidence.similarity == 0.0

    def test_adapt_stale_face(self):
        decision = MockIdentityDecision(
            track_id="track_005",
            identity_id="alice",
            similarity=0.85,
            quality=0.88,
            binding_state="STALE",
            margin=0.12
        )
        
        evidence = self.adapter.adapt(decision)
        
        assert evidence.status == EvidenceStatus.UNKNOWN
        assert evidence.identity_id == "alice"

    def test_adapt_low_quality_face_gate(self):
        decision = MockIdentityDecision(
            track_id="track_006",
            identity_id="diana",
            similarity=0.82,
            quality=0.40,
            binding_state="CONFIRMED_WEAK",
            margin=0.10
        )
        
        evidence = self.adapter.adapt(decision)
        
        assert evidence.quality == 0.40

    def test_adapt_preserves_timestamp(self):
        decision = MockIdentityDecision(
            track_id="track_007",
            identity_id="eve",
            similarity=0.80,
            quality=0.85,
            binding_state="CONFIRMED_STRONG",
            margin=0.10
        )
        
        before_time = time.time()
        evidence = self.adapter.adapt(decision)
        after_time = time.time()
        
        assert before_time <= evidence.timestamp <= after_time

    def test_adapt_none_input_handling(self):
        evidence = self.adapter.adapt(None)
        
        assert evidence is not None
        assert evidence.status == EvidenceStatus.MISSING
        assert evidence.identity_id is None
        assert evidence.similarity == 0.0


class TestGaitAdapter:

    def setup_method(self):
        self.adapter = GaitAdapter()

    def test_adapter_initialization(self):
        assert self.adapter is not None
        assert hasattr(self.adapter, 'adapt')

    def test_adapt_confirmed_gait(self):
        gait_state = MockGaitTrackState(
            track_id="track_001",
            identity_id="alice",
            similarity=0.78,
            margin=0.08,
            confidence=0.75,
            quality=0.72,
            state="CONFIRMED",
            sequence_length=45
        )
        
        evidence = self.adapter.adapt(gait_state)
        
        assert evidence.track_id == "track_001"
        assert evidence.identity_id == "alice"
        assert evidence.similarity == 0.78
        assert evidence.margin == 0.08
        assert evidence.quality == 0.72
        assert evidence.sequence_length == 45
        assert evidence.status == EvidenceStatus.CONFIRMED_WEAK

    def test_adapt_tentative_gait(self):
        gait_state = MockGaitTrackState(
            track_id="track_002",
            identity_id="bob",
            similarity=0.70,
            margin=0.05,
            confidence=0.65,
            quality=0.68,
            state="EVALUATING",
            sequence_length=32
        )
        
        evidence = self.adapter.adapt(gait_state)
        
        assert evidence.status == EvidenceStatus.TENTATIVE
        assert evidence.identity_id == "bob"

    def test_adapt_collecting_gait(self):
        gait_state = MockGaitTrackState(
            track_id="track_003",
            identity_id=None,
            similarity=0.0,
            margin=0.0,
            confidence=0.0,
            quality=0.30,
            state="COLLECTING",
            sequence_length=15
        )
        
        evidence = self.adapter.adapt(gait_state)
        
        assert evidence.status == EvidenceStatus.MISSING
        assert evidence.identity_id is None
        assert evidence.sequence_length == 15

    def test_adapt_minimum_sequence_length_gate(self):
        gait_state = MockGaitTrackState(
            track_id="track_004",
            identity_id="charlie",
            similarity=0.80,
            margin=0.10,
            confidence=0.75,
            quality=0.70,
            state="CONFIRMED",
            sequence_length=20
        )
        
        evidence = self.adapter.adapt(gait_state)
        
        assert evidence.status in [EvidenceStatus.MISSING, EvidenceStatus.TENTATIVE]

    def test_adapt_gait_margin_safety_check(self):
        gait_state = MockGaitTrackState(
            track_id="track_005",
            identity_id="diana",
            similarity=0.76,
            margin=0.12,
            confidence=0.72,
            quality=0.70,
            state="CONFIRMED",
            sequence_length=40
        )
        
        evidence = self.adapter.adapt(gait_state)
        
        assert evidence.margin == 0.12
        assert evidence.status in [EvidenceStatus.CONFIRMED_WEAK, EvidenceStatus.TENTATIVE]

    def test_adapt_low_margin_gait(self):
        gait_state = MockGaitTrackState(
            track_id="track_006",
            identity_id="eve",
            similarity=0.68,
            margin=0.02,
            confidence=0.60,
            quality=0.65,
            state="CONFIRMED",
            sequence_length=38
        )
        
        evidence = self.adapter.adapt(gait_state)
        
        assert evidence.margin == 0.02

    def test_adapt_preserves_sequence_length(self):
        gait_state = MockGaitTrackState(
            track_id="track_007",
            identity_id="frank",
            similarity=0.75,
            margin=0.08,
            confidence=0.70,
            quality=0.70,
            state="CONFIRMED",
            sequence_length=52
        )
        
        evidence = self.adapter.adapt(gait_state)
        
        assert evidence.sequence_length == 52

    def test_adapt_unsure_gait(self):
        gait_state = MockGaitTrackState(
            track_id="track_008",
            identity_id="grace",
            similarity=0.62,
            margin=0.03,
            confidence=0.55,
            quality=0.60,
            state="UNSURE",
            sequence_length=35
        )
        
        evidence = self.adapter.adapt(gait_state)
        
        assert evidence.status == EvidenceStatus.TENTATIVE
        assert evidence.confidence is not None or evidence.similarity is not None

    def test_adapt_none_input_handling(self):
        evidence = self.adapter.adapt(None)
        
        assert evidence is not None
        assert evidence.status == EvidenceStatus.MISSING
        assert evidence.identity_id is None


class TestSourceAuthAdapter:

    def setup_method(self):
        self.adapter = SourceAuthAdapter()

    def test_adapter_initialization(self):
        assert self.adapter is not None
        assert hasattr(self.adapter, 'adapt')

    def test_adapt_real_spoof_score(self):
        scores = MockSourceAuthScores(
            track_id="track_001",
            realness_score=0.92,
            spoof_confidence=0.05
        )
        
        evidence = self.adapter.adapt(scores)
        
        assert evidence.track_id == "track_001"
        assert evidence.realness_score == 0.92
        assert evidence.state in [
            SourceAuthState.REAL,
            SourceAuthState.LIKELY_REAL,
        ]

    def test_adapt_spoof_score(self):
        scores = MockSourceAuthScores(
            track_id="track_002",
            realness_score=0.15,
            spoof_confidence=0.85
        )
        
        evidence = self.adapter.adapt(scores)
        
        assert evidence.realness_score == 0.15
        assert evidence.state in [
            SourceAuthState.SPOOF,
            SourceAuthState.LIKELY_SPOOF,
        ]

    def test_adapt_uncertain_spoof_score(self):
        scores = MockSourceAuthScores(
            track_id="track_003",
            realness_score=0.50,
            spoof_confidence=0.50
        )
        
        evidence = self.adapter.adapt(scores)
        
        assert evidence.realness_score == 0.50
        assert evidence.state == SourceAuthState.UNCERTAIN

    def test_adapt_high_confidence_real(self):
        scores = MockSourceAuthScores(
            track_id="track_004",
            realness_score=0.98,
            spoof_confidence=0.01
        )
        
        evidence = self.adapter.adapt(scores)
        
        assert evidence.state == SourceAuthState.REAL
        assert evidence.realness_score == 0.98

    def test_adapt_high_confidence_spoof(self):
        scores = MockSourceAuthScores(
            track_id="track_005",
            realness_score=0.05,
            spoof_confidence=0.95
        )
        
        evidence = self.adapter.adapt(scores)
        
        assert evidence.state == SourceAuthState.SPOOF
        assert evidence.realness_score == 0.05

    def test_adapt_none_input_handling(self):
        evidence = self.adapter.adapt(None)
        
        assert evidence is not None
        assert evidence.state == SourceAuthState.MISSING


class TestAdapterNonInvasiveness:

    def test_face_adapter_does_not_modify_decision(self):
        adapter = FaceAdapter()
        original = MockIdentityDecision(
            track_id="track_001",
            identity_id="alice",
            similarity=0.85,
            quality=0.90,
            binding_state="CONFIRMED_STRONG",
            margin=0.15
        )
        
        orig_values = (
            original.track_id,
            original.identity_id,
            original.similarity,
            original.quality,
            original.binding_state,
            original.margin
        )
        
        adapter.adapt(original)
        
        new_values = (
            original.track_id,
            original.identity_id,
            original.similarity,
            original.quality,
            original.binding_state,
            original.margin
        )
        
        assert orig_values == new_values

    def test_gait_adapter_does_not_modify_state(self):
        adapter = GaitAdapter()
        original = MockGaitTrackState(
            track_id="track_001",
            identity_id="alice",
            similarity=0.78,
            margin=0.08,
            confidence=0.75,
            quality=0.72,
            state="CONFIRMED",
            sequence_length=45
        )
        
        orig_values = (
            original.track_id,
            original.identity_id,
            original.similarity,
            original.margin,
            original.confidence,
            original.quality,
            original.state,
            original.sequence_length
        )
        
        adapter.adapt(original)
        
        new_values = (
            original.track_id,
            original.identity_id,
            original.similarity,
            original.margin,
            original.confidence,
            original.quality,
            original.state,
            original.sequence_length
        )
        
        assert orig_values == new_values

    def test_source_auth_adapter_does_not_modify_scores(self):
        adapter = SourceAuthAdapter()
        original = MockSourceAuthScores(
            track_id="track_001",
            realness_score=0.92,
            spoof_confidence=0.05
        )
        
        orig_values = (
            original.track_id,
            original.realness_score,
            original.spoof_confidence
        )
        
        adapter.adapt(original)
        
        new_values = (
            original.track_id,
            original.realness_score,
            original.spoof_confidence
        )
        
        assert orig_values == new_values


class TestAdapterBatchProcessing:

    def test_face_adapter_batch_processing(self):
        adapter = FaceAdapter()
        
        decisions = [
            MockIdentityDecision("track_001", "alice", 0.88, 0.92, "CONFIRMED_STRONG", 0.15),
            MockIdentityDecision("track_002", "bob", 0.72, 0.70, "CONFIRMED_WEAK", 0.08),
            MockIdentityDecision("track_003", None, 0.0, 0.40, "UNKNOWN", 0.0),
        ]
        
        evidences = [adapter.adapt(d) for d in decisions]
        
        assert len(evidences) == 3
        assert evidences[0].identity_id == "alice"
        assert evidences[1].identity_id == "bob"
        assert evidences[2].identity_id is None

    def test_gait_adapter_batch_processing(self):
        adapter = GaitAdapter()
        
        states = [
            MockGaitTrackState("track_001", "alice", 0.78, 0.08, 0.75, 0.72, "CONFIRMED", 45),
            MockGaitTrackState("track_002", "bob", 0.70, 0.05, 0.65, 0.68, "EVALUATING", 32),
            MockGaitTrackState("track_003", None, 0.0, 0.0, 0.0, 0.30, "COLLECTING", 15),
        ]
        
        evidences = [adapter.adapt(s) for s in states]
        
        assert len(evidences) == 3
        assert evidences[0].status in [EvidenceStatus.CONFIRMED_WEAK, EvidenceStatus.TENTATIVE]
        assert evidences[1].status == EvidenceStatus.TENTATIVE
        assert evidences[2].status == EvidenceStatus.MISSING


class TestAdapterErrorHandling:

    def test_face_adapter_missing_fields_handling(self):
        adapter = FaceAdapter()
        
        partial = Mock()
        partial.track_id = "track_001"
        partial.identity_id = "alice"
        
        try:
            evidence = adapter.adapt(partial)
            assert evidence is not None
            assert evidence.status in [EvidenceStatus.MISSING, EvidenceStatus.UNKNOWN]
        except Exception as e:
            pytest.fail(f"Adapter should handle missing fields gracefully: {e}")

    def test_gait_adapter_missing_fields_handling(self):
        adapter = GaitAdapter()
        
        partial = Mock()
        partial.track_id = "track_001"
        
        try:
            evidence = adapter.adapt(partial)
            assert evidence is not None
            assert evidence.status in [EvidenceStatus.MISSING, EvidenceStatus.UNKNOWN]
        except Exception as e:
            pytest.fail(f"Adapter should handle missing fields gracefully: {e}")

    def test_adapters_idempotent(self):
        face_adapter = FaceAdapter()
        
        decision = MockIdentityDecision(
            "track_001", "alice", 0.88, 0.92, "CONFIRMED_STRONG", 0.15
        )
        
        evidence1 = face_adapter.adapt(decision)
        evidence2 = face_adapter.adapt(decision)
        
        assert evidence1.similarity == evidence2.similarity
        assert evidence1.status == evidence2.status
        assert evidence1.identity_id == evidence2.identity_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
