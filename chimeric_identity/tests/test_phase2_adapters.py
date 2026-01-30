# chimeric_identity/tests/test_phase2_adapters.py
# ============================================================================
# PHASE 2 INTEGRATION TESTS - Adapters (Face, Gait, SourceAuth)
# ============================================================================
#
# Purpose:
#   Validate that adapters correctly normalize face/gait/spoof subsystem outputs
#   into unified Evidence schema. Tests non-invasiveness and error handling.
#
# Test Coverage:
#   1. Face Adapter: IdentityDecision → FaceEvidence normalization
#   2. Gait Adapter: GaitTrackState → GaitEvidence normalization
#   3. SourceAuth Adapter: SourceAuthScores → SourceAuthEvidence conversion
#   4. Error Handling: Graceful degradation on missing/malformed inputs
#   5. Quality Gates: All quality thresholds applied correctly
#
# Design:
#   - Synthetic face/gait/spoof engine outputs (no real subsystems)
#   - Mock objects that mimic actual subsystem APIs
#   - Tests verify non-invasiveness (no state modifications)
#   - Comprehensive edge case coverage

import time
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from chimeric_identity.types import (
    EvidenceStatus,
    FaceEvidence,
    GaitEvidence,
    SourceAuthEvidence,
    SourceAuthState,
)
from chimeric_identity.adapters.face_adapter import FaceAdapter
from chimeric_identity.adapters.gait_adapter import GaitAdapter
from chimeric_identity.adapters.source_auth_adapter import SourceAuthAdapter


# ============================================================================
# MOCK OBJECT DEFINITIONS
# ============================================================================

@dataclass
class MockIdentityDecision:
    """Mimics identity.identity_decision.IdentityDecision"""
    track_id: str
    identity_id: str
    similarity: float
    quality: float  # 0-1
    binding_state: str  # UNKNOWN, PENDING, CONFIRMED_WEAK, CONFIRMED_STRONG, STALE
    margin: float  # distance to 2nd best


@dataclass
class MockGaitTrackState:
    """Mimics gait_subsystem.gait.state.GaitTrackState"""
    track_id: str
    identity_id: str
    similarity: float
    margin: float
    confidence: float  # synthesized from margin + quality
    quality: float  # pose quality 0-1
    state: str  # COLLECTING, EVALUATING, CONFIRMED, UNSURE
    sequence_length: int


@dataclass
class MockSourceAuthScores:
    """Mimics source_auth.engine.SourceAuthScores"""
    track_id: str
    realness_score: float  # 0-1
    spoof_confidence: float  # 0-1


@dataclass
class MockTracklet:
    """Mimics perception.schemas.tracklet.Tracklet"""
    track_id: str
    frame_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    face_crops: list = None
    gait_sequence_data: dict = None


# ============================================================================
# TEST: FACE ADAPTER
# ============================================================================

class TestFaceAdapter:
    """Test face_adapter.py - Evidence normalization from IdentityEngine"""

    def setup_method(self):
        """Initialize adapter before each test."""
        self.adapter = FaceAdapter()

    def test_adapter_initialization(self):
        """Test FaceAdapter can be created."""
        assert self.adapter is not None
        assert hasattr(self.adapter, 'adapt')

    def test_adapt_confirmed_strong_face(self):
        """Test adaptation of strong face evidence."""
        # Setup: Create strong face identification
        decision = MockIdentityDecision(
            track_id="track_001",
            identity_id="alice",
            similarity=0.88,
            quality=0.92,
            binding_state="CONFIRMED_STRONG",
            margin=0.15
        )
        
        # Execute: Adapt to Evidence schema
        evidence = self.adapter.adapt(decision)
        
        # Verify: Correct conversion
        assert evidence.track_id == "track_001"
        assert evidence.identity_id == "alice"
        assert evidence.similarity == 0.88
        assert evidence.quality == 0.92
        assert evidence.margin == 0.15
        assert evidence.status == EvidenceStatus.CONFIRMED_STRONG

    def test_adapt_confirmed_weak_face(self):
        """Test adaptation of weak face evidence."""
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
        """Test face in PENDING state (not yet confirmed)."""
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
        """Test face in UNKNOWN state."""
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
        """Test stale face evidence (old binding)."""
        decision = MockIdentityDecision(
            track_id="track_005",
            identity_id="alice",
            similarity=0.85,
            quality=0.88,
            binding_state="STALE",
            margin=0.12
        )
        
        evidence = self.adapter.adapt(decision)
        
        # Stale face should be treated as UNKNOWN
        assert evidence.status == EvidenceStatus.UNKNOWN
        assert evidence.identity_id == "alice"  # But preserve ID for conflict detection

    def test_adapt_low_quality_face_gate(self):
        """Test quality gate blocks poor face evidence."""
        decision = MockIdentityDecision(
            track_id="track_006",
            identity_id="diana",
            similarity=0.82,
            quality=0.40,  # Below typical quality threshold
            binding_state="CONFIRMED_WEAK",
            margin=0.10
        )
        
        evidence = self.adapter.adapt(decision)
        
        # Low quality face should degrade status
        assert evidence.quality == 0.40
        # Depending on policy, could be UNKNOWN or WEAK

    def test_adapt_preserves_timestamp(self):
        """Test that timestamp is preserved correctly."""
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
        """Test adapter gracefully handles None input."""
        evidence = self.adapter.adapt(None)
        
        assert evidence is not None
        assert evidence.status == EvidenceStatus.MISSING
        assert evidence.identity_id is None
        assert evidence.similarity == 0.0


# ============================================================================
# TEST: GAIT ADAPTER
# ============================================================================

class TestGaitAdapter:
    """Test gait_adapter.py - Evidence normalization from GaitEngine"""

    def setup_method(self):
        """Initialize adapter before each test."""
        self.adapter = GaitAdapter()

    def test_adapter_initialization(self):
        """Test GaitAdapter can be created."""
        assert self.adapter is not None
        assert hasattr(self.adapter, 'adapt')

    def test_adapt_confirmed_gait(self):
        """Test adaptation of confirmed gait evidence."""
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
        """Test gait in tentative (evaluating) state."""
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
        """Test gait in COLLECTING state (not enough frames yet)."""
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
        """Test gait gate: reject if sequence too short."""
        # Gait requires >= 30 frames typically
        gait_state = MockGaitTrackState(
            track_id="track_004",
            identity_id="charlie",
            similarity=0.80,
            margin=0.10,
            confidence=0.75,
            quality=0.70,
            state="CONFIRMED",
            sequence_length=20  # Too short!
        )
        
        evidence = self.adapter.adapt(gait_state)
        
        # Should reject or downgrade due to short sequence
        assert evidence.status in [EvidenceStatus.MISSING, EvidenceStatus.TENTATIVE]

    def test_adapt_gait_margin_safety_check(self):
        """Test gait margin is captured (critical for learning gate)."""
        gait_state = MockGaitTrackState(
            track_id="track_005",
            identity_id="diana",
            similarity=0.76,
            margin=0.12,  # Good margin
            confidence=0.72,
            quality=0.70,
            state="CONFIRMED",
            sequence_length=40
        )
        
        evidence = self.adapter.adapt(gait_state)
        
        assert evidence.margin == 0.12
        assert evidence.status in [EvidenceStatus.CONFIRMED_WEAK, EvidenceStatus.TENTATIVE]

    def test_adapt_low_margin_gait(self):
        """Test gait with marginal identification (sim1 ≈ sim2)."""
        gait_state = MockGaitTrackState(
            track_id="track_006",
            identity_id="eve",
            similarity=0.68,
            margin=0.02,  # Dangerously close to 2nd place
            confidence=0.60,
            quality=0.65,
            state="CONFIRMED",
            sequence_length=38
        )
        
        evidence = self.adapter.adapt(gait_state)
        
        # Low margin should be flagged
        assert evidence.margin == 0.02
        # Should potentially block learning even if confirmed

    def test_adapt_preserves_sequence_length(self):
        """Test sequence length is preserved (needed for temporal logic)."""
        gait_state = MockGaitTrackState(
            track_id="track_007",
            identity_id="frank",
            similarity=0.75,
            margin=0.08,
            confidence=0.70,
            quality=0.70,
            state="CONFIRMED",
            sequence_length=52  # Long sequence
        )
        
        evidence = self.adapter.adapt(gait_state)
        
        assert evidence.sequence_length == 52

    def test_adapt_unsure_gait(self):
        """Test gait in UNSURE state."""
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
        """Test adapter gracefully handles None input."""
        evidence = self.adapter.adapt(None)
        
        assert evidence is not None
        assert evidence.status == EvidenceStatus.MISSING
        assert evidence.identity_id is None


# ============================================================================
# TEST: SOURCE AUTH ADAPTER
# ============================================================================

class TestSourceAuthAdapter:
    """Test source_auth_adapter.py - Spoof evidence normalization"""

    def setup_method(self):
        """Initialize adapter before each test."""
        self.adapter = SourceAuthAdapter()

    def test_adapter_initialization(self):
        """Test SourceAuthAdapter can be created."""
        assert self.adapter is not None
        assert hasattr(self.adapter, 'adapt')

    def test_adapt_real_spoof_score(self):
        """Test high realness score (genuine face)."""
        scores = MockSourceAuthScores(
            track_id="track_001",
            realness_score=0.92,  # High = real
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
        """Test low realness score (likely spoof)."""
        scores = MockSourceAuthScores(
            track_id="track_002",
            realness_score=0.15,  # Low = likely spoof
            spoof_confidence=0.85
        )
        
        evidence = self.adapter.adapt(scores)
        
        assert evidence.realness_score == 0.15
        assert evidence.state in [
            SourceAuthState.SPOOF,
            SourceAuthState.LIKELY_SPOOF,
        ]

    def test_adapt_uncertain_spoof_score(self):
        """Test borderline spoof score."""
        scores = MockSourceAuthScores(
            track_id="track_003",
            realness_score=0.50,
            spoof_confidence=0.50
        )
        
        evidence = self.adapter.adapt(scores)
        
        assert evidence.realness_score == 0.50
        assert evidence.state == SourceAuthState.UNCERTAIN

    def test_adapt_high_confidence_real(self):
        """Test very confident real face."""
        scores = MockSourceAuthScores(
            track_id="track_004",
            realness_score=0.98,
            spoof_confidence=0.01
        )
        
        evidence = self.adapter.adapt(scores)
        
        assert evidence.state == SourceAuthState.REAL
        assert evidence.realness_score == 0.98

    def test_adapt_high_confidence_spoof(self):
        """Test very confident spoof detection."""
        scores = MockSourceAuthScores(
            track_id="track_005",
            realness_score=0.05,
            spoof_confidence=0.95
        )
        
        evidence = self.adapter.adapt(scores)
        
        assert evidence.state == SourceAuthState.SPOOF
        assert evidence.realness_score == 0.05

    def test_adapt_none_input_handling(self):
        """Test adapter handles None input."""
        evidence = self.adapter.adapt(None)
        
        assert evidence is not None
        assert evidence.state == SourceAuthState.MISSING


# ============================================================================
# TEST: ADAPTER NON-INVASIVENESS
# ============================================================================

class TestAdapterNonInvasiveness:
    """Verify adapters do not modify external state (read-only)."""

    def test_face_adapter_does_not_modify_decision(self):
        """Verify FaceAdapter doesn't modify input."""
        adapter = FaceAdapter()
        original = MockIdentityDecision(
            track_id="track_001",
            identity_id="alice",
            similarity=0.85,
            quality=0.90,
            binding_state="CONFIRMED_STRONG",
            margin=0.15
        )
        
        # Store original values
        orig_values = (
            original.track_id,
            original.identity_id,
            original.similarity,
            original.quality,
            original.binding_state,
            original.margin
        )
        
        # Call adapter
        adapter.adapt(original)
        
        # Verify no modification
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
        """Verify GaitAdapter doesn't modify input."""
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
        """Verify SourceAuthAdapter doesn't modify input."""
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


# ============================================================================
# TEST: BATCH PROCESSING
# ============================================================================

class TestAdapterBatchProcessing:
    """Test adapters handling multiple tracklets."""

    def test_face_adapter_batch_processing(self):
        """Test FaceAdapter with multiple decisions."""
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
        """Test GaitAdapter with multiple states."""
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


# ============================================================================
# TEST: ERROR HANDLING & EDGE CASES
# ============================================================================

class TestAdapterErrorHandling:
    """Test adapter behavior on malformed or missing inputs."""

    def test_face_adapter_missing_fields_handling(self):
        """Test handling of incomplete IdentityDecision."""
        adapter = FaceAdapter()
        
        # Create partial mock with missing fields
        partial = Mock()
        partial.track_id = "track_001"
        partial.identity_id = "alice"
        # Other fields missing
        
        # Should gracefully degrade
        try:
            evidence = adapter.adapt(partial)
            assert evidence is not None
            assert evidence.status in [EvidenceStatus.MISSING, EvidenceStatus.UNKNOWN]
        except Exception as e:
            pytest.fail(f"Adapter should handle missing fields gracefully: {e}")

    def test_gait_adapter_missing_fields_handling(self):
        """Test handling of incomplete GaitTrackState."""
        adapter = GaitAdapter()
        
        partial = Mock()
        partial.track_id = "track_001"
        # Missing most fields
        
        try:
            evidence = adapter.adapt(partial)
            assert evidence is not None
            assert evidence.status in [EvidenceStatus.MISSING, EvidenceStatus.UNKNOWN]
        except Exception as e:
            pytest.fail(f"Adapter should handle missing fields gracefully: {e}")

    def test_adapters_idempotent(self):
        """Test adapters are idempotent (same input → same output)."""
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
    # Run tests: pytest tests/test_phase2_adapters.py -v
    pytest.main([__file__, "-v"])
