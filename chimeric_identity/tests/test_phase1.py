# chimeric_identity/tests/test_phase1.py
# ============================================================================
# PHASE 1 UNIT TESTS - Types, State Machine, Evidence Accumulator
# ============================================================================
#
# Purpose:
#   Validate core chimeric logic without needing real face/gait engines.
#   Tests state transitions, evidence buffering, hysteresis.
#
# Test Coverage:
#   1. Types: Evidence creation, validation, freshness checks
#   2. State Machine: All state transitions (UNKNOWN → CONFIRMED, etc.)
#   3. Evidence Accumulator: Buffering, pruning, hysteresis
#
# Design:
#   - Synthetic evidence (no real camera/detection needed)
#   - Deterministic timestamps (controlled time progression)
#   - Clear assertion messages (easy debugging)

import time
from chimeric_identity.types import (
    ChimericState,
    ChimericReason,
    FaceEvidence,
    GaitEvidence,
    SourceAuthEvidence,
    EvidenceStatus,
    SourceAuthState,
    validate_evidence_consistency,
)
from chimeric_identity.state_machine import (
    ChimericStateMachine,
    TrackChimericState,
)
from chimeric_identity.evidence_accumulator import (
    EvidenceAccumulator,
    HysteresisConfig,
    EvidenceWindow,
)


# ============================================================================
# TEST HELPERS
# ============================================================================

def create_face_evidence(
    identity_id: str = "alice",
    similarity: float = 0.85,
    quality: float = 0.90,
    status: EvidenceStatus = EvidenceStatus.CONFIRMED_STRONG,
    margin: float = 0.15,
    timestamp: float = None
) -> FaceEvidence:
    """Create synthetic face evidence for testing."""
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
    sequence_length: int = 35,
    timestamp: float = None
) -> GaitEvidence:
    """Create synthetic gait evidence for testing."""
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


# ============================================================================
# TEST TYPES
# ============================================================================

def test_face_evidence_freshness():
    """Test face evidence freshness checks."""
    now = time.time()
    
    # Fresh evidence
    face_ev = create_face_evidence(timestamp=now - 1.0)
    assert face_ev.is_fresh(now), "Evidence should be fresh within 2s"
    
    # Stale evidence
    face_ev_stale = create_face_evidence(timestamp=now - 3.0)
    assert not face_ev_stale.is_fresh(now), "Evidence should be stale after 2s"
    
    print("✓ test_face_evidence_freshness passed")


def test_gait_evidence_status():
    """Test gait evidence status checks."""
    gait_collecting = create_gait_evidence(
        status=EvidenceStatus.COLLECTING,
        sequence_length=15
    )
    assert gait_collecting.is_collecting()
    assert not gait_collecting.has_sufficient_data()
    
    gait_confirmed = create_gait_evidence(
        status=EvidenceStatus.CONFIRMED_STRONG,
        sequence_length=35
    )
    assert gait_confirmed.is_confirmed()
    assert gait_confirmed.has_sufficient_data()
    
    print("✓ test_gait_evidence_status passed")


def test_evidence_consistency():
    """Test evidence consistency validation."""
    face_alice = create_face_evidence(identity_id="alice")
    gait_alice = create_gait_evidence(identity_id="alice")
    gait_bob = create_gait_evidence(identity_id="bob")
    
    # Same person → consistent
    assert validate_evidence_consistency(face_alice, gait_alice)
    
    # Different person → inconsistent
    assert not validate_evidence_consistency(face_alice, gait_bob)
    
    # One None → consistent (no conflict)
    assert validate_evidence_consistency(face_alice, None)
    assert validate_evidence_consistency(None, gait_alice)
    
    print("✓ test_evidence_consistency passed")


# ============================================================================
# TEST STATE MACHINE
# ============================================================================

def test_state_machine_unknown_to_confirmed():
    """Test UNKNOWN → CONFIRMED transition via face."""
    sm = ChimericStateMachine()
    now = time.time()
    
    face_ev = create_face_evidence(
        identity_id="alice",
        status=EvidenceStatus.CONFIRMED_STRONG,
        timestamp=now
    )
    
    state, reason, identity = sm.update_state(
        track_id=1,
        face_ev=face_ev,
        gait_ev=None,
        source_auth_ev=None,
        now=now
    )
    
    assert state == ChimericState.CONFIRMED
    assert reason == ChimericReason.FACE_CONFIRMED_DOMINATES
    assert identity == "alice"
    
    print("✓ test_state_machine_unknown_to_confirmed passed")


def test_state_machine_unknown_to_tentative():
    """Test UNKNOWN → TENTATIVE transition via gait (no face)."""
    sm = ChimericStateMachine()
    now = time.time()
    
    gait_ev = create_gait_evidence(
        identity_id="bob",
        status=EvidenceStatus.CONFIRMED_WEAK,
        timestamp=now
    )
    
    state, reason, identity = sm.update_state(
        track_id=2,
        face_ev=None,
        gait_ev=gait_ev,
        source_auth_ev=None,
        now=now
    )
    
    assert state == ChimericState.TENTATIVE
    assert reason == ChimericReason.GAIT_TENTATIVE_NO_FACE_ANCHOR
    assert identity == "bob"
    
    print("✓ test_state_machine_unknown_to_tentative passed")


def test_state_machine_tentative_to_confirmed():
    """Test TENTATIVE → CONFIRMED when face confirms gait hypothesis."""
    sm = ChimericStateMachine()
    now = time.time()
    
    # Step 1: Gait proposes "charlie" (TENTATIVE)
    gait_ev = create_gait_evidence(
        identity_id="charlie",
        status=EvidenceStatus.CONFIRMED_WEAK,
        timestamp=now
    )
    state, _, _ = sm.update_state(2, None, gait_ev, None, now)
    assert state == ChimericState.TENTATIVE
    
    # Step 2: Face confirms same "charlie" (CONFIRMED)
    face_ev = create_face_evidence(
        identity_id="charlie",
        status=EvidenceStatus.CONFIRMED_STRONG,
        timestamp=now + 0.5
    )
    state, reason, identity = sm.update_state(2, face_ev, gait_ev, None, now + 0.5)
    
    assert state == ChimericState.CONFIRMED
    assert reason == ChimericReason.FACE_CONFIRMED_WITH_GAIT_SUPPORT
    assert identity == "charlie"
    
    print("✓ test_state_machine_tentative_to_confirmed passed")


def test_state_machine_conflict():
    """Test CONFIRMED → HOLD_CONFLICT when face/gait disagree."""
    sm = ChimericStateMachine()
    now = time.time()
    
    # Step 1: Face confirms "alice"
    face_alice = create_face_evidence(
        identity_id="alice",
        status=EvidenceStatus.CONFIRMED_STRONG,
        timestamp=now
    )
    state, _, _ = sm.update_state(3, face_alice, None, None, now)
    assert state == ChimericState.CONFIRMED
    
    # Step 2: Gait proposes "bob" (conflict!)
    gait_bob = create_gait_evidence(
        identity_id="bob",
        status=EvidenceStatus.CONFIRMED_WEAK,
        timestamp=now + 1.0
    )
    state, reason, identity = sm.update_state(3, face_alice, gait_bob, None, now + 1.0)
    
    assert state == ChimericState.HOLD_CONFLICT
    assert reason == ChimericReason.FACE_GAIT_CONFLICT_HOLD
    assert identity is None  # No output during conflict
    
    print("✓ test_state_machine_conflict passed")


def test_state_machine_spoof_detection():
    """Test spoof detection triggers HOLD_CONFLICT."""
    sm = ChimericStateMachine()
    now = time.time()
    
    face_ev = create_face_evidence(identity_id="alice", timestamp=now)
    spoof_ev = SourceAuthEvidence(
        realness_score=0.1,
        state=SourceAuthState.SPOOF,
        timestamp=now
    )
    
    state, reason, identity = sm.update_state(4, face_ev, None, spoof_ev, now)
    
    assert state == ChimericState.HOLD_CONFLICT
    assert reason == ChimericReason.POSSIBLE_SPOOF_DETECTED
    assert identity is None
    
    print("✓ test_state_machine_spoof_detection passed")


# ============================================================================
# TEST EVIDENCE ACCUMULATOR
# ============================================================================

def test_evidence_window_pruning():
    """Test automatic pruning of old evidence."""
    window = EvidenceWindow(window_sec=2.0)
    now = time.time()
    
    # Add evidence at different times
    for i in range(5):
        ev = create_face_evidence(timestamp=now - (4 - i))
        window.add(ev, now)
    
    # Should keep only last 2 seconds
    recent = window.get_recent(now)
    assert len(recent) <= 3, f"Expected ≤3 recent, got {len(recent)}"
    
    # All recent evidence should be within 2s
    for ev in recent:
        assert ev.age_seconds(now) <= 2.0
    
    print("✓ test_evidence_window_pruning passed")


def test_evidence_stability():
    """Test stability tracking (primary identity + count)."""
    window = EvidenceWindow(window_sec=3.0)
    now = time.time()
    
    # Add 3 "alice" samples and 1 "bob" sample
    for i in range(3):
        ev = create_face_evidence(identity_id="alice", timestamp=now - i * 0.5)
        window.add(ev, now)
    
    ev_bob = create_face_evidence(identity_id="bob", timestamp=now - 1.5)
    window.add(ev_bob, now)
    
    # Get stability
    primary_id, count, quality = window.get_stability(now)
    
    assert primary_id == "alice", f"Expected alice, got {primary_id}"
    assert count == 3, f"Expected count=3, got {count}"
    
    print("✓ test_evidence_stability passed")


def test_hysteresis_confirm():
    """Test hysteresis: initial confirm passes lower threshold."""
    config = HysteresisConfig(
        confirm_threshold=0.75,
        switch_threshold=0.85,
        margin_bonus=0.10
    )
    
    # Initial confirm (no current identity)
    assert config.can_confirm(similarity=0.76, margin=0.06)
    assert not config.can_confirm(similarity=0.74, margin=0.06)
    
    print("✓ test_hysteresis_confirm passed")


def test_hysteresis_switch():
    """Test hysteresis: switching requires higher threshold."""
    config = HysteresisConfig(
        confirm_threshold=0.75,
        switch_threshold=0.85,
        margin_bonus=0.10
    )
    
    # Switching requires stricter evidence
    assert config.can_switch(
        new_similarity=0.87,
        new_margin=0.16,
        current_identity="alice"
    )
    
    # Lower similarity fails
    assert not config.can_switch(
        new_similarity=0.78,
        new_margin=0.10,
        current_identity="alice"
    )
    
    print("✓ test_hysteresis_switch passed")


def test_accumulator_integration():
    """Test full accumulator workflow."""
    accum = EvidenceAccumulator(track_id=10)
    now = time.time()
    
    # Add face evidence
    face_ev = create_face_evidence(identity_id="alice", timestamp=now)
    accum.add_face_evidence(face_ev, now)
    
    # Add gait evidence
    gait_ev = create_gait_evidence(identity_id="alice", timestamp=now + 0.5)
    accum.add_gait_evidence(gait_ev, now + 0.5)
    
    # Check stability
    face_id, face_count, _ = accum.get_face_stability(now + 0.5)
    gait_id, gait_count, _ = accum.get_gait_stability(now + 0.5)
    
    assert face_id == "alice"
    assert gait_id == "alice"
    assert face_count == 1
    assert gait_count == 1
    
    print("✓ test_accumulator_integration passed")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all Phase 1 unit tests."""
    print("\n" + "="*60)
    print("CHIMERIC PHASE 1 UNIT TESTS")
    print("="*60 + "\n")
    
    print("--- Types Tests ---")
    test_face_evidence_freshness()
    test_gait_evidence_status()
    test_evidence_consistency()
    
    print("\n--- State Machine Tests ---")
    test_state_machine_unknown_to_confirmed()
    test_state_machine_unknown_to_tentative()
    test_state_machine_tentative_to_confirmed()
    test_state_machine_conflict()
    test_state_machine_spoof_detection()
    
    print("\n--- Evidence Accumulator Tests ---")
    test_evidence_window_pruning()
    test_evidence_stability()
    test_hysteresis_confirm()
    test_hysteresis_switch()
    test_accumulator_integration()
    
    print("\n" + "="*60)
    print("ALL PHASE 1 TESTS PASSED ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
