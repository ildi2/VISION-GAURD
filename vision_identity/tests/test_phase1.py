
import time
from vision_identity.types import (
    VisionState,
    VisionReason,
    FaceEvidence,
    GaitEvidence,
    SourceAuthEvidence,
    EvidenceStatus,
    SourceAuthState,
    validate_evidence_consistency,
)
from vision_identity.state_machine import (
    VisionStateMachine,
    TrackVisionState,
)
from vision_identity.evidence_accumulator import (
    EvidenceAccumulator,
    HysteresisConfig,
    EvidenceWindow,
)


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
    sequence_length: int = 35,
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


def test_face_evidence_freshness():
    now = time.time()
    
    face_ev = create_face_evidence(timestamp=now - 1.0)
    assert face_ev.is_fresh(now), "Evidence should be fresh within 2s"
    
    face_ev_stale = create_face_evidence(timestamp=now - 3.0)
    assert not face_ev_stale.is_fresh(now), "Evidence should be stale after 2s"
    
    print("✓ test_face_evidence_freshness passed")


def test_gait_evidence_status():
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
    face_alice = create_face_evidence(identity_id="alice")
    gait_alice = create_gait_evidence(identity_id="alice")
    gait_bob = create_gait_evidence(identity_id="bob")
    
    assert validate_evidence_consistency(face_alice, gait_alice)
    
    assert not validate_evidence_consistency(face_alice, gait_bob)
    
    assert validate_evidence_consistency(face_alice, None)
    assert validate_evidence_consistency(None, gait_alice)
    
    print("✓ test_evidence_consistency passed")


def test_state_machine_unknown_to_confirmed():
    sm = VisionStateMachine()
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
    
    assert state == VisionState.CONFIRMED
    assert reason == VisionReason.FACE_CONFIRMED_DOMINATES
    assert identity == "alice"
    
    print("✓ test_state_machine_unknown_to_confirmed passed")


def test_state_machine_unknown_to_tentative():
    sm = VisionStateMachine()
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
    
    assert state == VisionState.TENTATIVE
    assert reason == VisionReason.GAIT_TENTATIVE_NO_FACE_ANCHOR
    assert identity == "bob"
    
    print("✓ test_state_machine_unknown_to_tentative passed")


def test_state_machine_tentative_to_confirmed():
    sm = VisionStateMachine()
    now = time.time()
    
    gait_ev = create_gait_evidence(
        identity_id="charlie",
        status=EvidenceStatus.CONFIRMED_WEAK,
        timestamp=now
    )
    state, _, _ = sm.update_state(2, None, gait_ev, None, now)
    assert state == VisionState.TENTATIVE
    
    face_ev = create_face_evidence(
        identity_id="charlie",
        status=EvidenceStatus.CONFIRMED_STRONG,
        timestamp=now + 0.5
    )
    state, reason, identity = sm.update_state(2, face_ev, gait_ev, None, now + 0.5)
    
    assert state == VisionState.CONFIRMED
    assert reason == VisionReason.FACE_CONFIRMED_WITH_GAIT_SUPPORT
    assert identity == "charlie"
    
    print("✓ test_state_machine_tentative_to_confirmed passed")


def test_state_machine_conflict():
    sm = VisionStateMachine()
    now = time.time()
    
    face_alice = create_face_evidence(
        identity_id="alice",
        status=EvidenceStatus.CONFIRMED_STRONG,
        timestamp=now
    )
    state, _, _ = sm.update_state(3, face_alice, None, None, now)
    assert state == VisionState.CONFIRMED
    
    gait_bob = create_gait_evidence(
        identity_id="bob",
        status=EvidenceStatus.CONFIRMED_WEAK,
        timestamp=now + 1.0
    )
    state, reason, identity = sm.update_state(3, face_alice, gait_bob, None, now + 1.0)
    
    assert state == VisionState.HOLD_CONFLICT
    assert reason == VisionReason.FACE_GAIT_CONFLICT_HOLD
    assert identity is None
    
    print("✓ test_state_machine_conflict passed")


def test_state_machine_spoof_detection():
    sm = VisionStateMachine()
    now = time.time()
    
    face_ev = create_face_evidence(identity_id="alice", timestamp=now)
    spoof_ev = SourceAuthEvidence(
        realness_score=0.1,
        state=SourceAuthState.SPOOF,
        timestamp=now
    )
    
    state, reason, identity = sm.update_state(4, face_ev, None, spoof_ev, now)
    
    assert state == VisionState.HOLD_CONFLICT
    assert reason == VisionReason.POSSIBLE_SPOOF_DETECTED
    assert identity is None
    
    print("✓ test_state_machine_spoof_detection passed")


def test_evidence_window_pruning():
    window = EvidenceWindow(window_sec=2.0)
    now = time.time()
    
    for i in range(5):
        ev = create_face_evidence(timestamp=now - (4 - i))
        window.add(ev, now)
    
    recent = window.get_recent(now)
    assert len(recent) <= 3, f"Expected ≤3 recent, got {len(recent)}"
    
    for ev in recent:
        assert ev.age_seconds(now) <= 2.0
    
    print("✓ test_evidence_window_pruning passed")


def test_evidence_stability():
    window = EvidenceWindow(window_sec=3.0)
    now = time.time()
    
    for i in range(3):
        ev = create_face_evidence(identity_id="alice", timestamp=now - i * 0.5)
        window.add(ev, now)
    
    ev_bob = create_face_evidence(identity_id="bob", timestamp=now - 1.5)
    window.add(ev_bob, now)
    
    primary_id, count, quality = window.get_stability(now)
    
    assert primary_id == "alice", f"Expected alice, got {primary_id}"
    assert count == 3, f"Expected count=3, got {count}"
    
    print("✓ test_evidence_stability passed")


def test_hysteresis_confirm():
    config = HysteresisConfig(
        confirm_threshold=0.75,
        switch_threshold=0.85,
        margin_bonus=0.10
    )
    
    assert config.can_confirm(similarity=0.76, margin=0.06)
    assert not config.can_confirm(similarity=0.74, margin=0.06)
    
    print("✓ test_hysteresis_confirm passed")


def test_hysteresis_switch():
    config = HysteresisConfig(
        confirm_threshold=0.75,
        switch_threshold=0.85,
        margin_bonus=0.10
    )
    
    assert config.can_switch(
        new_similarity=0.87,
        new_margin=0.16,
        current_identity="alice"
    )
    
    assert not config.can_switch(
        new_similarity=0.78,
        new_margin=0.10,
        current_identity="alice"
    )
    
    print("✓ test_hysteresis_switch passed")


def test_accumulator_integration():
    accum = EvidenceAccumulator(track_id=10)
    now = time.time()
    
    face_ev = create_face_evidence(identity_id="alice", timestamp=now)
    accum.add_face_evidence(face_ev, now)
    
    gait_ev = create_gait_evidence(identity_id="alice", timestamp=now + 0.5)
    accum.add_gait_evidence(gait_ev, now + 0.5)
    
    face_id, face_count, _ = accum.get_face_stability(now + 0.5)
    gait_id, gait_count, _ = accum.get_gait_stability(now + 0.5)
    
    assert face_id == "alice"
    assert gait_id == "alice"
    assert face_count == 1
    assert gait_count == 1
    
    print("✓ test_accumulator_integration passed")


def run_all_tests():
    print("\n" + "="*60)
    print("VISION IDENTITY PHASE 1 UNIT TESTS")
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
