#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from privacy.policy_fsm import PolicyFSM, PolicyState, PolicyAction, TrackPolicyState


@dataclass
class MockDecision:
    track_id: int
    identity_id: Optional[str] = None
    category: str = "unknown"
    binding_state: Optional[str] = None
    confidence: float = 0.0
    id_source: str = "U"


def create_resident_confirmed(track_id: int, pid: str = "p_0001") -> MockDecision:
    return MockDecision(
        track_id=track_id,
        identity_id=pid,
        category="resident",
        binding_state="CONFIRMED_STRONG",
        confidence=0.85,
        id_source="F",
    )


def create_resident_weak(track_id: int, pid: str = "p_0001") -> MockDecision:
    return MockDecision(
        track_id=track_id,
        identity_id=pid,
        category="resident",
        binding_state="CONFIRMED_WEAK",
        confidence=0.65,
        id_source="F",
    )


def create_unknown(track_id: int) -> MockDecision:
    return MockDecision(
        track_id=track_id,
        identity_id=None,
        category="unknown",
        binding_state="UNKNOWN",
        confidence=0.0,
        id_source="U",
    )


def create_gps_carry(track_id: int, pid: str = "p_0001") -> MockDecision:
    return MockDecision(
        track_id=track_id,
        identity_id=pid,
        category="resident",
        binding_state="GPS_CARRY",
        confidence=0.65,
        id_source="G",
    )


def create_pending(track_id: int) -> MockDecision:
    return MockDecision(
        track_id=track_id,
        identity_id=None,
        category="resident",
        binding_state="PENDING",
        confidence=0.5,
        id_source="F",
    )


def test_1_initial_state_unknown_visible():
    print("\n=== TEST 1: Initial State UNKNOWN_VISIBLE ===")
    
    cfg = type('Cfg', (), {
        'grace_sec': 5.0,
        'reacquire_sec': 10.0,
        'unlock_allowed': False,
        'authorized_categories': ['resident'],
        'require_confirmed_binding': True,
    })()
    
    fsm = PolicyFSM(cfg)
    
    ts = time.time()
    decisions = [create_unknown(track_id=1)]
    track_ids = {1}
    
    actions = fsm.update(ts, track_ids, decisions)
    
    assert 1 in actions, "Track 1 should have an action"
    assert actions[1] == PolicyAction.SHOW, "Unknown track should be SHOW"
    
    state_obj = fsm.get_track_state(1)
    assert state_obj is not None
    assert state_obj.state == PolicyState.UNKNOWN_VISIBLE
    
    print("  ✅ New tracks start in UNKNOWN_VISIBLE with SHOW action")
    return True


def test_2_lock_on_authorized():
    print("\n=== TEST 2: Lock on Authorized ===")
    
    cfg = type('Cfg', (), {
        'grace_sec': 5.0,
        'reacquire_sec': 10.0,
        'unlock_allowed': False,
        'authorized_categories': ['resident'],
        'require_confirmed_binding': True,
    })()
    
    fsm = PolicyFSM(cfg)
    ts = time.time()
    
    actions = fsm.update(ts, {1}, [create_unknown(1)])
    assert actions[1] == PolicyAction.SHOW, "Unknown should be SHOW"
    
    ts += 0.2
    actions = fsm.update(ts, {1}, [create_resident_confirmed(1)])
    
    assert actions[1] == PolicyAction.REDACT, "Authorized should be REDACT"
    
    state_obj = fsm.get_track_state(1)
    assert state_obj.state == PolicyState.AUTHORIZED_LOCKED_REDACT
    assert state_obj.locked_since_ts is not None
    
    print("  ✅ Track locked when authorized (resident + CONFIRMED_STRONG)")
    return True


def test_3_lock_persists_through_gps_carry():
    print("\n=== TEST 3: Lock Persists Through GPS Carry ===")
    
    cfg = type('Cfg', (), {
        'grace_sec': 5.0,
        'reacquire_sec': 10.0,
        'unlock_allowed': False,
        'authorized_categories': ['resident'],
        'require_confirmed_binding': True,
    })()
    
    fsm = PolicyFSM(cfg)
    ts = time.time()
    
    actions = fsm.update(ts, {1}, [create_resident_confirmed(1)])
    assert actions[1] == PolicyAction.REDACT
    
    for i in range(4):
        ts += 0.2
        actions = fsm.update(ts, {1}, [create_gps_carry(1)])
        assert actions[1] == PolicyAction.REDACT, f"GPS carry frame {i+2} should still REDACT"
    
    state_obj = fsm.get_track_state(1)
    assert state_obj.state == PolicyState.AUTHORIZED_LOCKED_REDACT
    
    print("  ✅ Lock persists through GPS carry frames")
    return True


def test_4_lock_persists_when_authorized_signal_false():
    print("\n=== TEST 4: Lock Persists (Grace Period) ===")
    
    cfg = type('Cfg', (), {
        'grace_sec': 5.0,
        'reacquire_sec': 10.0,
        'unlock_allowed': False,
        'authorized_categories': ['resident'],
        'require_confirmed_binding': True,
    })()
    
    fsm = PolicyFSM(cfg)
    ts = time.time()
    
    actions = fsm.update(ts, {1}, [create_resident_confirmed(1)])
    assert actions[1] == PolicyAction.REDACT
    
    for i in range(10):
        ts += 0.2
        actions = fsm.update(ts, {1}, [create_unknown(1)])
        assert actions[1] == PolicyAction.REDACT, f"Frame {i+2}: Lock should persist (fail-closed)"
    
    state_obj = fsm.get_track_state(1)
    assert state_obj.state == PolicyState.AUTHORIZED_LOCKED_REDACT
    assert state_obj.grace_until_ts is not None, "Grace timer should be set"
    
    print("  ✅ Lock persists when authorized signal becomes false (grace)")
    return True


def test_5_reacquire_on_track_disappear():
    print("\n=== TEST 5: Reacquire on Track Disappear ===")
    
    cfg = type('Cfg', (), {
        'grace_sec': 5.0,
        'reacquire_sec': 10.0,
        'unlock_allowed': False,
        'authorized_categories': ['resident'],
        'require_confirmed_binding': True,
    })()
    
    fsm = PolicyFSM(cfg)
    ts = time.time()
    
    actions = fsm.update(ts, {1}, [create_resident_confirmed(1)])
    assert actions[1] == PolicyAction.REDACT
    
    ts += 0.2
    actions = fsm.update(ts, set(), [])
    
    assert 1 in actions, "Missing track should still have action"
    assert actions[1] == PolicyAction.REDACT, "REACQUIRE should still REDACT"
    
    state_obj = fsm.get_track_state(1)
    assert state_obj.state == PolicyState.REACQUIRE_REDACT
    
    print("  ✅ Track enters REACQUIRE_REDACT when it disappears")
    return True


def test_6_reacquire_returns_to_locked():
    print("\n=== TEST 6: Reacquire Returns to Locked ===")
    
    cfg = type('Cfg', (), {
        'grace_sec': 5.0,
        'reacquire_sec': 10.0,
        'unlock_allowed': False,
        'authorized_categories': ['resident'],
        'require_confirmed_binding': True,
    })()
    
    fsm = PolicyFSM(cfg)
    ts = time.time()
    
    actions = fsm.update(ts, {1}, [create_resident_confirmed(1)])
    assert actions[1] == PolicyAction.REDACT
    
    for _ in range(2):
        ts += 0.2
        actions = fsm.update(ts, set(), [])
    
    state_obj = fsm.get_track_state(1)
    assert state_obj.state == PolicyState.REACQUIRE_REDACT
    
    ts += 0.2
    actions = fsm.update(ts, {1}, [create_resident_confirmed(1)])
    
    assert actions[1] == PolicyAction.REDACT, "Reappeared track should REDACT"
    
    state_obj = fsm.get_track_state(1)
    assert state_obj.state == PolicyState.AUTHORIZED_LOCKED_REDACT
    
    print("  ✅ Track returns to AUTHORIZED_LOCKED_REDACT when reappears")
    return True


def test_7_no_single_frame_reveal():
    print("\n=== TEST 7: No Single-Frame Reveal ===")
    
    cfg = type('Cfg', (), {
        'grace_sec': 5.0,
        'reacquire_sec': 10.0,
        'unlock_allowed': False,
        'authorized_categories': ['resident'],
        'require_confirmed_binding': True,
    })()
    
    fsm = PolicyFSM(cfg)
    ts = time.time()
    
    actions = fsm.update(ts, {1}, [create_resident_confirmed(1)])
    locked_at = ts
    
    reveal_count = 0
    for i in range(100):
        ts += 0.1
        
        if i % 10 == 0:
            dec = create_unknown(1)
        elif i % 7 == 0:
            dec = create_pending(1)
        elif i % 5 == 0:
            dec = create_gps_carry(1)
        else:
            dec = create_resident_confirmed(1)
        
        actions = fsm.update(ts, {1}, [dec])
        
        if actions.get(1) == PolicyAction.SHOW:
            reveal_count += 1
            print(f"  ❌ REVEAL at frame {i+2}!")
    
    assert reveal_count == 0, f"Had {reveal_count} reveals after lock!"
    
    state_obj = fsm.get_track_state(1)
    assert state_obj.state == PolicyState.AUTHORIZED_LOCKED_REDACT
    
    print(f"  ✅ No reveals in 100 frames after lock (grace={cfg.grace_sec}s)")
    return True


def test_8_unlock_allowed_false_blocks_unlock():
    print("\n=== TEST 8: unlock_allowed=False Blocks Unlock ===")
    
    cfg = type('Cfg', (), {
        'grace_sec': 0.5,
        'reacquire_sec': 1.0,
        'unlock_allowed': False,
        'authorized_categories': ['resident'],
        'require_confirmed_binding': True,
    })()
    
    fsm = PolicyFSM(cfg)
    ts = time.time()
    
    actions = fsm.update(ts, {1}, [create_resident_confirmed(1)])
    assert actions[1] == PolicyAction.REDACT
    
    for _ in range(20):
        ts += 0.1
        actions = fsm.update(ts, set(), [])
    
    state_obj = fsm.get_track_state(1)
    assert state_obj.state == PolicyState.ENDED_COOLDOWN
    assert actions[1] == PolicyAction.REDACT
    
    ts += 0.1
    actions = fsm.update(ts, {1}, [create_unknown(1)])
    
    assert actions[1] == PolicyAction.REDACT, "Should not unlock even when track reappears as unknown"
    
    print("  ✅ unlock_allowed=False prevents unlock")
    return True


def test_9_require_confirmed_binding():
    print("\n=== TEST 9: require_confirmed_binding Blocks Pre-Confirmation Lock ===")
    
    cfg = type('Cfg', (), {
        'grace_sec': 5.0,
        'reacquire_sec': 10.0,
        'unlock_allowed': False,
        'authorized_categories': ['resident'],
        'require_confirmed_binding': True,
    })()
    
    fsm = PolicyFSM(cfg)
    ts = time.time()
    
    actions = fsm.update(ts, {1}, [create_pending(1)])
    
    assert actions[1] == PolicyAction.SHOW, "PENDING should not trigger lock"
    
    state_obj = fsm.get_track_state(1)
    assert state_obj.state == PolicyState.UNKNOWN_VISIBLE
    
    ts += 0.2
    actions = fsm.update(ts, {1}, [create_resident_weak(1)])
    
    assert actions[1] == PolicyAction.REDACT, "CONFIRMED_WEAK should trigger lock"
    
    state_obj = fsm.get_track_state(1)
    assert state_obj.state == PolicyState.AUTHORIZED_LOCKED_REDACT
    
    print("  ✅ require_confirmed_binding blocks lock until confirmed")
    return True


def test_10_policy_info_for_audit():
    print("\n=== TEST 10: Policy Info for Audit ===")
    
    cfg = type('Cfg', (), {
        'grace_sec': 5.0,
        'reacquire_sec': 10.0,
        'unlock_allowed': False,
        'authorized_categories': ['resident'],
        'require_confirmed_binding': True,
    })()
    
    fsm = PolicyFSM(cfg)
    ts = time.time()
    
    actions = fsm.update(ts, {1}, [create_unknown(1)])
    
    info = fsm.get_track_policy_info(1, ts)
    assert info["policy_state"] == "UNKNOWN_VISIBLE"
    assert info["is_redacted"] == False
    assert info["redaction_method"] == "none"
    
    ts += 0.2
    lock_ts = ts
    actions = fsm.update(ts, {1}, [create_resident_confirmed(1)])
    
    ts += 2.0
    actions = fsm.update(ts, {1}, [create_resident_confirmed(1)])
    
    info = fsm.get_track_policy_info(1, ts)
    assert info["policy_state"] == "AUTHORIZED_LOCKED_REDACT"
    assert info["is_redacted"] == True
    assert info["redaction_method"] == "bbox_blur"
    assert info["lock_age_sec"] is not None
    assert info["lock_age_sec"] >= 1.9
    
    print("  ✅ Policy info correct for audit")
    return True


def test_11_fsm_stats():
    print("\n=== TEST 11: FSM Statistics ===")
    
    cfg = type('Cfg', (), {
        'grace_sec': 5.0,
        'reacquire_sec': 10.0,
        'unlock_allowed': False,
        'authorized_categories': ['resident'],
        'require_confirmed_binding': True,
    })()
    
    fsm = PolicyFSM(cfg)
    ts = time.time()
    
    for track_id in range(1, 5):
        ts += 0.1
        fsm.update(ts, {track_id}, [create_unknown(track_id)])
    
    ts += 0.1
    fsm.update(ts, {1, 2}, [
        create_resident_confirmed(1),
        create_resident_confirmed(2),
    ])
    
    stats = fsm.get_stats()
    
    assert stats["total_tracks_managed"] == 4
    assert stats["locks_created"] == 2
    assert stats["state_counts"]["AUTHORIZED_LOCKED_REDACT"] == 2
    assert stats["state_counts"]["UNKNOWN_VISIBLE"] == 2
    
    print(f"  Stats: {stats}")
    print("  ✅ FSM statistics tracked correctly")
    return True


def test_12_gps_carry_authorizes_redaction():
    print("\n=== TEST 12: GPS_CARRY Authorizes Redaction ===")
    
    cfg = type('Cfg', (), {
        'grace_sec': 5.0,
        'reacquire_sec': 10.0,
        'unlock_allowed': False,
        'authorized_categories': ['resident'],
        'require_confirmed_binding': True,
    })()
    
    fsm = PolicyFSM(cfg)
    ts = time.time()
    
    actions = fsm.update(ts, {1}, [create_gps_carry(1)])
    assert actions[1] == PolicyAction.REDACT, "GPS_CARRY should authorize redaction"
    
    state_obj = fsm.get_track_state(1)
    assert state_obj.state == PolicyState.AUTHORIZED_LOCKED_REDACT, \
        "GPS_CARRY should lock track to REDACT state"
    
    print("  ✅ GPS_CARRY authorizes and locks track")
    
    assert fsm._is_authorized("resident", "GPS_CARRY") == True, \
        "_is_authorized should return True for GPS_CARRY"
    assert fsm._is_authorized("resident", "CONFIRMED_WEAK") == True, \
        "_is_authorized should return True for CONFIRMED_WEAK"
    assert fsm._is_authorized("resident", "CONFIRMED_STRONG") == True, \
        "_is_authorized should return True for CONFIRMED_STRONG"
    assert fsm._is_authorized("resident", "PENDING") == False, \
        "_is_authorized should return False for PENDING"
    assert fsm._is_authorized("resident", "UNKNOWN") == False, \
        "_is_authorized should return False for UNKNOWN"
    
    print("  ✅ _is_authorized accepts GPS_CARRY, CONFIRMED_WEAK, CONFIRMED_STRONG")
    
    ts += 0.2
    actions = fsm.update(ts, {1}, [create_resident_confirmed(1)])
    assert actions[1] == PolicyAction.REDACT
    
    ts += 0.2
    actions = fsm.update(ts, {1}, [create_gps_carry(1)])
    assert actions[1] == PolicyAction.REDACT
    
    print("  ✅ Transitions between face/GPS preserve REDACT")
    return True


def main():
    print("=" * 60)
    print("M3 POLICY FSM UNIT TESTS")
    print("=" * 60)
    
    all_passed = True
    tests = [
        test_1_initial_state_unknown_visible,
        test_2_lock_on_authorized,
        test_3_lock_persists_through_gps_carry,
        test_4_lock_persists_when_authorized_signal_false,
        test_5_reacquire_on_track_disappear,
        test_6_reacquire_returns_to_locked,
        test_7_no_single_frame_reveal,
        test_8_unlock_allowed_false_blocks_unlock,
        test_9_require_confirmed_binding,
        test_10_policy_info_for_audit,
        test_11_fsm_stats,
        test_12_gps_carry_authorizes_redaction,
    ]
    
    for test_fn in tests:
        try:
            if not test_fn():
                all_passed = False
        except Exception as e:
            print(f"  ❌ {test_fn.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("=== ALL M3 TESTS PASSED ===")
    else:
        print("=== SOME M3 TESTS FAILED ===")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
