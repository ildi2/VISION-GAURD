
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from unittest import TestCase

from identity.continuity_binder import ContinuityBinder
from schemas import Tracklet, IdentityDecision


def make_track(
    track_id: int,
    bbox: Tuple[float, float, float, float],
    age_frames: int = 15,
    confidence: float = 0.8,
    lost_frames: int = 0,
    embedding: Optional[np.ndarray] = None
) -> Tracklet:
    return Tracklet(
        track_id=track_id,
        camera_id="cam1",
        last_frame_id=0,
        last_box=bbox,
        confidence=confidence,
        age_frames=age_frames,
        lost_frames=lost_frames,
        embedding=embedding
    )


def make_face_decision(
    track_id: int,
    person_id: str,
    binding_state: str = "CONFIRMED_STRONG"
) -> IdentityDecision:
    return IdentityDecision(
        track_id=track_id,
        identity_id=person_id,
        category="resident",
        confidence=0.9,
        reason="face_match",
        binding_state=binding_state
    )


def make_unknown_decision(track_id: int) -> IdentityDecision:
    return IdentityDecision(
        track_id=track_id,
        identity_id=None,
        category="unknown",
        confidence=0.0,
        reason="no_face",
        binding_state="UNKNOWN"
    )


def make_embedding(seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    vec = np.random.randn(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


def setup_binder(
    min_track_age_frames: int = 10,
    appearance_distance_threshold: float = 0.35,
    max_bbox_displacement_px: int = 600,
    min_bbox_overlap: float = 0.1,
    track_health_max_lost_frames: int = 2,
    grace_window_sec: float = 1.0,
    shadow_mode: bool = False
) -> ContinuityBinder:
    cfg = type('Config', (), {
        'continuity': {
            'min_track_age_frames': min_track_age_frames,
            'appearance_distance_threshold': appearance_distance_threshold,
            'appearance_ema_alpha': 0.3,
            'appearance_safe_zone_frames': 5,
            'max_bbox_displacement_frac': 0.25,
            'max_bbox_displacement_px': max_bbox_displacement_px,
            'min_bbox_overlap': min_bbox_overlap,
            'track_health_min_confidence': 0.5,
            'track_health_max_lost_frames': track_health_max_lost_frames,
            'face_contradiction_threshold': 3,
            'grace_window_sec': grace_window_sec,
            'grace_max_candidates': 5,
            'shadow_mode': shadow_mode
        }
    })()
    
    binder = ContinuityBinder(cfg)
    binder.set_frame_dimensions(1920, 1080)
    return binder


def assert_id_source(
    decisions: List[IdentityDecision],
    track_id: int,
    expected_source: str,
    scenario: str
):
    decision = next((d for d in decisions if d.track_id == track_id), None)
    assert decision is not None, f"[{scenario}] Track {track_id} not found in decisions"
    assert decision.id_source == expected_source, (
        f"[{scenario}] Track {track_id} expected id_source={expected_source}, "
        f"got {decision.id_source}"
    )


def assert_identity(
    decisions: List[IdentityDecision],
    track_id: int,
    expected_person_id: Optional[str],
    scenario: str
):
    decision = next((d for d in decisions if d.track_id == track_id), None)
    assert decision is not None, f"[{scenario}] Track {track_id} not found in decisions"
    assert decision.identity_id == expected_person_id, (
        f"[{scenario}] Track {track_id} expected identity_id={expected_person_id}, "
        f"got {decision.identity_id}"
    )


class TestScenario1BasicGPSContinuity(TestCase):
    
    def test_basic_gps_continuity(self):
        binder = setup_binder()
        alice_embedding = make_embedding(seed=1)
        
        base_ts = 1000.0
        
        for frame_idx in range(11):
            ts = base_ts + frame_idx * 0.033
            
            track = make_track(
                track_id=1,
                bbox=(100, 100, 200, 200),
                age_frames=10 + frame_idx,
                embedding=alice_embedding
            )
            
            decision = make_face_decision(track_id=1, person_id="alice")
            
            decisions = binder.apply(ts, [track], [decision])
            
            assert_id_source(decisions, track_id=1, expected_source="F",
                           scenario=f"Scenario1-Phase1-Frame{frame_idx}")
            assert_identity(decisions, track_id=1, expected_person_id="alice",
                          scenario=f"Scenario1-Phase1-Frame{frame_idx}")
        
        for frame_idx in range(11, 21):
            ts = base_ts + frame_idx * 0.033
            
            track = make_track(
                track_id=1,
                bbox=(105, 105, 205, 205),
                age_frames=10 + frame_idx,
                embedding=alice_embedding
            )
            
            decision = make_unknown_decision(track_id=1)
            
            decisions = binder.apply(ts, [track], [decision])
            
            assert_id_source(decisions, track_id=1, expected_source="G",
                           scenario=f"Scenario1-Phase2-Frame{frame_idx}")
            assert_identity(decisions, track_id=1, expected_person_id="alice",
                          scenario=f"Scenario1-Phase2-Frame{frame_idx}")
        
        for frame_idx in range(21, 31):
            ts = base_ts + frame_idx * 0.033
            
            track = make_track(
                track_id=1,
                bbox=(110, 110, 210, 210),
                age_frames=10 + frame_idx,
                embedding=alice_embedding
            )
            
            decision = make_face_decision(track_id=1, person_id="alice")
            
            decisions = binder.apply(ts, [track], [decision])
            
            assert_id_source(decisions, track_id=1, expected_source="F",
                           scenario=f"Scenario1-Phase3-Frame{frame_idx}")
            assert_identity(decisions, track_id=1, expected_person_id="alice",
                          scenario=f"Scenario1-Phase3-Frame{frame_idx}")


class TestScenario2AppearanceBreak(TestCase):
    
    def test_appearance_break_tracker_switch(self):
        binder = setup_binder(appearance_distance_threshold=0.35)
        
        alice_embedding = make_embedding(seed=1)
        bob_embedding = make_embedding(seed=2)
        
        distance = 1.0 - np.dot(alice_embedding, bob_embedding)
        assert distance > 0.35, "Test embeddings must be sufficiently different"
        
        base_ts = 1000.0
        
        for frame_idx in range(11):
            ts = base_ts + frame_idx * 0.033
            
            track = make_track(
                track_id=1,
                bbox=(100, 100, 200, 200),
                age_frames=10 + frame_idx,
                embedding=alice_embedding
            )
            
            if frame_idx < 5:
                decision = make_face_decision(track_id=1, person_id="alice")
            else:
                decision = make_unknown_decision(track_id=1)
            
            decisions = binder.apply(ts, [track], [decision])
            
            assert_identity(decisions, track_id=1, expected_person_id="alice",
                          scenario=f"Scenario2-Phase1-Frame{frame_idx}")
        
        ts = base_ts + 11 * 0.033
        
        track = make_track(
            track_id=1,
            bbox=(105, 105, 205, 205),
            age_frames=21,
            embedding=bob_embedding
        )
        
        decision = make_unknown_decision(track_id=1)
        
        decisions = binder.apply(ts, [track], [decision])
        
        assert_id_source(decisions, track_id=1, expected_source="U",
                       scenario="Scenario2-Phase2-AppearanceBreak")
        assert_identity(decisions, track_id=1, expected_person_id=None,
                      scenario="Scenario2-Phase2-AppearanceBreak")
        
        assert 1 not in binder.memories, "Memory should be deleted after appearance break"


class TestScenario3GraceReattachment(TestCase):
    
    def test_grace_reattachment_success(self):
        binder = setup_binder(grace_window_sec=1.0)
        
        alice_embedding = make_embedding(seed=1)
        base_ts = 1000.0
        
        for frame_idx in range(11):
            ts = base_ts + frame_idx * 0.033
            
            track = make_track(
                track_id=1,
                bbox=(100, 100, 200, 200),
                age_frames=10 + frame_idx,
                embedding=alice_embedding
            )
            
            if frame_idx < 5:
                decision = make_face_decision(track_id=1, person_id="alice")
            else:
                decision = make_unknown_decision(track_id=1)
            
            decisions = binder.apply(ts, [track], [decision])
            
            assert_identity(decisions, track_id=1, expected_person_id="alice",
                          scenario=f"Scenario3-Phase1-Frame{frame_idx}")
        
        ts = base_ts + 11 * 0.033
        
        decisions = binder.apply(ts, [], [])
        
        assert 1 not in binder.memories, "Track 1 should be removed from active memories"
        assert 1 in binder.recently_lost, "Track 1 should be in grace pool"
        
        grace_memory = binder.recently_lost[1]
        assert grace_memory.person_id == "alice", "Grace memory should preserve alice identity"
        assert grace_memory.lost_at_ts == ts, "Grace memory should record loss timestamp"
        
        for frame_idx in range(12, 16):
            ts = base_ts + frame_idx * 0.033
            
            decisions = binder.apply(ts, [], [])
            
            assert 1 in binder.recently_lost, f"Grace memory should persist at frame {frame_idx}"
        
        ts = base_ts + 16 * 0.033
        
        track = make_track(
            track_id=2,
            bbox=(110, 110, 210, 210),
            age_frames=1,
            embedding=alice_embedding
        )
        
        decision = make_unknown_decision(track_id=2)
        
        decisions = binder.apply(ts, [track], [decision])
        
        assert_id_source(decisions, track_id=2, expected_source="G",
                       scenario="Scenario3-Phase4-GraceReattachment")
        assert_identity(decisions, track_id=2, expected_person_id="alice",
                      scenario="Scenario3-Phase4-GraceReattachment")
        
        assert 1 not in binder.recently_lost, "Grace memory should be removed after reattachment"
        assert 2 in binder.memories, "New track 2 should have active memory"
        
        new_memory = binder.memories[2]
        assert new_memory.person_id == "alice", "Memory should preserve alice identity"
        assert new_memory.track_id == 2, "Memory should point to new track_id"
        assert new_memory.original_track_id == 1, "Memory should remember original track_id"


class TestScenario4GraceExpiry(TestCase):
    
    def test_grace_expiry_too_long(self):
        binder = setup_binder(grace_window_sec=1.0)
        
        alice_embedding = make_embedding(seed=1)
        base_ts = 1000.0
        
        for frame_idx in range(11):
            ts = base_ts + frame_idx * 0.033
            
            track = make_track(
                track_id=1,
                bbox=(100, 100, 200, 200),
                age_frames=10 + frame_idx,
                embedding=alice_embedding
            )
            
            if frame_idx < 5:
                decision = make_face_decision(track_id=1, person_id="alice")
            else:
                decision = make_unknown_decision(track_id=1)
            
            decisions = binder.apply(ts, [track], [decision])
            assert_identity(decisions, track_id=1, expected_person_id="alice",
                          scenario=f"Scenario4-Phase1-Frame{frame_idx}")
        
        ts_loss = base_ts + 11 * 0.033
        decisions = binder.apply(ts_loss, [], [])
        
        assert 1 in binder.recently_lost, "Track 1 should be in grace pool"
        
        
        ts_after_grace = ts_loss + 1.5
        
        decisions = binder.apply(ts_after_grace, [], [])
        
        assert 1 not in binder.recently_lost, "Grace memory should expire after 1.0s"
        
        ts_return = ts_loss + 2.0
        
        track = make_track(
            track_id=2,
            bbox=(110, 110, 210, 210),
            age_frames=1,
            embedding=alice_embedding
        )
        
        decision = make_unknown_decision(track_id=2)
        
        decisions = binder.apply(ts_return, [track], [decision])
        
        assert_id_source(decisions, track_id=2, expected_source="U",
                       scenario="Scenario4-Phase4-GraceExpired")
        assert_identity(decisions, track_id=2, expected_person_id=None,
                      scenario="Scenario4-Phase4-GraceExpired")
        
        assert 2 not in binder.memories, "No memory should exist for track 2 (too young)"


class TestScenario5BBoxTeleport(TestCase):
    
    def test_bbox_teleport_detected(self):
        binder = setup_binder(
            max_bbox_displacement_px=600,
            min_bbox_overlap=0.1
        )
        
        alice_embedding = make_embedding(seed=1)
        base_ts = 1000.0
        
        for frame_idx in range(11):
            ts = base_ts + frame_idx * 0.033
            
            track = make_track(
                track_id=1,
                bbox=(100, 100, 200, 200),
                age_frames=10 + frame_idx,
                embedding=alice_embedding
            )
            
            if frame_idx < 5:
                decision = make_face_decision(track_id=1, person_id="alice")
            else:
                decision = make_unknown_decision(track_id=1)
            
            decisions = binder.apply(ts, [track], [decision])
            assert_identity(decisions, track_id=1, expected_person_id="alice",
                          scenario=f"Scenario5-Phase1-Frame{frame_idx}")
        
        ts = base_ts + 11 * 0.033
        
        track = make_track(
            track_id=1,
            bbox=(1500, 800, 1600, 900),
            age_frames=21,
            embedding=alice_embedding
        )
        
        decision = make_unknown_decision(track_id=1)
        
        decisions = binder.apply(ts, [track], [decision])
        
        assert_id_source(decisions, track_id=1, expected_source="U",
                       scenario="Scenario5-Phase2-BBoxTeleport")
        assert_identity(decisions, track_id=1, expected_person_id=None,
                      scenario="Scenario5-Phase2-BBoxTeleport")
        
        assert 1 not in binder.memories, "Memory should be deleted after bbox teleport"


class TestScenario6TrackHealthBreak(TestCase):
    
    def test_track_health_break_lost_frames(self):
        binder = setup_binder(track_health_max_lost_frames=2)
        
        alice_embedding = make_embedding(seed=1)
        base_ts = 1000.0
        
        for frame_idx in range(11):
            ts = base_ts + frame_idx * 0.033
            
            track = make_track(
                track_id=1,
                bbox=(100, 100, 200, 200),
                age_frames=10 + frame_idx,
                lost_frames=0,
                embedding=alice_embedding
            )
            
            if frame_idx < 5:
                decision = make_face_decision(track_id=1, person_id="alice")
            else:
                decision = make_unknown_decision(track_id=1)
            
            decisions = binder.apply(ts, [track], [decision])
            assert_identity(decisions, track_id=1, expected_person_id="alice",
                          scenario=f"Scenario6-Phase1-Frame{frame_idx}")
        
        ts = base_ts + 11 * 0.033
        
        track = make_track(
            track_id=1,
            bbox=(105, 105, 205, 205),
            age_frames=21,
            lost_frames=3,
            embedding=alice_embedding
        )
        
        decision = make_unknown_decision(track_id=1)
        
        decisions = binder.apply(ts, [track], [decision])
        
        assert_id_source(decisions, track_id=1, expected_source="U",
                       scenario="Scenario6-Phase2-HealthBreak")
        assert_identity(decisions, track_id=1, expected_person_id=None,
                      scenario="Scenario6-Phase2-HealthBreak")
        
        assert 1 not in binder.memories, "Memory should be deleted after health break"


if __name__ == "__main__":
    import unittest
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*80)
    print("PHASE 8 SCENARIO TESTS: EXECUTION SUMMARY")
    print("="*80)
    print(f"Total tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80)
    
    if result.wasSuccessful():
        print("✅ ALL SCENARIO TESTS PASSED")
        print("\nPhase 8 validation complete:")
        print("  ✓ Basic GPS continuity (person turns away)")
        print("  ✓ Appearance break (tracker switches persons)")
        print("  ✓ Grace reattachment (brief signal loss)")
        print("  ✓ Grace expiry (too long away)")
        print("  ✓ BBox teleport detection")
        print("  ✓ Track health break (lost_frames)")
    else:
        print("❌ SOME TESTS FAILED")
        exit(1)
