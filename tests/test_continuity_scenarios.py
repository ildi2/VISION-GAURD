# tests/test_continuity_scenarios.py
"""
PHASE 8 TESTS: End-to-End Validation Scenarios

Comprehensive scenario tests validating GPS-like continuity behavior across:
1. Basic GPS continuity (person turns away, identity persists)
2. Appearance break (tracker switches to different person)
3. Grace reattachment (brief signal loss, track_id change, successful reattachment)
4. Grace expiry (person leaves too long, no reattachment)
5. BBox teleport (tracker jumps across frame, continuity breaks)
6. Track health break (lost_frames exceeds threshold)

Test Architecture:
    - Timeline simulation (frame-by-frame progression)
    - Mock objects (Tracklet, IdentityDecision created programmatically)
    - Deterministic behavior (no randomness, fixed timestamps)
    - No camera dependencies (pure logic validation)
    - Fast execution (<5 seconds total)

Phase 8 Goal: Validate all continuity behaviors work correctly end-to-end.

Author: GaitGuard Team
Date: 2026-01-31
Status: Production-Ready
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from unittest import TestCase

from identity.continuity_binder import ContinuityBinder
from schemas import Tracklet, IdentityDecision


# ============================================================================
# HELPER FUNCTIONS: Timeline Simulation
# ============================================================================

def make_track(
    track_id: int,
    bbox: Tuple[float, float, float, float],
    age_frames: int = 15,
    confidence: float = 0.8,
    lost_frames: int = 0,
    embedding: Optional[np.ndarray] = None
) -> Tracklet:
    """
    Create mock tracklet for scenario testing.
    
    Args:
        track_id: Track identifier
        bbox: Bounding box (x1, y1, x2, y2)
        age_frames: Track age (default 15 = stable)
        confidence: Tracker confidence (default 0.8 = healthy)
        lost_frames: Lost frames count (default 0 = active)
        embedding: Face embedding vector (optional)
    
    Returns:
        Tracklet object ready for continuity binder
    """
    return Tracklet(
        track_id=track_id,
        camera_id="cam1",
        last_frame_id=0,  # Not used by continuity binder
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
    """
    Create face-confirmed identity decision.
    
    Args:
        track_id: Track identifier
        person_id: Person identifier (e.g., "alice")
        binding_state: Binding state (default CONFIRMED_STRONG)
    
    Returns:
        IdentityDecision with face confirmation
    """
    return IdentityDecision(
        track_id=track_id,
        identity_id=person_id,
        category="resident",
        confidence=0.9,
        reason="face_match",
        binding_state=binding_state
    )


def make_unknown_decision(track_id: int) -> IdentityDecision:
    """
    Create unknown identity decision (no face evidence).
    
    Args:
        track_id: Track identifier
    
    Returns:
        IdentityDecision with unknown state
    """
    return IdentityDecision(
        track_id=track_id,
        identity_id=None,
        category="unknown",
        confidence=0.0,
        reason="no_face",
        binding_state="UNKNOWN"
    )


def make_embedding(seed: int = 0) -> np.ndarray:
    """
    Create deterministic L2-normalized embedding vector.
    
    Args:
        seed: Random seed for reproducibility
    
    Returns:
        512-D L2-normalized embedding vector
    """
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
    """
    Create configured continuity binder for testing.
    
    Args:
        All continuity configuration parameters with production defaults
    
    Returns:
        Configured ContinuityBinder instance
    """
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
    binder.set_frame_dimensions(1920, 1080)  # 1080p resolution
    return binder


def assert_id_source(
    decisions: List[IdentityDecision],
    track_id: int,
    expected_source: str,
    scenario: str
):
    """
    Assert identity source marker is correct.
    
    Args:
        decisions: List of identity decisions
        track_id: Track to check
        expected_source: Expected id_source ("F", "G", "U")
        scenario: Scenario description for error messages
    
    Raises:
        AssertionError if id_source doesn't match
    """
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
    """
    Assert identity assignment is correct.
    
    Args:
        decisions: List of identity decisions
        track_id: Track to check
        expected_person_id: Expected person_id (None for unknown)
        scenario: Scenario description for error messages
    
    Raises:
        AssertionError if identity doesn't match
    """
    decision = next((d for d in decisions if d.track_id == track_id), None)
    assert decision is not None, f"[{scenario}] Track {track_id} not found in decisions"
    assert decision.identity_id == expected_person_id, (
        f"[{scenario}] Track {track_id} expected identity_id={expected_person_id}, "
        f"got {decision.identity_id}"
    )


# ============================================================================
# SCENARIO 1: Basic GPS Continuity
# ============================================================================

class TestScenario1BasicGPSContinuity(TestCase):
    """
    Test basic GPS-like continuity: face assigns, tracking carries.
    
    Timeline:
        Frames 0-10: Track 1 with face confirmation (identity=alice)
        Frames 11-20: Track 1 turns away (no face, GPS carries identity)
        Frames 21-30: Track 1 turns back (face re-confirms alice)
    
    Expected Behavior:
        - Frames 0-10: id_source="F" (face-assigned)
        - Frames 11-20: id_source="G" (GPS-carried, identity persists)
        - Frames 21-30: id_source="F" (face re-confirmed)
        - Identity never lost throughout entire sequence
    """
    
    def test_basic_gps_continuity(self):
        """Person turns away, identity persists via GPS carry."""
        binder = setup_binder()
        alice_embedding = make_embedding(seed=1)
        
        # Timeline: 3 phases (face → GPS → face)
        base_ts = 1000.0
        
        # ===== Phase 1: Frames 0-10 (Face Confirmation) =====
        for frame_idx in range(11):  # 0-10 inclusive
            ts = base_ts + frame_idx * 0.033  # 30 FPS
            
            # Track 1 with face evidence
            track = make_track(
                track_id=1,
                bbox=(100, 100, 200, 200),
                age_frames=10 + frame_idx,  # Grows over time
                embedding=alice_embedding
            )
            
            # Face engine confirms alice
            decision = make_face_decision(track_id=1, person_id="alice")
            
            # Apply continuity
            decisions = binder.apply(ts, [track], [decision])
            
            # Verify: Face-assigned
            assert_id_source(decisions, track_id=1, expected_source="F",
                           scenario=f"Scenario1-Phase1-Frame{frame_idx}")
            assert_identity(decisions, track_id=1, expected_person_id="alice",
                          scenario=f"Scenario1-Phase1-Frame{frame_idx}")
        
        # ===== Phase 2: Frames 11-20 (GPS Carry - No Face) =====
        for frame_idx in range(11, 21):  # 11-20 inclusive
            ts = base_ts + frame_idx * 0.033
            
            # Track 1 turns away (no face evidence, but same embedding)
            track = make_track(
                track_id=1,
                bbox=(105, 105, 205, 205),  # Small movement
                age_frames=10 + frame_idx,
                embedding=alice_embedding  # Still alice's appearance
            )
            
            # No face confirmation (unknown decision)
            decision = make_unknown_decision(track_id=1)
            
            # Apply continuity
            decisions = binder.apply(ts, [track], [decision])
            
            # Verify: GPS-carried (identity persists without face)
            assert_id_source(decisions, track_id=1, expected_source="G",
                           scenario=f"Scenario1-Phase2-Frame{frame_idx}")
            assert_identity(decisions, track_id=1, expected_person_id="alice",
                          scenario=f"Scenario1-Phase2-Frame{frame_idx}")
        
        # ===== Phase 3: Frames 21-30 (Face Re-confirmation) =====
        for frame_idx in range(21, 31):  # 21-30 inclusive
            ts = base_ts + frame_idx * 0.033
            
            # Track 1 turns back (face evidence returns)
            track = make_track(
                track_id=1,
                bbox=(110, 110, 210, 210),
                age_frames=10 + frame_idx,
                embedding=alice_embedding
            )
            
            # Face engine confirms alice again
            decision = make_face_decision(track_id=1, person_id="alice")
            
            # Apply continuity
            decisions = binder.apply(ts, [track], [decision])
            
            # Verify: Face-assigned again
            assert_id_source(decisions, track_id=1, expected_source="F",
                           scenario=f"Scenario1-Phase3-Frame{frame_idx}")
            assert_identity(decisions, track_id=1, expected_person_id="alice",
                          scenario=f"Scenario1-Phase3-Frame{frame_idx}")


# ============================================================================
# SCENARIO 2: Appearance Break (Tracker Switches Persons)
# ============================================================================

class TestScenario2AppearanceBreak(TestCase):
    """
    Test appearance guard: tracker switches to different person.
    
    Timeline:
        Frames 0-10: Track 1 is alice (face confirmed, GPS carries)
        Frame 11: Track 1 suddenly has bob's embedding (tracker error)
    
    Expected Behavior:
        - Frames 0-10: identity=alice, id_source="F"/"G"
        - Frame 11: Appearance guard fails (distance > threshold)
        - Frame 11: Memory deleted, identity lost
        - Frame 11: id_source="U", identity_id=None
    """
    
    def test_appearance_break_tracker_switch(self):
        """Tracker switches to different person, continuity breaks."""
        binder = setup_binder(appearance_distance_threshold=0.35)
        
        alice_embedding = make_embedding(seed=1)
        bob_embedding = make_embedding(seed=2)
        
        # Verify embeddings are different (distance > threshold)
        distance = 1.0 - np.dot(alice_embedding, bob_embedding)
        assert distance > 0.35, "Test embeddings must be sufficiently different"
        
        base_ts = 1000.0
        
        # ===== Phase 1: Frames 0-10 (Alice established) =====
        for frame_idx in range(11):
            ts = base_ts + frame_idx * 0.033
            
            track = make_track(
                track_id=1,
                bbox=(100, 100, 200, 200),
                age_frames=10 + frame_idx,
                embedding=alice_embedding
            )
            
            if frame_idx < 5:
                # First 5 frames: face confirms alice
                decision = make_face_decision(track_id=1, person_id="alice")
            else:
                # Frames 5-10: GPS carries alice (no face)
                decision = make_unknown_decision(track_id=1)
            
            decisions = binder.apply(ts, [track], [decision])
            
            # Verify alice is bound
            assert_identity(decisions, track_id=1, expected_person_id="alice",
                          scenario=f"Scenario2-Phase1-Frame{frame_idx}")
        
        # ===== Phase 2: Frame 11 (Appearance Break) =====
        ts = base_ts + 11 * 0.033
        
        # Track 1 suddenly has bob's embedding (tracker switched persons)
        track = make_track(
            track_id=1,
            bbox=(105, 105, 205, 205),  # Small bbox movement (no teleport)
            age_frames=21,  # Still stable
            embedding=bob_embedding  # CRITICAL: Different embedding
        )
        
        # No face confirmation
        decision = make_unknown_decision(track_id=1)
        
        # Apply continuity
        decisions = binder.apply(ts, [track], [decision])
        
        # Verify: Appearance guard fails, memory deleted
        assert_id_source(decisions, track_id=1, expected_source="U",
                       scenario="Scenario2-Phase2-AppearanceBreak")
        assert_identity(decisions, track_id=1, expected_person_id=None,
                      scenario="Scenario2-Phase2-AppearanceBreak")
        
        # Verify memory is deleted (next frame won't carry)
        assert 1 not in binder.memories, "Memory should be deleted after appearance break"


# ============================================================================
# SCENARIO 3: Grace Reattachment (Brief Signal Loss)
# ============================================================================

class TestScenario3GraceReattachment(TestCase):
    """
    Test grace reattachment: person leaves briefly, returns with new track_id.
    
    Timeline:
        Frames 0-10: Track 1 is alice (face confirmed)
        Frame 11: Track 1 disappears (person exits frame briefly)
        Frames 12-15: No tracks (grace window open, memory in grace pool)
        Frame 16: Track 2 appears (alice returns, new track_id from tracker)
    
    Expected Behavior:
        - Frame 11: Track 1 moved to grace pool (lost_at_ts set)
        - Frame 16: Grace reattachment succeeds (embedding match, bbox proximity)
        - Frame 16: Memory reattached to track 2, identity=alice, id_source="G"
        - Original memory deleted from grace pool
    """
    
    def test_grace_reattachment_success(self):
        """Person exits briefly, returns with new track_id, reattaches."""
        binder = setup_binder(grace_window_sec=1.0)  # 1 second grace window
        
        alice_embedding = make_embedding(seed=1)
        base_ts = 1000.0
        
        # ===== Phase 1: Frames 0-10 (Alice established on track 1) =====
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
        
        # ===== Phase 2: Frame 11 (Track disappears) =====
        ts = base_ts + 11 * 0.033
        
        # No tracks (person exits frame)
        decisions = binder.apply(ts, [], [])
        
        # Verify track 1 moved to grace pool
        assert 1 not in binder.memories, "Track 1 should be removed from active memories"
        assert 1 in binder.recently_lost, "Track 1 should be in grace pool"
        
        grace_memory = binder.recently_lost[1]
        assert grace_memory.person_id == "alice", "Grace memory should preserve alice identity"
        assert grace_memory.lost_at_ts == ts, "Grace memory should record loss timestamp"
        
        # ===== Phase 3: Frames 12-15 (Grace window open, no tracks) =====
        for frame_idx in range(12, 16):
            ts = base_ts + frame_idx * 0.033
            
            # Still no tracks (person still gone, grace window open)
            decisions = binder.apply(ts, [], [])
            
            # Verify grace memory still exists (within 1 second window)
            assert 1 in binder.recently_lost, f"Grace memory should persist at frame {frame_idx}"
        
        # ===== Phase 4: Frame 16 (Alice returns with new track_id=2) =====
        ts = base_ts + 16 * 0.033  # ~0.5 seconds after loss (within 1.0s grace)
        
        # Track 2 appears (new track_id from tracker, but same person)
        track = make_track(
            track_id=2,  # CRITICAL: New track_id
            bbox=(110, 110, 210, 210),  # Near previous location (grace proximity check)
            age_frames=1,  # Fresh track (too young to carry normally)
            embedding=alice_embedding  # Same embedding (grace embedding check)
        )
        
        # No face confirmation (relies on grace reattachment)
        decision = make_unknown_decision(track_id=2)
        
        # Apply continuity
        decisions = binder.apply(ts, [track], [decision])
        
        # Verify: Grace reattachment successful
        assert_id_source(decisions, track_id=2, expected_source="G",
                       scenario="Scenario3-Phase4-GraceReattachment")
        assert_identity(decisions, track_id=2, expected_person_id="alice",
                      scenario="Scenario3-Phase4-GraceReattachment")
        
        # Verify grace pool cleanup
        assert 1 not in binder.recently_lost, "Grace memory should be removed after reattachment"
        assert 2 in binder.memories, "New track 2 should have active memory"
        
        # Verify memory transferred correctly
        new_memory = binder.memories[2]
        assert new_memory.person_id == "alice", "Memory should preserve alice identity"
        assert new_memory.track_id == 2, "Memory should point to new track_id"
        assert new_memory.original_track_id == 1, "Memory should remember original track_id"


# ============================================================================
# SCENARIO 4: Grace Expiry (Too Long Away)
# ============================================================================

class TestScenario4GraceExpiry(TestCase):
    """
    Test grace expiry: person leaves too long, no reattachment.
    
    Timeline:
        Frames 0-10: Track 1 is alice (face confirmed)
        Frame 11: Track 1 disappears
        Frames 12-50: No tracks (grace window expires at frame ~41, 1.0s)
        Frame 51: Track 2 appears (alice returns, but too late)
    
    Expected Behavior:
        - Frame 11: Track 1 moved to grace pool
        - Frame ~41: Grace memory expires (>1.0s elapsed)
        - Frame 51: No grace reattachment (expired), treated as new unknown
        - Frame 51: id_source="U", identity_id=None (must get face again)
    """
    
    def test_grace_expiry_too_long(self):
        """Person leaves too long, grace expires, no reattachment."""
        binder = setup_binder(grace_window_sec=1.0)
        
        alice_embedding = make_embedding(seed=1)
        base_ts = 1000.0
        
        # ===== Phase 1: Frames 0-10 (Alice established) =====
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
        
        # ===== Phase 2: Frame 11 (Track disappears) =====
        ts_loss = base_ts + 11 * 0.033
        decisions = binder.apply(ts_loss, [], [])
        
        assert 1 in binder.recently_lost, "Track 1 should be in grace pool"
        
        # ===== Phase 3: Frames 12-50 (Grace window expires) =====
        # Grace window is 1.0 seconds
        # At 30 FPS, 1.0 seconds = 30 frames
        # Loss at frame 11 (~0.36s), expiry at frame 11+30=41 (~1.36s total)
        
        # Simulate passage of time (>1.0 seconds)
        ts_after_grace = ts_loss + 1.5  # 1.5 seconds later
        
        # Apply with no tracks (grace expiry cleanup happens)
        decisions = binder.apply(ts_after_grace, [], [])
        
        # Verify grace memory expired and cleaned up
        assert 1 not in binder.recently_lost, "Grace memory should expire after 1.0s"
        
        # ===== Phase 4: Frame 51 (Alice returns too late) =====
        ts_return = ts_loss + 2.0  # 2 seconds after loss (too late)
        
        # Track 2 appears (alice returns, new track_id)
        track = make_track(
            track_id=2,
            bbox=(110, 110, 210, 210),
            age_frames=1,  # Fresh track
            embedding=alice_embedding  # Same person
        )
        
        decision = make_unknown_decision(track_id=2)
        
        # Apply continuity
        decisions = binder.apply(ts_return, [track], [decision])
        
        # Verify: No grace reattachment (expired), treated as unknown
        assert_id_source(decisions, track_id=2, expected_source="U",
                       scenario="Scenario4-Phase4-GraceExpired")
        assert_identity(decisions, track_id=2, expected_person_id=None,
                      scenario="Scenario4-Phase4-GraceExpired")
        
        # Verify no memory created (needs face confirmation first)
        assert 2 not in binder.memories, "No memory should exist for track 2 (too young)"


# ============================================================================
# SCENARIO 5: BBox Teleport Detection
# ============================================================================

class TestScenario5BBoxTeleport(TestCase):
    """
    Test bbox teleport guard: track jumps across frame.
    
    Timeline:
        Frames 0-10: Track 1 is alice at (100, 100, 200, 200)
        Frame 11: Track 1 suddenly at (1500, 800, 1600, 900) (tracker glitch)
    
    Expected Behavior:
        - Frames 0-10: identity=alice, GPS carry established
        - Frame 11: BBox teleport detected (center distance >600px, IoU ~0)
        - Frame 11: BBox guard fails, memory deleted
        - Frame 11: id_source="U", identity_id=None
    """
    
    def test_bbox_teleport_detected(self):
        """Track jumps across frame (tracker glitch), continuity breaks."""
        binder = setup_binder(
            max_bbox_displacement_px=600,  # Maximum 600px movement
            min_bbox_overlap=0.1           # Minimum 0.1 IoU
        )
        
        alice_embedding = make_embedding(seed=1)
        base_ts = 1000.0
        
        # ===== Phase 1: Frames 0-10 (Alice established at left side) =====
        for frame_idx in range(11):
            ts = base_ts + frame_idx * 0.033
            
            track = make_track(
                track_id=1,
                bbox=(100, 100, 200, 200),  # Left side of frame
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
        
        # ===== Phase 2: Frame 11 (BBox Teleport) =====
        ts = base_ts + 11 * 0.033
        
        # Track 1 suddenly jumps to right side (tracker ID collision/glitch)
        track = make_track(
            track_id=1,
            bbox=(1500, 800, 1600, 900),  # Right side, ~1400px away
            age_frames=21,  # Still stable
            embedding=alice_embedding  # Same embedding (not appearance issue)
        )
        
        decision = make_unknown_decision(track_id=1)
        
        # Apply continuity
        decisions = binder.apply(ts, [track], [decision])
        
        # Verify: BBox teleport detected, continuity broken
        assert_id_source(decisions, track_id=1, expected_source="U",
                       scenario="Scenario5-Phase2-BBoxTeleport")
        assert_identity(decisions, track_id=1, expected_person_id=None,
                      scenario="Scenario5-Phase2-BBoxTeleport")
        
        # Verify memory deleted
        assert 1 not in binder.memories, "Memory should be deleted after bbox teleport"


# ============================================================================
# SCENARIO 6: Track Health Break (Lost Frames)
# ============================================================================

class TestScenario6TrackHealthBreak(TestCase):
    """
    Test track health guard: lost_frames exceeds threshold.
    
    Timeline:
        Frames 0-10: Track 1 is alice (face confirmed, GPS carries)
        Frame 11: Track 1 has lost_frames=3 (tracker struggling, threshold=2)
    
    Expected Behavior:
        - Frames 0-10: identity=alice, healthy track
        - Frame 11: Track health guard fails (lost_frames=3 > threshold=2)
        - Frame 11: Memory deleted, continuity broken
        - Frame 11: id_source="U", identity_id=None
    """
    
    def test_track_health_break_lost_frames(self):
        """Track becomes unhealthy (lost_frames), continuity breaks."""
        binder = setup_binder(track_health_max_lost_frames=2)  # Max 2 lost frames
        
        alice_embedding = make_embedding(seed=1)
        base_ts = 1000.0
        
        # ===== Phase 1: Frames 0-10 (Alice established, healthy) =====
        for frame_idx in range(11):
            ts = base_ts + frame_idx * 0.033
            
            track = make_track(
                track_id=1,
                bbox=(100, 100, 200, 200),
                age_frames=10 + frame_idx,
                lost_frames=0,  # Healthy (no lost frames)
                embedding=alice_embedding
            )
            
            if frame_idx < 5:
                decision = make_face_decision(track_id=1, person_id="alice")
            else:
                decision = make_unknown_decision(track_id=1)
            
            decisions = binder.apply(ts, [track], [decision])
            assert_identity(decisions, track_id=1, expected_person_id="alice",
                          scenario=f"Scenario6-Phase1-Frame{frame_idx}")
        
        # ===== Phase 2: Frame 11 (Track becomes unhealthy) =====
        ts = base_ts + 11 * 0.033
        
        # Track 1 struggling (lost_frames=3 > threshold=2)
        track = make_track(
            track_id=1,
            bbox=(105, 105, 205, 205),  # Small movement (no teleport)
            age_frames=21,  # Still stable age
            lost_frames=3,  # CRITICAL: Exceeds threshold (2)
            embedding=alice_embedding  # Same appearance
        )
        
        decision = make_unknown_decision(track_id=1)
        
        # Apply continuity
        decisions = binder.apply(ts, [track], [decision])
        
        # Verify: Track health guard fails, continuity broken
        assert_id_source(decisions, track_id=1, expected_source="U",
                       scenario="Scenario6-Phase2-HealthBreak")
        assert_identity(decisions, track_id=1, expected_person_id=None,
                      scenario="Scenario6-Phase2-HealthBreak")
        
        # Verify memory deleted
        assert 1 not in binder.memories, "Memory should be deleted after health break"


# ============================================================================
# EXECUTION SUMMARY
# ============================================================================

if __name__ == "__main__":
    import unittest
    
    # Run all scenario tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
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
