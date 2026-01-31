# tests/test_continuity_binder.py
"""
Comprehensive unit tests for continuity_binder.py (GPS-like identity continuity).

Test Coverage:
    A. Configuration Loading (dict, defaults, nested)
    B. Guard Logic Isolated (5 guards tested individually)
    C. End-to-End Carry Logic (face → GPS transitions)
    D. Grace Reattachment (brief signal loss recovery)
    E. Shadow Mode (observe-only diagnostics)
    F. Edge Cases (None values, zero embeddings, empty lists)

Testing Philosophy:
    - Test in isolation (no camera, no live pipeline)
    - Each guard tested individually before integration tests
    - Positive and negative test cases for each guard
    - Edge cases explicitly tested (defensive programming validation)
    - Shadow mode verified (no mutations)

Author: GaitGuard Team
Date: 2026-01-31
Status: Production-Ready
"""

import pytest
import numpy as np
import time

from identity.continuity_binder import (
    ContinuityBinder,
    ContinuityMemory,
    default_continuity_config
)
from schemas import Tracklet, IdentityDecision


# ============================================================================
# A. CONFIGURATION LOADING
# ============================================================================

class TestConfigurationLoading:
    """Test configuration extraction from various sources."""
    
    def test_config_loading_from_dict(self):
        """Config extracted from dict with nested continuity section."""
        cfg = type('Config', (), {
            'continuity': {
                'min_track_age_frames': 15,
                'appearance_distance_threshold': 0.40,
                'grace_window_sec': 2.0
            }
        })()
        
        binder = ContinuityBinder(cfg)
        
        assert binder.min_track_age_frames == 15
        assert binder.appearance_distance_threshold == 0.40
        assert binder.grace_window_sec == 2.0
    
    def test_config_loading_chimeric_nested(self):
        """Config extracted from chimeric.continuity nested structure."""
        continuity_obj = type('Continuity', (), {
            'min_track_age_frames': 12,
            'max_bbox_displacement_frac': 0.30
        })()
        
        chimeric_obj = type('Chimeric', (), {
            'continuity': continuity_obj
        })()
        
        cfg = type('Config', (), {
            'chimeric': chimeric_obj
        })()
        
        binder = ContinuityBinder(cfg)
        
        assert binder.min_track_age_frames == 12
        assert binder.max_bbox_displacement_frac == 0.30
    
    def test_config_loading_defaults(self):
        """All defaults loaded when config section missing."""
        cfg = type('Config', (), {})()
        
        binder = ContinuityBinder(cfg)
        
        # Verify all defaults
        assert binder.min_track_age_frames == 10
        assert binder.appearance_distance_threshold == 0.35
        assert binder.appearance_ema_alpha == 0.3
        assert binder.appearance_safe_zone_frames == 5
        assert binder.max_bbox_displacement_frac == 0.25
        assert binder.max_bbox_displacement_px == 600
        assert binder.min_bbox_overlap == 0.1
        assert binder.track_health_min_confidence == 0.5
        assert binder.track_health_max_lost_frames == 2
        assert binder.face_contradiction_threshold == 3
        assert binder.grace_window_sec == 1.0
        assert binder.grace_max_candidates == 5
        assert binder.shadow_mode is False
    
    def test_config_partial_override(self):
        """Partial config override, rest use defaults."""
        cfg = type('Config', (), {
            'continuity': {
                'min_track_age_frames': 20,
                'grace_window_sec': 3.0
                # Other params will use defaults
            }
        })()
        
        binder = ContinuityBinder(cfg)
        
        # Overridden values
        assert binder.min_track_age_frames == 20
        assert binder.grace_window_sec == 3.0
        
        # Default values
        assert binder.appearance_distance_threshold == 0.35
        assert binder.shadow_mode is False


# ============================================================================
# B. GUARD LOGIC (ISOLATED)
# ============================================================================

class TestGuardTrackStability:
    """Test Guard 1: Track stability (age threshold)."""
    
    def test_track_stability_pass(self):
        """Track with age_frames >= threshold passes."""
        cfg = type('Config', (), {
            'continuity': {'min_track_age_frames': 10}
        })()
        binder = ContinuityBinder(cfg)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200), confidence=0.8,
            age_frames=15  # >= 10
        )
        
        assert binder._track_stable(track) is True
    
    def test_track_stability_pass_exact_threshold(self):
        """Track with age_frames == threshold passes (boundary case)."""
        cfg = type('Config', (), {
            'continuity': {'min_track_age_frames': 10}
        })()
        binder = ContinuityBinder(cfg)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200), confidence=0.8,
            age_frames=10  # == 10
        )
        
        assert binder._track_stable(track) is True
    
    def test_track_stability_fail(self):
        """Track with age_frames < threshold fails."""
        cfg = type('Config', (), {
            'continuity': {'min_track_age_frames': 10}
        })()
        binder = ContinuityBinder(cfg)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200), confidence=0.8,
            age_frames=5  # < 10
        )
        
        assert binder._track_stable(track) is False


class TestGuardTrackHealth:
    """Test Guard 2c: Track health (confidence + lost_frames)."""
    
    def test_track_health_pass(self):
        """Track with good confidence and low lost_frames passes."""
        cfg = type('Config', (), {
            'continuity': {
                'track_health_min_confidence': 0.5,
                'track_health_max_lost_frames': 2
            }
        })()
        binder = ContinuityBinder(cfg)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200),
            confidence=0.7,  # >= 0.5
            lost_frames=1    # <= 2
        )
        
        assert binder._track_healthy(track) is True
    
    def test_track_health_fail_confidence(self):
        """Track with low confidence fails."""
        cfg = type('Config', (), {
            'continuity': {
                'track_health_min_confidence': 0.5,
                'track_health_max_lost_frames': 2
            }
        })()
        binder = ContinuityBinder(cfg)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200),
            confidence=0.3,  # < 0.5
            lost_frames=0
        )
        
        assert binder._track_healthy(track) is False
    
    def test_track_health_fail_lost_frames(self):
        """Track with too many lost_frames fails."""
        cfg = type('Config', (), {
            'continuity': {
                'track_health_min_confidence': 0.5,
                'track_health_max_lost_frames': 2
            }
        })()
        binder = ContinuityBinder(cfg)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200),
            confidence=0.8,  # Good
            lost_frames=5    # > 2
        )
        
        assert binder._track_healthy(track) is False


class TestGuardBboxStability:
    """Test Guard 2b: BBox stability (center distance + IoU)."""
    
    def test_bbox_stable_small_movement(self):
        """Small bbox movement passes stability guard."""
        cfg = type('Config', (), {
            'continuity': {
                'max_bbox_displacement_px': 600,
                'min_bbox_overlap': 0.1
            }
        })()
        binder = ContinuityBinder(cfg)
        binder.set_frame_dimensions(1920, 1080)  # 1080p
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            last_bbox=(100, 100, 200, 200)
        )
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(110, 110, 210, 210),  # Small movement (10px)
            confidence=0.8
        )
        
        assert binder._bbox_stable(track, memory) is True
    
    def test_bbox_stable_large_movement_good_iou(self):
        """Large center distance but good IoU passes (e.g., bbox size change)."""
        cfg = type('Config', (), {
            'continuity': {
                'max_bbox_displacement_px': 100,  # Small threshold
                'min_bbox_overlap': 0.1
            }
        })()
        binder = ContinuityBinder(cfg)
        binder.set_frame_dimensions(1920, 1080)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            last_bbox=(100, 100, 200, 200)  # 100x100 box
        )
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(50, 50, 250, 250),  # Larger box (200x200), center moved 50px
            confidence=0.8
        )
        
        # Center moved significantly, but high IoU (overlapping)
        assert binder._bbox_stable(track, memory) is True
    
    def test_bbox_stable_fail_teleport(self):
        """Large bbox jump with low IoU fails (teleport detected)."""
        cfg = type('Config', (), {
            'continuity': {
                'max_bbox_displacement_px': 600,
                'min_bbox_overlap': 0.1
            }
        })()
        binder = ContinuityBinder(cfg)
        binder.set_frame_dimensions(1920, 1080)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            last_bbox=(100, 100, 200, 200)
        )
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(1000, 1000, 1100, 1100),  # Huge jump (teleport)
            confidence=0.8
        )
        
        assert binder._bbox_stable(track, memory) is False
    
    def test_bbox_stable_no_previous_bbox(self):
        """No previous bbox → pass guard (cannot compare)."""
        cfg = type('Config', (), {
            'continuity': {
                'max_bbox_displacement_px': 600,
                'min_bbox_overlap': 0.1
            }
        })()
        binder = ContinuityBinder(cfg)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            last_bbox=None  # No previous bbox
        )
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200),
            confidence=0.8
        )
        
        assert binder._bbox_stable(track, memory) is True


class TestGuardAppearanceConsistency:
    """Test Guard 2: Appearance consistency (embedding distance + EMA)."""
    
    def test_appearance_consistent_same_embedding(self):
        """Identical embeddings pass consistency check."""
        cfg = type('Config', (), {
            'continuity': {
                'appearance_distance_threshold': 0.35,
                'appearance_safe_zone_frames': 0  # Disable safe zone
            }
        })()
        binder = ContinuityBinder(cfg)
        
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            last_embedding=embedding.copy(),
            safe_zone_counter=10  # Outside safe zone
        )
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200), confidence=0.8,
            embedding=embedding.copy()
        )
        
        consistent, distance = binder._appearance_consistent(track, memory)
        
        assert consistent is True
        assert distance < 0.01  # Nearly zero distance
    
    def test_appearance_consistent_similar_embedding(self):
        """Similar embeddings pass consistency check."""
        cfg = type('Config', (), {
            'continuity': {
                'appearance_distance_threshold': 0.35,
                'appearance_safe_zone_frames': 0
            }
        })()
        binder = ContinuityBinder(cfg)
        
        # Create similar embeddings (small random noise)
        base_embedding = np.random.rand(512).astype(np.float32)
        base_embedding /= np.linalg.norm(base_embedding)
        
        embedding1 = base_embedding.copy()
        embedding2 = base_embedding + np.random.randn(512).astype(np.float32) * 0.01  # Small noise (0.01 scale)
        embedding2 /= np.linalg.norm(embedding2)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            last_embedding=embedding1,
            safe_zone_counter=10
        )
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200), confidence=0.8,
            embedding=embedding2
        )
        
        consistent, distance = binder._appearance_consistent(track, memory)
        
        assert consistent is True
        assert distance < 0.35  # Below threshold
    
    def test_appearance_inconsistent_different_embedding(self):
        """Very different embeddings fail consistency check."""
        cfg = type('Config', (), {
            'continuity': {
                'appearance_distance_threshold': 0.35,
                'appearance_safe_zone_frames': 0
            }
        })()
        binder = ContinuityBinder(cfg)
        
        # Create opposite embeddings (high cosine distance)
        embedding1 = np.random.rand(512).astype(np.float32)
        embedding1 /= np.linalg.norm(embedding1)
        
        embedding2 = -embedding1  # Opposite direction, distance ≈ 2.0
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            last_embedding=embedding1,
            safe_zone_counter=10
        )
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200), confidence=0.8,
            embedding=embedding2
        )
        
        consistent, distance = binder._appearance_consistent(track, memory)
        
        assert consistent is False
        assert distance > 0.35  # Above threshold
    
    def test_appearance_skip_no_embedding(self):
        """No embedding → skip guard (pass with 0 distance)."""
        cfg = type('Config', (), {
            'continuity': {
                'appearance_distance_threshold': 0.35
            }
        })()
        binder = ContinuityBinder(cfg)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            last_embedding=None  # No embedding
        )
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200), confidence=0.8,
            embedding=None  # No embedding
        )
        
        consistent, distance = binder._appearance_consistent(track, memory)
        
        # Skip guard when no embedding
        assert consistent is True
        assert distance == 0.0


class TestGuardFaceConsistency:
    """Test Guard 3: Face consistency (persistent contradiction detection)."""
    
    def test_face_consistent_no_face_evidence(self):
        """No face evidence → no contradiction, pass guard."""
        cfg = type('Config', (), {
            'continuity': {
                'face_contradiction_threshold': 3
            }
        })()
        binder = ContinuityBinder(cfg)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            face_contradiction_counter=0
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,  # No face evidence
            binding_state="UNKNOWN",
            confidence=0.0
        )
        
        assert binder._face_consistent(decision, memory) is True
        assert memory.face_contradiction_counter == 0  # No change
    
    def test_face_consistent_matching_face(self):
        """Face confirms same identity → reset counter, pass guard."""
        cfg = type('Config', (), {
            'continuity': {
                'face_contradiction_threshold': 3
            }
        })()
        binder = ContinuityBinder(cfg)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            face_contradiction_counter=2  # Had some contradictions
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",  # Matches memory
            binding_state="CONFIRMED_STRONG",
            confidence=0.92
        )
        
        assert binder._face_consistent(decision, memory) is True
        assert memory.face_contradiction_counter == 0  # Reset
    
    def test_face_inconsistent_single_contradiction(self):
        """Single face contradiction → increment counter, still pass."""
        cfg = type('Config', (), {
            'continuity': {
                'face_contradiction_threshold': 3
            }
        })()
        binder = ContinuityBinder(cfg)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            face_contradiction_counter=0
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id="bob",  # Different person
            binding_state="CONFIRMED_STRONG",
            confidence=0.90
        )
        
        assert binder._face_consistent(decision, memory) is True  # Still pass
        assert memory.face_contradiction_counter == 1  # Incremented
    
    def test_face_inconsistent_persistent_contradiction(self):
        """Persistent contradiction (>= threshold) → fail guard."""
        cfg = type('Config', (), {
            'continuity': {
                'face_contradiction_threshold': 3
            }
        })()
        binder = ContinuityBinder(cfg)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            face_contradiction_counter=2  # Already 2 contradictions
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id="bob",  # Third contradiction
            binding_state="CONFIRMED_STRONG",
            confidence=0.90
        )
        
        assert binder._face_consistent(decision, memory) is False  # Fail
        assert memory.face_contradiction_counter == 3  # Threshold reached
    
    def test_face_consistent_pending_not_contradiction(self):
        """PENDING binding (not CONFIRMED) → no contradiction counted."""
        cfg = type('Config', (), {
            'continuity': {
                'face_contradiction_threshold': 3
            }
        })()
        binder = ContinuityBinder(cfg)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            face_contradiction_counter=0
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id="bob",  # Different, but not confirmed
            binding_state="PENDING",
            confidence=0.70
        )
        
        # PENDING doesn't count as strong evidence
        assert binder._face_consistent(decision, memory) is True
        assert memory.face_contradiction_counter == 0  # Decremented (decay)


# ============================================================================
# C. END-TO-END CARRY LOGIC
# ============================================================================

class TestEndToEndCarry:
    """Test full carry scenarios (face → GPS transitions)."""
    
    def test_carry_persists_while_track_alive(self):
        """Identity persists while track_id exists and guards pass."""
        cfg = type('Config', (), {
            'continuity': {
                'min_track_age_frames': 5,
                'track_health_min_confidence': 0.5,
                'track_health_max_lost_frames': 2,
                'appearance_safe_zone_frames': 0  # Disable for test simplicity
            }
        })()
        binder = ContinuityBinder(cfg)
        
        # Frame 1: Face confirms identity
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=1,
            last_box=(100, 100, 200, 200), confidence=0.8,
            age_frames=10,
            embedding=embedding.copy()
        )
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_STRONG",
            confidence=0.92
        )
        
        decisions = binder.apply(ts=1.0, tracks=[track], decisions=[decision])
        
        assert decisions[0].id_source == "F"  # Face assigned
        assert 1 in binder.memories  # Memory created
        
        # Frames 2-10: No face evidence, but track still alive
        for frame_idx in range(2, 11):
            track.age_frames = 10 + frame_idx
            track.last_frame_id = frame_idx
            track.last_box = (100 + frame_idx, 100 + frame_idx, 200 + frame_idx, 200 + frame_idx)
            
            # No face evidence (unknown decision)
            decision = IdentityDecision(
                track_id=1,
                identity_id=None,
                binding_state="UNKNOWN",
                confidence=0.0
            )
            
            decisions = binder.apply(ts=float(frame_idx), tracks=[track], decisions=[decision])
            
            # Identity carried via GPS
            assert decisions[0].identity_id == "alice"
            assert decisions[0].id_source == "G"  # GPS carry
            assert 1 in binder.memories  # Memory still exists
    
    def test_no_carry_for_young_tracks(self):
        """Memory exists but track too young → no carry."""
        cfg = type('Config', (), {
            'continuity': {
                'min_track_age_frames': 10
            }
        })()
        binder = ContinuityBinder(cfg)
        
        # Manually create memory (simulate past face confirmation)
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        binder.memories[1] = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=1.0,
            last_bbox=(100, 100, 200, 200),
            last_embedding=embedding.copy()
        )
        
        # Young track (age_frames=5 < 10)
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=5,
            last_box=(105, 105, 205, 205), confidence=0.8,
            age_frames=5,  # Too young
            embedding=embedding.copy()
        )
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0
        )
        
        decisions = binder.apply(ts=2.0, tracks=[track], decisions=[decision])
        
        # No carry (track too young)
        assert decisions[0].identity_id is None
        assert decisions[0].id_source == "U"
    
    def test_appearance_break_removes_memory(self):
        """Appearance distance > threshold → memory deleted."""
        cfg = type('Config', (), {
            'continuity': {
                'min_track_age_frames': 5,
                'appearance_distance_threshold': 0.35,
                'appearance_safe_zone_frames': 0
            }
        })()
        binder = ContinuityBinder(cfg)
        
        # Create memory with specific embedding
        original_embedding = np.random.rand(512).astype(np.float32)
        original_embedding /= np.linalg.norm(original_embedding)
        
        binder.memories[1] = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=1.0,
            last_bbox=(100, 100, 200, 200),
            last_embedding=original_embedding.copy(),
            safe_zone_counter=10  # Outside safe zone
        )
        
        # Track with very different embedding (different person)
        different_embedding = -original_embedding  # Opposite, distance ≈ 2.0
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=10,
            last_box=(105, 105, 205, 205), confidence=0.8,
            age_frames=10,
            embedding=different_embedding
        )
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0
        )
        
        decisions = binder.apply(ts=2.0, tracks=[track], decisions=[decision])
        
        # Memory removed due to appearance break
        assert 1 not in binder.memories
        assert decisions[0].id_source == "U"
    
    def test_bbox_teleport_removes_memory(self):
        """BBox teleport → memory deleted."""
        cfg = type('Config', (), {
            'continuity': {
                'min_track_age_frames': 5,
                'max_bbox_displacement_px': 600,
                'min_bbox_overlap': 0.1
            }
        })()
        binder = ContinuityBinder(cfg)
        binder.set_frame_dimensions(1920, 1080)
        
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        binder.memories[1] = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=1.0,
            last_bbox=(100, 100, 200, 200),
            last_embedding=embedding.copy()
        )
        
        # Track teleported to far location
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=10,
            last_box=(1500, 800, 1600, 900),  # Far away
            confidence=0.8,
            age_frames=10,
            embedding=embedding.copy()  # Same appearance
        )
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0
        )
        
        decisions = binder.apply(ts=2.0, tracks=[track], decisions=[decision])
        
        # Memory removed due to bbox teleport
        assert 1 not in binder.memories
        assert decisions[0].id_source == "U"


# ============================================================================
# D. GRACE REATTACHMENT
# ============================================================================

class TestGraceReattachment:
    """Test grace reattachment logic (brief signal loss recovery)."""
    
    def test_grace_reattachment_one_to_one(self):
        """Track disappears, then reappears → grace reattachment."""
        cfg = type('Config', (), {
            'continuity': {
                'grace_window_sec': 1.0,
                'min_track_age_frames': 5,
                'max_bbox_displacement_px': 600,
                'appearance_distance_threshold': 0.35
            }
        })()
        binder = ContinuityBinder(cfg)
        binder.set_frame_dimensions(1920, 1080)
        
        # Frame 1: Face confirms
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=1,
            last_box=(100, 100, 200, 200), confidence=0.8,
            age_frames=10,
            embedding=embedding.copy()
        )
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_STRONG",
            confidence=0.92
        )
        
        binder.apply(ts=1.0, tracks=[track], decisions=[decision])
        assert 1 in binder.memories
        
        # Frame 2: Track disappears (not in tracks list)
        binder.apply(ts=1.5, tracks=[], decisions=[])
        assert 1 not in binder.memories  # Moved to grace pool
        assert 1 in binder.recently_lost  # Keyed by original_track_id
        
        # Frame 3: New track appears nearby (likely same person)
        new_track = Tracklet(
            track_id=2, camera_id="cam1", last_frame_id=3,
            last_box=(110, 110, 210, 210),  # Close to original
            confidence=0.8,
            age_frames=2,  # Young track
            embedding=embedding.copy()  # Same appearance
        )
        new_decision = IdentityDecision(
            track_id=2,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0
        )
        
        decisions = binder.apply(ts=1.8, tracks=[new_track], decisions=[new_decision])
        
        # Grace reattachment successful
        assert 2 in binder.memories  # Old memory now attached to track 2
        assert 1 not in binder.recently_lost  # Removed from grace pool
        assert decisions[0].identity_id == "alice"  # Identity reattached
        assert decisions[0].id_source == "G"  # GPS carry
    
    def test_grace_reattachment_fails_too_far(self):
        """New track too far away → grace reattachment fails."""
        cfg = type('Config', (), {
            'continuity': {
                'grace_window_sec': 1.0,
                'min_track_age_frames': 5,
                'max_bbox_displacement_px': 100  # Small threshold
            }
        })()
        binder = ContinuityBinder(cfg)
        binder.set_frame_dimensions(1920, 1080)
        
        # Setup: Track 1 confirmed, then disappears
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=1,
            last_box=(100, 100, 200, 200), confidence=0.8,
            age_frames=10,
            embedding=embedding.copy()
        )
        decision = IdentityDecision(
            track_id=1, identity_id="alice",
            binding_state="CONFIRMED_STRONG", confidence=0.92
        )
        
        binder.apply(ts=1.0, tracks=[track], decisions=[decision])
        binder.apply(ts=1.5, tracks=[], decisions=[])  # Disappears
        
        # New track appears very far away
        new_track = Tracklet(
            track_id=2, camera_id="cam1", last_frame_id=3,
            last_box=(1000, 1000, 1100, 1100),  # Very far
            confidence=0.8,
            age_frames=2,
            embedding=embedding.copy()
        )
        new_decision = IdentityDecision(
            track_id=2, identity_id=None,
            binding_state="UNKNOWN", confidence=0.0
        )
        
        decisions = binder.apply(ts=1.8, tracks=[new_track], decisions=[new_decision])
        
        # Grace reattachment failed (too far)
        assert 2 not in binder.memories  # No memory attached
        assert 1 in binder.recently_lost  # Still in grace pool
        assert decisions[0].identity_id is None
    
    def test_grace_expires_after_window(self):
        """Grace memory expires after window → no reattachment."""
        cfg = type('Config', (), {
            'continuity': {
                'grace_window_sec': 1.0,  # 1 second window
                'min_track_age_frames': 5
            }
        })()
        binder = ContinuityBinder(cfg)
        
        # Setup: Track 1 confirmed, then disappears
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=1,
            last_box=(100, 100, 200, 200), confidence=0.8,
            age_frames=10,
            embedding=embedding.copy()
        )
        decision = IdentityDecision(
            track_id=1, identity_id="alice",
            binding_state="CONFIRMED_STRONG", confidence=0.92
        )
        
        binder.apply(ts=1.0, tracks=[track], decisions=[decision])
        binder.apply(ts=1.5, tracks=[], decisions=[])  # Disappears at t=1.5
        
        # New track appears AFTER grace window (t=1.5 + 1.5 = 3.0 > 1.0 window)
        new_track = Tracklet(
            track_id=2, camera_id="cam1", last_frame_id=3,
            last_box=(110, 110, 210, 210),
            confidence=0.8,
            age_frames=2,
            embedding=embedding.copy()
        )
        new_decision = IdentityDecision(
            track_id=2, identity_id=None,
            binding_state="UNKNOWN", confidence=0.0
        )
        
        decisions = binder.apply(ts=3.0, tracks=[new_track], decisions=[new_decision])
        
        # Grace expired, no reattachment
        assert 2 not in binder.memories
        assert 1 not in binder.recently_lost  # Cleaned up (expired)
        assert decisions[0].identity_id is None


# ============================================================================
# E. SHADOW MODE
# ============================================================================

class TestShadowMode:
    """Test shadow mode (observe-only diagnostics)."""
    
    def test_shadow_mode_no_identity_mutation(self):
        """Shadow mode annotates extra but leaves identity_id unchanged."""
        cfg = type('Config', (), {
            'continuity': {
                'shadow_mode': True,  # Shadow mode enabled
                'min_track_age_frames': 5
            }
        })()
        binder = ContinuityBinder(cfg)
        
        # Create memory
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        binder.memories[1] = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=1.0,
            last_bbox=(100, 100, 200, 200),
            last_embedding=embedding.copy()
        )
        
        # Track eligible for carry
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=10,
            last_box=(105, 105, 205, 205), confidence=0.8,
            age_frames=10,
            embedding=embedding.copy()
        )
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,  # Unknown from face engine
            binding_state="UNKNOWN",
            confidence=0.0
        )
        
        decisions = binder.apply(ts=2.0, tracks=[track], decisions=[decision])
        
        # Shadow mode: identity_id NOT changed
        assert decisions[0].identity_id is None
        # But extra contains diagnostic
        assert decisions[0].extra is not None
        assert decisions[0].extra.get('shadow_id_source') == "G"
        assert decisions[0].extra.get('would_carry') == "alice"
        # id_source still marked in extra (not schema field in shadow mode)
        assert decisions[0].extra.get('id_source') == "G"
    
    def test_shadow_mode_memory_still_updated(self):
        """Shadow mode still updates memory state (for diagnostics)."""
        cfg = type('Config', (), {
            'continuity': {
                'shadow_mode': True,
                'min_track_age_frames': 5
            }
        })()
        binder = ContinuityBinder(cfg)
        
        embedding1 = np.random.rand(512).astype(np.float32)
        embedding1 /= np.linalg.norm(embedding1)
        
        binder.memories[1] = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=1.0,
            last_bbox=(100, 100, 200, 200),
            last_embedding=embedding1.copy(),
            safe_zone_counter=0
        )
        
        embedding2 = np.random.rand(512).astype(np.float32)
        embedding2 /= np.linalg.norm(embedding2)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=10,
            last_box=(110, 110, 210, 210), confidence=0.8,
            age_frames=10,
            embedding=embedding2.copy()
        )
        decision = IdentityDecision(
            track_id=1, identity_id=None,
            binding_state="UNKNOWN", confidence=0.0
        )
        
        binder.apply(ts=2.0, tracks=[track], decisions=[decision])
        
        # Memory updated (last_bbox, last_embedding, safe_zone_counter)
        memory = binder.memories[1]
        assert memory.last_bbox == (110, 110, 210, 210)
        assert np.allclose(memory.last_embedding, embedding2)
        assert memory.safe_zone_counter == 1  # Incremented


# ============================================================================
# F. EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and defensive programming."""
    
    def test_empty_tracks_list(self):
        """Empty tracks list → cleanup only."""
        cfg = type('Config', (), {'continuity': {}})()
        binder = ContinuityBinder(cfg)
        
        # Create active memory
        binder.memories[1] = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice",
            confidence=0.9, last_face_ts=1.0
        )
        
        decisions = binder.apply(ts=2.0, tracks=[], decisions=[])
        
        # Memory moved to grace pool
        assert 1 not in binder.memories
        assert 1 in binder.recently_lost
        assert len(decisions) == 0
    
    def test_empty_decisions_list(self):
        """Empty decisions list → decisions created for all tracks."""
        cfg = type('Config', (), {
            'continuity': {'min_track_age_frames': 5}
        })()
        binder = ContinuityBinder(cfg)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=10,
            last_box=(100, 100, 200, 200), confidence=0.8,
            age_frames=10
        )
        
        decisions = binder.apply(ts=1.0, tracks=[track], decisions=[])
        
        # Decision created for track
        assert len(decisions) == 1
        assert decisions[0].track_id == 1
        assert decisions[0].id_source == "U"  # No memory, no face
    
    def test_decision_already_exists(self):
        """Decision already in list → no duplicate created."""
        cfg = type('Config', (), {'continuity': {}})()
        binder = ContinuityBinder(cfg)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=10,
            last_box=(100, 100, 200, 200), confidence=0.8,
            age_frames=10
        )
        decision = IdentityDecision(
            track_id=1, identity_id=None,
            binding_state="UNKNOWN", confidence=0.0
        )
        
        decisions = binder.apply(ts=1.0, tracks=[track], decisions=[decision])
        
        # Same decision object, no duplicate
        assert len(decisions) == 1
        assert decisions[0] is decision
    
    def test_zero_length_embedding(self):
        """Zero-length embedding → normalized to avoid division by zero."""
        cfg = type('Config', (), {'continuity': {}})()
        binder = ContinuityBinder(cfg)
        
        zero_embedding = np.zeros(512, dtype=np.float32)
        
        # Should not crash, handles division by zero
        normalized = binder._normalize_embedding_if_needed(zero_embedding)
        
        # Result should be defined (not NaN)
        assert not np.any(np.isnan(normalized))
    
    def test_frame_dimensions_not_set(self):
        """BBox threshold uses fallback when frame dimensions not set."""
        cfg = type('Config', (), {
            'continuity': {
                'max_bbox_displacement_px': 600  # Fallback value
            }
        })()
        binder = ContinuityBinder(cfg)
        
        # Don't call set_frame_dimensions()
        
        threshold = binder._get_bbox_displacement_threshold()
        
        # Should use fallback value
        assert threshold == 600


# ============================================================================
# PYTEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
