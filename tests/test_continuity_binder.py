
import pytest
import numpy as np
import time

from identity.continuity_binder import (
    ContinuityBinder,
    ContinuityMemory,
    default_continuity_config
)
from schemas import Tracklet, IdentityDecision


class TestConfigurationLoading:
    
    def test_config_loading_from_dict(self):
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
    
    def test_config_loading_vision_identity_nested(self):
        continuity_obj = type('Continuity', (), {
            'min_track_age_frames': 12,
            'max_bbox_displacement_frac': 0.30
        })()
        
        vision_identity_obj = type('VisionIdentity', (), {
            'continuity': continuity_obj
        })()
        
        cfg = type('Config', (), {
            'vision_identity': vision_identity_obj
        })()
        
        binder = ContinuityBinder(cfg)
        
        assert binder.min_track_age_frames == 12
        assert binder.max_bbox_displacement_frac == 0.30
    
    def test_config_loading_defaults(self):
        cfg = type('Config', (), {})()
        
        binder = ContinuityBinder(cfg)
        
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
        cfg = type('Config', (), {
            'continuity': {
                'min_track_age_frames': 20,
                'grace_window_sec': 3.0
            }
        })()
        
        binder = ContinuityBinder(cfg)
        
        assert binder.min_track_age_frames == 20
        assert binder.grace_window_sec == 3.0
        
        assert binder.appearance_distance_threshold == 0.35
        assert binder.shadow_mode is False


class TestGuardTrackStability:
    
    def test_track_stability_pass(self):
        cfg = type('Config', (), {
            'continuity': {'min_track_age_frames': 10}
        })()
        binder = ContinuityBinder(cfg)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200), confidence=0.8,
            age_frames=15
        )
        
        assert binder._track_stable(track) is True
    
    def test_track_stability_pass_exact_threshold(self):
        cfg = type('Config', (), {
            'continuity': {'min_track_age_frames': 10}
        })()
        binder = ContinuityBinder(cfg)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200), confidence=0.8,
            age_frames=10
        )
        
        assert binder._track_stable(track) is True
    
    def test_track_stability_fail(self):
        cfg = type('Config', (), {
            'continuity': {'min_track_age_frames': 10}
        })()
        binder = ContinuityBinder(cfg)
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200), confidence=0.8,
            age_frames=5
        )
        
        assert binder._track_stable(track) is False


class TestGuardTrackHealth:
    
    def test_track_health_pass(self):
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
            confidence=0.7,
            lost_frames=1
        )
        
        assert binder._track_healthy(track) is True
    
    def test_track_health_fail_confidence(self):
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
            confidence=0.3,
            lost_frames=0
        )
        
        assert binder._track_healthy(track) is False
    
    def test_track_health_fail_lost_frames(self):
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
            confidence=0.8,
            lost_frames=5
        )
        
        assert binder._track_healthy(track) is False


class TestGuardBboxStability:
    
    def test_bbox_stable_small_movement(self):
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
            last_box=(110, 110, 210, 210),
            confidence=0.8
        )
        
        assert binder._bbox_stable(track, memory) is True
    
    def test_bbox_stable_large_movement_good_iou(self):
        cfg = type('Config', (), {
            'continuity': {
                'max_bbox_displacement_px': 100,
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
            last_box=(50, 50, 250, 250),
            confidence=0.8
        )
        
        assert binder._bbox_stable(track, memory) is True
    
    def test_bbox_stable_fail_teleport(self):
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
            last_box=(1000, 1000, 1100, 1100),
            confidence=0.8
        )
        
        assert binder._bbox_stable(track, memory) is False
    
    def test_bbox_stable_no_previous_bbox(self):
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
            last_bbox=None
        )
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200),
            confidence=0.8
        )
        
        assert binder._bbox_stable(track, memory) is True


class TestGuardAppearanceConsistency:
    
    def test_appearance_consistent_same_embedding(self):
        cfg = type('Config', (), {
            'continuity': {
                'appearance_distance_threshold': 0.35,
                'appearance_safe_zone_frames': 0
            }
        })()
        binder = ContinuityBinder(cfg)
        
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            last_embedding=embedding.copy(),
            safe_zone_counter=10
        )
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200), confidence=0.8,
            embedding=embedding.copy()
        )
        
        consistent, distance = binder._appearance_consistent(track, memory)
        
        assert consistent is True
        assert distance < 0.01
    
    def test_appearance_consistent_similar_embedding(self):
        cfg = type('Config', (), {
            'continuity': {
                'appearance_distance_threshold': 0.35,
                'appearance_safe_zone_frames': 0
            }
        })()
        binder = ContinuityBinder(cfg)
        
        base_embedding = np.random.rand(512).astype(np.float32)
        base_embedding /= np.linalg.norm(base_embedding)
        
        embedding1 = base_embedding.copy()
        embedding2 = base_embedding + np.random.randn(512).astype(np.float32) * 0.01
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
        assert distance < 0.35
    
    def test_appearance_inconsistent_different_embedding(self):
        cfg = type('Config', (), {
            'continuity': {
                'appearance_distance_threshold': 0.35,
                'appearance_safe_zone_frames': 0
            }
        })()
        binder = ContinuityBinder(cfg)
        
        embedding1 = np.random.rand(512).astype(np.float32)
        embedding1 /= np.linalg.norm(embedding1)
        
        embedding2 = -embedding1
        
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
        assert distance > 0.35
    
    def test_appearance_skip_no_embedding(self):
        cfg = type('Config', (), {
            'continuity': {
                'appearance_distance_threshold': 0.35
            }
        })()
        binder = ContinuityBinder(cfg)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            last_embedding=None
        )
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=50,
            last_box=(100, 100, 200, 200), confidence=0.8,
            embedding=None
        )
        
        consistent, distance = binder._appearance_consistent(track, memory)
        
        assert consistent is True
        assert distance == 0.0


class TestGuardFaceConsistency:
    
    def test_face_consistent_no_face_evidence(self):
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
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0
        )
        
        assert binder._face_consistent(decision, memory) is True
        assert memory.face_contradiction_counter == 0
    
    def test_face_consistent_matching_face(self):
        cfg = type('Config', (), {
            'continuity': {
                'face_contradiction_threshold': 3
            }
        })()
        binder = ContinuityBinder(cfg)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            face_contradiction_counter=2
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_STRONG",
            confidence=0.92
        )
        
        assert binder._face_consistent(decision, memory) is True
        assert memory.face_contradiction_counter == 0
    
    def test_face_inconsistent_single_contradiction(self):
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
            identity_id="bob",
            binding_state="CONFIRMED_STRONG",
            confidence=0.90
        )
        
        assert binder._face_consistent(decision, memory) is True
        assert memory.face_contradiction_counter == 1
    
    def test_face_inconsistent_persistent_contradiction(self):
        cfg = type('Config', (), {
            'continuity': {
                'face_contradiction_threshold': 3
            }
        })()
        binder = ContinuityBinder(cfg)
        
        memory = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=10.0,
            face_contradiction_counter=2
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id="bob",
            binding_state="CONFIRMED_STRONG",
            confidence=0.90
        )
        
        assert binder._face_consistent(decision, memory) is False
        assert memory.face_contradiction_counter == 3
    
    def test_face_consistent_pending_not_contradiction(self):
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
            identity_id="bob",
            binding_state="PENDING",
            confidence=0.70
        )
        
        assert binder._face_consistent(decision, memory) is True
        assert memory.face_contradiction_counter == 0


class TestEndToEndCarry:
    
    def test_carry_persists_while_track_alive(self):
        cfg = type('Config', (), {
            'continuity': {
                'min_track_age_frames': 5,
                'track_health_min_confidence': 0.5,
                'track_health_max_lost_frames': 2,
                'appearance_safe_zone_frames': 0
            }
        })()
        binder = ContinuityBinder(cfg)
        
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
        
        assert decisions[0].id_source == "F"
        assert 1 in binder.memories
        
        for frame_idx in range(2, 11):
            track.age_frames = 10 + frame_idx
            track.last_frame_id = frame_idx
            track.last_box = (100 + frame_idx, 100 + frame_idx, 200 + frame_idx, 200 + frame_idx)
            
            decision = IdentityDecision(
                track_id=1,
                identity_id=None,
                binding_state="UNKNOWN",
                confidence=0.0
            )
            
            decisions = binder.apply(ts=float(frame_idx), tracks=[track], decisions=[decision])
            
            assert decisions[0].identity_id == "alice"
            assert decisions[0].id_source == "G"
            assert 1 in binder.memories
    
    def test_no_carry_for_young_tracks(self):
        cfg = type('Config', (), {
            'continuity': {
                'min_track_age_frames': 10
            }
        })()
        binder = ContinuityBinder(cfg)
        
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        binder.memories[1] = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=1.0,
            last_bbox=(100, 100, 200, 200),
            last_embedding=embedding.copy()
        )
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=5,
            last_box=(105, 105, 205, 205), confidence=0.8,
            age_frames=5,
            embedding=embedding.copy()
        )
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0
        )
        
        decisions = binder.apply(ts=2.0, tracks=[track], decisions=[decision])
        
        assert decisions[0].identity_id is None
        assert decisions[0].id_source == "U"
    
    def test_appearance_break_removes_memory(self):
        cfg = type('Config', (), {
            'continuity': {
                'min_track_age_frames': 5,
                'appearance_distance_threshold': 0.35,
                'appearance_safe_zone_frames': 0
            }
        })()
        binder = ContinuityBinder(cfg)
        
        original_embedding = np.random.rand(512).astype(np.float32)
        original_embedding /= np.linalg.norm(original_embedding)
        
        binder.memories[1] = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=1.0,
            last_bbox=(100, 100, 200, 200),
            last_embedding=original_embedding.copy(),
            safe_zone_counter=10
        )
        
        different_embedding = -original_embedding
        
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
        
        assert 1 not in binder.memories
        assert decisions[0].id_source == "U"
    
    def test_bbox_teleport_removes_memory(self):
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
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=10,
            last_box=(1500, 800, 1600, 900),
            confidence=0.8,
            age_frames=10,
            embedding=embedding.copy()
        )
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0
        )
        
        decisions = binder.apply(ts=2.0, tracks=[track], decisions=[decision])
        
        assert 1 not in binder.memories
        assert decisions[0].id_source == "U"


class TestGraceReattachment:
    
    def test_grace_reattachment_one_to_one(self):
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
        
        binder.apply(ts=1.5, tracks=[], decisions=[])
        assert 1 not in binder.memories
        assert 1 in binder.recently_lost
        
        new_track = Tracklet(
            track_id=2, camera_id="cam1", last_frame_id=3,
            last_box=(110, 110, 210, 210),
            confidence=0.8,
            age_frames=2,
            embedding=embedding.copy()
        )
        new_decision = IdentityDecision(
            track_id=2,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0
        )
        
        decisions = binder.apply(ts=1.8, tracks=[new_track], decisions=[new_decision])
        
        assert 2 in binder.memories
        assert 1 not in binder.recently_lost
        assert decisions[0].identity_id == "alice"
        assert decisions[0].id_source == "G"
    
    def test_grace_reattachment_fails_too_far(self):
        cfg = type('Config', (), {
            'continuity': {
                'grace_window_sec': 1.0,
                'min_track_age_frames': 5,
                'max_bbox_displacement_px': 100
            }
        })()
        binder = ContinuityBinder(cfg)
        binder.set_frame_dimensions(1920, 1080)
        
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
        binder.apply(ts=1.5, tracks=[], decisions=[])
        
        new_track = Tracklet(
            track_id=2, camera_id="cam1", last_frame_id=3,
            last_box=(1000, 1000, 1100, 1100),
            confidence=0.8,
            age_frames=2,
            embedding=embedding.copy()
        )
        new_decision = IdentityDecision(
            track_id=2, identity_id=None,
            binding_state="UNKNOWN", confidence=0.0
        )
        
        decisions = binder.apply(ts=1.8, tracks=[new_track], decisions=[new_decision])
        
        assert 2 not in binder.memories
        assert 1 in binder.recently_lost
        assert decisions[0].identity_id is None
    
    def test_grace_expires_after_window(self):
        cfg = type('Config', (), {
            'continuity': {
                'grace_window_sec': 1.0,
                'min_track_age_frames': 5
            }
        })()
        binder = ContinuityBinder(cfg)
        
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
        binder.apply(ts=1.5, tracks=[], decisions=[])
        
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
        
        assert 2 not in binder.memories
        assert 1 not in binder.recently_lost
        assert decisions[0].identity_id is None


class TestShadowMode:
    
    def test_shadow_mode_no_identity_mutation(self):
        cfg = type('Config', (), {
            'continuity': {
                'shadow_mode': True,
                'min_track_age_frames': 5
            }
        })()
        binder = ContinuityBinder(cfg)
        
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        binder.memories[1] = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice", confidence=0.9,
            last_face_ts=1.0,
            last_bbox=(100, 100, 200, 200),
            last_embedding=embedding.copy()
        )
        
        track = Tracklet(
            track_id=1, camera_id="cam1", last_frame_id=10,
            last_box=(105, 105, 205, 205), confidence=0.8,
            age_frames=10,
            embedding=embedding.copy()
        )
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0
        )
        
        decisions = binder.apply(ts=2.0, tracks=[track], decisions=[decision])
        
        assert decisions[0].identity_id is None
        assert decisions[0].extra is not None
        assert decisions[0].extra.get('shadow_id_source') == "G"
        assert decisions[0].extra.get('would_carry') == "alice"
        assert decisions[0].extra.get('id_source') == "G"
    
    def test_shadow_mode_memory_still_updated(self):
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
        
        memory = binder.memories[1]
        assert memory.last_bbox == (110, 110, 210, 210)
        assert np.allclose(memory.last_embedding, embedding2)
        assert memory.safe_zone_counter == 1


class TestEdgeCases:
    
    def test_empty_tracks_list(self):
        cfg = type('Config', (), {'continuity': {}})()
        binder = ContinuityBinder(cfg)
        
        binder.memories[1] = ContinuityMemory(
            track_id=1, person_id="alice", label="Alice",
            confidence=0.9, last_face_ts=1.0
        )
        
        decisions = binder.apply(ts=2.0, tracks=[], decisions=[])
        
        assert 1 not in binder.memories
        assert 1 in binder.recently_lost
        assert len(decisions) == 0
    
    def test_empty_decisions_list(self):
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
        
        assert len(decisions) == 1
        assert decisions[0].track_id == 1
        assert decisions[0].id_source == "U"
    
    def test_decision_already_exists(self):
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
        
        assert len(decisions) == 1
        assert decisions[0] is decision
    
    def test_zero_length_embedding(self):
        cfg = type('Config', (), {'continuity': {}})()
        binder = ContinuityBinder(cfg)
        
        zero_embedding = np.zeros(512, dtype=np.float32)
        
        normalized = binder._normalize_embedding_if_needed(zero_embedding)
        
        assert not np.any(np.isnan(normalized))
    
    def test_frame_dimensions_not_set(self):
        cfg = type('Config', (), {
            'continuity': {
                'max_bbox_displacement_px': 600
            }
        })()
        binder = ContinuityBinder(cfg)
        
        
        threshold = binder._get_bbox_displacement_threshold()
        
        assert threshold == 600


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
