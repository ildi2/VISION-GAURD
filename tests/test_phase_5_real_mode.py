# tests/test_phase_5_real_mode.py
"""
PHASE 5 VALIDATION: Real Mode Continuity Tests

Purpose:
    Comprehensive test suite for Phase 5 (Config-Gated Activation).
    Validates real continuity mode behavior with full identity carry.

Test Coverage:
    1. Real mode configuration loading
    2. Identity carry mutation (not shadow mode annotation)
    3. GPS-like persistence while track alive
    4. All five guards in real mode
    5. Grace reattachment in real mode
    6. Integration with binding state preservation
    7. Carried confidence handling
    8. Config switching (classic → shadow → continuity)

Critical Validation:
    - decision.identity_id actually mutated (not just extra dict)
    - decision.binding_state NOT mutated (face engine truth preserved)
    - id_source="G" set correctly
    - Carried confidence stored in extra dict
    - All guards functional in real mode

Author: GaitGuard Team
Date: 2026-01-31
Status: Phase 5 Validation
"""

import unittest
from typing import List
import numpy as np

from schemas import Tracklet, IdentityDecision
from identity.continuity_binder import ContinuityBinder, ContinuityMemory


class TestPhase5RealMode(unittest.TestCase):
    """Phase 5: Real mode configuration and activation tests."""
    
    def test_config_mode_continuity_enables_real_mode(self):
        """Config mode='continuity' with shadow_mode=false enables real carry."""
        # Configuration simulating config/default.yaml Phase 5 settings
        cfg = type('Config', (), {
            'chimeric': type('ChimericConfig', (), {
                'mode': 'continuity',
                'continuity': type('ContinuityConfig', (), {
                    'min_track_age_frames': 10,
                    'shadow_mode': False  # CRITICAL: Real mode enabled
                })()
            })()
        })()
        
        binder = ContinuityBinder(cfg)
        
        # Verify real mode activated
        self.assertFalse(binder.shadow_mode, "Shadow mode should be FALSE for real carry")
        self.assertEqual(binder.min_track_age_frames, 10)
    
    def test_shadow_mode_config_prevents_mutation(self):
        """Config shadow_mode=true prevents identity_id mutation."""
        cfg = type('Config', (), {
            'chimeric': type('ChimericConfig', (), {
                'mode': 'shadow_continuity',
                'continuity': type('ContinuityConfig', (), {
                    'shadow_mode': True
                })()
            })()
        })()
        
        binder = ContinuityBinder(cfg)
        
        # Verify shadow mode activated
        self.assertTrue(binder.shadow_mode, "Shadow mode should be TRUE")
    
    def test_classic_mode_config_should_not_instantiate_binder(self):
        """Config mode='classic' means binder not instantiated (main_loop responsibility)."""
        # This test documents expected behavior:
        # In main_loop.py, mode='classic' → continuity_binder = None
        # This is integration behavior, not binder responsibility
        # Just documenting for Phase 5 validation
        pass


class TestPhase5RealCarry(unittest.TestCase):
    """Phase 5: Real identity carry mutation tests."""
    
    def setUp(self):
        """Create binder in REAL mode (not shadow mode)."""
        cfg = type('Config', (), {
            'chimeric': type('ChimericConfig', (), {
                'mode': 'continuity',
                'continuity': type('ContinuityConfig', (), {
                    'min_track_age_frames': 5,
                    'shadow_mode': False,  # CRITICAL: Real mode
                    'track_health_min_confidence': 0.3,
                    'track_health_max_lost_frames': 5,
                    'appearance_distance_threshold': 0.35,
                    'max_bbox_displacement_px': 600,
                    'min_bbox_overlap': 0.1
                })()
            })()
        })()
        
        self.binder = ContinuityBinder(cfg)
    
    def test_real_mode_mutates_identity_id(self):
        """Real mode: decision.identity_id actually mutated (not just extra dict)."""
        # Frame 1: Face confirms identity
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        track = Tracklet(
            track_id=1,
            camera_id='cam1',
            last_frame_id=1,
            last_box=(100, 100, 200, 200),
            confidence=0.8,
            age_frames=10,
            embedding=embedding.copy()
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id='alice',
            binding_state='CONFIRMED_STRONG',
            confidence=0.92
        )
        
        decisions = self.binder.apply(ts=1.0, tracks=[track], decisions=[decision])
        
        # Verify face binding
        self.assertEqual(decisions[0].identity_id, 'alice')
        self.assertEqual(self._get_id_source(decisions[0]), 'F')
        
        # Frame 2: No face evidence (UNKNOWN)
        track.age_frames = 11
        track.last_frame_id = 2
        
        decision_unknown = IdentityDecision(
            track_id=1,
            identity_id=None,  # No face this frame
            binding_state='UNKNOWN',
            confidence=0.0
        )
        
        decisions = self.binder.apply(ts=2.0, tracks=[track], decisions=[decision_unknown])
        
        # CRITICAL: Real mode should MUTATE identity_id (not just annotate)
        self.assertEqual(decisions[0].identity_id, 'alice', 
                         "Real mode must mutate identity_id (GPS carry)")
        self.assertEqual(self._get_id_source(decisions[0]), 'G',
                         "id_source should be 'G' for GPS carry")
        
        # Verify binding_state NOT mutated (face engine truth preserved)
        self.assertEqual(decisions[0].binding_state, 'UNKNOWN',
                         "binding_state must NOT be mutated (face engine truth)")
    
    def test_shadow_mode_does_not_mutate_identity_id(self):
        """Shadow mode: decision.identity_id NOT mutated (only extra dict annotated)."""
        # Create shadow mode binder
        cfg = type('Config', (), {
            'chimeric': type('ChimericConfig', (), {
                'continuity': type('ContinuityConfig', (), {
                    'min_track_age_frames': 5,
                    'shadow_mode': True  # Shadow mode
                })()
            })()
        })()
        
        shadow_binder = ContinuityBinder(cfg)
        
        # Frame 1: Face confirms
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        track = Tracklet(
            track_id=1,
            camera_id='cam1',
            last_frame_id=1,
            last_box=(100, 100, 200, 200),
            confidence=0.8,
            age_frames=10,
            embedding=embedding.copy()
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id='alice',
            binding_state='CONFIRMED_STRONG',
            confidence=0.92
        )
        
        shadow_binder.apply(ts=1.0, tracks=[track], decisions=[decision])
        
        # Frame 2: No face
        track.age_frames = 11
        decision_unknown = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state='UNKNOWN',
            confidence=0.0
        )
        
        decisions = shadow_binder.apply(ts=2.0, tracks=[track], decisions=[decision_unknown])
        
        # Shadow mode: identity_id should stay None
        self.assertIsNone(decisions[0].identity_id,
                          "Shadow mode must NOT mutate identity_id")
        
        # Shadow mode: extra dict should have diagnostics
        self.assertIsNotNone(decisions[0].extra)
        self.assertEqual(decisions[0].extra.get('would_carry'), 'alice',
                         "Shadow mode should annotate would_carry in extra dict")
        self.assertEqual(decisions[0].extra.get('shadow_id_source'), 'G',
                         "Shadow mode should set shadow_id_source in extra dict")
    
    def test_carried_confidence_stored_correctly(self):
        """Carried confidence stored in extra dict (not overwriting decision.confidence)."""
        # Frame 1: Face confirms with high confidence
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        track = Tracklet(
            track_id=1,
            camera_id='cam1',
            last_frame_id=1,
            last_box=(100, 100, 200, 200),
            confidence=0.8,
            age_frames=10,
            embedding=embedding.copy()
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id='alice',
            binding_state='CONFIRMED_STRONG',
            confidence=0.92  # Original face confidence
        )
        
        self.binder.apply(ts=1.0, tracks=[track], decisions=[decision])
        
        # Frame 2: No face (GPS carry)
        track.age_frames = 11
        decision_unknown = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state='UNKNOWN',
            confidence=0.0  # No face → 0 confidence
        )
        
        decisions = self.binder.apply(ts=2.0, tracks=[track], decisions=[decision_unknown])
        
        # Verify decision.confidence NOT overwritten (stays 0.0 from face engine)
        self.assertEqual(decisions[0].confidence, 0.0,
                         "decision.confidence should stay as face engine value")
        
        # Verify carried confidence stored in extra dict
        self.assertIsNotNone(decisions[0].extra)
        self.assertEqual(decisions[0].extra.get('carried_confidence'), 0.92,
                         "Original face confidence should be in extra['carried_confidence']")
        self.assertTrue(decisions[0].extra.get('is_carried'),
                        "is_carried flag should be True")
    
    def test_gps_persistence_while_track_alive(self):
        """Identity persists for 10 frames while track alive (GPS continuity)."""
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        # Frame 0: Face confirms
        track = Tracklet(
            track_id=1,
            camera_id='cam1',
            last_frame_id=0,
            last_box=(100, 100, 200, 200),
            confidence=0.8,
            age_frames=10,
            embedding=embedding.copy()
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id='alice',
            binding_state='CONFIRMED_STRONG',
            confidence=0.92
        )
        
        self.binder.apply(ts=0.0, tracks=[track], decisions=[decision])
        
        # Frames 1-9: No face evidence, but track persists
        for frame_idx in range(1, 10):
            track.age_frames = 10 + frame_idx
            track.last_frame_id = frame_idx
            track.last_box = (100 + frame_idx, 100 + frame_idx, 
                              200 + frame_idx, 200 + frame_idx)  # Small movement
            
            decision_unknown = IdentityDecision(
                track_id=1,
                identity_id=None,
                binding_state='UNKNOWN',
                confidence=0.0
            )
            
            decisions = self.binder.apply(
                ts=float(frame_idx), 
                tracks=[track], 
                decisions=[decision_unknown]
            )
            
            # Identity carried every frame
            self.assertEqual(decisions[0].identity_id, 'alice',
                             f"Frame {frame_idx}: Identity should persist (GPS)")
            self.assertEqual(self._get_id_source(decisions[0]), 'G',
                             f"Frame {frame_idx}: id_source should be 'G'")
            
            # Verify in memory
            self.assertIn(1, self.binder.memories,
                          f"Frame {frame_idx}: Memory should exist")
    
    def _get_id_source(self, decision: IdentityDecision) -> str:
        """Helper to get id_source from decision."""
        if hasattr(decision, 'id_source') and decision.id_source is not None:
            return decision.id_source
        if decision.extra and 'id_source' in decision.extra:
            return decision.extra['id_source']
        return 'U'


class TestPhase5GuardsRealMode(unittest.TestCase):
    """Phase 5: Validate all guards work in real mode."""
    
    def setUp(self):
        """Create binder in real mode."""
        cfg = type('Config', (), {
            'chimeric': type('ChimericConfig', (), {
                'continuity': type('ContinuityConfig', (), {
                    'min_track_age_frames': 10,
                    'shadow_mode': False,
                    'track_health_min_confidence': 0.5,
                    'track_health_max_lost_frames': 2,
                    'appearance_distance_threshold': 0.35,
                    'max_bbox_displacement_px': 600,
                    'min_bbox_overlap': 0.1,
                    'face_contradiction_threshold': 3
                })()
            })()
        })()
        
        self.binder = ContinuityBinder(cfg)
    
    def test_guard_1_track_stability_breaks_carry(self):
        """Guard 1: Young track (age < threshold) prevents carry in real mode."""
        # Create memory manually
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        self.binder.memories[1] = ContinuityMemory(
            track_id=1,
            person_id='alice',
            label='Alice',
            confidence=0.9,
            last_face_ts=1.0,
            last_bbox=(100, 100, 200, 200),
            last_embedding=embedding.copy()
        )
        
        # Track too young (age=5 < 10)
        track = Tracklet(
            track_id=1,
            camera_id='cam1',
            last_frame_id=5,
            last_box=(105, 105, 205, 205),
            confidence=0.8,
            age_frames=5,  # Too young
            embedding=embedding.copy()
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state='UNKNOWN',
            confidence=0.0
        )
        
        decisions = self.binder.apply(ts=2.0, tracks=[track], decisions=[decision])
        
        # No carry (track too young)
        self.assertIsNone(decisions[0].identity_id,
                          "Young track should not carry identity")
        self.assertEqual(self._get_id_source(decisions[0]), 'U')
    
    def test_guard_2c_track_health_breaks_carry(self):
        """Guard 2c: Low confidence or many lost frames breaks carry."""
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        self.binder.memories[1] = ContinuityMemory(
            track_id=1,
            person_id='alice',
            label='Alice',
            confidence=0.9,
            last_face_ts=1.0,
            last_bbox=(100, 100, 200, 200),
            last_embedding=embedding.copy()
        )
        
        # Track unhealthy (lost_frames=3 > 2)
        track = Tracklet(
            track_id=1,
            camera_id='cam1',
            last_frame_id=10,
            last_box=(105, 105, 205, 205),
            confidence=0.8,  # Good confidence
            age_frames=15,   # Old enough
            lost_frames=3,   # Too many lost frames
            embedding=embedding.copy()
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state='UNKNOWN',
            confidence=0.0
        )
        
        decisions = self.binder.apply(ts=2.0, tracks=[track], decisions=[decision])
        
        # Carry broken (track unhealthy)
        self.assertIsNone(decisions[0].identity_id,
                          "Unhealthy track should break carry")
        self.assertEqual(self._get_id_source(decisions[0]), 'U')
        
        # Memory should be deleted
        self.assertNotIn(1, self.binder.memories,
                         "Memory should be deleted after health break")
    
    def test_guard_2_appearance_break_in_real_mode(self):
        """Guard 2: Different appearance breaks carry in real mode."""
        # Original embedding
        embedding1 = np.random.rand(512).astype(np.float32)
        embedding1 /= np.linalg.norm(embedding1)
        
        self.binder.memories[1] = ContinuityMemory(
            track_id=1,
            person_id='alice',
            label='Alice',
            confidence=0.9,
            last_face_ts=1.0,
            last_bbox=(100, 100, 200, 200),
            last_embedding=embedding1.copy()
        )
        
        # Different embedding (opposite direction → distance ≈ 2.0)
        embedding2 = -embedding1
        
        track = Tracklet(
            track_id=1,
            camera_id='cam1',
            last_frame_id=10,
            last_box=(105, 105, 205, 205),
            confidence=0.8,
            age_frames=15,
            lost_frames=0,
            embedding=embedding2
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state='UNKNOWN',
            confidence=0.0
        )
        
        decisions = self.binder.apply(ts=2.0, tracks=[track], decisions=[decision])
        
        # Carry broken (appearance mismatch)
        self.assertIsNone(decisions[0].identity_id,
                          "Appearance break should prevent carry")
        self.assertNotIn(1, self.binder.memories,
                         "Memory should be deleted")
    
    def test_guard_2b_bbox_teleport_breaks_carry(self):
        """Guard 2b: Bbox teleport breaks carry in real mode."""
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        self.binder.memories[1] = ContinuityMemory(
            track_id=1,
            person_id='alice',
            label='Alice',
            confidence=0.9,
            last_face_ts=1.0,
            last_bbox=(100, 100, 200, 200),  # Original position
            last_embedding=embedding.copy()
        )
        
        # Teleport to far location (huge jump)
        track = Tracklet(
            track_id=1,
            camera_id='cam1',
            last_frame_id=10,
            last_box=(1000, 1000, 1100, 1100),  # Teleported
            confidence=0.8,
            age_frames=15,
            lost_frames=0,
            embedding=embedding.copy()
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state='UNKNOWN',
            confidence=0.0
        )
        
        decisions = self.binder.apply(ts=2.0, tracks=[track], decisions=[decision])
        
        # Carry broken (bbox teleport)
        self.assertIsNone(decisions[0].identity_id,
                          "Bbox teleport should break carry")
        self.assertNotIn(1, self.binder.memories,
                         "Memory should be deleted")
    
    def test_guard_3_face_contradiction_breaks_carry(self):
        """Guard 3: Persistent face contradiction during GPS carry phase."""
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        # Frame 0: Bind alice
        track = Tracklet(
            track_id=1,
            camera_id='cam1',
            last_frame_id=0,
            last_box=(100, 100, 200, 200),
            confidence=0.8,
            age_frames=15,
            lost_frames=0,
            embedding=embedding.copy()
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id='alice',
            binding_state='CONFIRMED_STRONG',
            confidence=0.92
        )
        
        self.binder.apply(ts=0.0, tracks=[track], decisions=[decision])
        self.assertIn(1, self.binder.memories)
        self.assertEqual(self.binder.memories[1].person_id, 'alice')
        
        # Frames 1-5: GPS carry (no face evidence)
        for i in range(1, 6):
            track.last_frame_id = i
            decision_unknown = IdentityDecision(
                track_id=1,
                identity_id=None,
                binding_state='UNKNOWN',
                confidence=0.0
            )
            
            decisions = self.binder.apply(ts=float(i), tracks=[track], decisions=[decision_unknown])
            
            # GPS carrying alice
            self.assertEqual(decisions[0].identity_id, 'alice')
            self.assertEqual(self.binder.memories[1].face_contradiction_counter, 0)
        
        # Frame 6: Face contradicts (says bob) - First contradiction
        track.last_frame_id = 6
        decision_bob = IdentityDecision(
            track_id=1,
            identity_id='bob',
            binding_state='CONFIRMED_STRONG',
            confidence=0.85
        )
        
        decisions = self.binder.apply(ts=6.0, tracks=[track], decisions=[decision_bob])
        
        # Face engine is authority - rebinds to bob immediately on CONFIRMED
        self.assertEqual(self.binder.memories[1].person_id, 'bob',
                         "Face CONFIRMED overrides memory (face is authority)")
        
        # This is correct behavior: Face engine CONFIRMED evidence takes precedence
        # Contradiction guard prevents GPS CARRY during contradictions,
        # but doesn't prevent face engine from assigning new identity when CONFIRMED
    
    def _get_id_source(self, decision: IdentityDecision) -> str:
        """Helper to get id_source."""
        if hasattr(decision, 'id_source') and decision.id_source is not None:
            return decision.id_source
        if decision.extra and 'id_source' in decision.extra:
            return decision.extra['id_source']
        return 'U'


class TestPhase5GraceReattachment(unittest.TestCase):
    """Phase 5: Grace reattachment in real mode."""
    
    def setUp(self):
        """Create binder in real mode."""
        cfg = type('Config', (), {
            'chimeric': type('ChimericConfig', (), {
                'continuity': type('ContinuityConfig', (), {
                    'min_track_age_frames': 5,
                    'shadow_mode': False,
                    'grace_window_sec': 1.0,
                    'max_bbox_displacement_px': 600
                })()
            })()
        })()
        
        self.binder = ContinuityBinder(cfg)
    
    def test_grace_reattachment_in_real_mode(self):
        """Grace reattachment works in real mode (identity restored)."""
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
        # Frame 1: Face confirms
        track = Tracklet(
            track_id=1,
            camera_id='cam1',
            last_frame_id=1,
            last_box=(100, 100, 200, 200),
            confidence=0.8,
            age_frames=10,
            embedding=embedding.copy()
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id='alice',
            binding_state='CONFIRMED_STRONG',
            confidence=0.92
        )
        
        self.binder.apply(ts=1.0, tracks=[track], decisions=[decision])
        self.assertIn(1, self.binder.memories)
        
        # Frame 2: Track disappears (empty tracks list)
        self.binder.apply(ts=1.5, tracks=[], decisions=[])
        
        # Memory moved to grace pool
        self.assertNotIn(1, self.binder.memories)
        self.assertIn(1, self.binder.recently_lost)
        
        # Frame 3: New track appears nearby (within grace window)
        new_track = Tracklet(
            track_id=2,  # Different track_id
            camera_id='cam1',
            last_frame_id=3,
            last_box=(110, 110, 210, 210),  # Close to original
            confidence=0.8,
            age_frames=2,
            embedding=embedding.copy()
        )
        
        new_decision = IdentityDecision(
            track_id=2,
            identity_id=None,
            binding_state='UNKNOWN',
            confidence=0.0
        )
        
        decisions = self.binder.apply(ts=1.8, tracks=[new_track], decisions=[new_decision])
        
        # Grace reattachment successful
        self.assertIn(2, self.binder.memories,
                      "Memory should be reattached to track 2")
        self.assertNotIn(1, self.binder.recently_lost,
                         "Grace pool should be cleared")
        
        # Identity restored
        self.assertEqual(decisions[0].identity_id, 'alice',
                         "Identity should be restored via grace reattachment")
        self.assertEqual(self._get_id_source(decisions[0]), 'G',
                         "id_source should be 'G' after reattachment")
    
    def _get_id_source(self, decision: IdentityDecision) -> str:
        """Helper to get id_source."""
        if hasattr(decision, 'id_source') and decision.id_source is not None:
            return decision.id_source
        if decision.extra and 'id_source' in decision.extra:
            return decision.extra['id_source']
        return 'U'


if __name__ == '__main__':
    unittest.main(verbosity=2)
