
import unittest
from typing import List
import numpy as np

from schemas import Tracklet, IdentityDecision
from identity.continuity_binder import ContinuityBinder, ContinuityMemory


class TestPhase5RealMode(unittest.TestCase):
    
    def test_config_mode_continuity_enables_real_mode(self):
        cfg = type('Config', (), {
            'vision_identity': type('VisionIdentityConfig', (), {
                'mode': 'continuity',
                'continuity': type('ContinuityConfig', (), {
                    'min_track_age_frames': 10,
                    'shadow_mode': False
                })()
            })()
        })()
        
        binder = ContinuityBinder(cfg)
        
        self.assertFalse(binder.shadow_mode, "Shadow mode should be FALSE for real carry")
        self.assertEqual(binder.min_track_age_frames, 10)
    
    def test_shadow_mode_config_prevents_mutation(self):
        cfg = type('Config', (), {
            'vision_identity': type('VisionIdentityConfig', (), {
                'mode': 'shadow_continuity',
                'continuity': type('ContinuityConfig', (), {
                    'shadow_mode': True
                })()
            })()
        })()
        
        binder = ContinuityBinder(cfg)
        
        self.assertTrue(binder.shadow_mode, "Shadow mode should be TRUE")
    
    def test_classic_mode_config_should_not_instantiate_binder(self):
        pass


class TestPhase5RealCarry(unittest.TestCase):
    
    def setUp(self):
        cfg = type('Config', (), {
            'vision_identity': type('VisionIdentityConfig', (), {
                'mode': 'continuity',
                'continuity': type('ContinuityConfig', (), {
                    'min_track_age_frames': 5,
                    'shadow_mode': False,
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
        
        self.assertEqual(decisions[0].identity_id, 'alice')
        self.assertEqual(self._get_id_source(decisions[0]), 'F')
        
        track.age_frames = 11
        track.last_frame_id = 2
        
        decision_unknown = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state='UNKNOWN',
            confidence=0.0
        )
        
        decisions = self.binder.apply(ts=2.0, tracks=[track], decisions=[decision_unknown])
        
        self.assertEqual(decisions[0].identity_id, 'alice', 
                         "Real mode must mutate identity_id (GPS carry)")
        self.assertEqual(self._get_id_source(decisions[0]), 'G',
                         "id_source should be 'G' for GPS carry")
        
        self.assertEqual(decisions[0].binding_state, 'UNKNOWN',
                         "binding_state must NOT be mutated (face engine truth)")
    
    def test_shadow_mode_does_not_mutate_identity_id(self):
        cfg = type('Config', (), {
            'vision_identity': type('VisionIdentityConfig', (), {
                'continuity': type('ContinuityConfig', (), {
                    'min_track_age_frames': 5,
                    'shadow_mode': True
                })()
            })()
        })()
        
        shadow_binder = ContinuityBinder(cfg)
        
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
        
        track.age_frames = 11
        decision_unknown = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state='UNKNOWN',
            confidence=0.0
        )
        
        decisions = shadow_binder.apply(ts=2.0, tracks=[track], decisions=[decision_unknown])
        
        self.assertIsNone(decisions[0].identity_id,
                          "Shadow mode must NOT mutate identity_id")
        
        self.assertIsNotNone(decisions[0].extra)
        self.assertEqual(decisions[0].extra.get('would_carry'), 'alice',
                         "Shadow mode should annotate would_carry in extra dict")
        self.assertEqual(decisions[0].extra.get('shadow_id_source'), 'G',
                         "Shadow mode should set shadow_id_source in extra dict")
    
    def test_carried_confidence_stored_correctly(self):
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
        
        self.binder.apply(ts=1.0, tracks=[track], decisions=[decision])
        
        track.age_frames = 11
        decision_unknown = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state='UNKNOWN',
            confidence=0.0
        )
        
        decisions = self.binder.apply(ts=2.0, tracks=[track], decisions=[decision_unknown])
        
        self.assertEqual(decisions[0].confidence, 0.0,
                         "decision.confidence should stay as face engine value")
        
        self.assertIsNotNone(decisions[0].extra)
        self.assertEqual(decisions[0].extra.get('carried_confidence'), 0.92,
                         "Original face confidence should be in extra['carried_confidence']")
        self.assertTrue(decisions[0].extra.get('is_carried'),
                        "is_carried flag should be True")
    
    def test_gps_persistence_while_track_alive(self):
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
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
        
        for frame_idx in range(1, 10):
            track.age_frames = 10 + frame_idx
            track.last_frame_id = frame_idx
            track.last_box = (100 + frame_idx, 100 + frame_idx, 
                              200 + frame_idx, 200 + frame_idx)
            
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
            
            self.assertEqual(decisions[0].identity_id, 'alice',
                             f"Frame {frame_idx}: Identity should persist (GPS)")
            self.assertEqual(self._get_id_source(decisions[0]), 'G',
                             f"Frame {frame_idx}: id_source should be 'G'")
            
            self.assertIn(1, self.binder.memories,
                          f"Frame {frame_idx}: Memory should exist")
    
    def _get_id_source(self, decision: IdentityDecision) -> str:
        if hasattr(decision, 'id_source') and decision.id_source is not None:
            return decision.id_source
        if decision.extra and 'id_source' in decision.extra:
            return decision.extra['id_source']
        return 'U'


class TestPhase5GuardsRealMode(unittest.TestCase):
    
    def setUp(self):
        cfg = type('Config', (), {
            'vision_identity': type('VisionIdentityConfig', (), {
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
        
        track = Tracklet(
            track_id=1,
            camera_id='cam1',
            last_frame_id=5,
            last_box=(105, 105, 205, 205),
            confidence=0.8,
            age_frames=5,
            embedding=embedding.copy()
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state='UNKNOWN',
            confidence=0.0
        )
        
        decisions = self.binder.apply(ts=2.0, tracks=[track], decisions=[decision])
        
        self.assertIsNone(decisions[0].identity_id,
                          "Young track should not carry identity")
        self.assertEqual(self._get_id_source(decisions[0]), 'U')
    
    def test_guard_2c_track_health_breaks_carry(self):
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
        
        track = Tracklet(
            track_id=1,
            camera_id='cam1',
            last_frame_id=10,
            last_box=(105, 105, 205, 205),
            confidence=0.8,
            age_frames=15,
            lost_frames=3,
            embedding=embedding.copy()
        )
        
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state='UNKNOWN',
            confidence=0.0
        )
        
        decisions = self.binder.apply(ts=2.0, tracks=[track], decisions=[decision])
        
        self.assertIsNone(decisions[0].identity_id,
                          "Unhealthy track should break carry")
        self.assertEqual(self._get_id_source(decisions[0]), 'U')
        
        self.assertNotIn(1, self.binder.memories,
                         "Memory should be deleted after health break")
    
    def test_guard_2_appearance_break_in_real_mode(self):
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
        
        self.assertIsNone(decisions[0].identity_id,
                          "Appearance break should prevent carry")
        self.assertNotIn(1, self.binder.memories,
                         "Memory should be deleted")
    
    def test_guard_2b_bbox_teleport_breaks_carry(self):
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
        
        track = Tracklet(
            track_id=1,
            camera_id='cam1',
            last_frame_id=10,
            last_box=(1000, 1000, 1100, 1100),
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
        
        self.assertIsNone(decisions[0].identity_id,
                          "Bbox teleport should break carry")
        self.assertNotIn(1, self.binder.memories,
                         "Memory should be deleted")
    
    def test_guard_3_face_contradiction_breaks_carry(self):
        embedding = np.random.rand(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        
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
        
        for i in range(1, 6):
            track.last_frame_id = i
            decision_unknown = IdentityDecision(
                track_id=1,
                identity_id=None,
                binding_state='UNKNOWN',
                confidence=0.0
            )
            
            decisions = self.binder.apply(ts=float(i), tracks=[track], decisions=[decision_unknown])
            
            self.assertEqual(decisions[0].identity_id, 'alice')
            self.assertEqual(self.binder.memories[1].face_contradiction_counter, 0)
        
        track.last_frame_id = 6
        decision_bob = IdentityDecision(
            track_id=1,
            identity_id='bob',
            binding_state='CONFIRMED_STRONG',
            confidence=0.85
        )
        
        decisions = self.binder.apply(ts=6.0, tracks=[track], decisions=[decision_bob])
        
        self.assertEqual(self.binder.memories[1].person_id, 'bob',
                         "Face CONFIRMED overrides memory (face is authority)")
        
    
    def _get_id_source(self, decision: IdentityDecision) -> str:
        if hasattr(decision, 'id_source') and decision.id_source is not None:
            return decision.id_source
        if decision.extra and 'id_source' in decision.extra:
            return decision.extra['id_source']
        return 'U'


class TestPhase5GraceReattachment(unittest.TestCase):
    
    def setUp(self):
        cfg = type('Config', (), {
            'vision_identity': type('VisionIdentityConfig', (), {
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
        
        self.binder.apply(ts=1.0, tracks=[track], decisions=[decision])
        self.assertIn(1, self.binder.memories)
        
        self.binder.apply(ts=1.5, tracks=[], decisions=[])
        
        self.assertNotIn(1, self.binder.memories)
        self.assertIn(1, self.binder.recently_lost)
        
        new_track = Tracklet(
            track_id=2,
            camera_id='cam1',
            last_frame_id=3,
            last_box=(110, 110, 210, 210),
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
        
        self.assertIn(2, self.binder.memories,
                      "Memory should be reattached to track 2")
        self.assertNotIn(1, self.binder.recently_lost,
                         "Grace pool should be cleared")
        
        self.assertEqual(decisions[0].identity_id, 'alice',
                         "Identity should be restored via grace reattachment")
        self.assertEqual(self._get_id_source(decisions[0]), 'G',
                         "id_source should be 'G' after reattachment")
    
    def _get_id_source(self, decision: IdentityDecision) -> str:
        if hasattr(decision, 'id_source') and decision.id_source is not None:
            return decision.id_source
        if decision.extra and 'id_source' in decision.extra:
            return decision.extra['id_source']
        return 'U'


if __name__ == '__main__':
    unittest.main(verbosity=2)
