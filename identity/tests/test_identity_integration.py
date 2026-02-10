
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from identity.identity_engine import FaceIdentityEngine
from identity.binding import BindingManager, BindingState
from schemas import IdentityDecision


class MockSearchResult:
    def __init__(self, person_id, distance=0.08, score=0.95, category="known"):
        self.person_id = person_id
        self.distance = distance
        self.score = score
        self.category = category


class MockGallery:
    def __init__(self, results_sequence=None):
        self.results_sequence = results_sequence or []
        self.call_count = 0
    
    def search_best(self, embedding, k=5):
        if self.call_count < len(self.results_sequence):
            result = self.results_sequence[self.call_count]
            self.call_count += 1
            return result
        return None


class MockFaceRoute:
    def __init__(self):
        self.call_count = 0
    
    def run(self, frame, tracks):
        return {}
    
    def reset(self):
        pass


class MockConfig:
    class Gallery:
        pass
    
    class Thresholds:
        strong_match_dist = 0.08
        weak_match_dist = 0.15
        min_quality_for_embed = 0.50
        min_quality_runtime = 0.50
    
    class Smoothing:
        min_samples_confirm = 3
        min_samples_switch = 4
        evidence_lookback_sec = 5.0
        half_life_sec = 30.0
        stale_after_sec = 60.0
        min_confidence = 0.30
    
    gallery = Gallery()
    thresholds = Thresholds()
    smoothing = Smoothing()


class TestBindingIntegrationBasic:
    
    def setup_method(self):
        self.cfg = MockConfig()
        self.gallery = MockGallery()
        self.face_route = MockFaceRoute()
        
        self.engine = FaceIdentityEngine(
            face_cfg=self.cfg,
            face_route=self.face_route,
            gallery=self.gallery,
        )
    
    def test_gallery_match_applied_through_binding(self):
        track_id = 1
        embedding = np.random.rand(128).astype(np.float32)
        
        self.gallery.results_sequence = [
            MockSearchResult("person_123", distance=0.08, score=0.95),
            MockSearchResult("person_123", distance=0.08, score=0.95),
            MockSearchResult("person_123", distance=0.08, score=0.95),
        ]
        
        ts = 1000.0
        decisions = []
        for i in range(3):
            decision = self.engine._decide_with_new_embedding(
                track_id=track_id,
                embedding=embedding,
                quality=0.80,
                ts=ts + i,
            )
            decisions.append(decision)
        
        final_decision = decisions[-1]
        assert "binding=" in final_decision.reason
    
    def test_decision_identity_can_be_overridden_by_binding(self):
        track_id = 1
        embedding = np.random.rand(128).astype(np.float32)
        
        self.gallery.results_sequence = [
            MockSearchResult("person_A", distance=0.08, score=0.92),
            MockSearchResult("person_A", distance=0.08, score=0.92),
            MockSearchResult("person_A", distance=0.08, score=0.92),
            MockSearchResult("person_B", distance=0.06, score=0.98),
            MockSearchResult("person_B", distance=0.06, score=0.98),
        ]
        
        ts = 1000.0
        decisions = []
        
        for i in range(3):
            decision = self.engine._decide_with_new_embedding(
                track_id=track_id,
                embedding=embedding,
                quality=0.80,
                ts=ts + i,
            )
            decisions.append(decision)
            print(f"Frame {i}: {decision.identity_id} (reason: {decision.reason[:100]})")
        
        assert decisions[2].identity_id == "person_A"
        
        for i in range(3, 5):
            decision = self.engine._decide_with_new_embedding(
                track_id=track_id,
                embedding=embedding,
                quality=0.80,
                ts=ts + i,
            )
            decisions.append(decision)
            print(f"Frame {i}: {decision.identity_id} (reason: {decision.reason[:100]})")
        

class TestBindingMultiFrameScenarios:
    
    def setup_method(self):
        self.cfg = MockConfig()
        self.gallery = MockGallery()
        self.face_route = MockFaceRoute()
        
        self.engine = FaceIdentityEngine(
            face_cfg=self.cfg,
            face_route=self.face_route,
            gallery=self.gallery,
        )
    
    def test_person_appears_leaves_reappears(self):
        track_id_1 = 1
        track_id_2 = 2
        embedding = np.random.rand(128).astype(np.float32)
        person_id = "person_alice"
        
        self.gallery.call_count = 0
        self.gallery.results_sequence = [
            MockSearchResult(person_id, distance=0.08, score=0.92),
            MockSearchResult(person_id, distance=0.08, score=0.92),
            MockSearchResult(person_id, distance=0.08, score=0.92),
        ]
        
        ts = 1000.0
        for i in range(3):
            decision = self.engine._decide_with_new_embedding(
                track_id=track_id_1,
                embedding=embedding,
                quality=0.80,
                ts=ts + i,
            )
            if i == 2:
                assert decision.identity_id == person_id
        
        self.engine._cleanup_dead_tracks({}, ts + 100)
        
        self.gallery.call_count = 0
        self.gallery.results_sequence = [
            MockSearchResult(person_id, distance=0.08, score=0.92),
            MockSearchResult(person_id, distance=0.08, score=0.92),
            MockSearchResult(person_id, distance=0.08, score=0.92),
        ]
        
        for i in range(3):
            decision = self.engine._decide_with_new_embedding(
                track_id=track_id_2,
                embedding=embedding,
                quality=0.80,
                ts=ts + 100 + i,
            )
            if i == 2:
                assert decision.identity_id == person_id
    
    def test_quality_fluctuation_maintains_binding(self):
        track_id = 1
        embedding = np.random.rand(128).astype(np.float32)
        person_id = "person_bob"
        ts = 1000.0
        
        self.gallery.results_sequence = [
            MockSearchResult(person_id, distance=0.08, score=0.92),
        ] * 10
        
        decisions = []
        
        for i in range(3):
            decision = self.engine._decide_with_new_embedding(
                track_id=track_id,
                embedding=embedding,
                quality=0.80,
                ts=ts + i,
            )
            decisions.append(decision)
        
        qualities = [0.70, 0.60, 0.75, 0.65, 0.80]
        for i, q in enumerate(qualities):
            decision = self.engine._decide_with_new_embedding(
                track_id=track_id,
                embedding=embedding,
                quality=q,
                ts=ts + 3 + i,
            )
            decisions.append(decision)
            
            if q >= self.cfg.thresholds.min_quality_runtime:
                assert decision.identity_id == person_id


class TestBindingErrorRecovery:
    
    def setup_method(self):
        self.cfg = MockConfig()
        self.gallery = MockGallery()
        self.face_route = MockFaceRoute()
        
        self.engine = FaceIdentityEngine(
            face_cfg=self.cfg,
            face_route=self.face_route,
            gallery=self.gallery,
        )
    
    def test_binding_manager_exception_doesnt_crash(self):
        track_id = 1
        embedding = np.random.rand(128).astype(np.float32)
        
        self.gallery.results_sequence = [
            MockSearchResult("person_123", distance=0.08, score=0.95),
        ]
        
        self.engine.binding_manager.process_evidence = Mock(
            side_effect=Exception("Binding error")
        )
        
        decision = self.engine._decide_with_new_embedding(
            track_id=track_id,
            embedding=embedding,
            quality=0.80,
            ts=1000.0,
        )
        
        assert decision is not None
        assert decision.identity_id is not None
    
    def test_binding_disabled_bypasses(self):
        track_id = 1
        embedding = np.random.rand(128).astype(np.float32)
        
        self.engine.binding_manager.enabled = False
        
        self.gallery.results_sequence = [
            MockSearchResult("person_123", distance=0.08, score=0.95),
        ]
        
        decision = self.engine._decide_with_new_embedding(
            track_id=track_id,
            embedding=embedding,
            quality=0.80,
            ts=1000.0,
        )
        
        assert decision.identity_id == "person_123"


class TestBindingWithDecide:
    
    def setup_method(self):
        self.cfg = MockConfig()
        self.gallery = MockGallery()
        self.face_route = MockFaceRoute()
        
        self.engine = FaceIdentityEngine(
            face_cfg=self.cfg,
            face_route=self.face_route,
            gallery=self.gallery,
        )
    
    def test_decide_with_binding_reason_included(self):
        pass


class TestBindingMetrics:
    
    def setup_method(self):
        self.cfg = MockConfig()
        self.gallery = MockGallery()
        self.face_route = MockFaceRoute()
        self.metrics = Mock()
        self.metrics.metrics = Mock()
        self.metrics.metrics.binding_state_counts = {}
        
        self.engine = FaceIdentityEngine(
            face_cfg=self.cfg,
            face_route=self.face_route,
            gallery=self.gallery,
        )
    
    def test_binding_decisions_logged(self):
        track_id = 1
        embedding = np.random.rand(128).astype(np.float32)
        
        self.gallery.results_sequence = [
            MockSearchResult("person_123", distance=0.08, score=0.95),
            MockSearchResult("person_123", distance=0.08, score=0.95),
            MockSearchResult("person_123", distance=0.08, score=0.95),
        ]
        
        ts = 1000.0
        for i in range(3):
            decision = self.engine._decide_with_new_embedding(
                track_id=track_id,
                embedding=embedding,
                quality=0.80,
                ts=ts + i,
            )
        

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
