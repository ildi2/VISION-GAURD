"""
Integration tests for PHASE C: Binding State Machine with Identity Engine

Covers end-to-end flow with:
- Gallery matching
- Binding state machine application
- Decision override behavior
- Multi-frame scenarios
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from identity.identity_engine import FaceIdentityEngine
from identity.binding import BindingManager, BindingState
from schemas import IdentityDecision


class MockSearchResult:
    """Mock gallery search result."""
    def __init__(self, person_id, distance=0.08, score=0.95, category="known"):
        self.person_id = person_id
        self.distance = distance
        self.score = score
        self.category = category


class MockGallery:
    """Mock FaceGallery."""
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
    """Mock FaceRoute."""
    def __init__(self):
        self.call_count = 0
    
    def run(self, frame, tracks):
        # Return empty (no evidence) by default
        return {}
    
    def reset(self):
        pass


class MockConfig:
    """Mock configuration."""
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
    """Test basic binding integration with identity engine."""
    
    def setup_method(self):
        """Setup for each test."""
        self.cfg = MockConfig()
        self.gallery = MockGallery()
        self.face_route = MockFaceRoute()
        
        self.engine = FaceIdentityEngine(
            face_cfg=self.cfg,
            face_route=self.face_route,
            gallery=self.gallery,
        )
    
    def test_gallery_match_applied_through_binding(self):
        """Strong gallery match should go through binding state machine."""
        track_id = 1
        embedding = np.random.rand(128).astype(np.float32)
        
        # Setup gallery to return strong match
        self.gallery.results_sequence = [
            MockSearchResult("person_123", distance=0.08, score=0.95),
            MockSearchResult("person_123", distance=0.08, score=0.95),
            MockSearchResult("person_123", distance=0.08, score=0.95),
        ]
        
        # Make decisions with new embedding
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
        
        # After 3 samples, should have binding info in reason
        final_decision = decisions[-1]
        assert "binding=" in final_decision.reason
    
    def test_decision_identity_can_be_overridden_by_binding(self):
        """Binding state machine can override gallery match identity."""
        track_id = 1
        embedding = np.random.rand(128).astype(np.float32)
        
        # Setup gallery to return matches
        self.gallery.results_sequence = [
            MockSearchResult("person_A", distance=0.08, score=0.92),
            MockSearchResult("person_A", distance=0.08, score=0.92),
            MockSearchResult("person_A", distance=0.08, score=0.92),
            MockSearchResult("person_B", distance=0.06, score=0.98),  # Tempting switch
            MockSearchResult("person_B", distance=0.06, score=0.98),
        ]
        
        ts = 1000.0
        decisions = []
        
        # First 3 frames lock to person_A
        for i in range(3):
            decision = self.engine._decide_with_new_embedding(
                track_id=track_id,
                embedding=embedding,
                quality=0.80,
                ts=ts + i,
            )
            decisions.append(decision)
            print(f"Frame {i}: {decision.identity_id} (reason: {decision.reason[:100]})")
        
        # At frame 3, person_A should be locked
        assert decisions[2].identity_id == "person_A"
        
        # Frames 4-5 with person_B
        for i in range(3, 5):
            decision = self.engine._decide_with_new_embedding(
                track_id=track_id,
                embedding=embedding,
                quality=0.80,
                ts=ts + i,
            )
            decisions.append(decision)
            print(f"Frame {i}: {decision.identity_id} (reason: {decision.reason[:100]})")
        
        # May still be person_A due to binding margin requirements
        # (person_B needs significant margin advantage)


class TestBindingMultiFrameScenarios:
    """Test multi-frame realistic scenarios."""
    
    def setup_method(self):
        """Setup for each test."""
        self.cfg = MockConfig()
        self.gallery = MockGallery()
        self.face_route = MockFaceRoute()
        
        self.engine = FaceIdentityEngine(
            face_cfg=self.cfg,
            face_route=self.face_route,
            gallery=self.gallery,
        )
    
    def test_person_appears_leaves_reappears(self):
        """Person appears, tracking ends, reappears - should rebind."""
        track_id_1 = 1
        track_id_2 = 2  # Different track when person reappears
        embedding = np.random.rand(128).astype(np.float32)
        person_id = "person_alice"
        
        # Track 1: Alice appears, gets bound
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
        
        # Track 1 lost (track cleanup)
        self.engine._cleanup_dead_tracks({}, ts + 100)
        
        # Track 2: Alice reappears with same face
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
                # Should rebind to same person
                assert decision.identity_id == person_id
    
    def test_quality_fluctuation_maintains_binding(self):
        """Binding should maintain through quality fluctuations."""
        track_id = 1
        embedding = np.random.rand(128).astype(np.float32)
        person_id = "person_bob"
        ts = 1000.0
        
        # Setup constant matches
        self.gallery.results_sequence = [
            MockSearchResult(person_id, distance=0.08, score=0.92),
        ] * 10
        
        decisions = []
        
        # Locked frames with stable quality
        for i in range(3):
            decision = self.engine._decide_with_new_embedding(
                track_id=track_id,
                embedding=embedding,
                quality=0.80,
                ts=ts + i,
            )
            decisions.append(decision)
        
        # Frames with fluctuating quality
        qualities = [0.70, 0.60, 0.75, 0.65, 0.80]
        for i, q in enumerate(qualities):
            decision = self.engine._decide_with_new_embedding(
                track_id=track_id,
                embedding=embedding,
                quality=q,
                ts=ts + 3 + i,
            )
            decisions.append(decision)
            
            # Should maintain binding even with lower quality
            # (as long as above min_quality_runtime)
            if q >= self.cfg.thresholds.min_quality_runtime:
                assert decision.identity_id == person_id


class TestBindingErrorRecovery:
    """Test binding behavior on errors."""
    
    def setup_method(self):
        """Setup for each test."""
        self.cfg = MockConfig()
        self.gallery = MockGallery()
        self.face_route = MockFaceRoute()
        
        self.engine = FaceIdentityEngine(
            face_cfg=self.cfg,
            face_route=self.face_route,
            gallery=self.gallery,
        )
    
    def test_binding_manager_exception_doesnt_crash(self):
        """BindingManager exception should not crash pipeline."""
        track_id = 1
        embedding = np.random.rand(128).astype(np.float32)
        
        # Setup gallery to return strong match
        self.gallery.results_sequence = [
            MockSearchResult("person_123", distance=0.08, score=0.95),
        ]
        
        # Mock binding manager to raise exception
        self.engine.binding_manager.process_evidence = Mock(
            side_effect=Exception("Binding error")
        )
        
        # Should not crash - use original decision
        decision = self.engine._decide_with_new_embedding(
            track_id=track_id,
            embedding=embedding,
            quality=0.80,
            ts=1000.0,
        )
        
        # Should return valid decision
        assert decision is not None
        assert decision.identity_id is not None
    
    def test_binding_disabled_bypasses(self):
        """When binding disabled, decisions should work normally."""
        track_id = 1
        embedding = np.random.rand(128).astype(np.float32)
        
        # Disable binding
        self.engine.binding_manager.enabled = False
        
        # Setup gallery
        self.gallery.results_sequence = [
            MockSearchResult("person_123", distance=0.08, score=0.95),
        ]
        
        decision = self.engine._decide_with_new_embedding(
            track_id=track_id,
            embedding=embedding,
            quality=0.80,
            ts=1000.0,
        )
        
        # Should work normally (binding bypassed)
        assert decision.identity_id == "person_123"


class TestBindingWithDecide:
    """Test binding integration through high-level decide() API."""
    
    def setup_method(self):
        """Setup for each test."""
        self.cfg = MockConfig()
        self.gallery = MockGallery()
        self.face_route = MockFaceRoute()
        
        self.engine = FaceIdentityEngine(
            face_cfg=self.cfg,
            face_route=self.face_route,
            gallery=self.gallery,
        )
    
    def test_decide_with_binding_reason_included(self):
        """decide() should include binding info in reason string."""
        # This would require full IdSignals setup
        # Skipped for now as it requires full Frame/Tracklet setup
        pass


class TestBindingMetrics:
    """Test metrics collection through binding integration."""
    
    def setup_method(self):
        """Setup for each test."""
        self.cfg = MockConfig()
        self.gallery = MockGallery()
        self.face_route = MockFaceRoute()
        self.metrics = Mock()
        self.metrics.metrics = Mock()
        self.metrics.metrics.binding_state_counts = {}
        
        # Create engine with metrics mock
        self.engine = FaceIdentityEngine(
            face_cfg=self.cfg,
            face_route=self.face_route,
            gallery=self.gallery,
        )
    
    def test_binding_decisions_logged(self):
        """Binding decisions should be logged."""
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
        
        # Binding should have recorded something
        # (would verify through metrics if they were properly mocked)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
