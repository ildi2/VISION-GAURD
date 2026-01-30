"""
Tests for PHASE C: Binding State Machine

Covers:
- State transitions (UNKNOWN → PENDING → CONFIRMED)
- Margin enforcement
- Anti-lock-in mechanism
- Switching logic
- Configuration handling
- Metrics recording
"""

import pytest
import logging
from typing import Optional
from dataclasses import dataclass

from identity.binding import (
    BindingManager,
    BindingState,
    BindingConfig,
    BindingDecision,
    EvidenceRecord,
)


logger = logging.getLogger(__name__)


@dataclass
class MockMetrics:
    """Mock metrics collector for testing."""
    binding_state_counts: dict = None
    binding_confirmations: int = 0
    binding_downgrades: int = 0
    binding_switches_success: int = 0
    binding_anti_lock_triggers: int = 0
    
    def __post_init__(self):
        if self.binding_state_counts is None:
            self.binding_state_counts = {}
    
    def record_binding_confirmation(self):
        self.binding_confirmations += 1
    
    def record_binding_downgrade(self):
        self.binding_downgrades += 1
    
    def record_binding_switch(self, success: bool):
        if success:
            self.binding_switches_success += 1


@dataclass
class MockConfig:
    """Mock configuration for testing."""
    class Governance:
        class Binding:
            enabled = True
            class Confirmation:
                min_samples_strong = 3
                min_samples_weak = 5
                window_seconds = 3.0
                min_avg_score = 0.75
                min_avg_margin = 0.08
                min_quality_for_strong = 0.60
            
            class Switching:
                min_sustained_samples = 4
                margin_advantage = 0.12
                window_seconds = 2.0
                timeout_seconds = 5.0
            
            class Contradiction:
                threshold = 0.15
                counter_max = 5
                decay_per_second = 1.0
            
            confirmation = Confirmation()
            switching = Switching()
            contradiction = Contradiction()
        
        binding = Binding()
    
    governance = Governance()


class TestBindingStateTransitions:
    """Test state machine transitions."""
    
    def setup_method(self):
        """Setup for each test."""
        self.cfg = MockConfig()
        self.metrics = MockMetrics()
        self.engine = BindingManager(self.cfg, self.metrics)
    
    def test_unknown_to_pending_with_3_strong_samples(self):
        """Should transition UNKNOWN → PENDING with 3 strong samples."""
        track_id = 1
        person_id = "person_123"
        ts = 1000.0
        
        # Add 3 strong samples within window
        for i in range(3):
            result = self.engine.process_evidence(
                track_id=track_id,
                person_id=person_id,
                score=0.90,
                second_best_score=0.78,  # margin = 0.12
                quality=0.75,
                timestamp=ts + i * 0.5,
            )
        
        assert result.binding_state == BindingState.PENDING.value
        assert result.person_id == person_id
        assert result.confidence == 0.5
    
    def test_pending_to_confirmed_weak(self):
        """Should transition PENDING → CONFIRMED_WEAK with sustained samples."""
        track_id = 1
        person_id = "person_123"
        ts = 1000.0
        
        # Get to PENDING
        for i in range(3):
            self.engine.process_evidence(
                track_id=track_id,
                person_id=person_id,
                score=0.90,
                second_best_score=0.78,
                quality=0.75,
                timestamp=ts + i * 0.5,
            )
        
        # Continue with more samples to move to CONFIRMED_WEAK
        result = None
        for i in range(3, 5):
            result = self.engine.process_evidence(
                track_id=track_id,
                person_id=person_id,
                score=0.88,
                second_best_score=0.76,
                quality=0.75,
                timestamp=ts + i * 0.5,
            )
        
        assert result.binding_state == BindingState.CONFIRMED_WEAK.value
        assert result.confidence >= 0.70
    
    def test_reject_weak_samples(self):
        """Should reject samples below margin threshold."""
        track_id = 1
        person_id = "person_123"
        ts = 1000.0
        
        # Add weak sample (margin < 0.08)
        result = self.engine.process_evidence(
            track_id=track_id,
            person_id=person_id,
            score=0.80,
            second_best_score=0.75,  # margin = 0.05 < 0.08
            quality=0.75,
            timestamp=ts,
        )
        
        assert result.binding_state == BindingState.UNKNOWN.value
    
    def test_reject_low_quality_samples(self):
        """Should reject samples with low face quality."""
        track_id = 1
        person_id = "person_123"
        ts = 1000.0
        
        # Add low quality sample
        result = self.engine.process_evidence(
            track_id=track_id,
            person_id=person_id,
            score=0.90,
            second_best_score=0.78,
            quality=0.50,  # < 0.60 min
            timestamp=ts,
        )
        
        assert result.binding_state == BindingState.UNKNOWN.value
    
    def test_require_3_samples_for_lock(self):
        """Should require min_samples_strong to lock."""
        track_id = 1
        person_id = "person_123"
        ts = 1000.0
        
        # Add only 2 samples
        result = None
        for i in range(2):
            result = self.engine.process_evidence(
                track_id=track_id,
                person_id=person_id,
                score=0.90,
                second_best_score=0.78,
                quality=0.75,
                timestamp=ts + i * 0.5,
            )
        
        # Should still be UNKNOWN with 2 samples
        assert result.binding_state == BindingState.UNKNOWN.value
        
        # Add 3rd sample
        result = self.engine.process_evidence(
            track_id=track_id,
            person_id=person_id,
            score=0.90,
            second_best_score=0.78,
            quality=0.75,
            timestamp=ts + 2 * 0.5,
        )
        
        # Now should transition to PENDING
        assert result.binding_state == BindingState.PENDING.value


class TestMarginEnforcement:
    """Test margin-based evidence validation."""
    
    def setup_method(self):
        """Setup for each test."""
        self.cfg = MockConfig()
        self.metrics = MockMetrics()
        self.engine = BindingManager(self.cfg, self.metrics)
    
    def test_margin_threshold_enforcement(self):
        """Margin must be >= min_avg_margin."""
        track_id = 1
        ts = 1000.0
        
        # Score 0.80, second 0.77 → margin 0.03 (too low)
        result = self.engine.process_evidence(
            track_id=track_id,
            person_id="person_1",
            score=0.80,
            second_best_score=0.77,
            quality=0.75,
            timestamp=ts,
        )
        
        assert result.binding_state == BindingState.UNKNOWN.value
        
        # Score 0.90, second 0.81 → margin 0.09 (OK)
        result = self.engine.process_evidence(
            track_id=track_id,
            person_id="person_1",
            score=0.90,
            second_best_score=0.81,
            quality=0.75,
            timestamp=ts + 1,
        )
        
        # Should start accumulating
        assert result.binding_state in [BindingState.UNKNOWN.value, BindingState.PENDING.value]
    
    def test_switching_requires_margin_advantage(self):
        """Switching requires new identity to exceed current + margin_advantage."""
        track_id = 1
        ts = 1000.0
        
        # Lock to person_1
        for i in range(3):
            self.engine.process_evidence(
                track_id=track_id,
                person_id="person_1",
                score=0.90,
                second_best_score=0.78,
                quality=0.75,
                timestamp=ts + i,
            )
        
        # Try to switch to person_2 with insufficient advantage
        # person_1 avg ≈ 0.90, person_2 needs > 0.90 + 0.12 = 1.02 (impossible)
        for i in range(4):
            result = self.engine.process_evidence(
                track_id=track_id,
                person_id="person_2",
                score=0.95,
                second_best_score=0.85,  # margin 0.10
                quality=0.75,
                timestamp=ts + 3 + i,
            )
        
        # Should still be locked to person_1 (switch attempted but not enough margin)
        assert result.person_id == "person_1"


class TestAntiLockInMechanism:
    """Test contradiction detection and anti-lock-in."""
    
    def setup_method(self):
        """Setup for each test."""
        self.cfg = MockConfig()
        self.metrics = MockMetrics()
        self.engine = BindingManager(self.cfg, self.metrics)
    
    def test_contradiction_counter_increment(self):
        """Contradiction counter should increment on low scores."""
        track_id = 1
        ts = 1000.0
        
        # Lock to person_1
        for i in range(3):
            self.engine.process_evidence(
                track_id=track_id,
                person_id="person_1",
                score=0.90,
                second_best_score=0.78,
                quality=0.75,
                timestamp=ts + i,
            )
        
        track_state = self.engine._track_states[track_id]
        initial_counter = track_state.contradiction_counter
        
        # Add low score for locked person (contradiction)
        for i in range(3):
            self.engine.process_evidence(
                track_id=track_id,
                person_id="person_1",
                score=0.10,  # Very low for locked person
                second_best_score=0.05,
                quality=0.75,
                timestamp=ts + 3 + i,
            )
        
        # Counter should have increased
        assert track_state.contradiction_counter > initial_counter
    
    def test_downgrade_on_high_contradiction(self):
        """Should downgrade state when contradiction exceeds threshold."""
        track_id = 1
        ts = 1000.0
        
        # Get to CONFIRMED_WEAK
        for i in range(5):
            self.engine.process_evidence(
                track_id=track_id,
                person_id="person_1",
                score=0.88,
                second_best_score=0.76,
                quality=0.75,
                timestamp=ts + i,
            )
        
        track_state = self.engine._track_states[track_id]
        assert track_state.state == BindingState.CONFIRMED_WEAK
        
        # Generate contradictions
        for i in range(6):  # counter_max = 5
            self.engine.process_evidence(
                track_id=track_id,
                person_id="person_1",
                score=0.10,
                second_best_score=0.05,
                quality=0.75,
                timestamp=ts + 5 + i,
            )
        
        # Should have downgraded
        assert track_state.state == BindingState.PENDING
    
    def test_anti_lock_break(self):
        """After downgrade, system should accept new bindings."""
        track_id = 1
        ts = 1000.0
        
        # Lock to person_1
        for i in range(3):
            self.engine.process_evidence(
                track_id=track_id,
                person_id="person_1",
                score=0.90,
                second_best_score=0.78,
                quality=0.75,
                timestamp=ts + i,
            )
        
        # Generate contradictions to force downgrade
        for i in range(7):
            self.engine.process_evidence(
                track_id=track_id,
                person_id="person_1",
                score=0.10,
                second_best_score=0.05,
                quality=0.75,
                timestamp=ts + 3 + i,
            )
        
        track_state = self.engine._track_states[track_id]
        
        # Now add strong evidence for different person
        for i in range(3):
            result = self.engine.process_evidence(
                track_id=track_id,
                person_id="person_2",
                score=0.92,
                second_best_score=0.80,
                quality=0.75,
                timestamp=ts + 10 + i,
            )
        
        # Should rebind to person_2
        assert result.person_id == "person_2"


class TestConfigurationHandling:
    """Test configuration loading and defaults."""
    
    def test_disabled_binding_returns_bypass(self):
        """When disabled, binding should bypass."""
        cfg = MockConfig()
        cfg.governance.binding.enabled = False
        
        engine = BindingManager(cfg, None)
        result = engine.process_evidence(
            track_id=1,
            person_id="person_1",
            score=0.90,
            second_best_score=0.78,
            quality=0.75,
            timestamp=1000.0,
        )
        
        assert result.binding_state == "BYPASS"
    
    def test_missing_config_fallback(self):
        """Should use defaults when config missing."""
        engine = BindingManager(None, None)
        
        result = engine.process_evidence(
            track_id=1,
            person_id="person_1",
            score=0.90,
            second_best_score=0.78,
            quality=0.75,
            timestamp=1000.0,
        )
        
        # Should have bypassed gracefully
        assert result.binding_state == "BYPASS"
    
    def test_custom_thresholds(self):
        """Should apply custom thresholds from config."""
        cfg = MockConfig()
        cfg.governance.binding.confirmation.min_samples_strong = 2  # Lower threshold
        
        engine = BindingManager(cfg, None)
        
        # Should lock with only 2 samples instead of 3
        result = None
        for i in range(2):
            result = engine.process_evidence(
                track_id=1,
                person_id="person_1",
                score=0.90,
                second_best_score=0.78,
                quality=0.75,
                timestamp=1000.0 + i,
            )
        
        assert result.binding_state == BindingState.PENDING.value


class TestMetricsRecording:
    """Test metrics collection."""
    
    def test_confirmation_event_recorded(self):
        """Should record confirmation when transitioning to CONFIRMED."""
        cfg = MockConfig()
        metrics = MockMetrics()
        engine = BindingManager(cfg, metrics)
        
        # Get to CONFIRMED_WEAK
        for i in range(5):
            engine.process_evidence(
                track_id=1,
                person_id="person_1",
                score=0.88,
                second_best_score=0.76,
                quality=0.75,
                timestamp=1000.0 + i,
            )
        
        assert metrics.binding_confirmations > 0
    
    def test_state_count_updated(self):
        """Should track state counts."""
        cfg = MockConfig()
        metrics = MockMetrics()
        engine = BindingManager(cfg, metrics)
        
        # Add evidence
        for i in range(3):
            engine.process_evidence(
                track_id=1,
                person_id="person_1",
                score=0.90,
                second_best_score=0.78,
                quality=0.75,
                timestamp=1000.0 + i,
            )
        
        # Should have recorded state counts
        assert len(metrics.binding_state_counts) > 0


class TestErrorHandling:
    """Test error handling and safety."""
    
    def test_none_person_id_handled(self):
        """Should handle None person_id gracefully."""
        cfg = MockConfig()
        engine = BindingManager(cfg, None)
        
        result = engine.process_evidence(
            track_id=1,
            person_id=None,
            score=0.0,
            second_best_score=0.0,
            quality=0.0,
            timestamp=1000.0,
        )
        
        # Should not crash
        assert result is not None
    
    def test_invalid_timestamps_handled(self):
        """Should handle time reversals gracefully."""
        cfg = MockConfig()
        engine = BindingManager(cfg, None)
        
        # Add evidence at t=1000
        engine.process_evidence(
            track_id=1,
            person_id="person_1",
            score=0.90,
            second_best_score=0.78,
            quality=0.75,
            timestamp=1000.0,
        )
        
        # Add evidence at earlier time
        result = engine.process_evidence(
            track_id=1,
            person_id="person_1",
            score=0.90,
            second_best_score=0.78,
            quality=0.75,
            timestamp=999.0,  # Time went backward
        )
        
        # Should not crash
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
