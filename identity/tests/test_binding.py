
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
    
    def setup_method(self):
        self.cfg = MockConfig()
        self.metrics = MockMetrics()
        self.engine = BindingManager(self.cfg, self.metrics)
    
    def test_unknown_to_pending_with_3_strong_samples(self):
        track_id = 1
        person_id = "person_123"
        ts = 1000.0
        
        for i in range(3):
            result = self.engine.process_evidence(
                track_id=track_id,
                person_id=person_id,
                score=0.90,
                second_best_score=0.78,
                quality=0.75,
                timestamp=ts + i * 0.5,
            )
        
        assert result.binding_state == BindingState.PENDING.value
        assert result.person_id == person_id
        assert result.confidence == 0.5
    
    def test_pending_to_confirmed_weak(self):
        track_id = 1
        person_id = "person_123"
        ts = 1000.0
        
        for i in range(3):
            self.engine.process_evidence(
                track_id=track_id,
                person_id=person_id,
                score=0.90,
                second_best_score=0.78,
                quality=0.75,
                timestamp=ts + i * 0.5,
            )
        
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
        track_id = 1
        person_id = "person_123"
        ts = 1000.0
        
        result = self.engine.process_evidence(
            track_id=track_id,
            person_id=person_id,
            score=0.80,
            second_best_score=0.75,
            quality=0.75,
            timestamp=ts,
        )
        
        assert result.binding_state == BindingState.UNKNOWN.value
    
    def test_reject_low_quality_samples(self):
        track_id = 1
        person_id = "person_123"
        ts = 1000.0
        
        result = self.engine.process_evidence(
            track_id=track_id,
            person_id=person_id,
            score=0.90,
            second_best_score=0.78,
            quality=0.50,
            timestamp=ts,
        )
        
        assert result.binding_state == BindingState.UNKNOWN.value
    
    def test_require_3_samples_for_lock(self):
        track_id = 1
        person_id = "person_123"
        ts = 1000.0
        
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
        
        assert result.binding_state == BindingState.UNKNOWN.value
        
        result = self.engine.process_evidence(
            track_id=track_id,
            person_id=person_id,
            score=0.90,
            second_best_score=0.78,
            quality=0.75,
            timestamp=ts + 2 * 0.5,
        )
        
        assert result.binding_state == BindingState.PENDING.value


class TestMarginEnforcement:
    
    def setup_method(self):
        self.cfg = MockConfig()
        self.metrics = MockMetrics()
        self.engine = BindingManager(self.cfg, self.metrics)
    
    def test_margin_threshold_enforcement(self):
        track_id = 1
        ts = 1000.0
        
        result = self.engine.process_evidence(
            track_id=track_id,
            person_id="person_1",
            score=0.80,
            second_best_score=0.77,
            quality=0.75,
            timestamp=ts,
        )
        
        assert result.binding_state == BindingState.UNKNOWN.value
        
        result = self.engine.process_evidence(
            track_id=track_id,
            person_id="person_1",
            score=0.90,
            second_best_score=0.81,
            quality=0.75,
            timestamp=ts + 1,
        )
        
        assert result.binding_state in [BindingState.UNKNOWN.value, BindingState.PENDING.value]
    
    def test_switching_requires_margin_advantage(self):
        track_id = 1
        ts = 1000.0
        
        for i in range(3):
            self.engine.process_evidence(
                track_id=track_id,
                person_id="person_1",
                score=0.90,
                second_best_score=0.78,
                quality=0.75,
                timestamp=ts + i,
            )
        
        for i in range(4):
            result = self.engine.process_evidence(
                track_id=track_id,
                person_id="person_2",
                score=0.95,
                second_best_score=0.85,
                quality=0.75,
                timestamp=ts + 3 + i,
            )
        
        assert result.person_id == "person_1"


class TestAntiLockInMechanism:
    
    def setup_method(self):
        self.cfg = MockConfig()
        self.metrics = MockMetrics()
        self.engine = BindingManager(self.cfg, self.metrics)
    
    def test_contradiction_counter_increment(self):
        track_id = 1
        ts = 1000.0
        
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
        
        for i in range(3):
            self.engine.process_evidence(
                track_id=track_id,
                person_id="person_1",
                score=0.10,
                second_best_score=0.05,
                quality=0.75,
                timestamp=ts + 3 + i,
            )
        
        assert track_state.contradiction_counter > initial_counter
    
    def test_downgrade_on_high_contradiction(self):
        track_id = 1
        ts = 1000.0
        
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
        
        for i in range(6):
            self.engine.process_evidence(
                track_id=track_id,
                person_id="person_1",
                score=0.10,
                second_best_score=0.05,
                quality=0.75,
                timestamp=ts + 5 + i,
            )
        
        assert track_state.state == BindingState.PENDING
    
    def test_anti_lock_break(self):
        track_id = 1
        ts = 1000.0
        
        for i in range(3):
            self.engine.process_evidence(
                track_id=track_id,
                person_id="person_1",
                score=0.90,
                second_best_score=0.78,
                quality=0.75,
                timestamp=ts + i,
            )
        
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
        
        for i in range(3):
            result = self.engine.process_evidence(
                track_id=track_id,
                person_id="person_2",
                score=0.92,
                second_best_score=0.80,
                quality=0.75,
                timestamp=ts + 10 + i,
            )
        
        assert result.person_id == "person_2"


class TestConfigurationHandling:
    
    def test_disabled_binding_returns_bypass(self):
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
        engine = BindingManager(None, None)
        
        result = engine.process_evidence(
            track_id=1,
            person_id="person_1",
            score=0.90,
            second_best_score=0.78,
            quality=0.75,
            timestamp=1000.0,
        )
        
        assert result.binding_state == "BYPASS"
    
    def test_custom_thresholds(self):
        cfg = MockConfig()
        cfg.governance.binding.confirmation.min_samples_strong = 2
        
        engine = BindingManager(cfg, None)
        
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
    
    def test_confirmation_event_recorded(self):
        cfg = MockConfig()
        metrics = MockMetrics()
        engine = BindingManager(cfg, metrics)
        
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
        cfg = MockConfig()
        metrics = MockMetrics()
        engine = BindingManager(cfg, metrics)
        
        for i in range(3):
            engine.process_evidence(
                track_id=1,
                person_id="person_1",
                score=0.90,
                second_best_score=0.78,
                quality=0.75,
                timestamp=1000.0 + i,
            )
        
        assert len(metrics.binding_state_counts) > 0


class TestErrorHandling:
    
    def test_none_person_id_handled(self):
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
        
        assert result is not None
    
    def test_invalid_timestamps_handled(self):
        cfg = MockConfig()
        engine = BindingManager(cfg, None)
        
        engine.process_evidence(
            track_id=1,
            person_id="person_1",
            score=0.90,
            second_best_score=0.78,
            quality=0.75,
            timestamp=1000.0,
        )
        
        result = engine.process_evidence(
            track_id=1,
            person_id="person_1",
            score=0.90,
            second_best_score=0.78,
            quality=0.75,
            timestamp=999.0,
        )
        
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
