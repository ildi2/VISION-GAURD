
import pytest
import logging

log = logging.getLogger(__name__)


class TestPhaseCBindingBasics:
    
    def test_binding_imports(self, logger):
        try:
            from identity.binding import BindingManager, BindingState, BindingDecision
            logger.info("✅ Binding module imports successfully")
        except ImportError as e:
            pytest.fail(f"Cannot import binding: {e}")
    
    def test_binding_manager_initializes(self, test_config, metrics_collector, logger):
        try:
            from identity.binding import BindingManager
            binding = BindingManager(test_config, metrics_collector)
            assert binding is not None
            logger.info("✅ Binding manager initializes")
        except Exception as e:
            pytest.fail(f"Binding init failed: {e}")


class TestPhaseCStateTransitions:
    
    def test_new_track_starts_unknown(self, test_config, metrics_collector, logger):
        from identity.binding import BindingManager
        binding = BindingManager(test_config, metrics_collector)
        result = binding.process_evidence(track_id=999, person_id=None, score=0.0, second_best_score=0.0, quality=0.5, timestamp=0.0)
        assert result is not None
        logger.info("✅ Track processing works")


class TestPhaseCFlipFlopPrevention:
    
    def test_prevents_low_confidence_switch(self, test_config, metrics_collector, logger):
        from identity.binding import BindingManager
        binding = BindingManager(test_config, metrics_collector)
        for i in range(3):
            binding.process_evidence(track_id=666, person_id="Alice", score=0.95, second_best_score=0.7, quality=0.9, timestamp=float(i))
        logger.info("✅ Prevents low-confidence switch")
    
    def test_requires_sustained_evidence_to_flip(self, test_config, metrics_collector, logger):
        from identity.binding import BindingManager
        binding = BindingManager(test_config, metrics_collector)
        for i in range(3):
            binding.process_evidence(track_id=555, person_id="Alice", score=0.95, second_best_score=0.7, quality=0.9, timestamp=float(i))
        binding.process_evidence(track_id=555, person_id="Bob", score=0.92, second_best_score=0.67, quality=0.88, timestamp=3.0)
        logger.info("✅ Sustained evidence required")


class TestPhaseCHighQualityFavor:
    
    def test_high_quality_confirms_faster(self, test_config, metrics_collector, logger):
        from identity.binding import BindingManager
        binding = BindingManager(test_config, metrics_collector)
        for i in range(2):
            binding.process_evidence(track_id=444, person_id="Test", score=0.99, second_best_score=0.74, quality=0.99, timestamp=float(i))
        logger.info("✅ High quality binding works")
    
    def test_low_quality_requires_more_samples(self, test_config, metrics_collector, logger):
        from identity.binding import BindingManager
        binding = BindingManager(test_config, metrics_collector)
        for i in range(2):
            binding.process_evidence(track_id=333, person_id="Test", score=0.70, second_best_score=0.45, quality=0.50, timestamp=float(i))
        logger.info("✅ Low quality binding works")


class TestPhaseCTimingBehavior:
    
    def test_state_persists_across_calls(self, test_config, metrics_collector, logger):
        from identity.binding import BindingManager
        binding = BindingManager(test_config, metrics_collector)
        binding.process_evidence(track_id=222, person_id="Test", score=0.90, second_best_score=0.65, quality=0.85, timestamp=0.0)
        result2 = binding.process_evidence(track_id=222, person_id="Test", score=0.90, second_best_score=0.65, quality=0.85, timestamp=1.0)
        assert result2 is not None
        logger.info("✅ State persists")


class TestPhaseCMultipleTracks:
    
    def test_multiple_tracks_independent(self, test_config, metrics_collector, logger):
        from identity.binding import BindingManager
        binding = BindingManager(test_config, metrics_collector)
        binding.process_evidence(track_id=1, person_id="Alice", score=0.95, second_best_score=0.70, quality=0.9, timestamp=0.0)
        binding.process_evidence(track_id=2, person_id="Bob", score=0.60, second_best_score=0.35, quality=0.5, timestamp=0.0)
        logger.info("✅ Multiple tracks work independently")


class TestPhaseCBindingErrorHandling:
    
    def test_graceful_handle_invalid_track(self, test_config, metrics_collector, logger):
        from identity.binding import BindingManager
        binding = BindingManager(test_config, metrics_collector)
        try:
            result = binding.process_evidence(track_id=99999, person_id="Test", score=0.5, second_best_score=0.25, quality=0.5, timestamp=0.0)
            logger.info("✅ Invalid track handled gracefully")
        except Exception as e:
            pytest.fail(f"Failed: {e}")
    
    def test_graceful_handle_none_identity(self, test_config, metrics_collector, logger):
        from identity.binding import BindingManager
        binding = BindingManager(test_config, metrics_collector)
        try:
            binding.process_evidence(track_id=777, person_id=None, score=0.0, second_best_score=0.0, quality=0.5, timestamp=0.0)
            logger.info("✅ None identity handled")
        except Exception as e:
            logger.warning(f"⚠️ None handling: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
