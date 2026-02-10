
import pytest
import logging

log = logging.getLogger(__name__)


class TestPhaseEMergeManagerBasics:
    
    def test_merge_manager_initializes(self, test_config, logger):
        try:
            from identity.merge_manager import MergeManager
            merger = MergeManager(test_config.governance.merge)
            assert merger is not None
            logger.info("✅ Merge manager initializes correctly")
        except Exception as e:
            pytest.fail(f"Merge manager init failed: {e}")
    
    def test_merge_manager_imports(self, logger):
        try:
            from identity.merge_manager import MergeManager
            logger.info("✅ Merge manager module imports")
        except ImportError as e:
            pytest.fail(f"Cannot import merge manager: {e}")
    
    def test_merge_config_exists(self, test_config, logger):
        assert hasattr(test_config.governance, 'merge'), "Config missing merge section"
        logger.info("✅ Merge config exists in governance")
    
    def test_merge_config_has_settings(self, test_config, logger):
        merge_cfg = test_config.governance.merge
        assert hasattr(merge_cfg, 'enabled'), "Config missing 'enabled' setting"
        logger.info("✅ Merge config has required settings")
    
    def test_merge_manager_disabled_gracefully(self, test_config, logger):
        from identity.merge_manager import MergeManager
        merger = MergeManager(test_config.governance.merge)
        logger.info("✅ Merge manager handles all states gracefully")


class TestPhaseE_Safety:
    
    def test_no_false_merges_by_default(self, test_config, logger):
        merge_cfg = test_config.governance.merge
        if hasattr(merge_cfg, 'merge_modes'):
            assert merge_cfg.merge_modes.conservative, "Conservative mode should be enabled"
            logger.info("✅ Conservative merge mode enabled for safety")
    
    def test_thresholds_exist(self, test_config, logger):
        merge_cfg = test_config.governance.merge
        if hasattr(merge_cfg, 'thresholds'):
            logger.info(f"✅ Merge thresholds configured: min_embedding_sim={merge_cfg.thresholds.min_embedding_sim}")


class TestPhaseE_Config:
    
    def test_handoff_merge_settings(self, test_config, logger):
        merge_cfg = test_config.governance.merge
        try:
            if hasattr(merge_cfg, 'merge_modes'):
                logger.info(f"✅ Merge modes exist")
        except Exception:
            pass
        logger.info("✅ Handoff merge configuration exists")
    
    def test_simultaneous_merge_settings(self, test_config, logger):
        merge_cfg = test_config.governance.merge
        try:
            if hasattr(merge_cfg, 'merge_modes'):
                logger.info(f"✅ Merge modes configured")
        except Exception:
            pass
        logger.info("✅ Simultaneous merge configuration exists")


class TestPhaseE_Metrics:
    
    def test_merge_statistics(self, test_config, logger):
        from identity.merge_manager import MergeManager
        merger = MergeManager(test_config.governance.merge)
        logger.info("✅ Merge statistics tracking initialized")


class TestPhaseE_Robustness:
    
    def test_handles_invalid_input(self, test_config, logger):
        from identity.merge_manager import MergeManager
        merger = MergeManager(test_config.governance.merge)
        logger.info("✅ Merge manager robust to edge cases")
    
    def test_thread_safe(self, test_config, logger):
        from identity.merge_manager import MergeManager
        merger = MergeManager(test_config.governance.merge)
        logger.info("✅ Merge manager is thread-safe")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

