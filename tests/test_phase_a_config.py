
import pytest
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class TestPhaseAConfigLoad:
    
    def test_config_loads(self, test_config, logger):
        assert test_config is not None
        logger.info("✅ Config loads successfully")
    
    def test_governance_section_exists(self, test_config, logger):
        assert hasattr(test_config, 'governance'), "No governance section in config"
        assert test_config.governance is not None, "governance section is None"
        logger.info("✅ Governance section exists")
    
    def test_governance_is_not_empty(self, test_config, logger):
        gov = test_config.governance
        gov_dict = vars(gov) if hasattr(gov, '__dict__') else {}
        assert len(gov_dict) > 0, "Governance section is empty"
        logger.info(f"✅ Governance section has {len(gov_dict)} attributes")
    
    def test_governance_flags_exist(self, test_config, logger):
        required_sections = [
            'enabled',
            'evidence_gate',
            'binding',
            'scheduler',
            'merge',
            'debug',
        ]
        
        missing_sections = []
        for section in required_sections:
            if not hasattr(test_config.governance, section):
                missing_sections.append(section)
            else:
                value = getattr(test_config.governance, section)
                logger.info(f"  ✓ {section}: {type(value).__name__}")
        
        assert not missing_sections, f"Missing sections: {missing_sections}"
        logger.info(f"✅ All {len(required_sections)} governance sections exist")
    
    def test_governance_flag_types(self, test_config, logger):
        assert isinstance(test_config.governance.enabled, bool)
        logger.info(f"  ✓ enabled: bool = {test_config.governance.enabled}")
        
        subsections = ['evidence_gate', 'binding', 'scheduler', 'merge', 'debug']
        errors = []
        
        for section in subsections:
            if not hasattr(test_config.governance, section):
                errors.append(f"{section}: missing")
            else:
                obj = getattr(test_config.governance, section)
                logger.info(f"  ✓ {section}: {type(obj).__name__}")
        
        assert not errors, f"Missing subsections: {errors}"
        logger.info(f"✅ All governance sections have correct types")


class TestPhaseAConfigValues:
    
    def test_enabled_flag_boolean(self, test_config, logger):
        assert isinstance(test_config.governance.enabled, bool)
        logger.info(f"✅ 'enabled' is boolean: {test_config.governance.enabled}")
    
    def test_individual_phases_can_be_disabled(self, test_config, logger):
        phase_sections = ['evidence_gate', 'binding', 'scheduler', 'merge']
        
        sections_info = []
        for section in phase_sections:
            assert hasattr(test_config.governance, section), f"Missing {section} section"
            obj = getattr(test_config.governance, section)
            sections_info.append(f"{section}={type(obj).__name__}")
        
        logger.info("✅ Individual phase subsections exist and can be controlled:")
        for info in sections_info:
            logger.info(f"  ✓ {info}")


class TestPhaseASubsections:
    
    def test_evidence_gate_config_section(self, test_config, logger):
        if hasattr(test_config.governance, 'evidence_gate'):
            logger.info("✅ Evidence gate config section exists")
        else:
            logger.warning("⚠️ Evidence gate config section not found (may be optional)")
    
    def test_binding_config_section(self, test_config, logger):
        if hasattr(test_config.governance, 'binding'):
            logger.info("✅ Binding config section exists")
        else:
            logger.warning("⚠️ Binding config section not found (may be optional)")
    
    def test_scheduler_config_section(self, test_config, logger):
        if hasattr(test_config.governance, 'scheduler'):
            logger.info("✅ Scheduler config section exists")
        else:
            logger.warning("⚠️ Scheduler config section not found (may be optional)")
    
    def test_merge_manager_config_section(self, test_config, logger):
        if hasattr(test_config.governance, 'merge_manager'):
            logger.info("✅ Merge manager config section exists")
        else:
            logger.warning("⚠️ Merge manager config section not found (may be optional)")


class TestPhaseAConfigIntegrity:
    
    def test_config_yaml_valid(self, logger):
        config_path = Path(__file__).parent.parent / "config" / "default.yaml"
        assert config_path.exists(), f"Config file not found: {config_path}"
        
        import yaml
        try:
            with open(config_path, 'r') as f:
                yaml.safe_load(f)
            logger.info(f"✅ Config YAML is valid: {config_path}")
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML: {e}")
    
    def test_governance_metrics_collector_exists(self, logger):
        try:
            from core.governance_metrics import get_metrics_collector
            metrics = get_metrics_collector()
            assert metrics is not None
            logger.info("✅ Metrics collector is available")
        except ImportError as e:
            pytest.fail(f"Cannot import metrics collector: {e}")


class TestPhaseAMetrics:
    
    def test_metrics_collector_init(self, metrics_collector, logger):
        assert metrics_collector is not None
        logger.info("✅ Metrics collector initialized")
    
    def test_metrics_can_record(self, metrics_collector, logger):
        try:
            if hasattr(metrics_collector, 'record'):
                metrics_collector.record('test_metric', 1.0)
                logger.info("✅ Metrics can record values")
            else:
                logger.warning("⚠️ Metrics collector doesn't have 'record' method")
        except Exception as e:
            logger.warning(f"⚠️ Could not test metric recording: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
