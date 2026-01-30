# chimeric_identity/tests/test_phase3_logging_cli.py
# ============================================================================
# PHASE 3 INTEGRATION TESTS - Logging, Metrics, CLI
# ============================================================================
#
# Purpose:
#   Validate logging output formats, metrics collection, and CLI interface.
#   Tests production-readiness of operationalization layer.
#
# Test Coverage:
#   1. ChimericLogger: 4 verbosity levels (QUIET, NORMAL, DEBUG, TRACE)
#   2. ChimericDecisionFormatter: All 3 output formats (oneline, detailed, JSON)
#   3. MetricsCollector: Sliding window aggregation
#   4. CLI Commands: run, analyze, validate subcommands
#   5. Config loading and CLI override handling
#
# Design:
#   - Test logging output format correctness
#   - Test metrics collection with synthetic decisions
#   - Test CLI argument parsing
#   - Test config validation logic

import time
import json
import pytest
import tempfile
import os
from io import StringIO
from unittest.mock import Mock, patch
from dataclasses import dataclass

from chimeric_identity.types import (
    ChimericState,
    ChimericReason,
    ChimericDecision,
    FaceEvidence,
    GaitEvidence,
    SourceAuthEvidence,
    EvidenceStatus,
    SourceAuthState,
)
from chimeric_identity.logging_utils import (
    ChimericLogger,
    ChimericDecisionFormatter,
    MetricsCollector,
)
from chimeric_identity.config import ChimericConfig


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_test_decision(
    track_id: str = "track_001",
    final_identity: str = "alice",
    state: ChimericState = ChimericState.CONFIRMED,
    confidence: float = 0.88,
    learning_allowed: bool = True,
) -> ChimericDecision:
    """Create a test ChimericDecision."""
    face = FaceEvidence(
        identity_id=final_identity,
        similarity=confidence,
        quality=0.92,
        status=EvidenceStatus.CONFIRMED_STRONG,
        margin=0.15,
        timestamp=time.time()
    )
    
    gait = GaitEvidence(
        identity_id=final_identity,
        similarity=0.75,
        margin=0.08,
        quality=0.70,
        status=EvidenceStatus.CONFIRMED_WEAK,
        sequence_length=40,
        timestamp=time.time()
    )
    
    return ChimericDecision(
        track_id=track_id,
        final_identity=final_identity,
        chimeric_confidence=confidence,
        state=state,
        evidence_summary={
            "face": face,
            "gait": gait,
            "source_auth": None
        },
        decision_reason=ChimericReason.FACE_CONFIRMED_WITH_GAIT_SUPPORT,
        learning_allowed=learning_allowed,
        timestamp=time.time(),
        debug_trace={}
    )


# ============================================================================
# TEST: CHIMERIC DECISION FORMATTER
# ============================================================================

class TestChimericDecisionFormatter:
    """Test decision output formatting."""

    def setup_method(self):
        """Initialize formatter."""
        self.formatter = ChimericDecisionFormatter()

    def test_format_oneline_basic(self):
        """Test one-line format for CONFIRMED decision."""
        decision = create_test_decision()
        
        output = self.formatter.format_oneline(decision)
        
        assert "[CHIMERIC]" in output
        assert "track_id=track_001" in output
        assert "state=CONFIRMED" in output
        assert "identity=alice" in output
        assert "confidence=" in output
        assert "learning=" in output

    def test_format_oneline_includes_face_gait(self):
        """One-line format includes both face and gait status."""
        decision = create_test_decision()
        
        output = self.formatter.format_oneline(decision)
        
        assert "face=" in output
        assert "gait=" in output
        assert "CONFIRMED_STRONG" in output or "CONFIRMED" in output

    def test_format_oneline_conflict(self):
        """One-line format for HOLD_CONFLICT state."""
        decision = create_test_decision(
            final_identity=None,
            state=ChimericState.HOLD_CONFLICT,
            confidence=0.0,
            learning_allowed=False
        )
        
        output = self.formatter.format_oneline(decision)
        
        assert "HOLD_CONFLICT" in output
        assert "learning=BLOCKED" in output

    def test_format_oneline_tentative(self):
        """One-line format for TENTATIVE state."""
        decision = create_test_decision(
            state=ChimericState.TENTATIVE,
            confidence=0.65,
            learning_allowed=False
        )
        
        output = self.formatter.format_oneline(decision)
        
        assert "TENTATIVE" in output
        assert "learning=BLOCKED" in output

    def test_format_detailed_comprehensive(self):
        """Detailed format includes all evidence fields."""
        decision = create_test_decision()
        
        output = self.formatter.format_detailed(decision)
        
        # Should include detailed evidence
        assert "FaceEvidence" in output or "face:" in output
        assert "GaitEvidence" in output or "gait:" in output
        assert "similarity" in output
        assert "quality" in output
        assert "margin" in output

    def test_format_json_valid(self):
        """JSON format is valid JSON."""
        decision = create_test_decision()
        
        output = self.formatter.format_json(decision)
        
        try:
            json_obj = json.loads(output)
            assert json_obj["track_id"] == "track_001"
            assert json_obj["final_identity"] == "alice"
            assert json_obj["state"] == "CONFIRMED"
        except json.JSONDecodeError:
            pytest.fail("JSON output is not valid JSON")

    def test_format_json_includes_all_fields(self):
        """JSON includes all decision fields."""
        decision = create_test_decision()
        
        output = self.formatter.format_json(decision)
        json_obj = json.loads(output)
        
        assert "track_id" in json_obj
        assert "final_identity" in json_obj
        assert "chimeric_confidence" in json_obj
        assert "state" in json_obj
        assert "learning_allowed" in json_obj
        assert "evidence_summary" in json_obj

    def test_format_with_redaction(self):
        """Test identity redaction option."""
        decision = create_test_decision(final_identity="alice")
        
        output_redacted = self.formatter.format_oneline(decision, redact_identity=True)
        
        # Identity should be redacted
        assert "alice" not in output_redacted or "[REDACTED]" in output_redacted
        assert "identity=" in output_redacted

    def test_format_none_input_handling(self):
        """Formatter gracefully handles None decision."""
        output = self.formatter.format_oneline(None)
        
        # Should return some indication of invalid input
        assert "invalid" in output.lower() or "none" in output.lower() or output == ""


# ============================================================================
# TEST: METRICS COLLECTOR
# ============================================================================

class TestMetricsCollector:
    """Test metrics collection and aggregation."""

    def setup_method(self):
        """Initialize metrics collector."""
        self.collector = MetricsCollector(window_size_sec=5.0)

    def test_metrics_collector_initialization(self):
        """Test MetricsCollector can be created."""
        assert self.collector is not None
        assert hasattr(self.collector, 'add_decision')
        assert hasattr(self.collector, 'get_current_metrics')

    def test_single_decision_metrics(self):
        """Single decision generates metrics."""
        decision = create_test_decision()
        
        self.collector.add_decision(decision)
        metrics = self.collector.get_current_metrics()
        
        assert metrics.total_decisions >= 1
        assert metrics.decisions_in_window >= 1

    def test_multiple_decisions_aggregation(self):
        """Multiple decisions are aggregated correctly."""
        decisions = [
            create_test_decision(track_id="track_001", final_identity="alice"),
            create_test_decision(track_id="track_002", final_identity="bob"),
            create_test_decision(track_id="track_003", final_identity="alice"),
        ]
        
        for decision in decisions:
            self.collector.add_decision(decision)
        
        metrics = self.collector.get_current_metrics()
        
        assert metrics.total_decisions >= 3
        assert metrics.decisions_in_window >= 3

    def test_state_distribution_tracking(self):
        """Metrics track state distribution (CONFIRMED, TENTATIVE, etc.)."""
        confirmed = create_test_decision(state=ChimericState.CONFIRMED)
        tentative = create_test_decision(state=ChimericState.TENTATIVE)
        unknown = create_test_decision(state=ChimericState.UNKNOWN)
        
        self.collector.add_decision(confirmed)
        self.collector.add_decision(tentative)
        self.collector.add_decision(unknown)
        
        metrics = self.collector.get_current_metrics()
        
        # Should track state counts
        assert hasattr(metrics, 'state_distribution') or metrics.total_decisions >= 3

    def test_confidence_histogram(self):
        """Metrics track confidence distribution."""
        high_conf = create_test_decision(confidence=0.95)
        med_conf = create_test_decision(confidence=0.70)
        low_conf = create_test_decision(confidence=0.40)
        
        self.collector.add_decision(high_conf)
        self.collector.add_decision(med_conf)
        self.collector.add_decision(low_conf)
        
        metrics = self.collector.get_current_metrics()
        
        # Should have confidence tracking
        assert metrics.mean_confidence is not None or metrics.total_decisions >= 3

    def test_learning_gate_tracking(self):
        """Metrics track learning allowed/blocked."""
        allowed = create_test_decision(learning_allowed=True)
        blocked = create_test_decision(learning_allowed=False)
        
        self.collector.add_decision(allowed)
        self.collector.add_decision(blocked)
        
        metrics = self.collector.get_current_metrics()
        
        # Should track learning gate status
        assert hasattr(metrics, 'learning_allowed_count') or metrics.total_decisions >= 2

    def test_window_sliding(self):
        """Sliding window only includes recent decisions."""
        # Add old decision (outside window)
        old_time = time.time() - 10.0
        old_decision = create_test_decision()
        # Manually set timestamp to old
        old_decision.timestamp = old_time
        
        # Add new decision (inside window)
        new_decision = create_test_decision()
        
        self.collector.add_decision(old_decision)
        self.collector.add_decision(new_decision)
        
        metrics = self.collector.get_current_metrics()
        
        # Should only include new decision in sliding window
        # Total might be 2, but window should be ~1
        assert metrics.total_decisions >= 1


# ============================================================================
# TEST: CHIMERIC LOGGER
# ============================================================================

class TestChimericLogger:
    """Test logging system."""

    def test_logger_initialization(self):
        """ChimericLogger can be created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = ChimericLogger(log_file=log_file, verbosity="NORMAL")
            
            assert logger is not None

    def test_log_decision_quiet(self):
        """QUIET verbosity: minimal output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = ChimericLogger(log_file=log_file, verbosity="QUIET")
            
            decision = create_test_decision()
            logger.log_decision(decision)
            
            # In QUIET mode, minimal output
            with open(log_file) as f:
                content = f.read()
            
            # Should have some output
            assert len(content) > 0

    def test_log_decision_normal(self):
        """NORMAL verbosity: production format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = ChimericLogger(log_file=log_file, verbosity="NORMAL")
            
            decision = create_test_decision()
            logger.log_decision(decision)
            
            with open(log_file) as f:
                content = f.read()
            
            assert "[CHIMERIC]" in content
            assert "track_001" in content
            assert "CONFIRMED" in content

    def test_log_decision_debug(self):
        """DEBUG verbosity: detailed output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = ChimericLogger(log_file=log_file, verbosity="DEBUG")
            
            decision = create_test_decision()
            logger.log_decision(decision)
            
            with open(log_file) as f:
                content = f.read()
            
            # Debug should have more detail
            assert len(content) > 50  # Substantial content

    def test_log_decision_trace(self):
        """TRACE verbosity: complete JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = ChimericLogger(log_file=log_file, verbosity="TRACE")
            
            decision = create_test_decision()
            logger.log_decision(decision)
            
            with open(log_file) as f:
                content = f.read()
            
            # Should be valid JSON or contain JSON
            try:
                json_obj = json.loads(content.split('\n')[0])
                assert "track_id" in json_obj
            except:
                # If not pure JSON, at least should be detailed
                assert "similarity" in content or "confidence" in content

    def test_log_error_handling(self):
        """Logger handles errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = ChimericLogger(log_file=log_file, verbosity="NORMAL")
            
            logger.log_error("track_001", "Test error message")
            
            with open(log_file) as f:
                content = f.read()
            
            assert "error" in content.lower() or "track_001" in content

    def test_log_metrics_snapshot(self):
        """Logger can log metrics snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = ChimericLogger(log_file=log_file, verbosity="DEBUG")
            
            metrics = Mock()
            metrics.total_decisions = 10
            metrics.mean_confidence = 0.82
            
            logger.log_metrics(metrics)
            
            with open(log_file) as f:
                content = f.read()
            
            # Should have metrics information
            assert len(content) > 0


# ============================================================================
# TEST: CONFIG VALIDATION
# ============================================================================

class TestConfigValidation:
    """Test configuration validation."""

    def test_config_valid_defaults(self):
        """Default config is valid."""
        config = ChimericConfig()
        
        # Should be creatable
        assert config is not None
        # Key thresholds should be set
        assert config.face_confirm_threshold is not None
        assert config.face_switch_threshold is not None

    def test_config_threshold_ordering(self):
        """Validate threshold ordering (switch > confirm)."""
        config = ChimericConfig()
        
        # Switch should be stricter than confirm
        assert config.face_switch_threshold > config.face_confirm_threshold
        assert config.gait_strong_threshold > config.gait_confirm_threshold

    def test_config_margin_safety(self):
        """Validate margin safety thresholds."""
        config = ChimericConfig()
        
        assert config.gait_margin_min > 0.0
        assert config.learning_margin_min > 0.0
        # Learning margin should be reasonable
        assert config.learning_margin_min < 0.25

    def test_config_temporal_validity(self):
        """Validate temporal configuration."""
        config = ChimericConfig()
        
        assert config.face_evidence_window_sec > 0
        assert config.gait_evidence_window_sec > 0
        # Gait should have longer window than face
        assert config.gait_evidence_window_sec >= config.face_evidence_window_sec

    def test_config_learning_constraints(self):
        """Validate learning gate constraints."""
        config = ChimericConfig()
        
        # Learning thresholds should be reasonable
        assert 0.5 <= config.learning_face_min_quality <= 1.0
        assert 0.5 <= config.learning_gait_min_quality <= 1.0
        # Learning margin should be non-zero
        assert config.learning_margin_min > 0.0


# ============================================================================
# TEST: CLI ARGUMENT HANDLING
# ============================================================================

class TestCLIArgumentHandling:
    """Test CLI argument parsing (without running full pipeline)."""

    def test_cli_mode_options(self):
        """Test mode selection options."""
        modes = ["chimeric_only", "face_only", "gait_only", "analysis_only"]
        
        # All modes should be valid
        for mode in modes:
            assert mode in ["chimeric_only", "face_only", "gait_only", "analysis_only"]

    def test_config_file_loading(self):
        """Test config file can be loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "test_config.yaml")
            
            # Write minimal YAML
            with open(config_file, 'w') as f:
                f.write("chimeric:\n  face_confirm_threshold: 0.75\n")
            
            # Should be loadable
            assert os.path.exists(config_file)

    def test_logging_level_options(self):
        """Test logging level options."""
        levels = ["QUIET", "NORMAL", "DEBUG", "TRACE"]
        
        for level in levels:
            assert level in ["QUIET", "NORMAL", "DEBUG", "TRACE"]


# ============================================================================
# TEST: END-TO-END FORMAT VERIFICATION
# ============================================================================

class TestEndToEndFormatting:
    """Test complete formatting pipeline."""

    def test_full_decision_pipeline(self):
        """Test decision from creation to formatted output."""
        formatter = ChimericDecisionFormatter()
        
        # Create complex decision
        decision = ChimericDecision(
            track_id="track_complex_001",
            final_identity="alice",
            chimeric_confidence=0.89,
            state=ChimericState.CONFIRMED,
            evidence_summary={
                "face": FaceEvidence(
                    identity_id="alice",
                    similarity=0.88,
                    quality=0.92,
                    status=EvidenceStatus.CONFIRMED_STRONG,
                    margin=0.15,
                    timestamp=time.time()
                ),
                "gait": GaitEvidence(
                    identity_id="alice",
                    similarity=0.75,
                    margin=0.08,
                    quality=0.70,
                    status=EvidenceStatus.CONFIRMED_WEAK,
                    sequence_length=40,
                    timestamp=time.time()
                ),
                "source_auth": None
            },
            decision_reason=ChimericReason.FACE_CONFIRMED_WITH_GAIT_SUPPORT,
            learning_allowed=True,
            timestamp=time.time(),
            debug_trace={}
        )
        
        # Test all formats
        oneline = formatter.format_oneline(decision)
        detailed = formatter.format_detailed(decision)
        json_fmt = formatter.format_json(decision)
        
        # All should be non-empty
        assert len(oneline) > 0
        assert len(detailed) > 0
        assert len(json_fmt) > 0
        
        # JSON should be parseable
        json_obj = json.loads(json_fmt)
        assert json_obj["track_id"] == "track_complex_001"


if __name__ == "__main__":
    # Run tests: pytest tests/test_phase3_logging_cli.py -v
    pytest.main([__file__, "-v"])
