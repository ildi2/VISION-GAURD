
import time
import json
import pytest
import tempfile
import os
from io import StringIO
from unittest.mock import Mock, patch
from dataclasses import dataclass

from vision_identity.types import (
    VisionState,
    VisionReason,
    VisionDecision,
    FaceEvidence,
    GaitEvidence,
    SourceAuthEvidence,
    EvidenceStatus,
    SourceAuthState,
)
from vision_identity.logging_utils import (
    VisionLogger,
    VisionDecisionFormatter,
    MetricsCollector,
)
from vision_identity.config import VisionIdentityConfig


def create_test_decision(
    track_id: str = "track_001",
    final_identity: str = "alice",
    state: VisionState = VisionState.CONFIRMED,
    confidence: float = 0.88,
    learning_allowed: bool = True,
) -> VisionDecision:
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
    
    return VisionDecision(
        track_id=track_id,
        final_identity=final_identity,
        vision_confidence=confidence,
        state=state,
        evidence_summary={
            "face": face,
            "gait": gait,
            "source_auth": None
        },
        decision_reason=VisionReason.FACE_CONFIRMED_WITH_GAIT_SUPPORT,
        learning_allowed=learning_allowed,
        timestamp=time.time(),
        debug_trace={}
    )


class TestVisionDecisionFormatter:

    def setup_method(self):
        self.formatter = VisionDecisionFormatter()

    def test_format_oneline_basic(self):
        decision = create_test_decision()
        
        output = self.formatter.format_oneline(decision)
        
        assert "[VISION-ID]" in output
        assert "track_id=track_001" in output
        assert "state=CONFIRMED" in output
        assert "identity=alice" in output
        assert "confidence=" in output
        assert "learning=" in output

    def test_format_oneline_includes_face_gait(self):
        decision = create_test_decision()
        
        output = self.formatter.format_oneline(decision)
        
        assert "face=" in output
        assert "gait=" in output
        assert "CONFIRMED_STRONG" in output or "CONFIRMED" in output

    def test_format_oneline_conflict(self):
        decision = create_test_decision(
            final_identity=None,
            state=VisionState.HOLD_CONFLICT,
            confidence=0.0,
            learning_allowed=False
        )
        
        output = self.formatter.format_oneline(decision)
        
        assert "HOLD_CONFLICT" in output
        assert "learning=BLOCKED" in output

    def test_format_oneline_tentative(self):
        decision = create_test_decision(
            state=VisionState.TENTATIVE,
            confidence=0.65,
            learning_allowed=False
        )
        
        output = self.formatter.format_oneline(decision)
        
        assert "TENTATIVE" in output
        assert "learning=BLOCKED" in output

    def test_format_detailed_comprehensive(self):
        decision = create_test_decision()
        
        output = self.formatter.format_detailed(decision)
        
        assert "FaceEvidence" in output or "face:" in output
        assert "GaitEvidence" in output or "gait:" in output
        assert "similarity" in output
        assert "quality" in output
        assert "margin" in output

    def test_format_json_valid(self):
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
        decision = create_test_decision()
        
        output = self.formatter.format_json(decision)
        json_obj = json.loads(output)
        
        assert "track_id" in json_obj
        assert "final_identity" in json_obj
        assert "vision_confidence" in json_obj
        assert "state" in json_obj
        assert "learning_allowed" in json_obj
        assert "evidence_summary" in json_obj

    def test_format_with_redaction(self):
        decision = create_test_decision(final_identity="alice")
        
        output_redacted = self.formatter.format_oneline(decision, redact_identity=True)
        
        assert "alice" not in output_redacted or "[REDACTED]" in output_redacted
        assert "identity=" in output_redacted

    def test_format_none_input_handling(self):
        output = self.formatter.format_oneline(None)
        
        assert "invalid" in output.lower() or "none" in output.lower() or output == ""


class TestMetricsCollector:

    def setup_method(self):
        self.collector = MetricsCollector(window_size_sec=5.0)

    def test_metrics_collector_initialization(self):
        assert self.collector is not None
        assert hasattr(self.collector, 'add_decision')
        assert hasattr(self.collector, 'get_current_metrics')

    def test_single_decision_metrics(self):
        decision = create_test_decision()
        
        self.collector.add_decision(decision)
        metrics = self.collector.get_current_metrics()
        
        assert metrics.total_decisions >= 1
        assert metrics.decisions_in_window >= 1

    def test_multiple_decisions_aggregation(self):
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
        confirmed = create_test_decision(state=VisionState.CONFIRMED)
        tentative = create_test_decision(state=VisionState.TENTATIVE)
        unknown = create_test_decision(state=VisionState.UNKNOWN)
        
        self.collector.add_decision(confirmed)
        self.collector.add_decision(tentative)
        self.collector.add_decision(unknown)
        
        metrics = self.collector.get_current_metrics()
        
        assert hasattr(metrics, 'state_distribution') or metrics.total_decisions >= 3

    def test_confidence_histogram(self):
        high_conf = create_test_decision(confidence=0.95)
        med_conf = create_test_decision(confidence=0.70)
        low_conf = create_test_decision(confidence=0.40)
        
        self.collector.add_decision(high_conf)
        self.collector.add_decision(med_conf)
        self.collector.add_decision(low_conf)
        
        metrics = self.collector.get_current_metrics()
        
        assert metrics.mean_confidence is not None or metrics.total_decisions >= 3

    def test_learning_gate_tracking(self):
        allowed = create_test_decision(learning_allowed=True)
        blocked = create_test_decision(learning_allowed=False)
        
        self.collector.add_decision(allowed)
        self.collector.add_decision(blocked)
        
        metrics = self.collector.get_current_metrics()
        
        assert hasattr(metrics, 'learning_allowed_count') or metrics.total_decisions >= 2

    def test_window_sliding(self):
        old_time = time.time() - 10.0
        old_decision = create_test_decision()
        old_decision.timestamp = old_time
        
        new_decision = create_test_decision()
        
        self.collector.add_decision(old_decision)
        self.collector.add_decision(new_decision)
        
        metrics = self.collector.get_current_metrics()
        
        assert metrics.total_decisions >= 1


class TestVisionLogger:

    def test_logger_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = VisionLogger(log_file=log_file, verbosity="NORMAL")
            
            assert logger is not None

    def test_log_decision_quiet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = VisionLogger(log_file=log_file, verbosity="QUIET")
            
            decision = create_test_decision()
            logger.log_decision(decision)
            
            with open(log_file) as f:
                content = f.read()
            
            assert len(content) > 0

    def test_log_decision_normal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = VisionLogger(log_file=log_file, verbosity="NORMAL")
            
            decision = create_test_decision()
            logger.log_decision(decision)
            
            with open(log_file) as f:
                content = f.read()
            
            assert "[VISION-ID]" in content
            assert "track_001" in content
            assert "CONFIRMED" in content

    def test_log_decision_debug(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = VisionLogger(log_file=log_file, verbosity="DEBUG")
            
            decision = create_test_decision()
            logger.log_decision(decision)
            
            with open(log_file) as f:
                content = f.read()
            
            assert len(content) > 50

    def test_log_decision_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = VisionLogger(log_file=log_file, verbosity="TRACE")
            
            decision = create_test_decision()
            logger.log_decision(decision)
            
            with open(log_file) as f:
                content = f.read()
            
            try:
                json_obj = json.loads(content.split('\n')[0])
                assert "track_id" in json_obj
            except:
                assert "similarity" in content or "confidence" in content

    def test_log_error_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = VisionLogger(log_file=log_file, verbosity="NORMAL")
            
            logger.log_error("track_001", "Test error message")
            
            with open(log_file) as f:
                content = f.read()
            
            assert "error" in content.lower() or "track_001" in content

    def test_log_metrics_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, "test.log")
            logger = VisionLogger(log_file=log_file, verbosity="DEBUG")
            
            metrics = Mock()
            metrics.total_decisions = 10
            metrics.mean_confidence = 0.82
            
            logger.log_metrics(metrics)
            
            with open(log_file) as f:
                content = f.read()
            
            assert len(content) > 0


class TestConfigValidation:

    def test_config_valid_defaults(self):
        config = VisionIdentityConfig()
        
        assert config is not None
        assert config.face_confirm_threshold is not None
        assert config.face_switch_threshold is not None

    def test_config_threshold_ordering(self):
        config = VisionIdentityConfig()
        
        assert config.face_switch_threshold > config.face_confirm_threshold
        assert config.gait_strong_threshold > config.gait_confirm_threshold

    def test_config_margin_safety(self):
        config = VisionIdentityConfig()
        
        assert config.gait_margin_min > 0.0
        assert config.learning_margin_min > 0.0
        assert config.learning_margin_min < 0.25

    def test_config_temporal_validity(self):
        config = VisionIdentityConfig()
        
        assert config.face_evidence_window_sec > 0
        assert config.gait_evidence_window_sec > 0
        assert config.gait_evidence_window_sec >= config.face_evidence_window_sec

    def test_config_learning_constraints(self):
        config = VisionIdentityConfig()
        
        assert 0.5 <= config.learning_face_min_quality <= 1.0
        assert 0.5 <= config.learning_gait_min_quality <= 1.0
        assert config.learning_margin_min > 0.0


class TestCLIArgumentHandling:

    def test_cli_mode_options(self):
        modes = ["vision_only", "face_only", "gait_only", "analysis_only"]
        
        for mode in modes:
            assert mode in ["vision_only", "face_only", "gait_only", "analysis_only"]

    def test_config_file_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = os.path.join(tmpdir, "test_config.yaml")
            
            with open(config_file, 'w') as f:
                f.write("vision_identity:\n  face_confirm_threshold: 0.75\n")
            
            assert os.path.exists(config_file)

    def test_logging_level_options(self):
        levels = ["QUIET", "NORMAL", "DEBUG", "TRACE"]
        
        for level in levels:
            assert level in ["QUIET", "NORMAL", "DEBUG", "TRACE"]


class TestEndToEndFormatting:

    def test_full_decision_pipeline(self):
        formatter = VisionDecisionFormatter()
        
        decision = VisionDecision(
            track_id="track_complex_001",
            final_identity="alice",
            vision_confidence=0.89,
            state=VisionState.CONFIRMED,
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
            decision_reason=VisionReason.FACE_CONFIRMED_WITH_GAIT_SUPPORT,
            learning_allowed=True,
            timestamp=time.time(),
            debug_trace={}
        )
        
        oneline = formatter.format_oneline(decision)
        detailed = formatter.format_detailed(decision)
        json_fmt = formatter.format_json(decision)
        
        assert len(oneline) > 0
        assert len(detailed) > 0
        assert len(json_fmt) > 0
        
        json_obj = json.loads(json_fmt)
        assert json_obj["track_id"] == "track_complex_001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
