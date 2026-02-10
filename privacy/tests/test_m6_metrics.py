
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pytest

from privacy.metrics import (
    PrivacyMetricsEngine,
    PrivacyMetricsConfig,
    MetricsWriter,
    LeakageDetector,
    TrackTimingState,
    TrackFlickerState,
    compute_mask_iou,
    compute_mask_area,
    create_metrics_engine,
)


class TestMaskIoU:
    
    def test_identical_masks_iou_1(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        
        iou = compute_mask_iou(mask, mask.copy())
        assert iou == pytest.approx(1.0, abs=0.001), f"Expected IoU=1.0, got {iou}"
    
    def test_disjoint_masks_iou_0(self):
        mask_a = np.zeros((100, 100), dtype=np.uint8)
        mask_a[0:25, 0:25] = 255
        
        mask_b = np.zeros((100, 100), dtype=np.uint8)
        mask_b[75:100, 75:100] = 255
        
        iou = compute_mask_iou(mask_a, mask_b)
        assert iou == pytest.approx(0.0, abs=0.001), f"Expected IoU=0.0, got {iou}"
    
    def test_partial_overlap_iou(self):
        mask_a = np.zeros((100, 100), dtype=np.uint8)
        mask_a[0:50, 0:50] = 255
        
        mask_b = np.zeros((100, 100), dtype=np.uint8)
        mask_b[25:75, 25:75] = 255
        
        iou = compute_mask_iou(mask_a, mask_b)
        assert 0.0 < iou < 1.0, f"Expected 0 < IoU < 1, got {iou}"
        assert iou == pytest.approx(625 / 4375, abs=0.01), f"IoU calculation incorrect"
    
    def test_empty_masks_iou_1(self):
        mask_a = np.zeros((100, 100), dtype=np.uint8)
        mask_b = np.zeros((100, 100), dtype=np.uint8)
        
        iou = compute_mask_iou(mask_a, mask_b)
        assert iou == pytest.approx(1.0, abs=0.001), "Two empty masks should have IoU=1.0"
    
    def test_one_empty_mask_iou_0(self):
        mask_a = np.zeros((100, 100), dtype=np.uint8)
        mask_a[25:75, 25:75] = 255
        
        mask_b = np.zeros((100, 100), dtype=np.uint8)
        
        iou = compute_mask_iou(mask_a, mask_b)
        assert iou == pytest.approx(0.0, abs=0.001), "Empty+Filled masks should have IoU=0.0"
    
    def test_none_mask_returns_0(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        
        assert compute_mask_iou(None, mask) == 0.0
        assert compute_mask_iou(mask, None) == 0.0
        assert compute_mask_iou(None, None) == 0.0


class TestMaskArea:
    
    def test_area_computation(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        
        area = compute_mask_area(mask)
        assert area == 2500
    
    def test_empty_mask_area_0(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        area = compute_mask_area(mask)
        assert area == 0
    
    def test_none_mask_area_0(self):
        area = compute_mask_area(None)
        assert area == 0


class TestTrackTimingState:
    
    def test_timing_state_initialization(self):
        state = TrackTimingState(track_id=1, first_seen_ts=1000.0)
        assert state.track_id == 1
        assert state.first_seen_ts == 1000.0
        assert state.lock_ts is None
        assert state.first_redacted_emit_ts is None
    
    def test_time_to_lock_calculation(self):
        state = TrackTimingState(track_id=1, first_seen_ts=1000.0)
        state.lock_ts = 1002.5
        
        ttl = state.get_time_to_lock()
        assert ttl == pytest.approx(2.5, abs=0.001)
    
    def test_time_to_redact_calculation(self):
        state = TrackTimingState(track_id=1, first_seen_ts=1000.0)
        state.lock_ts = 1002.0
        state.first_redacted_emit_ts = 1005.0
        
        ttr = state.get_time_to_redacted_emit()
        assert ttr == pytest.approx(5.0, abs=0.001)
    
    def test_incomplete_timing_returns_none(self):
        state = TrackTimingState(track_id=1, first_seen_ts=1000.0)
        assert state.get_time_to_lock() is None
        
        state.lock_ts = 1002.0
        assert state.get_time_to_lock() == pytest.approx(2.0, abs=0.001)
        assert state.get_time_to_redacted_emit() is None


class TestTrackFlickerState:
    
    def test_initial_flicker_state(self):
        state = TrackFlickerState(track_id=1)
        assert state.last_mask is None
        assert state.iou_values == []
        assert state.area_values == []
    
    def test_add_mask_records_history(self):
        state = TrackFlickerState(track_id=1)
        
        mask1 = np.zeros((100, 100), dtype=np.uint8)
        mask1[25:75, 25:75] = 255
        
        iou1 = state.add_mask(mask1)
        assert iou1 is None
        assert len(state.area_values) == 1
        
        mask2 = np.zeros((100, 100), dtype=np.uint8)
        mask2[30:70, 30:70] = 255
        
        iou2 = state.add_mask(mask2)
        assert iou2 is not None
        assert 0.0 < iou2 < 1.0
        
        assert len(state.iou_values) == 1
        assert len(state.area_values) == 2
    
    def test_mean_iou_calculation(self):
        state = TrackFlickerState(track_id=1)
        
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        
        state.add_mask(mask)
        state.add_mask(mask.copy())
        state.add_mask(mask.copy())
        
        mean_iou = state.get_mean_iou()
        assert mean_iou is not None
        assert mean_iou == pytest.approx(1.0, abs=0.01)


class TestLeakageDetector:
    
    def test_detector_initialization_with_config(self):
        cfg = PrivacyMetricsConfig(
            leakage_enabled=True,
            leakage_backend="opencv_haar",
        )
        detector = LeakageDetector(cfg)
        assert detector is not None
        assert detector._enabled
    
    def test_detector_disabled(self):
        cfg = PrivacyMetricsConfig(leakage_enabled=False)
        detector = LeakageDetector(cfg)
        assert not detector._enabled
    
    def test_detect_faces_on_blank_image(self):
        cfg = PrivacyMetricsConfig(leakage_enabled=True, leakage_backend="opencv_haar")
        detector = LeakageDetector(cfg)
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        
        faces = detector.detect_faces(blank)
        assert faces == []
    
    def test_stub_backend_returns_empty(self):
        cfg = PrivacyMetricsConfig(leakage_enabled=True, leakage_backend="stub")
        detector = LeakageDetector(cfg)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        faces = detector.detect_faces(img)
        assert faces == []


class TestPrivacyMetricsEngine:
    
    def test_engine_initialization(self):
        tmpdir = tempfile.mkdtemp()
        cfg = PrivacyMetricsConfig(
            enabled=True,
            log_path=str(Path(tmpdir) / "metrics.jsonl"),
            leakage_enabled=False,
            flicker_enabled=True,
            timing_enabled=True,
            utility_enabled=True,
        )
        
        engine = PrivacyMetricsEngine(cfg, delay_sec=3.0)
        assert engine is not None
        engine.finalize_and_summarize()
    
    def test_finalize_returns_summary(self):
        tmpdir = tempfile.mkdtemp()
        cfg = PrivacyMetricsConfig(
            enabled=True,
            log_path=str(Path(tmpdir) / "metrics.jsonl"),
            leakage_enabled=False,
            flicker_enabled=True,
            timing_enabled=True,
            utility_enabled=True,
        )
        
        engine = PrivacyMetricsEngine(cfg, delay_sec=3.0)
        summary = engine.finalize_and_summarize()
        
        assert summary is not None
        assert isinstance(summary, dict)
        assert "metrics_enabled" in summary
        assert "disabled_due_to_error" in summary
        assert "flicker" in summary
    
    def test_fail_open_on_bad_path(self):
        cfg = PrivacyMetricsConfig(
            enabled=True,
            log_path="/nonexistent/path/that/will/fail.jsonl",
            leakage_enabled=False,
            flicker_enabled=True,
            timing_enabled=True,
            utility_enabled=True,
        )
        
        engine = PrivacyMetricsEngine(cfg, delay_sec=3.0)
        summary = engine.finalize_and_summarize()
        assert summary is not None


class TestCreateMetricsEngine:
    
    def test_create_returns_engine(self):
        class Config:
            enabled = True
            log_path = "test_metrics.jsonl"
            flush_interval_sec = 1.0
            leakage_enabled = False
            leakage_backend = "none"
            haar_scale_factor = 1.1
            haar_min_neighbors = 5
            overlap_threshold = 0.3
            flicker_enabled = True
            timing_enabled = True
            utility_enabled = True
        
        engine = create_metrics_engine(Config(), delay_sec=3.0)
        assert engine is not None
        assert isinstance(engine, PrivacyMetricsEngine)
        engine.finalize_and_summarize()
    
    def test_create_with_none_config_returns_engine(self):
        engine = create_metrics_engine(None, delay_sec=3.0)
        assert engine is not None
        engine.finalize_and_summarize()


class TestMetricsWriter:
    
    def test_writes_valid_jsonl(self):
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir) / "test.jsonl"
        
        writer = MetricsWriter(str(path), flush_interval_sec=0.0)
        writer.write_entry({"type": "timing", "track_id": 1, "time_to_lock": 2.5})
        writer.write_entry({"type": "flicker", "track_id": 1, "iou": 0.85})
        writer.close()
        
        with open(path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        for line in lines:
            entry = json.loads(line)
            assert "type" in entry
            assert "track_id" in entry
    
    def test_handles_write_errors_gracefully(self):
        writer = MetricsWriter("/nonexistent/path/file.jsonl", flush_interval_sec=0.0)
        
        try:
            writer.write_entry({"test": "data"})
        except Exception:
            pass
        writer.close()
    
    def test_is_active_property(self):
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir) / "test.jsonl"
        
        writer = MetricsWriter(str(path), flush_interval_sec=0.0)
        assert writer.is_active
        
        writer.close()
        assert not writer.is_active


class TestM6Integration:
    
    def test_full_lifecycle_basic(self):
        tmpdir = tempfile.mkdtemp()
        
        cfg = PrivacyMetricsConfig(
            enabled=True,
            log_path=str(Path(tmpdir) / "metrics.jsonl"),
            flush_interval_sec=0.0,
            leakage_enabled=False,
            flicker_enabled=True,
            timing_enabled=True,
            utility_enabled=True,
        )
        
        engine = PrivacyMetricsEngine(cfg, delay_sec=3.0)
        
        base_ts = time.time()
        for i in range(5):
            ts = base_ts + i * 0.033
            
            engine.on_ingest(
                frame_ts=ts,
                frame_id=i,
                track_ids_present={1, 2},
                policy_info_by_track={
                    1: {"policy_state": "AUTHORIZED_LOCKED_REDACT" if i >= 2 else "UNKNOWN_VISIBLE"},
                    2: {"policy_state": "UNKNOWN_VISIBLE"},
                },
            )
            
            mask1 = np.zeros((480, 640), dtype=np.uint8)
            mask1[100:300, 200:400] = 255
            engine.on_masks(
                frame_ts=ts,
                frame_id=i,
                track_id=1,
                raw_mask=mask1,
                stable_mask=mask1,
                bbox=[200, 100, 200, 200],
            )
        
        summary = engine.finalize_and_summarize()
        
        assert summary is not None
        assert "metrics_enabled" in summary
        assert "flicker" in summary
        assert "timing" in summary
        
        log_path = Path(cfg.log_path)
        assert log_path.exists(), "Metrics log file should exist"


class TestLeakageOverlap:
    
    def test_face_fully_inside_redacted_bbox_is_leakage(self):
        cfg = PrivacyMetricsConfig(
            leakage_enabled=True,
            leakage_backend="stub",
            overlap_threshold=0.2,
        )
        detector = LeakageDetector(cfg)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        redacted_regions = [{"track_id": 1, "bbox": (100, 100, 300, 300)}]
        
        detector._stub_faces = [(120, 120, 200, 200)]
        detector._backend = "stub"
        
        original_detect = detector.detect_faces
        detector.detect_faces = lambda frame: detector._stub_faces
        
        result = detector.check_leakage(frame, redacted_regions)
        
        assert result["faces_detected"] == 1
        assert result["faces_in_redacted"] == 1
        assert result["leakage_flag"] == True
    
    def test_face_outside_redacted_bbox_is_not_leakage(self):
        cfg = PrivacyMetricsConfig(
            leakage_enabled=True,
            leakage_backend="stub",
            overlap_threshold=0.2,
        )
        detector = LeakageDetector(cfg)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        redacted_regions = [{"track_id": 1, "bbox": (0, 0, 100, 100)}]
        
        detector._stub_faces = [(400, 400, 500, 500)]
        detector.detect_faces = lambda frame: detector._stub_faces
        
        result = detector.check_leakage(frame, redacted_regions)
        
        assert result["faces_detected"] == 1
        assert result["faces_in_redacted"] == 0
        assert result["leakage_flag"] == False
    
    def test_no_redacted_regions_no_crash(self):
        cfg = PrivacyMetricsConfig(
            leakage_enabled=True,
            leakage_backend="stub",
            overlap_threshold=0.2,
        )
        detector = LeakageDetector(cfg)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        redacted_regions = []
        
        detector._stub_faces = [(100, 100, 200, 200)]
        detector.detect_faces = lambda frame: detector._stub_faces
        
        result = detector.check_leakage(frame, redacted_regions)
        
        assert result["faces_detected"] == 1
        assert result["faces_in_redacted"] == 0
        assert result["leakage_flag"] == False


class TestAuditBboxIntegration:
    
    def test_build_track_audit_entry_includes_bbox(self):
        from privacy.audit import build_track_audit_entry
        
        entry = build_track_audit_entry(
            track_id=1,
            policy_state="AUTHORIZED_LOCKED_REDACT",
            redaction_method="mask_blur",
            identity_id="person_123",
            id_source="F",
            authorized_signal=True,
            decision_category="resident",
            decision_binding_state="CONFIRMED_STRONG",
            is_redacted=True,
            bbox=[100.0, 150.0, 300.0, 400.0],
        )
        
        assert "bbox" in entry
        assert entry["bbox"] == [100.0, 150.0, 300.0, 400.0]
    
    def test_build_track_audit_entry_no_bbox_when_none(self):
        from privacy.audit import build_track_audit_entry
        
        entry = build_track_audit_entry(
            track_id=1,
            policy_state="UNKNOWN_VISIBLE",
            redaction_method="none",
            identity_id=None,
            id_source="U",
            authorized_signal=False,
            decision_category="unknown",
            decision_binding_state=None,
            is_redacted=False,
            bbox=None,
        )
        
        assert "bbox" not in entry
    
    def test_bbox_converts_tuple_to_list(self):
        from privacy.audit import build_track_audit_entry
        
        entry = build_track_audit_entry(
            track_id=1,
            policy_state="AUTHORIZED_LOCKED_REDACT",
            redaction_method="bbox_blur",
            identity_id="person_456",
            id_source="G",
            authorized_signal=True,
            decision_category="resident",
            decision_binding_state="CONFIRMED_WEAK",
            is_redacted=True,
            bbox=(50, 60, 200, 300),
        )
        
        assert "bbox" in entry
        assert isinstance(entry["bbox"], list)
        assert entry["bbox"] == [50, 60, 200, 300]


class TestTimingDeterministic:
    
    def test_timing_lifecycle_deterministic(self):
        tmpdir = tempfile.mkdtemp()
        cfg = PrivacyMetricsConfig(
            enabled=True,
            log_path=str(Path(tmpdir) / "metrics.jsonl"),
            leakage_enabled=False,
            flicker_enabled=False,
            timing_enabled=True,
            utility_enabled=False,
        )
        
        engine = PrivacyMetricsEngine(cfg, delay_sec=3.0)
        
        base_ts = 1000.0
        
        engine.on_ingest(
            frame_ts=base_ts + 0.0,
            frame_id=0,
            track_ids_present={1},
            policy_info_by_track={1: {"policy_state": "UNKNOWN_VISIBLE"}},
        )
        
        engine.on_ingest(
            frame_ts=base_ts + 0.2,
            frame_id=1,
            track_ids_present={1},
            policy_info_by_track={1: {"policy_state": "AUTHORIZED_LOCKED_REDACT"}},
        )
        
        engine.on_emit(
            emit_ts=base_ts + 3.2,
            frame_id=0,
            privacy_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            per_track_redaction_info={1: {"is_redacted": True, "track_id": 1}},
            redacted_regions=[],
        )
        
        summary = engine.finalize_and_summarize()
        
        assert "timing" in summary
        timing = summary["timing"]
        
        assert "time_to_lock" in timing
        assert timing["time_to_lock"]["mean_sec"] == pytest.approx(0.2, abs=0.01)
        
        assert "time_to_redacted_emit" in timing
        assert timing["time_to_redacted_emit"]["mean_sec"] == pytest.approx(3.2, abs=0.01)


class TestFlickerBoundedMemory:
    
    def test_alternating_masks_yield_low_iou(self):
        state = TrackFlickerState(track_id=1)
        
        mask_a = np.zeros((100, 100), dtype=np.uint8)
        mask_a[0:50, 0:50] = 255
        
        mask_b = np.zeros((100, 100), dtype=np.uint8)
        mask_b[50:100, 50:100] = 255
        
        state.add_mask(mask_a)
        iou = state.add_mask(mask_b)
        
        assert iou is not None
        assert iou == pytest.approx(0.0, abs=0.01), "Disjoint masks should have IoU=0"
        
        area_std = state.get_area_std()
        assert area_std is not None
    
    def test_long_run_does_not_grow_unbounded(self):
        state = TrackFlickerState(track_id=1)
        
        assert state.max_history == 100
        
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255
        
        for i in range(10000):
            varied_mask = mask.copy()
            if i % 2 == 0:
                varied_mask[20:30, 20:30] = 255
            state.add_mask(varied_mask)
        
        assert len(state.iou_values) <= state.max_history
        assert len(state.area_values) <= state.max_history
        
        mean_iou = state.get_mean_iou()
        assert mean_iou is not None
        assert 0.0 <= mean_iou <= 1.0


class TestStableMaskPreference:
    
    def test_stable_mask_used_when_both_provided(self):
        tmpdir = tempfile.mkdtemp()
        cfg = PrivacyMetricsConfig(
            enabled=True,
            log_path=str(Path(tmpdir) / "metrics.jsonl"),
            leakage_enabled=False,
            flicker_enabled=True,
            timing_enabled=False,
        )
        
        engine = PrivacyMetricsEngine(cfg, delay_sec=3.0)
        
        raw_mask = np.zeros((100, 100), dtype=np.uint8)
        raw_mask[0:50, 0:50] = 255
        
        stable_mask = np.zeros((100, 100), dtype=np.uint8)
        stable_mask[50:100, 50:100] = 255
        
        engine.on_masks(
            frame_ts=0.0,
            frame_id=0,
            track_id=1,
            raw_mask=raw_mask,
            stable_mask=stable_mask,
            bbox=None,
        )
        
        engine.on_masks(
            frame_ts=0.1,
            frame_id=1,
            track_id=1,
            raw_mask=raw_mask,
            stable_mask=stable_mask,
            bbox=None,
        )
        
        state = engine._track_flicker.get(1)
        assert state is not None
        mean_iou = state.get_mean_iou()
        
        assert mean_iou is not None
    
    def test_stable_mask_preference_distinct_sequence(self):
        tmpdir = tempfile.mkdtemp()
        cfg = PrivacyMetricsConfig(
            enabled=True,
            log_path=str(Path(tmpdir) / "metrics.jsonl"),
            leakage_enabled=False,
            flicker_enabled=True,
            timing_enabled=False,
        )
        
        engine = PrivacyMetricsEngine(cfg, delay_sec=3.0)
        
        stable = np.zeros((100, 100), dtype=np.uint8)
        stable[25:75, 25:75] = 255
        
        raw1 = np.zeros((100, 100), dtype=np.uint8)
        raw1[0:50, 0:50] = 255
        
        raw2 = np.zeros((100, 100), dtype=np.uint8)
        raw2[50:100, 50:100] = 255
        
        engine.on_masks(
            frame_ts=0.0, frame_id=0, track_id=1,
            raw_mask=raw1, stable_mask=stable, bbox=None,
        )
        
        engine.on_masks(
            frame_ts=0.1, frame_id=1, track_id=1,
            raw_mask=raw2, stable_mask=stable, bbox=None,
        )
        
        state = engine._track_flicker[1]
        mean_iou = state.get_mean_iou()
        
        assert mean_iou == pytest.approx(1.0, abs=0.01), \
            f"Expected IoU=1.0 (stable_mask preference), got {mean_iou}"
    
    def test_raw_mask_fallback_when_stable_none(self):
        tmpdir = tempfile.mkdtemp()
        cfg = PrivacyMetricsConfig(
            enabled=True,
            log_path=str(Path(tmpdir) / "metrics.jsonl"),
            leakage_enabled=False,
            flicker_enabled=True,
            timing_enabled=False,
        )
        
        engine = PrivacyMetricsEngine(cfg, delay_sec=3.0)
        
        raw = np.zeros((100, 100), dtype=np.uint8)
        raw[25:75, 25:75] = 255
        
        engine.on_masks(
            frame_ts=0.0, frame_id=0, track_id=1,
            raw_mask=raw, stable_mask=None, bbox=None,
        )
        engine.on_masks(
            frame_ts=0.1, frame_id=1, track_id=1,
            raw_mask=raw, stable_mask=None, bbox=None,
        )
        
        state = engine._track_flicker[1]
        mean_iou = state.get_mean_iou()
        assert mean_iou == pytest.approx(1.0, abs=0.01), \
            "Identical raw masks should yield IoU=1.0"


class TestUtilityMixedFrames:
    
    def test_mixed_frames_counted_correctly(self):
        tmpdir = tempfile.mkdtemp()
        cfg = PrivacyMetricsConfig(
            enabled=True,
            log_path=str(Path(tmpdir) / "metrics.jsonl"),
            leakage_enabled=False,
            flicker_enabled=False,
            timing_enabled=False,
            utility_enabled=True,
        )
        
        engine = PrivacyMetricsEngine(cfg, delay_sec=3.0)
        
        engine.on_emit(
            emit_ts=1000.0,
            frame_id=0,
            privacy_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            per_track_redaction_info={
                1: {"is_redacted": True, "track_id": 1},
                2: {"is_redacted": False, "track_id": 2},
            },
            redacted_regions=[],
        )
        
        engine.on_emit(
            emit_ts=1001.0,
            frame_id=1,
            privacy_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            per_track_redaction_info={
                1: {"is_redacted": True, "track_id": 1},
            },
            redacted_regions=[],
        )
        
        engine.on_emit(
            emit_ts=1002.0,
            frame_id=2,
            privacy_frame=np.zeros((480, 640, 3), dtype=np.uint8),
            per_track_redaction_info={
                3: {"is_redacted": False, "track_id": 3},
            },
            redacted_regions=[],
        )
        
        summary = engine.finalize_and_summarize()
        
        assert "utility" in summary
        utility = summary["utility"]
        
        assert utility["redacted_frames"] == 2
        assert utility["visible_frames"] == 2
        
        assert utility["frames_analyzed"] == 3
        assert utility["frames_with_any_redacted"] == 2
        assert utility["frames_with_any_visible"] == 2
        assert utility["frames_with_both"] == 1


class TestFailOpenBehavior:
    
    def test_invalid_writer_path_disables_writer_not_engine(self):
        cfg = PrivacyMetricsConfig(
            enabled=True,
            log_path="/nonexistent/deeply/nested/path/metrics.jsonl",
            leakage_enabled=False,
            flicker_enabled=True,
            timing_enabled=True,
            utility_enabled=True,
        )
        
        engine = PrivacyMetricsEngine(cfg, delay_sec=3.0)
        
        
        engine.on_ingest(
            frame_ts=1000.0,
            frame_id=0,
            track_ids_present={1},
            policy_info_by_track={1: {"policy_state": "UNKNOWN_VISIBLE"}},
        )
        
        summary = engine.finalize_and_summarize()
        assert summary is not None
    
    def test_leakage_detector_failure_disables_leakage_only(self):
        cfg = PrivacyMetricsConfig(
            enabled=True,
            log_path=str(Path(tempfile.mkdtemp()) / "metrics.jsonl"),
            leakage_enabled=True,
            leakage_backend="opencv_haar",
            flicker_enabled=True,
            timing_enabled=True,
            utility_enabled=True,
        )
        
        engine = PrivacyMetricsEngine(cfg, delay_sec=3.0)
        
        engine._leakage_detector._disabled_due_to_error = True
        
        engine.on_ingest(
            frame_ts=1000.0,
            frame_id=0,
            track_ids_present={1},
            policy_info_by_track={1: {"policy_state": "AUTHORIZED_LOCKED_REDACT"}},
        )
        
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:200, 100:200] = 255
        engine.on_masks(
            frame_ts=1000.0,
            frame_id=0,
            track_id=1,
            raw_mask=mask,
            stable_mask=mask,
            bbox=[100, 100, 100, 100],
        )
        
        summary = engine.finalize_and_summarize()
        
        assert "flicker" in summary
        assert "timing" in summary
    
    def test_on_masks_exception_disables_metrics(self):
        cfg = PrivacyMetricsConfig(
            enabled=True,
            log_path=str(Path(tempfile.mkdtemp()) / "metrics.jsonl"),
            flicker_enabled=True,
        )
        
        engine = PrivacyMetricsEngine(cfg, delay_sec=3.0)
        
        engine._track_flicker = None
        
        engine.on_masks(
            frame_ts=1000.0,
            frame_id=0,
            track_id=1,
            raw_mask=np.zeros((100, 100), dtype=np.uint8),
            stable_mask=None,
            bbox=None,
        )
        
        assert engine._disabled_due_to_error == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

