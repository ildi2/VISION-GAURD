
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from privacy.segmenter import (
    BaseSegmenter,
    NoneSegmenter,
    MaskResult,
    MaskSource,
    SegmenterConfig,
    create_segmenter,
)
from privacy.audit import build_track_audit_entry


@dataclass
class MockTracklet:
    track_id: int
    last_box: tuple
    
    @property
    def bbox(self):
        return self.last_box


@dataclass  
class MockFrame:
    image: np.ndarray
    ts: float
    frame_id: int = 0


@dataclass
class MockConfig:
    enabled: bool = False
    backend: str = "none"
    model_path: str = ""
    imgsz: int = 640
    conf: float = 0.35
    iou_threshold: float = 0.5
    iou_assoc_min: float = 0.25
    run_every_n_frames: int = 1
    min_mask_area_ratio: float = 0.01
    max_mask_area_ratio: float = 5.0
    min_mask_quality: float = 0.3
    dilate_mask_px: int = 3
    device: str = "cpu"


class TestMaskResult:
    
    def test_mask_result_no_mask(self):
        result = MaskResult(track_id=1)
        assert not result.is_valid
        assert result.mask is None
        assert result.quality_score == 0.0
        assert result.source == MaskSource.NONE
    
    def test_mask_result_with_mask(self):
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        result = MaskResult(
            track_id=1,
            mask=mask,
            quality_score=0.85,
            source=MaskSource.YOLO_SEG,
            mask_area_ratio=0.5,
        )
        assert result.is_valid
        assert result.mask is not None
        assert result.quality_score == 0.85
        assert result.source == MaskSource.YOLO_SEG
    
    def test_mask_result_to_audit_dict(self):
        result = MaskResult(
            track_id=1,
            mask=np.ones((10, 10), dtype=np.uint8),
            quality_score=0.9,
            source=MaskSource.YOLO_SEG,
            bbox_iou=0.75,
            mask_area_ratio=0.6,
        )
        
        audit = result.to_audit_dict()
        
        assert audit["mask_used"] is True
        assert audit["mask_source"] == "yolo_seg"
        assert audit["mask_quality"] == 0.9
        assert audit["mask_area_ratio"] == 0.6
        assert audit["bbox_iou"] == 0.75


class TestNoneSegmenter:
    
    def test_none_segmenter_returns_empty(self):
        cfg = SegmenterConfig()
        segmenter = NoneSegmenter(cfg)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = [MockTracklet(track_id=1, last_box=(100, 100, 200, 200))]
        
        results = segmenter.segment(frame, tracks)
        
        assert results == {}
    
    def test_none_segmenter_loads_successfully(self):
        cfg = SegmenterConfig()
        segmenter = NoneSegmenter(cfg)
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        segmenter.segment(frame, [])
        
        assert segmenter._initialized is True


class TestSegmenterFactory:
    
    def test_factory_disabled_returns_none_segmenter(self):
        cfg = MockConfig(enabled=False, backend="none")
        segmenter = create_segmenter(cfg)
        
        assert isinstance(segmenter, NoneSegmenter)
    
    def test_factory_backend_none_returns_none_segmenter(self):
        cfg = MockConfig(enabled=True, backend="none")
        segmenter = create_segmenter(cfg)
        
        assert isinstance(segmenter, NoneSegmenter)
    
    def test_factory_unknown_backend_returns_none_segmenter(self):
        cfg = MockConfig(enabled=True, backend="unknown_backend")
        segmenter = create_segmenter(cfg)
        
        assert isinstance(segmenter, NoneSegmenter)
    
    def test_factory_sam2_not_implemented(self):
        cfg = MockConfig(enabled=True, backend="sam2")
        segmenter = create_segmenter(cfg)
        
        assert isinstance(segmenter, NoneSegmenter)


class TestQualityGates:
    
    def test_quality_gate_low_score(self):
        cfg = SegmenterConfig(min_mask_quality=0.5)
        segmenter = NoneSegmenter(cfg)
        
        result = MaskResult(
            track_id=1,
            mask=np.ones((10, 10), dtype=np.uint8),
            quality_score=0.3,
            source=MaskSource.YOLO_SEG,
        )
        
        passes = segmenter._passes_quality_gates(result)
        assert not passes
    
    def test_quality_gate_high_score(self):
        cfg = SegmenterConfig(min_mask_quality=0.3)
        segmenter = NoneSegmenter(cfg)
        
        result = MaskResult(
            track_id=1,
            mask=np.ones((10, 10), dtype=np.uint8),
            quality_score=0.7,
            source=MaskSource.YOLO_SEG,
            mask_area_ratio=0.5,
        )
        
        passes = segmenter._passes_quality_gates(result)
        assert passes
    
    def test_quality_gate_area_ratio_too_low(self):
        cfg = SegmenterConfig(min_mask_area_ratio=0.1)
        segmenter = NoneSegmenter(cfg)
        
        result = MaskResult(
            track_id=1,
            mask=np.ones((10, 10), dtype=np.uint8),
            quality_score=0.8,
            source=MaskSource.YOLO_SEG,
            mask_area_ratio=0.05,
        )
        
        passes = segmenter._passes_quality_gates(result)
        assert not passes
    
    def test_quality_gate_area_ratio_too_high(self):
        cfg = SegmenterConfig(max_mask_area_ratio=5.0)
        segmenter = NoneSegmenter(cfg)
        
        result = MaskResult(
            track_id=1,
            mask=np.ones((10, 10), dtype=np.uint8),
            quality_score=0.8,
            source=MaskSource.YOLO_SEG,
            mask_area_ratio=10.0,
        )
        
        passes = segmenter._passes_quality_gates(result)
        assert not passes


class TestAuditM4Fields:
    
    def test_audit_entry_mask_used(self):
        entry = build_track_audit_entry(
            track_id=1,
            policy_state="AUTHORIZED_LOCKED_REDACT",
            redaction_method="mask_blur",
            identity_id="R001",
            id_source="F",
            authorized_signal=True,
            decision_category="resident",
            decision_binding_state="CONFIRMED_STRONG",
            is_redacted=True,
            mask_used=True,
            mask_source="yolo_seg",
            mask_quality=0.85,
            fallback_to_bbox=False,
        )
        
        assert entry["mask_used"] is True
        assert entry["mask_source"] == "yolo_seg"
        assert entry["mask_quality"] == 0.85
        assert entry["fallback_to_bbox"] is False
        assert entry["redaction_method"] == "mask_blur"
    
    def test_audit_entry_fallback_to_bbox(self):
        entry = build_track_audit_entry(
            track_id=2,
            policy_state="AUTHORIZED_LOCKED_REDACT",
            redaction_method="bbox_blur",
            identity_id="R002",
            id_source="F",
            authorized_signal=True,
            decision_category="resident",
            decision_binding_state="CONFIRMED_WEAK",
            is_redacted=True,
            mask_used=False,
            mask_source="none",
            mask_quality=None,
            fallback_to_bbox=True,
        )
        
        assert entry["mask_used"] is False
        assert entry["mask_source"] == "none"
        assert entry["fallback_to_bbox"] is True
        assert entry["redaction_method"] == "bbox_blur"
    
    def test_audit_entry_no_redaction_no_mask_fields(self):
        entry = build_track_audit_entry(
            track_id=3,
            policy_state="UNKNOWN_VISIBLE",
            redaction_method="none",
            identity_id=None,
            id_source="U",
            authorized_signal=False,
            decision_category="unknown",
            decision_binding_state=None,
            is_redacted=False,
        )
        
        assert "mask_used" not in entry or entry.get("mask_used") is False
        assert entry["redaction_method"] == "none"


class TestMaskDilation:
    
    def test_mask_dilation_increases_area(self):
        import cv2
        
        cfg = SegmenterConfig(dilate_mask_px=5)
        segmenter = NoneSegmenter(cfg)
        
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        
        original_area = np.sum(mask > 127)
        
        dilated = segmenter._dilate_mask(mask)
        dilated_area = np.sum(dilated > 127)
        
        assert dilated_area > original_area


class TestFrameSkip:
    
    def test_frame_skip_respects_setting(self):
        cfg = SegmenterConfig(run_every_n_frames=3)
        segmenter = NoneSegmenter(cfg)
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracks = [MockTracklet(track_id=1, last_box=(10, 10, 50, 50))]
        
        results = []
        for _ in range(5):
            results.append(segmenter.segment(frame, tracks))
        
        assert segmenter._frame_counter == 5


class TestExceptionSafety:
    
    def test_segment_catches_exceptions(self):
        
        class BrokenSegmenter(BaseSegmenter):
            def _load_model(self) -> bool:
                return True
            
            def _segment_internal(self, frame_bgr, tracks):
                raise RuntimeError("Intentional test error")
        
        cfg = SegmenterConfig()
        segmenter = BrokenSegmenter(cfg)
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracks = [MockTracklet(track_id=1, last_box=(10, 10, 50, 50))]
        
        result = segmenter.segment(frame, tracks)
        assert result == {}
    
    def test_segment_with_empty_tracks(self):
        cfg = SegmenterConfig()
        segmenter = NoneSegmenter(cfg)
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = segmenter.segment(frame, [])
        
        assert result == {}


class TestMaskSourceEnum:
    
    def test_mask_source_values(self):
        assert MaskSource.NONE.value == "none"
        assert MaskSource.YOLO_SEG.value == "yolo_seg"
        assert MaskSource.SAM2.value == "sam2"
        assert MaskSource.FALLBACK.value == "fallback"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
