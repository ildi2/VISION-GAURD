
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from schemas.tracklet import Tracklet

log = logging.getLogger("privacy.segmenter")


class MaskSource(Enum):
    NONE = "none"
    YOLO_SEG = "yolo_seg"
    SAM2 = "sam2"
    FALLBACK = "fallback"


@dataclass
class MaskResult:
    track_id: int
    mask: Optional[np.ndarray] = None
    quality_score: float = 0.0
    source: MaskSource = MaskSource.NONE
    bbox_iou: float = 0.0
    mask_area_ratio: float = 0.0
    
    @property
    def is_valid(self) -> bool:
        return self.mask is not None and self.quality_score > 0.0
    
    def to_audit_dict(self) -> Dict[str, Any]:
        return {
            "mask_used": self.is_valid,
            "mask_source": self.source.value,
            "mask_quality": round(self.quality_score, 3),
            "mask_area_ratio": round(self.mask_area_ratio, 3) if self.mask_area_ratio > 0 else None,
            "bbox_iou": round(self.bbox_iou, 3) if self.bbox_iou > 0 else None,
        }


@dataclass
class SegmenterConfig:
    enabled: bool = False
    backend: str = "none"
    model_path: str = ""
    imgsz: int = 640
    conf_threshold: float = 0.35
    iou_threshold: float = 0.5
    iou_assoc_min: float = 0.25
    run_every_n_frames: int = 1
    
    min_mask_area_ratio: float = 0.01
    max_mask_area_ratio: float = 5.0
    min_mask_quality: float = 0.3
    
    dilate_mask_px: int = 3
    device: str = "auto"


class BaseSegmenter(ABC):
    
    def __init__(self, cfg: SegmenterConfig) -> None:
        self._cfg = cfg
        self._initialized = False
        self._frame_counter = 0
    
    @abstractmethod
    def _load_model(self) -> bool:
        pass
    
    @abstractmethod
    def _segment_internal(
        self,
        frame_bgr: np.ndarray,
        tracks: List["Tracklet"],
    ) -> Dict[int, MaskResult]:
        pass
    
    def segment(
        self,
        frame_bgr: np.ndarray,
        tracks: List["Tracklet"],
    ) -> Dict[int, MaskResult]:
        try:
            self._frame_counter += 1
            
            if self._cfg.run_every_n_frames > 1:
                if self._frame_counter % self._cfg.run_every_n_frames != 0:
                    return {}
            
            if not self._initialized:
                if not self._load_model():
                    log.warning("Segmenter model failed to load, returning empty")
                    return {}
                self._initialized = True
            
            if not tracks:
                return {}
            
            results = self._segment_internal(frame_bgr, tracks)
            
            gated_results = {}
            for track_id, result in results.items():
                if self._passes_quality_gates(result):
                    if result.mask is not None and self._cfg.dilate_mask_px > 0:
                        result.mask = self._dilate_mask(result.mask)
                    gated_results[track_id] = result
                else:
                    gated_results[track_id] = MaskResult(
                        track_id=track_id,
                        mask=None,
                        quality_score=0.0,
                        source=MaskSource.NONE,
                    )
            
            return gated_results
            
        except Exception as e:
            log.exception("Segmenter.segment() failed (returning empty): %s", e)
            return {}
    
    def _passes_quality_gates(self, result: MaskResult) -> bool:
        if result.mask is None:
            return False
        
        if result.quality_score < self._cfg.min_mask_quality:
            log.debug(
                "Mask quality %.2f below threshold %.2f for track %d",
                result.quality_score, self._cfg.min_mask_quality, result.track_id,
            )
            return False
        
        if result.mask_area_ratio < self._cfg.min_mask_area_ratio:
            log.debug(
                "Mask area ratio %.3f below threshold %.3f for track %d",
                result.mask_area_ratio, self._cfg.min_mask_area_ratio, result.track_id,
            )
            return False
        
        if result.mask_area_ratio > self._cfg.max_mask_area_ratio:
            log.debug(
                "Mask area ratio %.3f above threshold %.3f for track %d",
                result.mask_area_ratio, self._cfg.max_mask_area_ratio, result.track_id,
            )
            return False
        
        return True
    
    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        try:
            import cv2
            kernel_size = self._cfg.dilate_mask_px * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            dilated = cv2.dilate(mask, kernel, iterations=1)
            return dilated
        except Exception:
            return mask


class NoneSegmenter(BaseSegmenter):
    
    def _load_model(self) -> bool:
        log.info("NoneSegmenter initialized (segmentation disabled)")
        return True
    
    def _segment_internal(
        self,
        frame_bgr: np.ndarray,
        tracks: List["Tracklet"],
    ) -> Dict[int, MaskResult]:
        return {}


class YOLOSegmenter(BaseSegmenter):
    
    def __init__(self, cfg: SegmenterConfig) -> None:
        super().__init__(cfg)
        self._model = None
        self._device = None
    
    def _load_model(self) -> bool:
        try:
            from ultralytics import YOLO
            import torch
            
            if self._cfg.device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self._cfg.device
            
            model_path = self._cfg.model_path
            if not model_path:
                model_path = "yolov8n-seg.pt"
            
            log.info("Loading YOLO segmentation model: %s (device=%s)", model_path, self._device)
            
            self._model = YOLO(model_path)
            
            dummy = np.zeros((self._cfg.imgsz, self._cfg.imgsz, 3), dtype=np.uint8)
            self._model.predict(
                dummy,
                device=self._device,
                verbose=False,
                imgsz=self._cfg.imgsz,
            )
            
            log.info("YOLO segmentation model loaded successfully")
            return True
            
        except ImportError:
            log.error("ultralytics not installed, YOLO segmentation unavailable")
            return False
        except Exception as e:
            log.exception("Failed to load YOLO segmentation model: %s", e)
            return False
    
    def _segment_internal(
        self,
        frame_bgr: np.ndarray,
        tracks: List["Tracklet"],
    ) -> Dict[int, MaskResult]:
        if self._model is None:
            return {}
        
        h, w = frame_bgr.shape[:2]
        
        results = self._model.predict(
            frame_bgr,
            device=self._device,
            verbose=False,
            imgsz=self._cfg.imgsz,
            conf=self._cfg.conf_threshold,
            iou=self._cfg.iou_threshold,
            classes=[0],
        )
        
        if not results or len(results) == 0:
            return {}
        
        result = results[0]
        
        if result.masks is None or result.boxes is None:
            return {}
        
        det_boxes = result.boxes.xyxy.cpu().numpy()
        det_confs = result.boxes.conf.cpu().numpy()
        masks_data = result.masks.data.cpu().numpy()
        
        if len(det_boxes) == 0:
            return {}
        
        track_bboxes: Dict[int, Tuple[float, float, float, float]] = {}
        for track in tracks:
            tid = getattr(track, "track_id", -1)
            bbox = getattr(track, "last_box", None) or getattr(track, "bbox", None)
            if bbox is not None and tid >= 0:
                track_bboxes[tid] = tuple(bbox)
        
        if not track_bboxes:
            return {}
        
        mask_results: Dict[int, MaskResult] = {}
        used_det_indices = set()
        
        for track_id, track_bbox in track_bboxes.items():
            best_iou = 0.0
            best_det_idx = -1
            
            for det_idx, det_box in enumerate(det_boxes):
                if det_idx in used_det_indices:
                    continue
                
                iou = self._compute_iou(track_bbox, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_iou >= self._cfg.iou_assoc_min and best_det_idx >= 0:
                used_det_indices.add(best_det_idx)
                
                mask_small = masks_data[best_det_idx]
                mask_full = self._resize_mask(mask_small, w, h)
                
                bbox_area = max(1, (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1]))
                mask_area = np.sum(mask_full > 127)
                area_ratio = mask_area / bbox_area
                
                mask_results[track_id] = MaskResult(
                    track_id=track_id,
                    mask=mask_full,
                    quality_score=float(det_confs[best_det_idx]),
                    source=MaskSource.YOLO_SEG,
                    bbox_iou=best_iou,
                    mask_area_ratio=area_ratio,
                )
            else:
                mask_results[track_id] = MaskResult(
                    track_id=track_id,
                    mask=None,
                    quality_score=0.0,
                    source=MaskSource.NONE,
                )
        
        return mask_results
    
    def _compute_iou(
        self,
        box1: Tuple[float, float, float, float],
        box2: np.ndarray,
    ) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area
    
    def _resize_mask(self, mask_small: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        import cv2
        
        if mask_small.max() <= 1.0:
            mask_small = (mask_small * 255).astype(np.uint8)
        
        mask_full = cv2.resize(
            mask_small,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )
        
        _, mask_binary = cv2.threshold(mask_full, 127, 255, cv2.THRESH_BINARY)
        
        return mask_binary.astype(np.uint8)


def create_segmenter(cfg: Any) -> BaseSegmenter:
    enabled = getattr(cfg, "enabled", False)
    backend = getattr(cfg, "backend", "none")
    
    if not enabled or backend == "none":
        return NoneSegmenter(SegmenterConfig())
    
    seg_cfg = SegmenterConfig(
        enabled=enabled,
        backend=backend,
        model_path=getattr(cfg, "model_path", ""),
        imgsz=getattr(cfg, "imgsz", 640),
        conf_threshold=getattr(cfg, "conf", 0.35),
        iou_threshold=getattr(cfg, "iou_threshold", 0.5),
        iou_assoc_min=getattr(cfg, "iou_assoc_min", 0.25),
        run_every_n_frames=getattr(cfg, "run_every_n_frames", 1),
        min_mask_area_ratio=getattr(cfg, "min_mask_area_ratio", 0.01),
        max_mask_area_ratio=getattr(cfg, "max_mask_area_ratio", 5.0),
        min_mask_quality=getattr(cfg, "min_mask_quality", 0.3),
        dilate_mask_px=getattr(cfg, "dilate_mask_px", 3),
        device=getattr(cfg, "device", "auto"),
    )
    
    if backend == "yolo_seg":
        log.info("Creating YOLO segmenter (backend=%s)", backend)
        return YOLOSegmenter(seg_cfg)
    elif backend == "sam2":
        log.warning("SAM2 backend not implemented in M4, falling back to none")
        return NoneSegmenter(seg_cfg)
    else:
        log.warning("Unknown segmentation backend '%s', falling back to none", backend)
        return NoneSegmenter(seg_cfg)
