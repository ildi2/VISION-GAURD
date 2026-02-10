
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("privacy.mask_stabilizer")


class StabilizationMethod(Enum):
    UNION_ONLY = "union_only"
    MAJORITY_VOTE = "majority_vote"


@dataclass
class StabilizationConfig:
    enabled: bool = False
    method: str = "union_only"
    history_size: int = 5
    mask_ttl_sec: float = 0.75
    max_shrink_ratio: float = 0.85
    morph_close_px: int = 3
    morph_open_px: int = 0


@dataclass
class StabilizationResult:
    track_id: int
    mask: Optional[np.ndarray]
    is_stable: bool = False
    
    method_used: str = "none"
    shrink_detected: bool = False
    ttl_reuse: bool = False
    original_area: int = 0
    stable_area: int = 0
    area_change_ratio: float = 1.0
    
    @property
    def is_valid(self) -> bool:
        return self.mask is not None
    
    def to_audit_dict(self) -> Dict[str, Any]:
        return {
            "stable_mask_used": self.is_stable,
            "stabilization_method": self.method_used,
            "shrink_detected": self.shrink_detected,
            "ttl_reuse": self.ttl_reuse,
            "mask_area": self.original_area,
            "stable_mask_area": self.stable_area,
        }


@dataclass
class TrackMaskHistory:
    track_id: int
    history_size: int = 5
    
    entries: List[Tuple[float, np.ndarray, int]] = field(default_factory=list)
    
    last_stable_mask: Optional[np.ndarray] = None
    last_stable_ts: float = 0.0
    last_stable_area: int = 0
    
    last_stable_bbox: Optional[Tuple[float, float, float, float]] = None
    
    def add_entry(self, ts: float, mask: np.ndarray) -> None:
        area = int(np.sum(mask > 127))
        
        self.entries.append((ts, mask.copy(), area))
        
        if len(self.entries) > self.history_size:
            self.entries.pop(0)
    
    def update_stable(self, mask: np.ndarray, ts: float, bbox: Optional[Tuple[float, float, float, float]] = None) -> None:
        self.last_stable_mask = mask.copy()
        self.last_stable_ts = ts
        self.last_stable_area = int(np.sum(mask > 127))
        self.last_stable_bbox = bbox
    
    def get_last_entry(self) -> Optional[Tuple[float, np.ndarray, int]]:
        if self.entries:
            return self.entries[-1]
        return None
    
    def get_history_masks(self) -> List[np.ndarray]:
        return [entry[1] for entry in self.entries]
    
    def get_history_age_sec(self, current_ts: float) -> float:
        if not self.entries:
            return float("inf")
        return current_ts - self.entries[-1][0]
    
    def get_stable_age_sec(self, current_ts: float) -> float:
        if self.last_stable_mask is None:
            return float("inf")
        return current_ts - self.last_stable_ts
    
    def clear(self) -> None:
        self.entries.clear()
        self.last_stable_mask = None
        self.last_stable_ts = 0.0
        self.last_stable_area = 0
        self.last_stable_bbox = None


class MaskStabilizer:
    
    def __init__(self, cfg: StabilizationConfig) -> None:
        self._cfg = cfg
        self._enabled = cfg.enabled
        
        self._track_histories: Dict[int, TrackMaskHistory] = {}
        
        self._total_updates = 0
        self._shrink_detections = 0
        self._ttl_reuses = 0
        self._ttl_reuse_rejected_position = 0
        self._union_applications = 0
        
        try:
            self._method = StabilizationMethod(cfg.method)
        except ValueError:
            log.warning("Unknown stabilization method '%s', falling back to union_only", cfg.method)
            self._method = StabilizationMethod.UNION_ONLY
        
        self._close_kernel = None
        self._open_kernel = None
        
        if cfg.morph_close_px > 0:
            ksize = cfg.morph_close_px * 2 + 1
            self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        
        if cfg.morph_open_px > 0:
            ksize = cfg.morph_open_px * 2 + 1
            self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        
        log.info(
            "MaskStabilizer initialized: enabled=%s, method=%s, history_size=%d, ttl=%.2fs",
            self._enabled, self._method.value, cfg.history_size, cfg.mask_ttl_sec,
        )
    
    def update(
        self,
        track_id: int,
        mask: Optional[np.ndarray],
        ts: float,
        current_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> StabilizationResult:
        self._total_updates += 1
        
        try:
            if track_id not in self._track_histories:
                self._track_histories[track_id] = TrackMaskHistory(
                    track_id=track_id,
                    history_size=self._cfg.history_size,
                )
            
            history = self._track_histories[track_id]
            
            if mask is None:
                return self._handle_missing_mask(track_id, history, ts, current_bbox)
            
            if mask.size == 0:
                return self._handle_missing_mask(track_id, history, ts, current_bbox)
            
            current_area = int(np.sum(mask > 127))
            
            shrink_detected = False
            position_moved = False
            if history.last_stable_mask is not None and history.last_stable_area > 0:
                area_ratio = current_area / history.last_stable_area
                if area_ratio < self._cfg.max_shrink_ratio:
                    if current_bbox is not None and history.last_stable_bbox is not None:
                        position_iou = self._compute_bbox_iou(history.last_stable_bbox, current_bbox)
                        if position_iou < 0.85:
                            position_moved = True
                            log.debug(
                                "M5: Shrink suppressed (movement): track=%d area_ratio=%.2f bbox_iou=%.3f",
                                track_id, area_ratio, position_iou,
                            )
                        else:
                            shrink_detected = True
                            self._shrink_detections += 1
                    else:
                        shrink_detected = True
                        self._shrink_detections += 1
            
            if self._method == StabilizationMethod.UNION_ONLY:
                stable_mask = self._stabilize_union_only(mask, history, shrink_detected)
            elif self._method == StabilizationMethod.MAJORITY_VOTE:
                stable_mask = self._stabilize_majority_vote(mask, history)
            else:
                stable_mask = mask.copy()
            
            stable_mask = self._apply_morphology(stable_mask)
            
            stable_area = int(np.sum(stable_mask > 127))
            
            history.add_entry(ts, mask)
            history.update_stable(stable_mask, ts, bbox=current_bbox)
            
            return StabilizationResult(
                track_id=track_id,
                mask=stable_mask,
                is_stable=True,
                method_used=self._method.value,
                shrink_detected=shrink_detected,
                ttl_reuse=False,
                original_area=current_area,
                stable_area=stable_area,
                area_change_ratio=stable_area / max(1, current_area),
            )
            
        except Exception as e:
            log.warning("MaskStabilizer.update() failed for track %d: %s", track_id, e)
            if mask is not None:
                return StabilizationResult(
                    track_id=track_id,
                    mask=mask,
                    is_stable=False,
                    method_used="passthrough",
                    original_area=int(np.sum(mask > 127)) if mask.size > 0 else 0,
                    stable_area=int(np.sum(mask > 127)) if mask.size > 0 else 0,
                )
            else:
                return StabilizationResult(track_id=track_id, mask=None, is_stable=False)
    
    def _handle_missing_mask(
        self,
        track_id: int,
        history: TrackMaskHistory,
        ts: float,
        current_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> StabilizationResult:
        if history.last_stable_mask is not None:
            age_sec = history.get_stable_age_sec(ts)
            
            if age_sec <= self._cfg.mask_ttl_sec:
                if current_bbox is not None and history.last_stable_bbox is not None:
                    iou = self._compute_bbox_iou(history.last_stable_bbox, current_bbox)
                    
                    if iou < 0.3:
                        self._ttl_reuse_rejected_position += 1
                        log.debug(
                            "M5: TTL reuse REJECTED for track=%d | IoU=%.3f < 0.3 (position changed significantly)",
                            track_id, iou,
                        )
                        return StabilizationResult(
                            track_id=track_id,
                            mask=None,
                            is_stable=False,
                            method_used="ttl_reuse_rejected",
                            shrink_detected=False,
                            ttl_reuse=False,
                        )
                
                self._ttl_reuses += 1
                
                log.debug(
                    "M5: TTL reuse ACCEPTED for track=%d | age=%.2fs, IoU=%.3f",
                    track_id, age_sec,
                    self._compute_bbox_iou(history.last_stable_bbox, current_bbox) if (current_bbox and history.last_stable_bbox) else -1.0,
                )
                
                return StabilizationResult(
                    track_id=track_id,
                    mask=history.last_stable_mask.copy(),
                    is_stable=True,
                    method_used="ttl_reuse",
                    shrink_detected=False,
                    ttl_reuse=True,
                    original_area=0,
                    stable_area=history.last_stable_area,
                )
        
        return StabilizationResult(
            track_id=track_id,
            mask=None,
            is_stable=False,
            method_used="none",
            ttl_reuse=False,
        )
    
    def _stabilize_union_only(
        self,
        mask: np.ndarray,
        history: TrackMaskHistory,
        shrink_detected: bool,
    ) -> np.ndarray:
        if not shrink_detected or history.last_stable_mask is None:
            return mask.copy()
        
        prev_stable = history.last_stable_mask
        if prev_stable.shape != mask.shape:
            prev_stable = cv2.resize(
                prev_stable, (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        
        union_mask = cv2.bitwise_or(mask, prev_stable)
        self._union_applications += 1
        
        return union_mask
    
    def _stabilize_majority_vote(
        self,
        mask: np.ndarray,
        history: TrackMaskHistory,
    ) -> np.ndarray:
        masks = history.get_history_masks()
        
        if len(masks) < 3:
            return mask.copy()
        
        all_masks = masks + [mask]
        
        h, w = mask.shape[:2]
        aligned_masks = []
        
        for m in all_masks:
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            aligned_masks.append((m > 127).astype(np.uint8))
        
        vote_sum = np.zeros((h, w), dtype=np.int32)
        for m in aligned_masks:
            vote_sum += m
        
        threshold = len(aligned_masks) // 2
        stable_mask = (vote_sum >= threshold).astype(np.uint8) * 255
        
        stable_mask = cv2.bitwise_or(stable_mask, mask)
        
        return stable_mask
    
    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        result = mask
        
        if self._close_kernel is not None:
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, self._close_kernel)
        
        if self._open_kernel is not None:
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, self._open_kernel)
        
        return result
    
    def get_track_ttl_mask(self, track_id: int, ts: float) -> Optional[np.ndarray]:
        if track_id not in self._track_histories:
            return None
        
        history = self._track_histories[track_id]
        
        if history.last_stable_mask is None:
            return None
        
        age_sec = history.get_stable_age_sec(ts)
        if age_sec <= self._cfg.mask_ttl_sec:
            return history.last_stable_mask.copy()
        
        return None
    
    def cleanup_stale_tracks(self, active_track_ids: set, current_ts: float) -> int:
        stale_ids = []
        
        for track_id, history in self._track_histories.items():
            if track_id not in active_track_ids:
                age = history.get_stable_age_sec(current_ts)
                if age > 5.0:
                    stale_ids.append(track_id)
        
        for track_id in stale_ids:
            del self._track_histories[track_id]
        
        return len(stale_ids)
    
    def _compute_bbox_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float],
    ) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - intersection_area
        
        if union_area <= 0:
            return 0.0
        
        return intersection_area / union_area
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "method": self._method.value,
            "total_updates": self._total_updates,
            "shrink_detections": self._shrink_detections,
            "ttl_reuses": self._ttl_reuses,
            "ttl_reuse_rejected_position": self._ttl_reuse_rejected_position,
            "union_applications": self._union_applications,
            "active_tracks": len(self._track_histories),
        }
    
    @property
    def enabled(self) -> bool:
        return self._enabled


def create_stabilizer(cfg: Any) -> MaskStabilizer:
    stab_cfg = StabilizationConfig(
        enabled=getattr(cfg, "enabled", False),
        method=getattr(cfg, "method", "union_only"),
        history_size=getattr(cfg, "history_size", 5),
        mask_ttl_sec=getattr(cfg, "mask_ttl_sec", 0.75),
        max_shrink_ratio=getattr(cfg, "max_shrink_ratio", 0.85),
        morph_close_px=getattr(cfg, "morph_close_px", 3),
        morph_open_px=getattr(cfg, "morph_open_px", 0),
    )
    
    return MaskStabilizer(stab_cfg)
