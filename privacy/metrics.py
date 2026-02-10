
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import statistics

import cv2
import numpy as np

log = logging.getLogger("privacy.metrics")


@dataclass
class PrivacyMetricsConfig:
    enabled: bool = True
    log_path: str = "privacy_output/privacy_metrics.jsonl"
    flush_interval_sec: float = 1.0
    
    leakage_enabled: bool = True
    leakage_backend: str = "opencv_haar"
    haar_scale_factor: float = 1.1
    haar_min_neighbors: int = 5
    haar_min_size: Tuple[int, int] = (30, 30)
    overlap_threshold: float = 0.2
    
    flicker_enabled: bool = True
    
    timing_enabled: bool = True
    
    utility_enabled: bool = True


class MetricsWriter:
    
    def __init__(
        self,
        log_path: str,
        flush_interval_sec: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self._log_path = log_path
        self._flush_interval_sec = flush_interval_sec
        self._enabled = enabled
        self._disabled_due_to_error = False
        
        self._file = None
        self._last_flush_ts = 0.0
        self._entries_written = 0
        
        if self._enabled:
            self._open()
    
    def _open(self) -> None:
        try:
            path = Path(self._log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            self._file = open(path, "a", encoding="utf-8")
            log.info("Metrics log opened: %s", self._log_path)
        except Exception as e:
            log.warning("Failed to open metrics log '%s': %s (disabling)", self._log_path, e)
            self._disabled_due_to_error = True
            self._file = None
    
    def write_entry(self, entry: Dict[str, Any]) -> bool:
        if not self._enabled or self._disabled_due_to_error or self._file is None:
            return False
        
        try:
            line = json.dumps(entry, default=str) + "\n"
            self._file.write(line)
            self._entries_written += 1
            
            now = time.time()
            if now - self._last_flush_ts >= self._flush_interval_sec:
                self._file.flush()
                self._last_flush_ts = now
            
            return True
        except Exception as e:
            log.warning("Metrics write failed: %s (disabling)", e)
            self._disabled_due_to_error = True
            return False
    
    def flush(self) -> None:
        if self._file is not None and not self._disabled_due_to_error:
            try:
                self._file.flush()
            except Exception:
                pass
    
    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
                log.info("Metrics log closed: %s (entries: %d)", self._log_path, self._entries_written)
            except Exception as e:
                log.warning("Failed to close metrics log: %s", e)
            finally:
                self._file = None
    
    @property
    def is_active(self) -> bool:
        return self._enabled and not self._disabled_due_to_error and self._file is not None
    
    @property
    def entries_written(self) -> int:
        return self._entries_written


class LeakageDetector:
    
    def __init__(self, cfg: PrivacyMetricsConfig) -> None:
        self._cfg = cfg
        self._enabled = cfg.leakage_enabled
        self._backend = cfg.leakage_backend
        self._overlap_threshold = cfg.overlap_threshold
        self._disabled_due_to_error = False
        
        self._face_cascade = None
        if self._enabled and self._backend == "opencv_haar":
            self._load_haar_cascade()
    
    def _load_haar_cascade(self) -> None:
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                raise RuntimeError("Failed to load Haar cascade")
            log.debug("Leakage detector loaded: %s", cascade_path)
        except Exception as e:
            log.warning("Failed to load Haar cascade: %s (disabling leakage)", e)
            self._disabled_due_to_error = True
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if not self._enabled or self._disabled_due_to_error:
            return []
        
        if self._backend == "stub":
            return []
        
        if self._backend == "none":
            return []
        
        if self._face_cascade is None:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=self._cfg.haar_scale_factor,
                minNeighbors=self._cfg.haar_min_neighbors,
                minSize=self._cfg.haar_min_size,
            )
            
            result = []
            for (x, y, w, h) in faces:
                result.append((x, y, x + w, y + h))
            
            return result
        except Exception as e:
            log.debug("Face detection failed: %s", e)
            return []
    
    def check_leakage(
        self,
        frame: np.ndarray,
        redacted_regions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        result = {
            "faces_detected": 0,
            "faces_in_redacted": 0,
            "leakage_flag": False,
            "leakage_details": [],
        }
        
        if not self._enabled or self._disabled_due_to_error:
            return result
        
        faces = self.detect_faces(frame)
        result["faces_detected"] = len(faces)
        
        if not faces or not redacted_regions:
            return result
        
        for face_bbox in faces:
            for region in redacted_regions:
                if self._check_overlap(face_bbox, region):
                    result["faces_in_redacted"] += 1
                    result["leakage_flag"] = True
                    result["leakage_details"].append({
                        "face_bbox": face_bbox,
                        "track_id": region.get("track_id"),
                    })
                    break
        
        return result
    
    def _check_overlap(
        self,
        face_bbox: Tuple[int, int, int, int],
        region: Dict[str, Any],
    ) -> bool:
        fx1, fy1, fx2, fy2 = face_bbox
        face_area = (fx2 - fx1) * (fy2 - fy1)
        if face_area <= 0:
            return False
        
        mask = region.get("mask")
        if mask is not None:
            mask_roi = mask[fy1:fy2, fx1:fx2]
            if mask_roi.size > 0:
                overlap_pixels = np.sum(mask_roi > 127)
                overlap_ratio = overlap_pixels / face_area
                return overlap_ratio >= self._overlap_threshold
        
        bbox = region.get("bbox")
        if bbox is not None:
            bx1, by1, bx2, by2 = bbox
            
            ix1 = max(fx1, bx1)
            iy1 = max(fy1, by1)
            ix2 = min(fx2, bx2)
            iy2 = min(fy2, by2)
            
            if ix2 <= ix1 or iy2 <= iy1:
                return False
            
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            overlap_ratio = intersection_area / face_area
            return overlap_ratio >= self._overlap_threshold
        
        return False
    
    def inject_stub_detector(self, stub_faces: List[Tuple[int, int, int, int]]) -> None:
        self._stub_faces = stub_faces
        self._backend = "stub"
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled and not self._disabled_due_to_error


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    if mask1 is None or mask2 is None:
        return 0.0
    
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    m1 = mask1 > 127
    m2 = mask2 > 127
    
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection) / float(union)


def compute_mask_area(mask: Optional[np.ndarray]) -> int:
    if mask is None:
        return 0
    return int(np.sum(mask > 127))


@dataclass
class TrackTimingState:
    track_id: int
    first_seen_ts: float = 0.0
    lock_ts: Optional[float] = None
    first_redacted_emit_ts: Optional[float] = None
    
    was_locked: bool = False
    was_emitted_redacted: bool = False
    
    def get_time_to_lock(self) -> Optional[float]:
        if self.lock_ts is None:
            return None
        return self.lock_ts - self.first_seen_ts
    
    def get_time_to_redacted_emit(self) -> Optional[float]:
        if self.first_redacted_emit_ts is None:
            return None
        return self.first_redacted_emit_ts - self.first_seen_ts


@dataclass
class TrackFlickerState:
    track_id: int
    last_mask: Optional[np.ndarray] = None
    last_area: int = 0
    
    iou_values: List[float] = field(default_factory=list)
    area_values: List[int] = field(default_factory=list)
    
    max_history: int = 100
    
    def add_mask(self, mask: Optional[np.ndarray]) -> Optional[float]:
        current_area = compute_mask_area(mask)
        iou = None
        
        if self.last_mask is not None and mask is not None:
            iou = compute_mask_iou(self.last_mask, mask)
            self.iou_values.append(iou)
            
            if len(self.iou_values) > self.max_history:
                self.iou_values = self.iou_values[-self.max_history:]
        
        self.area_values.append(current_area)
        if len(self.area_values) > self.max_history:
            self.area_values = self.area_values[-self.max_history:]
        
        self.last_mask = mask.copy() if mask is not None else None
        self.last_area = current_area
        
        return iou
    
    def get_mean_iou(self) -> Optional[float]:
        if not self.iou_values:
            return None
        return statistics.mean(self.iou_values)
    
    def get_p05_iou(self) -> Optional[float]:
        if len(self.iou_values) < 2:
            return None
        sorted_vals = sorted(self.iou_values)
        idx = max(0, int(len(sorted_vals) * 0.05))
        return sorted_vals[idx]
    
    def get_area_std(self) -> Optional[float]:
        if len(self.area_values) < 2:
            return None
        return statistics.stdev(self.area_values)
    
    def get_area_cv(self) -> Optional[float]:
        if len(self.area_values) < 2:
            return None
        mean = statistics.mean(self.area_values)
        if mean == 0:
            return None
        return statistics.stdev(self.area_values) / mean


class PrivacyMetricsEngine:
    
    def __init__(self, cfg: PrivacyMetricsConfig, delay_sec: float = 3.0) -> None:
        self._cfg = cfg
        self._enabled = cfg.enabled
        self._delay_sec = delay_sec
        self._disabled_due_to_error = False
        
        self._writer = MetricsWriter(
            log_path=cfg.log_path,
            flush_interval_sec=cfg.flush_interval_sec,
            enabled=cfg.enabled,
        )
        
        self._leakage_detector = LeakageDetector(cfg)
        
        self._track_timing: Dict[int, TrackTimingState] = {}
        
        self._track_flicker: Dict[int, TrackFlickerState] = {}
        
        self._frames_analyzed = 0
        self._total_faces_detected = 0
        self._total_leakage_events = 0
        self._total_redacted_frames = 0
        self._total_visible_frames = 0
        
        self._frames_with_any_redacted = 0
        self._frames_with_any_visible = 0
        self._frames_with_both = 0
        
        self._time_to_lock_samples: List[float] = []
        self._time_to_redacted_emit_samples: List[float] = []
        
        self._init_ts = time.time()
        
        if self._enabled:
            log.info(
                "PrivacyMetricsEngine initialized | leakage=%s, flicker=%s, timing=%s, utility=%s",
                cfg.leakage_enabled,
                cfg.flicker_enabled,
                cfg.timing_enabled,
                cfg.utility_enabled,
            )
    
    def on_ingest(
        self,
        frame_ts: float,
        frame_id: int,
        track_ids_present: Set[int],
        policy_info_by_track: Dict[int, Dict[str, Any]],
    ) -> None:
        if not self._enabled or self._disabled_due_to_error:
            return
        
        try:
            if not self._cfg.timing_enabled:
                return
            
            for track_id in track_ids_present:
                if track_id not in self._track_timing:
                    self._track_timing[track_id] = TrackTimingState(
                        track_id=track_id,
                        first_seen_ts=frame_ts,
                    )
                
                timing = self._track_timing[track_id]
                
                policy_info = policy_info_by_track.get(track_id, {})
                policy_state = policy_info.get("policy_state", "UNKNOWN_VISIBLE")
                
                if policy_state == "AUTHORIZED_LOCKED_REDACT" and not timing.was_locked:
                    timing.lock_ts = frame_ts
                    timing.was_locked = True
                    
                    ttl = timing.get_time_to_lock()
                    if ttl is not None:
                        self._time_to_lock_samples.append(ttl)
                        
        except Exception as e:
            log.warning("on_ingest failed: %s (disabling metrics)", e)
            self._disabled_due_to_error = True
    
    def on_masks(
        self,
        frame_ts: float,
        frame_id: int,
        track_id: int,
        raw_mask: Optional[np.ndarray],
        stable_mask: Optional[np.ndarray],
        bbox: Optional[List[float]],
    ) -> None:
        if not self._enabled or self._disabled_due_to_error:
            return
        
        try:
            if not self._cfg.flicker_enabled:
                return
            
            mask = stable_mask if stable_mask is not None else raw_mask
            
            if track_id not in self._track_flicker:
                self._track_flicker[track_id] = TrackFlickerState(track_id=track_id)
            
            flicker = self._track_flicker[track_id]
            
            iou = flicker.add_mask(mask)
            
        except Exception as e:
            log.warning("on_masks failed: %s (disabling metrics)", e)
            self._disabled_due_to_error = True
    
    def on_emit(
        self,
        emit_ts: float,
        frame_id: int,
        privacy_frame: np.ndarray,
        per_track_redaction_info: Dict[int, Dict[str, Any]],
        redacted_regions: List[Dict[str, Any]],
    ) -> None:
        if not self._enabled or self._disabled_due_to_error:
            return
        
        try:
            self._frames_analyzed += 1
            
            redacted_count = 0
            visible_count = 0
            
            for track_id, info in per_track_redaction_info.items():
                is_redacted = info.get("is_redacted", False)
                if is_redacted:
                    redacted_count += 1
                    self._total_redacted_frames += 1
                    
                    if self._cfg.timing_enabled and track_id in self._track_timing:
                        timing = self._track_timing[track_id]
                        if timing.was_locked and not timing.was_emitted_redacted:
                            timing.first_redacted_emit_ts = emit_ts
                            timing.was_emitted_redacted = True
                            
                            ttre = timing.get_time_to_redacted_emit()
                            if ttre is not None:
                                self._time_to_redacted_emit_samples.append(ttre)
                else:
                    visible_count += 1
                    self._total_visible_frames += 1
            
            if redacted_count > 0 and visible_count > 0:
                self._frames_with_both += 1
            if redacted_count > 0:
                self._frames_with_any_redacted += 1
            if visible_count > 0:
                self._frames_with_any_visible += 1
            
            leakage_result = {
                "faces_detected": 0,
                "faces_in_redacted": 0,
                "leakage_flag": False,
            }
            
            if self._cfg.leakage_enabled and len(redacted_regions) > 0:
                leakage_result = self._leakage_detector.check_leakage(
                    privacy_frame, redacted_regions
                )
                self._total_faces_detected += leakage_result["faces_detected"]
                if leakage_result["leakage_flag"]:
                    self._total_leakage_events += 1
            
            entry = {
                "type": "frame_metrics",
                "ts": emit_ts,
                "frame_id": frame_id,
                "emit_ts": emit_ts,
                "redacted_count": redacted_count,
                "visible_count": visible_count,
                "faces_detected_total": leakage_result["faces_detected"],
                "faces_detected_in_redacted": leakage_result["faces_in_redacted"],
                "leakage_flag": leakage_result["leakage_flag"],
            }
            
            self._writer.write_entry(entry)
            
        except Exception as e:
            log.warning("on_emit failed: %s (disabling metrics)", e)
            self._disabled_due_to_error = True
    
    def finalize_and_summarize(self) -> Dict[str, Any]:
        summary = {
            "type": "summary",
            "ts": time.time(),
            "session_duration_sec": time.time() - self._init_ts,
            "frames_analyzed": self._frames_analyzed,
            "metrics_enabled": self._enabled,
            "disabled_due_to_error": self._disabled_due_to_error,
        }
        
        if not self._enabled:
            return summary
        
        try:
            if self._cfg.leakage_enabled:
                leakage_rate = 0.0
                if self._total_redacted_frames > 0:
                    leakage_rate = self._total_leakage_events / self._total_redacted_frames
                
                summary["leakage"] = {
                    "total_faces_detected": self._total_faces_detected,
                    "total_leakage_events": self._total_leakage_events,
                    "leakage_rate": round(leakage_rate, 6),
                }
            
            if self._cfg.timing_enabled:
                timing_summary = {
                    "tracks_observed": len(self._track_timing),
                    "tracks_locked": sum(1 for t in self._track_timing.values() if t.was_locked),
                    "tracks_emitted_redacted": sum(1 for t in self._track_timing.values() if t.was_emitted_redacted),
                }
                
                if self._time_to_lock_samples:
                    sorted_ttl = sorted(self._time_to_lock_samples)
                    timing_summary["time_to_lock"] = {
                        "mean_sec": round(statistics.mean(sorted_ttl), 3),
                        "p50_sec": round(sorted_ttl[len(sorted_ttl) // 2], 3),
                        "p95_sec": round(sorted_ttl[int(len(sorted_ttl) * 0.95)], 3) if len(sorted_ttl) >= 2 else None,
                        "samples": len(sorted_ttl),
                    }
                
                if self._time_to_redacted_emit_samples:
                    sorted_ttre = sorted(self._time_to_redacted_emit_samples)
                    timing_summary["time_to_redacted_emit"] = {
                        "mean_sec": round(statistics.mean(sorted_ttre), 3),
                        "p50_sec": round(sorted_ttre[len(sorted_ttre) // 2], 3),
                        "p95_sec": round(sorted_ttre[int(len(sorted_ttre) * 0.95)], 3) if len(sorted_ttre) >= 2 else None,
                        "samples": len(sorted_ttre),
                        "expected_delay_sec": self._delay_sec,
                    }
                
                summary["timing"] = timing_summary
            
            if self._cfg.flicker_enabled:
                all_ious = []
                all_area_cvs = []
                
                for flicker in self._track_flicker.values():
                    all_ious.extend(flicker.iou_values)
                    cv = flicker.get_area_cv()
                    if cv is not None:
                        all_area_cvs.append(cv)
                
                flicker_summary = {
                    "tracks_measured": len(self._track_flicker),
                    "total_iou_samples": len(all_ious),
                }
                
                if all_ious:
                    sorted_ious = sorted(all_ious)
                    flicker_summary["iou"] = {
                        "mean": round(statistics.mean(sorted_ious), 4),
                        "p05": round(sorted_ious[max(0, int(len(sorted_ious) * 0.05))], 4),
                        "p50": round(sorted_ious[len(sorted_ious) // 2], 4),
                        "min": round(min(sorted_ious), 4),
                    }
                
                if all_area_cvs:
                    flicker_summary["area_cv"] = {
                        "mean": round(statistics.mean(all_area_cvs), 4),
                        "max": round(max(all_area_cvs), 4),
                    }
                
                summary["flicker"] = flicker_summary
            
            if self._cfg.utility_enabled:
                total = self._total_redacted_frames + self._total_visible_frames
                utility_summary = {
                    "total_track_frames": total,
                    "redacted_frames": self._total_redacted_frames,
                    "visible_frames": self._total_visible_frames,
                    "frames_analyzed": self._frames_analyzed,
                    "frames_with_any_redacted": self._frames_with_any_redacted,
                    "frames_with_any_visible": self._frames_with_any_visible,
                    "frames_with_both": self._frames_with_both,
                }
                
                if total > 0:
                    utility_summary["redacted_rate"] = round(self._total_redacted_frames / total, 4)
                    utility_summary["visible_rate"] = round(self._total_visible_frames / total, 4)
                
                if self._frames_analyzed > 0:
                    utility_summary["frames_redacted_rate"] = round(self._frames_with_any_redacted / self._frames_analyzed, 4)
                    utility_summary["frames_visible_rate"] = round(self._frames_with_any_visible / self._frames_analyzed, 4)
                    utility_summary["frames_mixed_rate"] = round(self._frames_with_both / self._frames_analyzed, 4)
                
                summary["utility"] = utility_summary
            
            self._writer.write_entry(summary)
            
            log.info("=== M6 Metrics Summary ===")
            log.info("Frames analyzed: %d", self._frames_analyzed)
            
            if "leakage" in summary:
                log.info(
                    "Leakage: events=%d, rate=%.4f",
                    summary["leakage"]["total_leakage_events"],
                    summary["leakage"]["leakage_rate"],
                )
            
            if "timing" in summary:
                t = summary["timing"]
                if "time_to_lock" in t:
                    log.info(
                        "Time-to-lock: mean=%.3fs, p50=%.3fs (n=%d)",
                        t["time_to_lock"]["mean_sec"],
                        t["time_to_lock"]["p50_sec"],
                        t["time_to_lock"]["samples"],
                    )
                if "time_to_redacted_emit" in t:
                    log.info(
                        "Time-to-redacted-emit: mean=%.3fs, p50=%.3fs (n=%d)",
                        t["time_to_redacted_emit"]["mean_sec"],
                        t["time_to_redacted_emit"]["p50_sec"],
                        t["time_to_redacted_emit"]["samples"],
                    )
            
            if "flicker" in summary and "iou" in summary["flicker"]:
                f = summary["flicker"]["iou"]
                log.info(
                    "Flicker (IoU): mean=%.4f, p05=%.4f, min=%.4f",
                    f["mean"], f["p05"], f["min"],
                )
            
            if "utility" in summary:
                u = summary["utility"]
                log.info(
                    "Utility: redacted=%d (%.2f%%), visible=%d (%.2f%%)",
                    u["redacted_frames"],
                    u.get("redacted_rate", 0) * 100,
                    u["visible_frames"],
                    u.get("visible_rate", 0) * 100,
                )
            
        except Exception as e:
            log.warning("finalize_and_summarize failed: %s", e)
            summary["error"] = str(e)
        
        self._writer.close()
        
        return summary
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "disabled_due_to_error": self._disabled_due_to_error,
            "frames_analyzed": self._frames_analyzed,
            "total_faces_detected": self._total_faces_detected,
            "total_leakage_events": self._total_leakage_events,
            "entries_written": self._writer.entries_written,
        }
    
    @property
    def is_active(self) -> bool:
        return self._enabled and not self._disabled_due_to_error


def create_metrics_engine(cfg: Any, delay_sec: float = 3.0) -> PrivacyMetricsEngine:
    metrics_cfg = PrivacyMetricsConfig(
        enabled=getattr(cfg, "enabled", True),
        log_path=getattr(cfg, "log_path", "privacy_output/privacy_metrics.jsonl"),
        flush_interval_sec=getattr(cfg, "flush_interval_sec", 1.0),
        leakage_enabled=getattr(cfg, "leakage_enabled", True),
        leakage_backend=getattr(cfg, "leakage_backend", "opencv_haar"),
        haar_scale_factor=getattr(cfg, "haar_scale_factor", 1.1),
        haar_min_neighbors=getattr(cfg, "haar_min_neighbors", 5),
        overlap_threshold=getattr(cfg, "overlap_threshold", 0.2),
        flicker_enabled=getattr(cfg, "flicker_enabled", True),
        timing_enabled=getattr(cfg, "timing_enabled", True),
        utility_enabled=getattr(cfg, "utility_enabled", True),
    )
    
    return PrivacyMetricsEngine(metrics_cfg, delay_sec)
