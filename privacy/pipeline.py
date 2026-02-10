
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import cv2
import numpy as np

from .audit import AuditWriter, build_audit_entry, build_track_audit_entry
from .delay_buffer import DelayBuffer, BufferItem
from .writer import PrivacyWriter
from .policy_fsm import PolicyFSM, PolicyAction, PolicyState
from .segmenter import create_segmenter, BaseSegmenter, MaskResult, MaskSource
from .mask_stabilizer import create_stabilizer, MaskStabilizer, StabilizationResult
from .metrics import create_metrics_engine, PrivacyMetricsEngine

if TYPE_CHECKING:
    from schemas import Frame
    from schemas.tracklet import Tracklet
    from schemas.identity_decision import IdentityDecision

log = logging.getLogger("privacy.pipeline")


class PrivacyPipeline:
    
    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        self._frame_count = 0
        self._latest_privacy_frame: Optional[np.ndarray] = None
        
        self._init_perf_ts = time.perf_counter()
        self._init_wall_ts = time.time()
        
        self._enabled = getattr(cfg, "enabled", True)
        self._delay_sec = getattr(cfg, "delay_sec", 3.0)
        
        self._redaction_style = getattr(cfg, "redaction_style", "blur")
        
        self._silhouette_cleanup = getattr(cfg, "silhouette_cleanup", False)
        self._silhouette_cleanup_kernel = None
        if self._silhouette_cleanup:
            _ksize = 3 * 2 + 1
            self._silhouette_cleanup_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (_ksize, _ksize)
            )
        
        output_cfg = getattr(cfg, "output", None)
        self._output_dir = getattr(output_cfg, "dir", "privacy_output") if output_cfg else "privacy_output"
        self._output_basename = getattr(output_cfg, "basename", "privacy_stream") if output_cfg else "privacy_stream"
        self._output_codec = getattr(output_cfg, "codec", "mp4v") if output_cfg else "mp4v"
        self._output_container = getattr(output_cfg, "container", "mp4") if output_cfg else "mp4"
        self._output_fps = getattr(output_cfg, "fps", 30) if output_cfg else 30
        
        policy_cfg = getattr(cfg, "policy", None)
        if policy_cfg:
            self._authorized_categories = getattr(policy_cfg, "authorized_categories", ["resident"])
            self._require_confirmed_binding = getattr(policy_cfg, "require_confirmed_binding", True)
        else:
            self._authorized_categories = ["resident"]
            self._require_confirmed_binding = True
        
        self._policy_fsm = PolicyFSM(policy_cfg) if policy_cfg else PolicyFSM(type('Cfg', (), {})())
        
        segmentation_cfg = getattr(cfg, "segmentation", None)
        self._segmenter: BaseSegmenter = create_segmenter(segmentation_cfg)
        self._segmentation_enabled = getattr(segmentation_cfg, "enabled", False) if segmentation_cfg else False
        
        self._mask_redactions = 0
        self._bbox_fallbacks = 0
        
        stabilization_cfg = getattr(cfg, "stabilization", None)
        self._stabilizer: MaskStabilizer = create_stabilizer(stabilization_cfg) if stabilization_cfg else create_stabilizer(type('Cfg', (), {'enabled': False})())
        self._stabilization_enabled = getattr(stabilization_cfg, "enabled", False) if stabilization_cfg else False
        
        self._stable_mask_uses = 0
        self._shrink_detections = 0
        self._ttl_reuses = 0
        
        metrics_cfg = getattr(cfg, "metrics", None)
        self._metrics_engine: Optional[PrivacyMetricsEngine] = create_metrics_engine(
            metrics_cfg, self._delay_sec
        )
        self._metrics_enabled = self._metrics_engine is not None
        
        ui_cfg = getattr(cfg, "ui", None)
        self._show_watermark = getattr(ui_cfg, "show_preview_watermark", True) if ui_cfg else True
        
        audit_cfg = getattr(cfg, "audit", None)
        audit_enabled = getattr(audit_cfg, "enabled", True) if audit_cfg else True
        audit_filename = getattr(audit_cfg, "filename", "privacy_audit.jsonl") if audit_cfg else "privacy_audit.jsonl"
        flush_interval = getattr(audit_cfg, "flush_interval_sec", 1.0) if audit_cfg else 1.0
        
        self._audit_writer = AuditWriter(
            output_dir=self._output_dir,
            filename=audit_filename,
            flush_interval_sec=flush_interval,
            enabled=audit_enabled,
        )
        
        max_buffer_frames = int(self._delay_sec * self._output_fps * 2)
        max_buffer_frames = max(max_buffer_frames, 100)
        
        self._delay_buffer = DelayBuffer(
            delay_sec=self._delay_sec,
            max_frames=max_buffer_frames,
            overflow_policy="drop_oldest",
        )
        
        self._writer = PrivacyWriter(
            output_dir=self._output_dir,
            basename=self._output_basename,
            fps=self._output_fps,
            codec=self._output_codec,
            container=self._output_container,
            frame_size=None,
            enabled=True,
        )
        
        self._frames_ingested = 0
        self._frames_emitted = 0
        self._frames_dropped = 0
        self._writer_disabled = False
        
        self._pre_write_buffer: List[Tuple[BufferItem, float]] = []
        self._pre_write_done = False
        self._PRE_WRITE_MIN_FRAMES = 12
        self._PRE_WRITE_MIN_SPAN_SEC = 1.5
        self._measured_fps: Optional[float] = None
        self._fps_measurement_done = False
        
        self._safe_identification_window_sec = self._delay_sec * 2.0
        
        self._face_cascade: Optional[Any] = None
        
        log.info(
            "PrivacyPipeline initialized (M5) | enabled=%s, delay_sec=%.1f, redaction_style=%s, audit=%s, output_dir=%s",
            self._enabled,
            self._delay_sec,
            self._redaction_style,
            audit_enabled,
            self._output_dir,
        )
        log.info(
            "M2 DelayBuffer: max_frames=%d | Writer: fps=%d, codec=%s",
            max_buffer_frames,
            self._output_fps,
            self._output_codec,
        )
        log.info(
            "M4 Segmentation: enabled=%s, backend=%s",
            self._segmentation_enabled,
            getattr(segmentation_cfg, "backend", "none") if segmentation_cfg else "none",
        )
        log.info(
            "M5 Stabilization: enabled=%s, method=%s",
            self._stabilization_enabled,
            getattr(stabilization_cfg, "method", "union_only") if stabilization_cfg else "union_only",
        )
        log.info(
            "M6 Metrics Engine: enabled=%s, leakage=%s, timing=%s, flicker=%s",
            self._metrics_enabled,
            getattr(metrics_cfg, "leakage_enabled", False) if metrics_cfg else False,
            getattr(metrics_cfg, "timing_enabled", False) if metrics_cfg else False,
            getattr(metrics_cfg, "flicker_enabled", False) if metrics_cfg else False,
        )
        
        if self._audit_writer.file_path:
            log.info("Audit log path: %s", self._audit_writer.file_path)
    
    def ingest(
        self,
        frame: "Frame",
        tracks: List["Tracklet"],
        decisions: List["IdentityDecision"],
    ) -> None:
        try:
            self._frame_count += 1
            self._frames_ingested += 1
            
            frame_perf_ts = getattr(frame, "ts", time.perf_counter())
            frame_id = getattr(frame, "frame_id", self._frame_count)
            
            frame_ts = self._init_wall_ts + (frame_perf_ts - self._init_perf_ts)
            
            current_wall_ts = time.time()
            
            decision_map: Dict[int, Any] = {}
            for dec in decisions:
                tid = getattr(dec, "track_id", None)
                if tid is not None:
                    decision_map[tid] = dec
            
            track_bbox_map: Dict[int, List[float]] = {}
            for track in tracks:
                tid = getattr(track, "track_id", -1)
                bbox = getattr(track, "last_box", None) or getattr(track, "bbox", None)
                if bbox is not None and tid >= 0:
                    track_bbox_map[tid] = list(bbox)
            
            track_ids_present: Set[int] = set(track_bbox_map.keys())
            
            if self._frame_count % 50 == 1:
                log.debug(
                    "M3 FSM input: tracks=%d, decisions=%d, track_ids=%s",
                    len(tracks), len(decisions), list(track_ids_present)[:5],
                )
            
            policy_actions = self._policy_fsm.update(
                frame_ts=current_wall_ts,
                track_ids_present=track_ids_present,
                decisions=decisions,
            )
            
            if self._metrics_enabled and self._metrics_engine:
                try:
                    policy_info_by_track: Dict[int, Dict[str, Any]] = {}
                    if policy_actions:
                        for tid, action in policy_actions.items():
                            policy_info_by_track[tid] = {
                                "policy_state": "AUTHORIZED_LOCKED_REDACT" if action == PolicyAction.REDACT else "UNKNOWN_VISIBLE"
                            }
                    
                    self._metrics_engine.on_ingest(
                        frame_ts=current_wall_ts,
                        frame_id=frame_id,
                        track_ids_present=track_ids_present,
                        policy_info_by_track=policy_info_by_track,
                    )
                except Exception as e:
                    log.debug("M6 on_ingest failed (ignored): %s", e)
            
            mask_results: Dict[int, MaskResult] = {}
            if self._segmentation_enabled and tracks:
                image = getattr(frame, "image", None)
                if image is not None:
                    mask_results = self._segmenter.segment(image, tracks)
            
            stabilization_results: Dict[int, StabilizationResult] = {}
            if self._stabilization_enabled and self._segmentation_enabled:
                for track_id in track_ids_present:
                    if track_id not in track_bbox_map:
                        continue
                    
                    mask_result = mask_results.get(track_id)
                    raw_mask = mask_result.mask if mask_result and mask_result.is_valid else None
                    
                    current_bbox = track_bbox_map.get(track_id)
                    if current_bbox is not None:
                        current_bbox = tuple(current_bbox)
                    
                    stab_result = self._stabilizer.update(track_id, raw_mask, current_wall_ts, current_bbox=current_bbox)
                    stabilization_results[track_id] = stab_result
                    
                    is_redact = policy_actions and policy_actions.get(track_id) == PolicyAction.REDACT
                    if is_redact:
                        if stab_result.is_stable:
                            self._stable_mask_uses += 1
                        if stab_result.shrink_detected:
                            self._shrink_detections += 1
                        if stab_result.ttl_reuse:
                            self._ttl_reuses += 1
            
            if self._metrics_enabled and self._metrics_engine:
                try:
                    for track_id in track_ids_present:
                        raw_mask = None
                        mask_result = mask_results.get(track_id)
                        if mask_result and mask_result.is_valid:
                            raw_mask = mask_result.mask
                        
                        stable_mask = None
                        stab_result = stabilization_results.get(track_id)
                        if stab_result and stab_result.is_valid and stab_result.mask is not None:
                            stable_mask = stab_result.mask
                        
                        bbox = None
                        for t in tracks:
                            if getattr(t, "track_id", -1) == track_id:
                                bbox = getattr(t, "bbox", None)
                                break
                        
                        self._metrics_engine.on_masks(
                            frame_ts=current_wall_ts,
                            frame_id=frame_id,
                            track_id=track_id,
                            raw_mask=raw_mask,
                            stable_mask=stable_mask,
                            bbox=bbox,
                        )
                except Exception as e:
                    log.debug("M6 on_masks failed (ignored): %s", e)
            
            track_entries: List[Dict[str, Any]] = []
            
            for track in tracks:
                track_id = getattr(track, "track_id", -1)
                dec = decision_map.get(track_id)
                
                if dec is not None:
                    identity_id = getattr(dec, "identity_id", None)
                    category = getattr(dec, "category", "unknown")
                    binding_state = getattr(dec, "binding_state", None)
                    id_source = getattr(dec, "id_source", "U")
                    
                    if id_source == "U" or id_source is None:
                        extra = getattr(dec, "extra", None)
                        if extra and isinstance(extra, dict):
                            id_source = extra.get("id_source", "U")
                else:
                    identity_id = None
                    category = "unknown"
                    binding_state = None
                    id_source = "U"
                
                authorized_signal = self._is_authorized(category, binding_state)
                
                policy_info = self._policy_fsm.get_track_policy_info(track_id, current_wall_ts)
                policy_state = policy_info.get("policy_state", "UNKNOWN_VISIBLE")
                is_redacted = policy_info.get("is_redacted", False)
                lock_age_sec = policy_info.get("lock_age_sec")
                grace_remaining_sec = policy_info.get("grace_remaining_sec")
                transition_count = policy_info.get("transition_count", 0)
                
                mask_result = mask_results.get(track_id)
                mask_used = False
                mask_source = None
                mask_quality = None
                fallback_to_bbox = False
                
                stab_result = stabilization_results.get(track_id)
                stable_mask_used = False
                stabilization_method = None
                shrink_detected = False
                ttl_reuse = False
                mask_area = None
                stable_mask_area = None
                
                _mask_method = "mask_silhouette" if self._redaction_style == "silhouette" else "mask_blur"
                _bbox_method = "bbox_blur"
                
                if is_redacted:
                    if stab_result is not None and stab_result.is_valid:
                        redaction_method = _mask_method
                        mask_used = True
                        stable_mask_used = True
                        stabilization_method = stab_result.method_used
                        shrink_detected = stab_result.shrink_detected
                        ttl_reuse = stab_result.ttl_reuse
                        mask_area = stab_result.original_area
                        stable_mask_area = stab_result.stable_area
                        if mask_result and mask_result.is_valid:
                            mask_source = mask_result.source.value
                            mask_quality = mask_result.quality_score
                        else:
                            mask_source = "ttl_cache" if ttl_reuse else "stabilizer"
                    elif mask_result is not None and mask_result.is_valid:
                        redaction_method = _mask_method
                        mask_used = True
                        mask_source = mask_result.source.value
                        mask_quality = mask_result.quality_score
                    else:
                        redaction_method = _bbox_method
                        if self._segmentation_enabled and policy_actions.get(track_id) == PolicyAction.REDACT:
                            fallback_to_bbox = True
                            mask_source = "none"
                else:
                    redaction_method = "none"
                
                track_entry = build_track_audit_entry(
                    track_id=track_id,
                    policy_state=policy_state,
                    redaction_method=redaction_method,
                    identity_id=identity_id,
                    id_source=id_source or "U",
                    authorized_signal=authorized_signal,
                    decision_category=category or "unknown",
                    decision_binding_state=binding_state,
                    is_redacted=is_redacted,
                    lock_age_sec=lock_age_sec,
                    grace_remaining_sec=grace_remaining_sec,
                    transition_count=transition_count,
                    mask_used=mask_used,
                    mask_source=mask_source,
                    mask_quality=mask_quality,
                    fallback_to_bbox=fallback_to_bbox,
                    stable_mask_used=stable_mask_used,
                    stabilization_method=stabilization_method,
                    shrink_detected=shrink_detected,
                    ttl_reuse=ttl_reuse,
                    mask_area=mask_area,
                    stable_mask_area=stable_mask_area,
                    bbox=track_bbox_map.get(track_id),
                )
                track_entries.append(track_entry)
            
            privacy_frame = self._generate_privacy_frame(
                frame, tracks, policy_actions, track_bbox_map, mask_results, stabilization_results, current_wall_ts
            )
            
            if privacy_frame is None:
                log.warning("Failed to generate privacy frame for frame_id=%d", frame_id)
                return
            
            audit_payload = {
                "track_entries": track_entries,
            }
            
            raw_image = getattr(frame, "image", None)
            rendering_data = {
                "track_ids_present": track_ids_present,
                "track_bbox_map": dict(track_bbox_map),
                "mask_results": mask_results,
                "stab_results": stabilization_results,
            }
            
            dropped = self._delay_buffer.push(
                frame_id=frame_id,
                ingest_ts=current_wall_ts,
                privacy_frame=privacy_frame,
                audit_payload=audit_payload,
                original_frame_ts=frame_ts,
                raw_image=raw_image,
                rendering_data=rendering_data,
            )
            
            if dropped > 0:
                self._frames_dropped += dropped
            
            emitted_items = self._delay_buffer.pop_eligible(current_wall_ts)
            
            for item in emitted_items:
                self._process_emitted_frame(item, current_wall_ts)
            
        except Exception as e:
            log.exception("Privacy pipeline ingest failed (continuing): %s", e)
    
    def _process_emitted_frame(self, item: BufferItem, emit_ts: float) -> None:
        try:
            self._frames_emitted += 1
            
            if not self._pre_write_done:
                self._pre_write_buffer.append((item, emit_ts))
                
                if item.privacy_frame is not None:
                    self._latest_privacy_frame = item.privacy_frame
                
                if len(self._pre_write_buffer) >= self._PRE_WRITE_MIN_FRAMES:
                    ts_list = [it.ingest_ts for it, _ in self._pre_write_buffer]
                    if len(ts_list) > 1:
                        span = ts_list[-1] - ts_list[0]
                        if span >= self._PRE_WRITE_MIN_SPAN_SEC:
                            self._flush_pre_write_buffer()
                return
            
            self._render_and_write_frame(item, emit_ts)
            
        except Exception as e:
            log.exception("Failed to process emitted frame %d: %s", item.frame_id, e)
    
    
    def _flush_pre_write_buffer(self) -> None:
        if not self._pre_write_buffer:
            self._pre_write_done = True
            return
        
        try:
            ts_list = [item.ingest_ts for item, _ in self._pre_write_buffer]
            span = ts_list[-1] - ts_list[0] if len(ts_list) > 1 else 0.0
            if span > 0.1 and len(ts_list) > 1:
                self._measured_fps = (len(ts_list) - 1) / span
                self._writer.set_fps(self._measured_fps)
                log.info(
                    "Pre-write FPS measured: %.2f fps from %d frames over %.2fs (config was %d)",
                    self._measured_fps, len(ts_list), span, self._output_fps,
                )
            else:
                log.warning(
                    "Pre-write FPS: insufficient data (frames=%d, span=%.2fs); "
                    "using config default %d fps",
                    len(ts_list), span, self._output_fps,
                )
            
            self._interpolate_missing_tracks()
            
            log.info(
                "Flushing pre-write buffer: %d frames "
                "(re-rendering with current FSM for retroactive + protective redaction)",
                len(self._pre_write_buffer),
            )
            for buf_item, buf_emit_ts in self._pre_write_buffer:
                self._render_and_write_frame(buf_item, buf_emit_ts)
            
        except Exception as e:
            log.exception("Pre-write buffer flush failed: %s", e)
        finally:
            self._pre_write_buffer.clear()
            self._pre_write_done = True
            self._fps_measurement_done = True
    
    def _interpolate_missing_tracks(self) -> None:
        n = len(self._pre_write_buffer)
        if n == 0:
            return
        
        for i in range(n):
            item, _ = self._pre_write_buffer[i]
            rd = item.rendering_data
            if rd is None:
                continue
            track_ids = rd.get("track_ids_present", set())
            if track_ids:
                continue
            
            nearest_rd = None
            for offset in range(1, n):
                for k in [i + offset, i - offset]:
                    if 0 <= k < n:
                        other_item, _ = self._pre_write_buffer[k]
                        other_rd = other_item.rendering_data
                        if other_rd and other_rd.get("track_ids_present"):
                            nearest_rd = other_rd
                            break
                if nearest_rd:
                    break
            
            if nearest_rd is not None:
                rd["interpolated_track_ids"] = nearest_rd.get("track_ids_present", set())
                rd["interpolated_bbox_map"] = nearest_rd.get("track_bbox_map", {})
                rd["interpolated_mask_results"] = nearest_rd.get("mask_results", {})
                rd["interpolated_stab_results"] = nearest_rd.get("stab_results", {})
                log.debug(
                    "Interpolated tracks for frame %d from neighbour (tracks=%s)",
                    item.frame_id, list(rd["interpolated_track_ids"])[:3],
                )
    
    
    def _render_and_write_frame(self, item: BufferItem, emit_ts: float) -> None:
        try:
            final_frame = None
            retroactive_track_ids = []
            
            if item.raw_image is not None and item.rendering_data is not None:
                final_frame, retroactive_track_ids = self._rerender_for_emission(item, emit_ts)
            
            if final_frame is None:
                final_frame = item.privacy_frame
            
            buffer_depth = self._delay_buffer.get_buffer_depth()
            lag_sec = emit_ts - item.ingest_ts
            
            audit_entry = build_audit_entry(
                frame_ts=item.original_frame_ts,
                frame_id=item.frame_id,
                emit_ts=emit_ts,
                track_entries=item.audit_payload.get("track_entries", []),
            )
            
            audit_entry["buffer_depth"] = buffer_depth
            audit_entry["lag_sec"] = round(lag_sec, 3)
            audit_entry["frames_emitted_total"] = self._frames_emitted
            
            if retroactive_track_ids:
                audit_entry["retroactive_redacted"] = retroactive_track_ids
                audit_entry["retroactive_count"] = len(retroactive_track_ids)
            
            self._audit_writer.write_entry(audit_entry)
            
            if not self._writer_disabled:
                success = self._writer.write(final_frame)
                if not success and self._writer.is_failed:
                    log.error("PrivacyWriter failed - disabling for remainder of run")
                    self._writer_disabled = True
            
            self._latest_privacy_frame = final_frame
            
            if self._metrics_enabled and self._metrics_engine:
                try:
                    track_entries = item.audit_payload.get("track_entries", [])
                    per_track_redaction_info: Dict[int, Dict[str, Any]] = {}
                    redacted_regions: List[Dict[str, Any]] = []
                    
                    for entry in track_entries:
                        tid = entry.get("track_id")
                        if tid is not None:
                            action = entry.get("policy_action", "REDACT")
                            is_redacted = action == "REDACT"
                            per_track_redaction_info[tid] = {
                                "is_redacted": is_redacted,
                                "track_id": tid,
                            }
                            
                            if is_redacted:
                                bbox = entry.get("bbox")
                                if bbox:
                                    redacted_regions.append({
                                        "track_id": tid,
                                        "bbox": bbox,
                                    })
                    
                    self._metrics_engine.on_emit(
                        emit_ts=emit_ts,
                        frame_id=item.frame_id,
                        privacy_frame=final_frame,
                        per_track_redaction_info=per_track_redaction_info,
                        redacted_regions=redacted_regions,
                    )
                except Exception as e:
                    log.debug("M6 on_emit failed (ignored): %s", e)
            
        except Exception as e:
            log.exception("Failed to render/write frame %d: %s", item.frame_id, e)
    
    
    def _should_redact_at_emission(
        self, track_id: int, emit_ts: float
    ) -> Tuple[bool, bool]:
        policy_info = self._policy_fsm.get_track_policy_info(track_id, emit_ts)
        is_redacted = policy_info.get("is_redacted", False)
        
        if is_redacted:
            return True, False
        
        first_seen_ts = policy_info.get("first_seen_ts")
        if first_seen_ts is not None and first_seen_ts > 0:
            track_age = emit_ts - first_seen_ts
            if track_age < self._safe_identification_window_sec:
                return True, True
        
        return False, False
    
    def _detect_faces_emergency(self, image: np.ndarray) -> List[List[int]]:
        try:
            if self._face_cascade is None:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
                if self._face_cascade.empty():
                    log.warning("Haar cascade failed to load; emergency face detection disabled")
                    return []
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            detections = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.15, minNeighbors=4, minSize=(30, 30),
            )
            
            bboxes: List[List[int]] = []
            for (x, y, w, h) in detections:
                cx, cy = x + w // 2, y + h // 2
                ew, eh = int(w * 1.5), int(h * 2.0)
                x1 = max(0, cx - ew // 2)
                y1 = max(0, cy - eh // 3)
                x2 = min(image.shape[1], cx + ew // 2)
                y2 = min(image.shape[0], cy + eh * 2 // 3)
                bboxes.append([x1, y1, x2, y2])
            return bboxes
        except Exception as e:
            log.debug("Emergency face detection failed: %s", e)
            return []
    
    def _rerender_for_emission(
        self,
        item: BufferItem,
        emit_ts: float,
    ) -> Tuple[Optional[np.ndarray], List[int]]:
        try:
            raw_image = item.raw_image
            rd = item.rendering_data
            
            if raw_image is None or rd is None:
                return None, []
            
            track_ids_present = rd.get("track_ids_present", set())
            track_bbox_map = rd.get("track_bbox_map", {})
            mask_results = rd.get("mask_results", {})
            stab_results = rd.get("stab_results", {})
            
            privacy_frame = raw_image.copy()
            
            retroactive_track_ids: List[int] = []
            protective_track_ids: List[int] = []
            stable_count = 0
            raw_count = 0
            bbox_count = 0
            
            for track_id in track_ids_present:
                needs_redaction, is_protective = self._should_redact_at_emission(
                    track_id, emit_ts,
                )
                
                if not needs_redaction:
                    continue
                
                if is_protective:
                    protective_track_ids.append(track_id)
                else:
                    original_entries = item.audit_payload.get("track_entries", [])
                    was_redacted_at_ingest = False
                    for entry in original_entries:
                        if entry.get("track_id") == track_id:
                            was_redacted_at_ingest = entry.get("is_redacted", False)
                            break
                    if not was_redacted_at_ingest:
                        retroactive_track_ids.append(track_id)
                
                bbox = track_bbox_map.get(track_id)
                
                stab_result = stab_results.get(track_id)
                if stab_result is not None and stab_result.is_valid:
                    self._apply_mask_redaction(privacy_frame, stab_result.mask, bbox=bbox)
                    stable_count += 1
                else:
                    mask_result = mask_results.get(track_id)
                    if mask_result is not None and mask_result.is_valid:
                        self._apply_mask_redaction(privacy_frame, mask_result.mask, bbox=bbox)
                        raw_count += 1
                    else:
                        if bbox is not None:
                            self._apply_bbox_redaction(privacy_frame, bbox)
                            bbox_count += 1
            
            interpolated_track_ids = rd.get("interpolated_track_ids", set())
            interpolated_bbox_map = rd.get("interpolated_bbox_map", {})
            interpolated_mask_results = rd.get("interpolated_mask_results", {})
            interpolated_stab_results = rd.get("interpolated_stab_results", {})
            
            for track_id in interpolated_track_ids:
                if track_id in track_ids_present:
                    continue
                
                needs_redaction, is_protective = self._should_redact_at_emission(
                    track_id, emit_ts,
                )
                if not needs_redaction:
                    continue
                
                bbox = interpolated_bbox_map.get(track_id)
                stab_result = interpolated_stab_results.get(track_id)
                mask_result = interpolated_mask_results.get(track_id)
                
                if stab_result is not None and stab_result.is_valid:
                    self._apply_mask_redaction(privacy_frame, stab_result.mask, bbox=bbox)
                    stable_count += 1
                elif mask_result is not None and mask_result.is_valid:
                    self._apply_mask_redaction(privacy_frame, mask_result.mask, bbox=bbox)
                    raw_count += 1
                elif bbox is not None:
                    self._apply_bbox_redaction(privacy_frame, bbox)
                    bbox_count += 1
                
                if is_protective:
                    protective_track_ids.append(track_id)
                else:
                    retroactive_track_ids.append(track_id)
            
            if not track_ids_present and not interpolated_track_ids:
                has_authorized = any(
                    ts_obj.state != PolicyState.UNKNOWN_VISIBLE
                    for ts_obj in self._policy_fsm._track_states.values()
                )
                if has_authorized:
                    face_bboxes = self._detect_faces_emergency(privacy_frame)
                    for fb in face_bboxes:
                        self._apply_bbox_redaction(privacy_frame, fb)
                        bbox_count += 1
                    if face_bboxes:
                        log.info(
                            "Emergency face redaction: frame_id=%d, faces=%d (no tracks, Haar fallback)",
                            item.frame_id, len(face_bboxes),
                        )
            
            all_retroactive = retroactive_track_ids + protective_track_ids
            if retroactive_track_ids:
                log.info(
                    "Retroactive redaction: frame_id=%d, tracks=%s (identified after capture)",
                    item.frame_id, retroactive_track_ids,
                )
            if protective_track_ids:
                log.info(
                    "Protective redaction: frame_id=%d, tracks=%s (still in identification window)",
                    item.frame_id, protective_track_ids,
                )
            
            if self._show_watermark:
                current_policy_actions = {}
                all_track_ids = track_ids_present | interpolated_track_ids
                for tid in all_track_ids:
                    needs, _ = self._should_redact_at_emission(tid, emit_ts)
                    if needs:
                        current_policy_actions[tid] = PolicyAction.REDACT
                    else:
                        current_policy_actions[tid] = PolicyAction.VISIBLE
                
                self._draw_watermark_m5(
                    privacy_frame,
                    current_policy_actions,
                    mask_results,
                    stab_results,
                )
            
            return privacy_frame, all_retroactive
            
        except Exception as e:
            log.warning("Retroactive re-rendering failed for frame %d: %s", item.frame_id, e)
            return None, []
    
    def _is_authorized(self, category: Optional[str], binding_state: Optional[str]) -> bool:
        if category is None:
            return False
        
        category_authorized = category in self._authorized_categories
        
        if not category_authorized:
            return False
        
        if self._require_confirmed_binding:
            if binding_state is None:
                return False
            confirmation_satisfied = binding_state in ("CONFIRMED_WEAK", "CONFIRMED_STRONG")
            return confirmation_satisfied
        else:
            return True
    
    def _generate_privacy_frame(
        self,
        frame: "Frame",
        tracks: Optional[List["Tracklet"]] = None,
        policy_actions: Optional[Dict[int, PolicyAction]] = None,
        track_bbox_map: Optional[Dict[int, List[float]]] = None,
        mask_results: Optional[Dict[int, MaskResult]] = None,
        stabilization_results: Optional[Dict[int, StabilizationResult]] = None,
        frame_ts: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        try:
            image = getattr(frame, "image", None)
            if image is None:
                return None
            
            privacy_frame = image.copy()
            
            if policy_actions and track_bbox_map:
                stable_mask_count = 0
                raw_mask_count = 0
                bbox_count = 0
                
                for track_id, action in policy_actions.items():
                    if action == PolicyAction.REDACT:
                        if track_id not in track_bbox_map:
                            continue
                        
                        bbox = track_bbox_map.get(track_id)
                        
                        stab_result = stabilization_results.get(track_id) if stabilization_results else None
                        
                        if stab_result is not None and stab_result.is_valid:
                            self._apply_mask_redaction(privacy_frame, stab_result.mask, bbox=bbox)
                            stable_mask_count += 1
                            self._mask_redactions += 1
                        else:
                            mask_result = mask_results.get(track_id) if mask_results else None
                            
                            if mask_result is not None and mask_result.is_valid:
                                self._apply_mask_redaction(privacy_frame, mask_result.mask, bbox=bbox)
                                raw_mask_count += 1
                                self._mask_redactions += 1
                            else:
                                if bbox is not None:
                                    self._apply_bbox_redaction(privacy_frame, bbox)
                                    bbox_count += 1
                                    if self._segmentation_enabled:
                                        self._bbox_fallbacks += 1
                
                if stable_mask_count > 0 or raw_mask_count > 0 or bbox_count > 0:
                    log.debug(
                        "M5: Applied redaction - stable_masks=%d, raw_masks=%d, bbox_fallback=%d",
                        stable_mask_count, raw_mask_count, bbox_count,
                    )
            
            if self._show_watermark:
                self._draw_watermark_m5(privacy_frame, policy_actions, mask_results, stabilization_results)
            
            return privacy_frame
            
        except Exception as e:
            log.warning("Failed to generate privacy frame: %s", e)
            return None
    

    def _apply_mask_redaction(self, image: np.ndarray, mask: np.ndarray, bbox: Optional[List[float]] = None) -> None:
        if self._redaction_style == "silhouette":
            self._apply_mask_silhouette(image, mask, bbox=bbox)
        else:
            self._apply_mask_blur(image, mask)

    def _apply_bbox_redaction(self, image: np.ndarray, bbox: List[float]) -> None:
        self._apply_bbox_blur(image, bbox)


    def _apply_mask_silhouette(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple = (0, 0, 0),
        feather_px: int = 0,
        bbox: Optional[List[float]] = None,
    ) -> None:
        try:
            h, w = image.shape[:2]
            
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            _BBOX_PAD = 10
            
            if bbox is not None:
                bx1 = max(0, int(bbox[0]) - _BBOX_PAD)
                by1 = max(0, int(bbox[1]) - _BBOX_PAD)
                bx2 = min(w, int(bbox[2]) + _BBOX_PAD)
                by2 = min(h, int(bbox[3]) + _BBOX_PAD)
                
                if bx2 <= bx1 or by2 <= by1:
                    return
                
                search_mask = mask[by1:by2, bx1:bx2]
                offset_x, offset_y = bx1, by1
            else:
                search_mask = mask
                offset_x, offset_y = 0, 0
            
            if self._silhouette_cleanup and self._silhouette_cleanup_kernel is not None:
                search_mask = self._cleanup_mask_roi(search_mask)
            
            search_mask = self._refine_silhouette_contours(search_mask)
            
            ys, xs = np.where(search_mask > 127)
            if len(xs) == 0 or len(ys) == 0:
                return
            
            sx1, sx2 = int(xs.min()), int(xs.max()) + 1
            sy1, sy2 = int(ys.min()), int(ys.max()) + 1
            
            x1 = offset_x + sx1
            y1 = offset_y + sy1
            x2 = offset_x + sx2
            y2 = offset_y + sy2
            
            mask_roi = search_mask[sy1:sy2, sx1:sx2]
            
            if feather_px > 0:
                k = feather_px * 2 + 1
                mask_soft = cv2.GaussianBlur(
                    mask_roi.astype(np.float32) / 255.0,
                    (k, k), 0,
                )
                if len(image.shape) == 3:
                    mask_soft = mask_soft[:, :, np.newaxis]
                
                roi = image[y1:y2, x1:x2].astype(np.float32)
                fill = np.full_like(roi, color, dtype=np.float32)
                blended = (fill * mask_soft + roi * (1.0 - mask_soft)).astype(np.uint8)
                image[y1:y2, x1:x2] = blended
            else:
                mask_bool = mask_roi > 127
                roi = image[y1:y2, x1:x2]
                roi[mask_bool] = color
            
        except Exception as e:
            log.warning("Failed to apply mask silhouette: %s", e)

    def _cleanup_mask_roi(self, mask_roi: np.ndarray) -> np.ndarray:
        try:
            cleaned = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, self._silhouette_cleanup_kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self._silhouette_cleanup_kernel)
            return cleaned
        except Exception:
            return mask_roi

    def _refine_silhouette_contours(self, mask_roi: np.ndarray) -> np.ndarray:
        try:
            h, w = mask_roi.shape[:2]
            if h == 0 or w == 0:
                return mask_roi
            
            binary = np.where(mask_roi > 127, np.uint8(255), np.uint8(0))
            
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
            
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(closed, dilate_kernel, iterations=1)
            
            blurred = cv2.GaussianBlur(dilated, (7, 7), sigmaX=2.0)
            
            smooth = np.where(blurred > 80, np.uint8(255), np.uint8(0))
            
            contours, _ = cv2.findContours(
                smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return mask_roi
            
            min_area = max(100, int(h * w * 0.005))
            
            refined = np.zeros((h, w), dtype=np.uint8)
            kept = 0
            
            for cnt in contours:
                if cv2.contourArea(cnt) < min_area:
                    continue
                
                cv2.drawContours(refined, [cnt], -1, 255, cv2.FILLED)
                kept += 1
            
            if kept == 0:
                return mask_roi
            
            return refined
            
        except Exception:
            return mask_roi


    def _apply_bbox_blur(self, image: np.ndarray, bbox: List[float]) -> None:
        try:
            h, w = image.shape[:2]
            
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))
            
            if x2 <= x1 or y2 <= y1:
                return
            
            roi = image[y1:y2, x1:x2]
            
            region_size = max(x2 - x1, y2 - y1)
            kernel_size = max(51, (region_size // 4) | 1)
            kernel_size = min(kernel_size, 101)
            
            blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
            
            image[y1:y2, x1:x2] = blurred
            
        except Exception as e:
            log.warning("Failed to apply bbox blur: %s", e)
    
    def _apply_mask_blur(self, image: np.ndarray, mask: np.ndarray) -> None:
        try:
            h, w = image.shape[:2]
            
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            ys, xs = np.where(mask > 127)
            if len(xs) == 0 or len(ys) == 0:
                return
            
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            
            region_size = max(x2 - x1, y2 - y1)
            kernel_size = max(51, (region_size // 4) | 1)
            kernel_size = min(kernel_size, 101)
            
            roi = image[y1:y2, x1:x2].copy()
            roi_blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
            
            mask_roi = mask[y1:y2, x1:x2]
            
            mask_float = (mask_roi > 127).astype(np.float32)
            if len(roi.shape) == 3:
                mask_float = mask_float[:, :, np.newaxis]
            
            blended_roi = (roi_blurred * mask_float + roi * (1 - mask_float)).astype(np.uint8)
            
            image[y1:y2, x1:x2] = blended_roi
            
        except Exception as e:
            log.warning("Failed to apply mask blur: %s", e)
    
    def _draw_watermark_m4(
        self,
        image: np.ndarray,
        policy_actions: Optional[Dict[int, PolicyAction]] = None,
        mask_results: Optional[Dict[int, MaskResult]] = None,
    ) -> None:
        try:
            h, w = image.shape[:2]
            
            redact_count = 0
            visible_count = 0
            mask_count = 0
            bbox_fallback_count = 0
            
            if policy_actions:
                for track_id, action in policy_actions.items():
                    if action == PolicyAction.REDACT:
                        redact_count += 1
                        mask_result = mask_results.get(track_id) if mask_results else None
                        if mask_result is not None and mask_result.is_valid:
                            mask_count += 1
                        else:
                            bbox_fallback_count += 1
                    else:
                        visible_count += 1
            
            if self._segmentation_enabled:
                text1 = "PRIVACY OUTPUT (M4 SEGMENTATION)"
                color = (255, 128, 0)
            else:
                text1 = "PRIVACY OUTPUT (M4 POLICY)"
                color = (0, 200, 255)
            
            text2 = f"Delay: {self._delay_sec:.1f}s | Emitted: {self._frames_emitted}"
            text3 = f"Buffer: {self._delay_buffer.get_buffer_depth()} | Redact: {redact_count} | Visible: {visible_count}"
            
            if self._segmentation_enabled and redact_count > 0:
                text4 = f"Masks: {mask_count} | Bbox Fallback: {bbox_fallback_count}"
            else:
                text4 = None
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            
            y1 = 25
            y2 = 48
            y3 = 71
            y4 = 94 if text4 else 0
            
            bg_height = 105 if text4 else 82
            cv2.rectangle(image, (5, 5), (450, bg_height), (0, 0, 0), -1)
            
            cv2.putText(image, text1, (10, y1), font, scale, color, thickness)
            cv2.putText(image, text2, (10, y2), font, scale * 0.85, color, thickness - 1)
            cv2.putText(image, text3, (10, y3), font, scale * 0.85, color, thickness - 1)
            if text4:
                cv2.putText(image, text4, (10, y4), font, scale * 0.85, color, thickness - 1)
            
        except Exception as e:
            log.warning("Failed to draw M4 watermark: %s", e)
    
    def _draw_watermark_m5(
        self,
        image: np.ndarray,
        policy_actions: Optional[Dict[int, PolicyAction]] = None,
        mask_results: Optional[Dict[int, MaskResult]] = None,
        stabilization_results: Optional[Dict[int, StabilizationResult]] = None,
    ) -> None:
        try:
            h, w = image.shape[:2]
            
            redact_count = 0
            visible_count = 0
            stable_mask_count = 0
            raw_mask_count = 0
            bbox_fallback_count = 0
            shrink_count = 0
            ttl_count = 0
            
            present_track_ids = set()
            if stabilization_results:
                present_track_ids = set(stabilization_results.keys())
            elif mask_results:
                present_track_ids = set(mask_results.keys())
            
            if policy_actions:
                for track_id, action in policy_actions.items():
                    if action == PolicyAction.REDACT:
                        if present_track_ids and track_id not in present_track_ids:
                            continue
                        
                        redact_count += 1
                        
                        stab_result = stabilization_results.get(track_id) if stabilization_results else None
                        if stab_result is not None and stab_result.is_valid:
                            stable_mask_count += 1
                            if stab_result.shrink_detected:
                                shrink_count += 1
                            if stab_result.ttl_reuse:
                                ttl_count += 1
                        else:
                            mask_result = mask_results.get(track_id) if mask_results else None
                            if mask_result is not None and mask_result.is_valid:
                                raw_mask_count += 1
                            else:
                                bbox_fallback_count += 1
                    else:
                        visible_count += 1
            
            if self._stabilization_enabled and self._segmentation_enabled:
                text1 = "PRIVACY OUTPUT (M5 STABILIZED)"
                color = (0, 255, 128)
            elif self._segmentation_enabled:
                text1 = "PRIVACY OUTPUT (M4 SEGMENTATION)"
                color = (255, 128, 0)
            else:
                text1 = "PRIVACY OUTPUT (M5 POLICY)"
                color = (0, 200, 255)
            
            text2 = f"Delay: {self._delay_sec:.1f}s | Emitted: {self._frames_emitted}"
            text3 = f"Buffer: {self._delay_buffer.get_buffer_depth()} | Redact: {redact_count} | Visible: {visible_count}"
            
            if self._stabilization_enabled and self._segmentation_enabled and redact_count > 0:
                text4 = f"Stable: {stable_mask_count} | Raw: {raw_mask_count} | Bbox: {bbox_fallback_count}"
                text5 = f"Shrink: {shrink_count} | TTL: {ttl_count}" if (shrink_count > 0 or ttl_count > 0) else None
            elif self._segmentation_enabled and redact_count > 0:
                text4 = f"Masks: {stable_mask_count + raw_mask_count} | Bbox Fallback: {bbox_fallback_count}"
                text5 = None
            else:
                text4 = None
                text5 = None
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            
            y1 = 25
            y2 = 48
            y3 = 71
            y4 = 94 if text4 else 0
            y5 = 117 if text5 else 0
            
            if text5:
                bg_height = 128
            elif text4:
                bg_height = 105
            else:
                bg_height = 82
            cv2.rectangle(image, (5, 5), (470, bg_height), (0, 0, 0), -1)
            
            cv2.putText(image, text1, (10, y1), font, scale, color, thickness)
            cv2.putText(image, text2, (10, y2), font, scale * 0.85, color, thickness - 1)
            cv2.putText(image, text3, (10, y3), font, scale * 0.85, color, thickness - 1)
            if text4:
                cv2.putText(image, text4, (10, y4), font, scale * 0.85, color, thickness - 1)
            if text5:
                cv2.putText(image, text5, (10, y5), font, scale * 0.85, color, thickness - 1)
            
        except Exception as e:
            log.warning("Failed to draw M5 watermark: %s", e)
    
    def _draw_watermark_m3(
        self,
        image: np.ndarray,
        policy_actions: Optional[Dict[int, PolicyAction]] = None,
    ) -> None:
        try:
            h, w = image.shape[:2]
            
            redact_count = 0
            visible_count = 0
            if policy_actions:
                for action in policy_actions.values():
                    if action == PolicyAction.REDACT:
                        redact_count += 1
                    else:
                        visible_count += 1
            
            text1 = "PRIVACY OUTPUT (M3 POLICY)"
            text2 = f"Delay: {self._delay_sec:.1f}s | Emitted: {self._frames_emitted}"
            text3 = f"Buffer: {self._delay_buffer.get_buffer_depth()} | Redact: {redact_count} | Visible: {visible_count}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            color = (0, 200, 255)
            
            y1 = 25
            y2 = 48
            y3 = 71
            
            cv2.rectangle(image, (5, 5), (410, 82), (0, 0, 0), -1)
            
            cv2.putText(image, text1, (10, y1), font, scale, color, thickness)
            cv2.putText(image, text2, (10, y2), font, scale * 0.85, color, thickness - 1)
            cv2.putText(image, text3, (10, y3), font, scale * 0.85, color, thickness - 1)
            
        except Exception as e:
            log.warning("Failed to draw M3 watermark: %s", e)
    
    def _draw_watermark_m2(self, image: np.ndarray) -> None:
        try:
            h, w = image.shape[:2]
            
            text1 = "PRIVACY OUTPUT (M2 DELAYED)"
            text2 = f"Delay: {self._delay_sec:.1f}s | Emitted: {self._frames_emitted}"
            text3 = f"Buffer: {self._delay_buffer.get_buffer_depth()} frames"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            color = (0, 255, 0)
            
            y1 = 25
            y2 = 48
            y3 = 71
            
            cv2.rectangle(image, (5, 5), (340, 82), (0, 0, 0), -1)
            
            cv2.putText(image, text1, (10, y1), font, scale, color, thickness)
            cv2.putText(image, text2, (10, y2), font, scale * 0.85, color, thickness - 1)
            cv2.putText(image, text3, (10, y3), font, scale * 0.85, color, thickness - 1)
            
        except Exception as e:
            log.warning("Failed to draw M2 watermark: %s", e)
    
    def get_latest_privacy_frame(self) -> Optional[np.ndarray]:
        return self._latest_privacy_frame
    
    def shutdown(self) -> None:
        try:
            log.info(
                "PrivacyPipeline shutting down | ingested=%d, emitted=%d, dropped=%d",
                self._frames_ingested,
                self._frames_emitted,
                self._frames_dropped,
            )
            
            remaining = self._delay_buffer.flush_all()
            flush_ts = time.time()
            
            log.info("Flushing %d remaining frames from delay buffer", len(remaining))
            
            for item in remaining:
                self._process_emitted_frame(item, flush_ts)
            
            if not self._pre_write_done and self._pre_write_buffer:
                log.info(
                    "Flushing pre-write buffer at shutdown: %d frames",
                    len(self._pre_write_buffer),
                )
                self._flush_pre_write_buffer()
            
            self._writer.close()
            
            writer_stats = self._writer.get_stats()
            if writer_stats.get("file_path"):
                log.info(
                    "Video written: %s | frames=%d",
                    writer_stats["file_path"],
                    writer_stats["frames_written"],
                )
            
            self._audit_writer.close()
            
            buffer_stats = self._delay_buffer.get_stats()
            log.info(
                "DelayBuffer stats: pushed=%d, emitted=%d, dropped=%d",
                buffer_stats["frames_pushed"],
                buffer_stats["frames_emitted"],
                buffer_stats["frames_dropped"],
            )
            
            fsm_stats = self._policy_fsm.get_stats()
            log.info(
                "PolicyFSM stats: tracks_managed=%d, transitions=%d, locks=%d, unlock_prevented=%d",
                fsm_stats["total_tracks_managed"],
                fsm_stats["total_transitions"],
                fsm_stats["locks_created"],
                fsm_stats["false_unlock_prevented"],
            )
            state_counts = fsm_stats.get("state_counts", {})
            if state_counts:
                log.info("PolicyFSM state counts: %s", state_counts)
            
            if self._segmentation_enabled:
                log.info(
                    "M4 Segmentation stats: mask_redactions=%d, bbox_fallbacks=%d",
                    self._mask_redactions,
                    self._bbox_fallbacks,
                )
            
            if self._stabilization_enabled:
                log.info(
                    "M5 Stabilization stats: stable_mask_uses=%d, shrink_detections=%d, ttl_reuses=%d",
                    self._stable_mask_uses,
                    self._shrink_detections,
                    self._ttl_reuses,
                )
            
            if self._metrics_enabled and self._metrics_engine:
                try:
                    summary = self._metrics_engine.finalize_and_summarize()
                    
                    leakage_events = summary.get("leakage", {}).get("total_leakage_events", 0)
                    mean_time_to_lock = summary.get("timing", {}).get("time_to_lock", {}).get("mean_sec", 0.0)
                    mean_time_to_redact = summary.get("timing", {}).get("time_to_redacted_emit", {}).get("mean_sec", 0.0)
                    mean_flicker_iou = summary.get("flicker", {}).get("iou", {}).get("mean", 0.0)
                    flicker_samples = summary.get("flicker", {}).get("total_iou_samples", 0)
                    utility_redacted_rate = summary.get("utility", {}).get("redacted_rate", 0.0)
                    
                    log.info(
                        "M6 Metrics: leakage_events=%d, mean_time_to_lock=%.3fs, mean_time_to_redact=%.3fs",
                        leakage_events,
                        mean_time_to_lock,
                        mean_time_to_redact,
                    )
                    log.info(
                        "M6 Metrics: mean_flicker_iou=%.3f, flicker_samples=%d, utility_redacted_rate=%.3f",
                        mean_flicker_iou,
                        flicker_samples,
                        utility_redacted_rate,
                    )
                except Exception as e:
                    log.warning("M6 finalize_and_summarize failed (ignored): %s", e)
            
        except Exception as e:
            log.exception("Privacy pipeline shutdown error (continuing): %s", e)
    
    @property
    def frame_count(self) -> int:
        return self._frame_count
    
    @property
    def frames_ingested(self) -> int:
        return self._frames_ingested
    
    @property
    def frames_emitted(self) -> int:
        return self._frames_emitted
    
    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped
    
    @property
    def audit_file_path(self) -> Optional[str]:
        path = self._audit_writer.file_path
        return str(path) if path else None
    
    @property
    def video_file_path(self) -> Optional[str]:
        path = self._writer.file_path
        return str(path) if path else None
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "enabled": self._enabled,
            "delay_sec": self._delay_sec,
            "frames_ingested": self._frames_ingested,
            "frames_emitted": self._frames_emitted,
            "frames_dropped": self._frames_dropped,
            "writer_disabled": self._writer_disabled,
            "buffer": self._delay_buffer.get_stats(),
            "writer": self._writer.get_stats(),
            "segmentation_enabled": self._segmentation_enabled,
            "mask_redactions": self._mask_redactions,
            "bbox_fallbacks": self._bbox_fallbacks,
            "stabilization_enabled": self._stabilization_enabled,
            "stable_mask_uses": self._stable_mask_uses,
            "shrink_detections": self._shrink_detections,
            "ttl_reuses": self._ttl_reuses,
        }
        return stats
