
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import numpy as np

from schemas import Tracklet, IdentityDecision

logger = logging.getLogger(__name__)


@dataclass
class ContinuityMemory:
    track_id: int
    person_id: str
    label: str
    confidence: float
    last_face_ts: float
    lost_at_ts: Optional[float] = None
    last_bbox: Optional[Tuple[float, float, float, float]] = None
    last_embedding: Optional[np.ndarray] = None
    embedding_ema: Optional[np.ndarray] = None
    safe_zone_counter: int = 0
    is_stale: bool = False
    face_contradiction_counter: int = 0
    original_track_id: Optional[int] = None


class ContinuityBinder:
    
    def __init__(self, cfg):
        if hasattr(cfg, 'vision_identity') and hasattr(cfg.vision_identity, 'continuity'):
            continuity_cfg = cfg.vision_identity.continuity
        elif hasattr(cfg, 'continuity'):
            continuity_cfg = cfg.continuity
        else:
            continuity_cfg = {}
        
        self.min_track_age_frames = self._get_config(continuity_cfg, 'min_track_age_frames', 10)
        self.appearance_distance_threshold = self._get_config(continuity_cfg, 'appearance_distance_threshold', 0.35)
        self.appearance_ema_alpha = self._get_config(continuity_cfg, 'appearance_ema_alpha', 0.3)
        self.appearance_safe_zone_frames = self._get_config(continuity_cfg, 'appearance_safe_zone_frames', 5)
        
        displacement_frac = self._get_config(continuity_cfg, 'max_bbox_displacement_fraction', None)
        if displacement_frac is None:
            displacement_frac = self._get_config(continuity_cfg, 'max_bbox_displacement_frac', 0.25)
        self.max_bbox_displacement_frac = displacement_frac
        self.max_bbox_displacement_fraction = displacement_frac
        self.max_bbox_displacement_px = self._get_config(continuity_cfg, 'max_bbox_displacement_px', 600)
        self.min_bbox_overlap = self._get_config(continuity_cfg, 'min_bbox_overlap', 0.1)
        
        health_min_conf = self._get_config(continuity_cfg, 'min_track_confidence', None)
        if health_min_conf is None:
            health_min_conf = self._get_config(continuity_cfg, 'track_health_min_confidence', 0.5)
        self.track_health_min_confidence = health_min_conf
        
        health_max_lost = self._get_config(continuity_cfg, 'max_lost_frames', None)
        if health_max_lost is None:
            health_max_lost = self._get_config(continuity_cfg, 'track_health_max_lost_frames', 2)
        self.track_health_max_lost_frames = health_max_lost
        
        face_contra = self._get_config(continuity_cfg, 'max_face_contradiction_count', None)
        if face_contra is None:
            face_contra = self._get_config(continuity_cfg, 'face_contradiction_threshold', 3)
        self.face_contradiction_threshold = face_contra
        
        grace_window = self._get_config(continuity_cfg, 'grace_window_seconds', None)
        if grace_window is None:
            grace_window = self._get_config(continuity_cfg, 'grace_window_sec', 1.0)
        self.grace_window_sec = grace_window
        self.grace_max_candidates = self._get_config(continuity_cfg, 'grace_max_candidates', 5)
        
        self.shadow_mode = self._get_config(continuity_cfg, 'shadow_mode', False)
        
        self.shadow_metrics = {
            'total_carries': 0,
            'total_binds': 0,
            'appearance_breaks': 0,
            'bbox_breaks': 0,
            'health_breaks': 0,
            'contradiction_breaks': 0,
            'grace_reattachments': 0,
            'spatial_transfers': 0,
            'young_track_skips': 0
        }
        self.shadow_metrics_log_interval_sec = self._get_config(
            continuity_cfg, 'shadow_metrics_log_interval_sec', 30.0
        )
        self._last_shadow_metrics_log_ts: Optional[float] = None
        
        self.frame_width: Optional[int] = None
        self.frame_height: Optional[int] = None
        self._bbox_displacement_threshold_px: Optional[float] = None
        
        self.memories: Dict[int, ContinuityMemory] = {}
        self.recently_lost: Dict[int, ContinuityMemory] = {}
        
        logger.info(
            "ContinuityBinder initialized | "
            f"min_age={self.min_track_age_frames} | "
            f"appearance_thresh={self.appearance_distance_threshold} | "
            f"bbox_frac={self.max_bbox_displacement_frac} | "
            f"grace_window={self.grace_window_sec}s | "
            f"shadow_mode={self.shadow_mode}"
        )
    
    def _get_config(self, cfg, key: str, default):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)
    
    def set_frame_dimensions(self, width: int, height: int):
        self.frame_width = width
        self.frame_height = height
        diagonal = np.sqrt(width**2 + height**2)
        self._bbox_displacement_threshold_px = self.max_bbox_displacement_frac * diagonal
        logger.info(
            f"Frame dimensions set: {width}x{height} | "
            f"Bbox displacement threshold: {self._bbox_displacement_threshold_px:.1f}px "
            f"(frac={self.max_bbox_displacement_frac}, fallback={self.max_bbox_displacement_px}px)"
        )
    
    def _get_bbox_displacement_threshold(self) -> float:
        if self._bbox_displacement_threshold_px is not None:
            return self._bbox_displacement_threshold_px
        return self.max_bbox_displacement_px
    
    def _normalize_embedding_if_needed(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if abs(norm - 1.0) > 1e-2:
            return embedding / (norm + 1e-8)
        return embedding
    
    
    def apply(
        self,
        ts: float,
        tracks: List[Tracklet],
        decisions: List[IdentityDecision]
    ) -> List[IdentityDecision]:
        decisions_by_track = {d.track_id: d for d in decisions}
        
        for track in tracks:
            tid = track.track_id
            decision = decisions_by_track.get(tid)
            
            if decision is None:
                decision = self._make_unknown_decision(tid)
                decisions.append(decision)
                decisions_by_track[tid] = decision
            
            memory = self.memories.get(tid)
            
            has_face_this_frame = getattr(track, 'has_face_this_frame', False)
            
            current_embedding = getattr(track, 'embedding', None)
            
            face_confirmed = (
                decision.identity_id is not None and
                decision.binding_state in ("CONFIRMED_WEAK", "CONFIRMED_STRONG") and
                has_face_this_frame
            )
            
            if face_confirmed:
                self._bind(decision, track, ts)
                self._set_id_source(decision, "F")
                continue
            
            if has_face_this_frame:
                self._set_id_source(decision, "F")
                
                if memory and decision.identity_id == memory.person_id:
                    memory.last_face_ts = ts
                    memory.last_bbox = track.last_box
                    if current_embedding is not None:
                        memory.last_embedding = current_embedding.copy()
                    memory.face_contradiction_counter = 0
                elif memory and decision.identity_id is not None and decision.identity_id != memory.person_id:
                    pass
                
                continue
            
            if memory is None:
                memory = self._attempt_grace_reattachment(track, ts)
                if memory:
                    old_tid = memory.original_track_id or memory.track_id
                    memory.track_id = tid
                    self.memories[tid] = memory
                    
                    if old_tid in self.recently_lost:
                        del self.recently_lost[old_tid]
                    
                    self._carry(decision, track, memory, ts)
                    continue
                
                spatial_memory = self._attempt_spatial_transfer(track, ts, tracks)
                if spatial_memory:
                    memory = ContinuityMemory(
                        track_id=tid,
                        person_id=spatial_memory.person_id,
                        label=spatial_memory.label,
                        confidence=spatial_memory.confidence * 0.9,
                        last_face_ts=spatial_memory.last_face_ts,
                        last_bbox=track.last_box,
                        last_embedding=getattr(track, 'embedding', None),
                        original_track_id=tid
                    )
                    self.memories[tid] = memory
                    
                    self._carry(decision, track, memory, ts)
                    
                    if self.shadow_mode:
                        self.shadow_metrics['spatial_transfers'] += 1
                    
                    logger.info(
                        f"SPATIAL TRANSFER: New track_id={tid} inherited identity from "
                        f"nearby track | person_id={spatial_memory.person_id}"
                    )
                    continue
            
            if memory:
                
                if not self._track_stable(track):
                    self._set_id_source(decision, "U")
                    if self.shadow_mode:
                        self.shadow_metrics['young_track_skips'] += 1
                    continue
                
                if not self._track_healthy(track):
                    del self.memories[tid]
                    self._set_id_source(decision, "U")
                    if self.shadow_mode:
                        self.shadow_metrics['health_breaks'] += 1
                    continue
                
                appearance_ok, distance = self._appearance_consistent(track, memory)
                if not appearance_ok:
                    del self.memories[tid]
                    self._set_id_source(decision, "U")
                    if self.shadow_mode:
                        self.shadow_metrics['appearance_breaks'] += 1
                    continue
                
                if not self._bbox_stable(track, memory):
                    del self.memories[tid]
                    self._set_id_source(decision, "U")
                    if self.shadow_mode:
                        self.shadow_metrics['bbox_breaks'] += 1
                    continue
                
                if not self._face_consistent(decision, memory):
                    del self.memories[tid]
                    self._set_id_source(decision, "U")
                    if self.shadow_mode:
                        self.shadow_metrics['contradiction_breaks'] += 1
                    continue
                
                self._carry(decision, track, memory, ts)
            else:
                self._set_id_source(decision, "U")
        
        active_track_ids = {t.track_id for t in tracks}
        for tid in list(self.memories.keys()):
            if tid not in active_track_ids:
                memory = self.memories.pop(tid)
                memory.lost_at_ts = ts
                grace_key = memory.original_track_id if memory.original_track_id else tid
                self.recently_lost[grace_key] = memory
        
        expired = [
            grace_key for grace_key, mem in self.recently_lost.items()
            if mem.lost_at_ts and (ts - mem.lost_at_ts) > self.grace_window_sec
        ]
        for grace_key in expired:
            del self.recently_lost[grace_key]
        
        if self.shadow_mode:
            if self._last_shadow_metrics_log_ts is None:
                self._last_shadow_metrics_log_ts = ts
            elif (ts - self._last_shadow_metrics_log_ts) >= self.shadow_metrics_log_interval_sec:
                self._log_shadow_metrics()
                self._last_shadow_metrics_log_ts = ts
        
        return decisions
    
    
    def _bind(self, decision: IdentityDecision, track: Tracklet, ts: float):
        current_embedding = getattr(track, 'embedding', None)
        
        memory = ContinuityMemory(
            track_id=track.track_id,
            person_id=decision.identity_id,
            label=getattr(decision, 'label', decision.identity_id or 'unknown'),
            confidence=decision.confidence,
            last_face_ts=ts,
            last_bbox=track.last_box,
            last_embedding=current_embedding.copy() if current_embedding is not None else None,
            original_track_id=track.track_id
        )
        
        self.memories[track.track_id] = memory
        
        if self.shadow_mode:
            self.shadow_metrics['total_binds'] += 1
        
        logger.info(
            f"BIND: track_id={track.track_id} → person_id={decision.identity_id} | "
            f"binding_state={decision.binding_state} | conf={decision.confidence:.2f}"
        )
    
    def _carry(self, decision: IdentityDecision, track: Tracklet, memory: ContinuityMemory, ts: float):
        if self.shadow_mode:
            if decision.extra is None:
                decision.extra = {}
            decision.extra['shadow_id_source'] = 'G'
            decision.extra['id_source'] = 'G'
            decision.extra['would_carry'] = memory.person_id
            decision.extra['carried_confidence'] = memory.confidence
        else:
            decision.identity_id = memory.person_id
            
            decision.confidence = memory.confidence
            
            decision.binding_state = "GPS_CARRY"
            
            if decision.extra is None:
                decision.extra = {}
            decision.extra['carried_confidence'] = memory.confidence
            decision.extra['is_carried'] = True
            decision.extra['original_binding_state'] = memory.label if hasattr(memory, 'binding_strength') else 'CARRIED'
        
        memory.last_bbox = track.last_box
        
        current_embedding = getattr(track, 'embedding', None)
        if current_embedding is not None:
            memory.last_embedding = current_embedding.copy()
        
        memory.safe_zone_counter += 1
        
        if self.shadow_mode:
            self.shadow_metrics['total_carries'] += 1
        else:
            logger.info(
                f"GPS CARRY: track_id={decision.track_id} → person_id={memory.person_id} | "
                f"conf={memory.confidence:.2f} | memory_age={memory.safe_zone_counter} frames"
            )
        
        if not self.shadow_mode:
            self._set_id_source(decision, "G")
    
    
    def _track_stable(self, track: Tracklet) -> bool:
        return track.age_frames >= self.min_track_age_frames
    
    def _track_healthy(self, track: Tracklet) -> bool:
        if track.confidence < self.track_health_min_confidence:
            return False
        
        if track.lost_frames > self.track_health_max_lost_frames:
            return False
        
        return True
    
    def _appearance_consistent(self, track: Tracklet, memory: ContinuityMemory) -> Tuple[bool, float]:
        current_embedding = getattr(track, 'embedding', None)
        if current_embedding is None or memory.last_embedding is None:
            logger.debug(f"Appearance guard skipped for track_id={track.track_id} (no embedding)")
            return (True, 0.0)
        
        current_embedding = self._normalize_embedding_if_needed(current_embedding)
        last_embedding = self._normalize_embedding_if_needed(memory.last_embedding)
        
        distance = 1.0 - np.dot(current_embedding, last_embedding)
        
        if memory.safe_zone_counter < self.appearance_safe_zone_frames:
            return (distance < self.appearance_distance_threshold, distance)
        
        if memory.embedding_ema is None:
            memory.embedding_ema = memory.last_embedding.copy()
        
        ema_normalized = self._normalize_embedding_if_needed(memory.embedding_ema)
        ema_distance = 1.0 - np.dot(current_embedding, ema_normalized)
        
        if ema_distance < self.appearance_distance_threshold:
            alpha = self.appearance_ema_alpha
            memory.embedding_ema = alpha * current_embedding + (1 - alpha) * ema_normalized
            memory.embedding_ema = self._normalize_embedding_if_needed(memory.embedding_ema)
            return (True, ema_distance)
        
        return (False, ema_distance)
    
    def _bbox_stable(self, track: Tracklet, memory: ContinuityMemory) -> bool:
        if memory.last_bbox is None:
            return True
        
        current_bbox = track.last_box
        prev_bbox = memory.last_bbox
        
        cx_curr = (current_bbox[0] + current_bbox[2]) / 2.0
        cy_curr = (current_bbox[1] + current_bbox[3]) / 2.0
        cx_prev = (prev_bbox[0] + prev_bbox[2]) / 2.0
        cy_prev = (prev_bbox[1] + prev_bbox[3]) / 2.0
        
        center_distance = np.sqrt((cx_curr - cx_prev)**2 + (cy_curr - cy_prev)**2)
        
        x1_inter = max(current_bbox[0], prev_bbox[0])
        y1_inter = max(current_bbox[1], prev_bbox[1])
        x2_inter = min(current_bbox[2], prev_bbox[2])
        y2_inter = min(current_bbox[3], prev_bbox[3])
        
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        curr_area = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
        prev_area = (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1])
        union_area = curr_area + prev_area - inter_area
        
        iou = inter_area / (union_area + 1e-8)
        
        displacement_threshold = self._get_bbox_displacement_threshold()
        
        if center_distance > displacement_threshold and iou < self.min_bbox_overlap:
            logger.warning(
                f"BBOX TELEPORT: track_id={track.track_id} | "
                f"center_dist={center_distance:.1f}px (thresh={displacement_threshold:.1f}) | "
                f"IoU={iou:.3f} (thresh={self.min_bbox_overlap})"
            )
            return False
        
        return True
    
    def _face_consistent(self, decision: IdentityDecision, memory: ContinuityMemory) -> bool:
        
        face_confirmed = (
            decision.identity_id is not None and
            decision.binding_state in ("CONFIRMED_WEAK", "CONFIRMED_STRONG")
        )
        
        if not face_confirmed:
            memory.face_contradiction_counter = max(0, memory.face_contradiction_counter - 1)
            return True
        
        if decision.identity_id != memory.person_id:
            memory.face_contradiction_counter += 1
            logger.warning(
                f"FACE CONTRADICTION: track_id={decision.track_id} | "
                f"face_id={decision.identity_id} vs memory_id={memory.person_id} | "
                f"counter={memory.face_contradiction_counter}/{self.face_contradiction_threshold}"
            )
            
            if memory.face_contradiction_counter >= self.face_contradiction_threshold:
                logger.warning(
                    f"PERSISTENT FACE CONTRADICTION: Breaking continuity for track_id={decision.track_id}"
                )
                return False
        else:
            memory.face_contradiction_counter = 0
        
        return True
    
    
    def _attempt_grace_reattachment(self, track: Tracklet, ts: float) -> Optional[ContinuityMemory]:
        if not self.recently_lost:
            return None
        
        candidates = []
        
        for old_tid, memory in self.recently_lost.items():
            if memory.lost_at_ts is None:
                continue
            if (ts - memory.lost_at_ts) > self.grace_window_sec:
                continue
            
            if memory.last_bbox is None:
                continue
            
            cx_new = (track.last_box[0] + track.last_box[2]) / 2.0
            cy_new = (track.last_box[1] + track.last_box[3]) / 2.0
            cx_old = (memory.last_bbox[0] + memory.last_bbox[2]) / 2.0
            cy_old = (memory.last_bbox[1] + memory.last_bbox[3]) / 2.0
            
            bbox_distance = np.sqrt((cx_new - cx_old)**2 + (cy_new - cy_old)**2)
            
            displacement_threshold = self._get_bbox_displacement_threshold()
            
            if bbox_distance > displacement_threshold:
                continue
            
            candidates.append((old_tid, memory, bbox_distance))
        
        if not candidates:
            return None
        
        if len(candidates) > self.grace_max_candidates:
            candidates.sort(key=lambda x: x[2])
            candidates = candidates[:self.grace_max_candidates]
        
        current_embedding = getattr(track, 'embedding', None)
        
        if current_embedding is None:
            best_old_tid, best_memory, _ = candidates[0]
            logger.info(
                f"GRACE REATTACH (bbox only): track_id {best_old_tid} → {track.track_id} | "
                f"person_id={best_memory.person_id}"
            )
            return best_memory
        
        best_memory = None
        best_distance = float('inf')
        
        for old_tid, memory, bbox_dist in candidates:
            if memory.last_embedding is None:
                continue
            
            current_emb = self._normalize_embedding_if_needed(current_embedding)
            last_emb = self._normalize_embedding_if_needed(memory.last_embedding)
            app_distance = 1.0 - np.dot(current_emb, last_emb)
            
            if app_distance < self.appearance_distance_threshold:
                if app_distance < best_distance:
                    best_distance = app_distance
                    best_memory = memory
        
        if best_memory:
            if self.shadow_mode:
                self.shadow_metrics['grace_reattachments'] += 1
            
            logger.info(
                f"GRACE REATTACH (appearance): track_id {best_memory.track_id} → {track.track_id} | "
                f"person_id={best_memory.person_id} | app_dist={best_distance:.3f}"
            )
            return best_memory
        
        return None
    
    def _attempt_spatial_transfer(
        self,
        new_track: Tracklet,
        ts: float,
        all_tracks: List[Tracklet]
    ) -> Optional[ContinuityMemory]:
        if new_track.age_frames >= self.min_track_age_frames:
            return None
        
        new_bbox = new_track.last_box
        new_cx = (new_bbox[0] + new_bbox[2]) / 2.0
        new_cy = (new_bbox[1] + new_bbox[3]) / 2.0
        
        proximity_threshold = self._get_bbox_displacement_threshold() * 0.5
        
        best_donor_memory = None
        best_distance = float('inf')
        
        for track in all_tracks:
            if track.track_id == new_track.track_id:
                continue
            
            memory = self.memories.get(track.track_id)
            if memory is None:
                continue
            
            donor_bbox = track.last_box
            donor_cx = (donor_bbox[0] + donor_bbox[2]) / 2.0
            donor_cy = (donor_bbox[1] + donor_bbox[3]) / 2.0
            
            distance = np.sqrt((new_cx - donor_cx)**2 + (new_cy - donor_cy)**2)
            
            if distance > proximity_threshold:
                continue
            
            if distance < best_distance:
                best_distance = distance
                best_donor_memory = memory
        
        for grace_key, memory in self.recently_lost.items():
            if memory.lost_at_ts is None:
                continue
            if (ts - memory.lost_at_ts) > 0.5:
                continue
            
            if memory.last_bbox is None:
                continue
            
            donor_cx = (memory.last_bbox[0] + memory.last_bbox[2]) / 2.0
            donor_cy = (memory.last_bbox[1] + memory.last_bbox[3]) / 2.0
            
            distance = np.sqrt((new_cx - donor_cx)**2 + (new_cy - donor_cy)**2)
            
            if distance > proximity_threshold:
                continue
            
            if distance < best_distance:
                best_distance = distance
                best_donor_memory = memory
                
                if grace_key in self.recently_lost:
                    del self.recently_lost[grace_key]
        
        return best_donor_memory
    
    
    def _make_unknown_decision(self, track_id: int) -> IdentityDecision:
        return IdentityDecision(
            track_id=track_id,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0,
            reason="no_face_evidence"
        )
    
    def _set_id_source(self, decision: IdentityDecision, source: str):
        if hasattr(decision, 'id_source'):
            decision.id_source = source
        else:
            if decision.extra is None:
                decision.extra = {}
            decision.extra['id_source'] = source
    
    def _get_id_source(self, decision: IdentityDecision) -> str:
        if hasattr(decision, 'id_source') and decision.id_source is not None:
            return decision.id_source
        if decision.extra and 'id_source' in decision.extra:
            return decision.extra['id_source']
        return "U"
    
    def _log_shadow_metrics(self):
        logger.info(
            "SHADOW MODE METRICS | "
            f"binds={self.shadow_metrics['total_binds']} | "
            f"carries={self.shadow_metrics['total_carries']} | "
            f"young_skips={self.shadow_metrics['young_track_skips']} | "
            f"app_breaks={self.shadow_metrics['appearance_breaks']} | "
            f"bbox_breaks={self.shadow_metrics['bbox_breaks']} | "
            f"health_breaks={self.shadow_metrics['health_breaks']} | "
            f"contradiction_breaks={self.shadow_metrics['contradiction_breaks']} | "
            f"grace_reattach={self.shadow_metrics['grace_reattachments']} | "
            f"spatial_transfers={self.shadow_metrics['spatial_transfers']} | "
            f"active_memories={len(self.memories)} | "
            f"grace_pool={len(self.recently_lost)}"
        )
    
    def get_shadow_metrics(self) -> dict:
        return self.shadow_metrics.copy()


def default_continuity_config():
    return {
        'min_track_age_frames': 10,
        'appearance_distance_threshold': 0.35,
        'appearance_ema_alpha': 0.3,
        'appearance_safe_zone_frames': 5,
        'max_bbox_displacement_frac': 0.25,
        'max_bbox_displacement_px': 600,
        'min_bbox_overlap': 0.1,
        'track_health_min_confidence': 0.5,
        'track_health_max_lost_frames': 2,
        'face_contradiction_threshold': 3,
        'grace_window_sec': 1.0,
        'grace_max_candidates': 5,
        'shadow_mode': False
    }
