
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional, Tuple, Dict, Any, Deque

import numpy as np

from schemas import FaceSample

logger = logging.getLogger(__name__)


class ReasonCode:
    
    ACCEPT_ALL_GATES = "passed_all_gates"
    
    REJECT_YAW_TOO_EXTREME = "yaw_too_extreme"
    REJECT_PITCH_TOO_EXTREME = "pitch_too_extreme"
    REJECT_TOO_DARK = "too_dark"
    REJECT_TOO_BRIGHT = "too_bright"
    REJECT_TOO_BLURRY = "too_blurry"
    
    HOLD_QUALITY_LOW_UNKNOWN = "quality_too_low_unknown"
    HOLD_QUALITY_LOW_CONFIRMED = "quality_too_low_confirmed"
    HOLD_QUALITY_LOW_STALE = "quality_too_low_stale"
    
    ERROR_MISSING_QUALITY = "error_missing_quality"
    ERROR_MISSING_BBOX = "error_missing_bbox"
    ERROR_INVALID_STATE = "error_invalid_state"
    ERROR_EXCEPTION = "error_exception"


class GateDecision:
    ACCEPT = "ACCEPT"
    HOLD = "HOLD"
    REJECT = "REJECT"


class EvidenceGate:
    
    def __init__(
        self,
        cfg: Optional[Any] = None,
        metrics_collector: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.metrics_collector = metrics_collector
        
        self.quality_buffers: Dict[int, Deque[float]] = {}
        self.quality_window_size = 5
        
        try:
            if cfg and hasattr(cfg, 'governance'):
                gov_config = cfg.governance
                if hasattr(gov_config, 'evidence_gate'):
                    eg_config = gov_config.evidence_gate
                    self.enabled = getattr(eg_config, 'enabled', True)
                    self.thresholds = getattr(eg_config, 'thresholds', None)
                else:
                    self.enabled = True
                    self.thresholds = None
            else:
                self.enabled = True
                self.thresholds = None
        except Exception as e:
            logger.warning(f"EvidenceGate init config error: {e}")
            self.enabled = True
            self.thresholds = None
        
        if self.thresholds is None:
            self.thresholds = self._default_thresholds()
        
        logger.info(
            f"EvidenceGate initialized | enabled={self.enabled} | "
            f"unknown_min_q={self._get_threshold('unknown_min_quality', 0.55)} | "
            f"quality_smoothing=enabled (window={self.quality_window_size} frames)"
        )
    
    def decide(
        self,
        face_sample: FaceSample,
        track_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        try:
            if not self.enabled:
                return (GateDecision.ACCEPT, "gate_disabled")
            
            if face_sample is None:
                self._record_decision(GateDecision.REJECT, ReasonCode.ERROR_MISSING_QUALITY, track_context)
                return (GateDecision.REJECT, ReasonCode.ERROR_MISSING_QUALITY)
            
            binding_state = track_context.get('binding_state', 'UNKNOWN') if track_context else 'UNKNOWN'
            track_id = track_context.get('track_id', -1) if track_context else -1
            
            try:
                quality = face_sample.clamped_quality() if hasattr(face_sample, 'clamped_quality') else float(face_sample.quality or 0.0)
            except Exception:
                quality = 0.0
            
            smoothed_quality = self._compute_smoothed_quality(quality, track_id)
            
            try:
                yaw = float(face_sample.yaw or 0.0)
                pitch = float(face_sample.pitch or 0.0)
            except Exception:
                yaw, pitch = 0.0, 0.0
            
            try:
                bbox = face_sample.bbox if hasattr(face_sample, 'bbox') else None
                brightness = self._compute_brightness_from_bbox(face_sample, bbox)
            except Exception:
                brightness = 0.5
            
            try:
                blur = self._compute_blur_from_sample(face_sample)
            except Exception:
                blur = 200.0
            
            result = self._check_geometric_filters(
                yaw=yaw,
                pitch=pitch,
                brightness=brightness,
                blur=blur,
                track_context=track_context
            )
            if result:
                return result
            
            result = self._check_quality_filters(
                quality=smoothed_quality,
                raw_quality=quality,
                binding_state=binding_state,
                track_context=track_context
            )
            if result:
                return result
            
            self._record_decision(GateDecision.ACCEPT, ReasonCode.ACCEPT_ALL_GATES, track_context)
            return (GateDecision.ACCEPT, ReasonCode.ACCEPT_ALL_GATES)
        
        except Exception as e:
            logger.error(f"EvidenceGate.decide exception: {e}", exc_info=True)
            self._record_decision(GateDecision.REJECT, ReasonCode.ERROR_EXCEPTION, track_context)
            return (GateDecision.REJECT, ReasonCode.ERROR_EXCEPTION)
    
    
    def _check_geometric_filters(
        self,
        yaw: float,
        pitch: float,
        brightness: float,
        blur: float,
        track_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[str, str]]:
        max_yaw = self._get_threshold('max_yaw_unknown', 40.0)
        if abs(yaw) > max_yaw:
            reason = ReasonCode.REJECT_YAW_TOO_EXTREME
            self._record_decision(GateDecision.REJECT, reason, track_context)
            return (GateDecision.REJECT, reason)
        
        max_pitch = self._get_threshold('max_pitch', 30.0)
        if abs(pitch) > max_pitch:
            reason = ReasonCode.REJECT_PITCH_TOO_EXTREME
            self._record_decision(GateDecision.REJECT, reason, track_context)
            return (GateDecision.REJECT, reason)
        
        min_brightness = self._get_threshold('min_brightness_normalized', 0.2)
        if brightness < min_brightness:
            reason = ReasonCode.REJECT_TOO_DARK
            self._record_decision(GateDecision.REJECT, reason, track_context)
            return (GateDecision.REJECT, reason)
        
        max_brightness = self._get_threshold('max_brightness_normalized', 0.9)
        if brightness > max_brightness:
            reason = ReasonCode.REJECT_TOO_BRIGHT
            self._record_decision(GateDecision.REJECT, reason, track_context)
            return (GateDecision.REJECT, reason)
        
        min_blur = self._get_threshold('min_blur_score', 200.0)
        if blur < min_blur:
            reason = ReasonCode.REJECT_TOO_BLURRY
            self._record_decision(GateDecision.REJECT, reason, track_context)
            return (GateDecision.REJECT, reason)
        
        return None
    
    def _check_quality_filters(
        self,
        quality: float,
        raw_quality: float,
        binding_state: str,
        track_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[str, str]]:
        if binding_state in ['UNKNOWN', 'PENDING']:
            threshold = self._get_threshold('unknown_min_quality', 0.55)
            if quality < threshold:
                reason = ReasonCode.HOLD_QUALITY_LOW_UNKNOWN
                track_id = track_context.get('track_id', -1) if track_context else -1
                logger.debug(
                    f"Quality rejected (UNKNOWN): track={track_id} | "
                    f"smoothed={quality:.3f} raw={raw_quality:.3f} threshold={threshold:.3f}"
                )
                self._record_decision(GateDecision.HOLD, reason, track_context)
                return (GateDecision.HOLD, reason)
        
        elif binding_state == 'CONFIRMED':
            threshold = self._get_threshold('confirmed_min_quality', 0.55)
            if quality < threshold:
                reason = ReasonCode.HOLD_QUALITY_LOW_CONFIRMED
                self._record_decision(GateDecision.HOLD, reason, track_context)
                return (GateDecision.HOLD, reason)
        
        elif binding_state == 'STALE':
            threshold = self._get_threshold('stale_min_quality', 0.45)
            if quality < threshold:
                reason = ReasonCode.HOLD_QUALITY_LOW_STALE
                self._record_decision(GateDecision.HOLD, reason, track_context)
                return (GateDecision.HOLD, reason)
        
        return None
    
    
    def _compute_smoothed_quality(self, raw_quality: float, track_id: int) -> float:
        try:
            if track_id not in self.quality_buffers:
                self.quality_buffers[track_id] = deque(maxlen=self.quality_window_size)
            
            buffer = self.quality_buffers[track_id]
            
            buffer.append(raw_quality)
            
            if len(buffer) >= self.quality_window_size:
                smoothed = np.mean(list(buffer))
                return float(smoothed)
            else:
                weights = np.arange(1, len(buffer) + 1, dtype=float)
                weighted_avg = np.average(list(buffer), weights=weights)
                return float(weighted_avg)
        
        except Exception as e:
            logger.warning(f"Quality smoothing error for track {track_id}: {e}")
            return raw_quality
    
    def cleanup_track_buffers(self, track_id: int) -> None:
        try:
            if track_id in self.quality_buffers:
                del self.quality_buffers[track_id]
        except Exception:
            pass
    
    
    def _compute_brightness_from_bbox(
        self,
        face_sample: FaceSample,
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> float:
        try:
            if face_sample.extra and 'brightness' in face_sample.extra:
                b = float(face_sample.extra['brightness'])
                return max(0.0, min(1.0, b))
        except Exception:
            pass
        
        return 0.5
    
    def _compute_blur_from_sample(self, face_sample: FaceSample) -> float:
        try:
            if face_sample.extra and 'blur' in face_sample.extra:
                b = float(face_sample.extra['blur'])
                return b
        except Exception:
            pass
        
        return 200.0
    
    def _record_decision(
        self,
        decision: str,
        reason: str,
        track_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            if not self.metrics_collector:
                return
            
            metrics = self.metrics_collector.metrics
            
            if decision == GateDecision.ACCEPT:
                metrics.record_face_accepted()
            elif decision == GateDecision.HOLD:
                metrics.record_face_held(reason)
            elif decision == GateDecision.REJECT:
                metrics.record_face_rejected(reason)
        except Exception as e:
            logger.warning(f"EvidenceGate: failed to record metrics: {e}")
    
    def _get_threshold(self, name: str, default: float) -> float:
        try:
            if self.thresholds is None:
                return default
            
            value = getattr(self.thresholds, name, default)
            return float(value)
        except Exception:
            return default
    
    def _default_thresholds(self) -> Any:
        class DefaultThresholds:
            unknown_min_quality = 0.55
            confirmed_min_quality = 0.55
            stale_min_quality = 0.45
            
            max_yaw_unknown = 40.0
            max_yaw_confirmed = 60.0
            max_pitch = 30.0
            
            min_brightness_normalized = 0.2
            max_brightness_normalized = 0.9
            
            min_blur_score = 200.0
        
        return DefaultThresholds()


_global_evidence_gate: Optional[EvidenceGate] = None


def get_evidence_gate(
    cfg: Optional[Any] = None,
    metrics_collector: Optional[Any] = None,
) -> EvidenceGate:
    global _global_evidence_gate
    
    if _global_evidence_gate is None:
        _global_evidence_gate = EvidenceGate(cfg, metrics_collector)
    
    return _global_evidence_gate


def reset_evidence_gate() -> None:
    global _global_evidence_gate
    _global_evidence_gate = None
