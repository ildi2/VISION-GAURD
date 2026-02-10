
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("privacy.delay_buffer")


@dataclass
class BufferItem:
    frame_id: int
    ingest_ts: float
    privacy_frame: np.ndarray
    audit_payload: Dict[str, Any]
    original_frame_ts: float
    raw_image: Optional[np.ndarray] = None
    rendering_data: Optional[Dict[str, Any]] = None


class DelayBuffer:
    
    def __init__(
        self,
        delay_sec: float = 3.0,
        max_frames: int = 300,
        overflow_policy: str = "drop_oldest",
    ) -> None:
        self._delay_sec = delay_sec
        self._max_frames = max_frames
        self._overflow_policy = overflow_policy
        
        self._buffer: Deque[BufferItem] = deque()
        
        self._frames_pushed = 0
        self._frames_emitted = 0
        self._frames_dropped = 0
        
        log.info(
            "DelayBuffer initialized | delay_sec=%.2f, max_frames=%d, policy=%s",
            delay_sec,
            max_frames,
            overflow_policy,
        )
    
    def push(
        self,
        frame_id: int,
        ingest_ts: float,
        privacy_frame: np.ndarray,
        audit_payload: Dict[str, Any],
        original_frame_ts: float,
        raw_image: Optional[np.ndarray] = None,
        rendering_data: Optional[Dict[str, Any]] = None,
    ) -> int:
        dropped = 0
        
        try:
            while len(self._buffer) >= self._max_frames:
                oldest = self._buffer.popleft()
                self._frames_dropped += 1
                dropped += 1
                log.warning(
                    "DelayBuffer overflow: dropped frame_id=%d (buffer=%d/%d)",
                    oldest.frame_id,
                    len(self._buffer),
                    self._max_frames,
                )
            
            item = BufferItem(
                frame_id=frame_id,
                ingest_ts=ingest_ts,
                privacy_frame=privacy_frame,
                audit_payload=audit_payload,
                original_frame_ts=original_frame_ts,
                raw_image=raw_image,
                rendering_data=rendering_data,
            )
            self._buffer.append(item)
            self._frames_pushed += 1
            
        except Exception as e:
            log.exception("DelayBuffer push failed: %s", e)
            dropped = -1
        
        return dropped
    
    def pop_eligible(self, current_ts: float) -> List[BufferItem]:
        eligible: List[BufferItem] = []
        
        try:
            threshold_ts = current_ts - self._delay_sec
            
            while self._buffer:
                oldest = self._buffer[0]
                
                if oldest.ingest_ts <= threshold_ts:
                    item = self._buffer.popleft()
                    eligible.append(item)
                    self._frames_emitted += 1
                else:
                    break
                    
        except Exception as e:
            log.exception("DelayBuffer pop_eligible failed: %s", e)
        
        return eligible
    
    def peek_oldest(self) -> Optional[BufferItem]:
        if self._buffer:
            return self._buffer[0]
        return None
    
    def flush_all(self) -> List[BufferItem]:
        remaining: List[BufferItem] = []
        
        try:
            while self._buffer:
                item = self._buffer.popleft()
                remaining.append(item)
                self._frames_emitted += 1
                
            log.info("DelayBuffer flushed: %d frames", len(remaining))
            
        except Exception as e:
            log.exception("DelayBuffer flush_all failed: %s", e)
        
        return remaining
    
    def get_buffer_depth(self) -> int:
        return len(self._buffer)
    
    def get_buffer_lag_sec(self, current_ts: float) -> float:
        if not self._buffer:
            return 0.0
        oldest = self._buffer[0]
        return current_ts - oldest.ingest_ts
    
    @property
    def delay_sec(self) -> float:
        return self._delay_sec
    
    @property
    def max_frames(self) -> int:
        return self._max_frames
    
    @property
    def frames_pushed(self) -> int:
        return self._frames_pushed
    
    @property
    def frames_emitted(self) -> int:
        return self._frames_emitted
    
    @property
    def frames_dropped(self) -> int:
        return self._frames_dropped
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "delay_sec": self._delay_sec,
            "max_frames": self._max_frames,
            "current_depth": len(self._buffer),
            "frames_pushed": self._frames_pushed,
            "frames_emitted": self._frames_emitted,
            "frames_dropped": self._frames_dropped,
        }
