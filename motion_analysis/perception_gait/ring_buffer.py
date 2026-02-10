from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Deque, List, Optional, Tuple
from collections import deque
import numpy as np

from schemas import Frame

logger = logging.getLogger(__name__)


@dataclass
class BufferEntry:
    ts: float
    frame_index: int
    bbox: Tuple[float, float, float, float]

    crop: Optional[np.ndarray] = None
    appearance: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None
    face_crop: Optional[np.ndarray] = None
    face_embedding: Optional[np.ndarray] = None
    face_quality: Optional[float] = None
    gait_vector: Optional[np.ndarray] = None
    event_feature: Optional[np.ndarray] = None


@dataclass
class RingBufferConfig:
    max_seconds: float = 2.0
    store_crops: bool = True
    max_crops_per_track: int = 60


class RingBuffer:

    def __init__(self, config: Optional[RingBufferConfig] = None):
        self.config = config or RingBufferConfig()
        self.buffers: Dict[int, Deque[BufferEntry]] = {}

    def add(
        self,
        track_id: int,
        ts: float,
        frame_index: int,
        bbox: Tuple[float, float, float, float],
        crop: Optional[np.ndarray] = None,
        appearance: Optional[np.ndarray] = None,
        pose: Optional[np.ndarray] = None,
        face_crop: Optional[np.ndarray] = None,
        face_embedding: Optional[np.ndarray] = None,
        face_quality: Optional[float] = None,
        gait_vector: Optional[np.ndarray] = None,
        event_feature: Optional[np.ndarray] = None,
    ) -> None:
        if track_id not in self.buffers:
            self.buffers[track_id] = deque()

        if not self.config.store_crops:
            crop = None
            face_crop = None

        if self.config.store_crops and crop is not None:
            if len(self.buffers[track_id]) >= self.config.max_crops_per_track:
                self.buffers[track_id].popleft()

        entry = BufferEntry(
            ts=ts,
            frame_index=frame_index,
            bbox=bbox,
            crop=crop,
            appearance=appearance,
            pose=pose,
            face_crop=face_crop,
            face_embedding=face_embedding,
            face_quality=face_quality,
            gait_vector=gait_vector,
            event_feature=event_feature,
        )

        self.buffers[track_id].append(entry)

        self._prune(track_id, ts)

    def get(self, track_id: int) -> List[BufferEntry]:
        if track_id not in self.buffers:
            return []
        return list(self.buffers[track_id])

    def get_latest(self, track_id: int) -> Optional[BufferEntry]:
        if track_id not in self.buffers or len(self.buffers[track_id]) == 0:
            return None
        return self.buffers[track_id][-1]

    def remove_track(self, track_id: int) -> None:
        if track_id in self.buffers:
            del self.buffers[track_id]

    def cleanup_dead_tracks(self, active_track_ids: List[int]) -> None:
        for tid in list(self.buffers.keys()):
            if tid not in active_track_ids:
                del self.buffers[tid]

    def _prune(self, track_id: int, current_ts: float) -> None:
        max_age = self.config.max_seconds
        buf = self.buffers.get(track_id)
        if buf is None:
            return

        while len(buf) > 0 and (current_ts - buf[0].ts) > max_age:
            buf.popleft()


    def get_last_n_entries(self, track_id: int, n: int) -> List[BufferEntry]:
        if track_id not in self.buffers:
            return []
        buf = self.buffers[track_id]
        return list(buf)[-n:]

    def get_time_window(
        self,
        track_id: int,
        window_sec: float,
        current_ts: float,
    ) -> List[BufferEntry]:
        if track_id not in self.buffers:
            return []

        buf = self.buffers[track_id]
        min_ts = current_ts - window_sec
        return [e for e in buf if e.ts >= min_ts]
