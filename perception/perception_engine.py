# perception/perception_engine.py
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from schemas import Frame, Tracklet
from core.interfaces import PerceptionEngine

from .detector import Detector, Detection
from .tracker_ocsort import OCSortTracker, Track
from .appearance import AppearanceExtractor
from .ring_buffer import RingBuffer, RingBufferConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal per-track state
# ---------------------------------------------------------------------------

class TrackState:
    """
    Holds:
      - public Tracklet object (required by schemas)
      - appearance_features (EMA updated inside tracker_ocsort)
    """

    def __init__(self, track_id: int, camera_id: str):
        self.tracklet = Tracklet(
            track_id=track_id,
            camera_id=camera_id,
            last_frame_id=0,
            last_box=(0.0, 0.0, 0.0, 0.0),
            confidence=0.0,
            age_frames=0,
            lost_frames=0,
            history_boxes=[],
        )
        self.appearance_feature: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Main Perception Engine (Phase 1)
# ---------------------------------------------------------------------------

class Phase1PerceptionEngine(PerceptionEngine):
    """
    Phase-1 perception pipeline.

    Per frame:
      1. Detection (YOLO)
      2. Appearance extraction (cheap classic CV)
      3. Tracking (OC-SORT + IoU + optional appearance fusion)
      4. Update Tracklets
      5. Update Ring Buffer
      6. Remove dead tracks

    Output:
      List[Tracklet] for all active tracks.
    """

    def __init__(
        self,
        detector: Optional[Detector] = None,
        tracker: Optional[OCSortTracker] = None,
        appearance: Optional[AppearanceExtractor] = None,
        ring_buffer: Optional[RingBuffer] = None,
        max_lost_frames: int = 30,
    ) -> None:
        super().__init__()

        # Modules
        self.detector: Detector = detector or Detector()
        self.tracker: OCSortTracker = tracker or OCSortTracker()
        self.appearance: AppearanceExtractor = appearance or AppearanceExtractor()
        self.ring_buffer: RingBuffer = ring_buffer or RingBuffer(RingBufferConfig())

        # Internal state for each track_id
        self._states: Dict[int, TrackState] = {}

        # When to drop a lost track
        self.max_lost_frames: int = int(max_lost_frames)

        logger.info("Phase-1 PerceptionEngine initialised.")

    # ------------------------------------------------------------------ #
    # REQUIRED API                                                       #
    # ------------------------------------------------------------------ #

    def process_frame(self, frame: Frame) -> List[Tracklet]:
        """
        Main function called every frame by core.main_loop.

        Returns:
            List[Tracklet] for all active (confirmed) tracks.
        """
        if frame.image is None:
            logger.warning("Phase1PerceptionEngine.process_frame: empty frame.image")
            return []

        # ---- Step 1: YOLO Detection (zero-copy, uses Frame.image directly) ----
        detections: List[Detection] = self.detector.detect(frame)

        # ---- Step 2: Appearance Features (one per detection) ----
        features = self.appearance.compute_features_for_detections(frame, detections)

        # ---- Step 3: OC-SORT Tracking ----
        tracks: List[Track] = self.tracker.update(detections, features)

        # ---- Step 4: Update Track States ----
        active_ids: Set[int] = set()
        for tr in tracks:
            tid = int(tr.track_id)
            active_ids.add(tid)

            # Convert bbox once per track per frame to a tuple[float, float, float, float]
            bbox_xyxy: Tuple[float, float, float, float] = tuple(
                float(v) for v in tr.bbox.tolist()
            )
            self._update_track_state(frame, tid, bbox_xyxy, float(tr.score))

        # ---- Step 5: Mark & Remove Lost Tracks ----
        self._increment_lost_and_prune(active_ids)

        # ---- Step 6: Update Ring Buffer ----
        # Use the already-updated Tracklet.last_box instead of recomputing bbox.
        for tr in tracks:
            tid = int(tr.track_id)
            state = self._states.get(tid)
            if state is None:
                continue

            self.ring_buffer.add(
                track_id=tid,
                ts=frame.ts,
                frame_index=frame.frame_id,
                bbox=state.tracklet.last_box,
                crop=None,        # Phase-1 doesn't persist crops yet
                appearance=None,  # can be wired later if needed
                pose=None,
            )

        # Return list of active tracklets
        return [state.tracklet for state in self._states.values()]

    # ------------------------------------------------------------------ #
    # INTERNAL HELPERS                                                   #
    # ------------------------------------------------------------------ #

    def _update_track_state(
        self,
        frame: Frame,
        track_id: int,
        bbox_xyxy: Tuple[float, float, float, float],
        score: float,
    ) -> None:
        """
        Create/update TrackState and Tracklet for a tracked object.
        """
        # Create if new
        if track_id not in self._states:
            self._states[track_id] = TrackState(
                track_id=track_id,
                camera_id=frame.camera_id,
            )

        state = self._states[track_id]
        t = state.tracklet

        # Update public Tracklet
        t.last_frame_id = frame.frame_id
        t.last_box = bbox_xyxy
        t.confidence = float(score)
        t.age_frames += 1
        t.lost_frames = 0

        # Keep short history of boxes (for future gait/trajectory logic)
        t.history_boxes.append(t.last_box)
        if len(t.history_boxes) > 60:
            t.history_boxes.pop(0)

        # Appearance feature from tracker’s EMA – not exposed yet in Phase 1
        state.appearance_feature = None

    def _increment_lost_and_prune(self, active_ids: Set[int]) -> None:
        """
        Increase lost counters for inactive tracks and remove dead ones.
        """
        to_remove: List[int] = []

        for tid, state in self._states.items():
            if tid not in active_ids:
                state.tracklet.lost_frames += 1
                if state.tracklet.lost_frames > self.max_lost_frames:
                    to_remove.append(tid)

        # Remove from internal state + ring buffer
        for tid in to_remove:
            self._states.pop(tid, None)
            self.ring_buffer.remove_track(tid)
            logger.debug("Phase-1: removed track %d (lost too long).", tid)
