"""
core/dummies.py

Dummy implementations of the engine interfaces.

These are used in Phase 0 so that the main pipeline can run end-to-end
without yet having real models (YOLO, face, gait, etc.).
"""

from __future__ import annotations

import time
from typing import List

from schemas import (
    Frame,
    Tracklet,
    IdSignals,
    IdentityDecision,
    EventFlags,
    Alert,
)
from .interfaces import (
    PerceptionEngine,
    IdentityEngine,
    EventsEngine,
    AlertEngine,
)


class DummyPerceptionEngine(PerceptionEngine):
    """
    Perception engine that doesn't detect anything.

    Later we will replace this with real YOLO + tracker.
    """

    def __init__(self) -> None:
        self._next_track_id = 0

    def process_frame(self, frame: Frame) -> List[Tracklet]:
        # Phase 0: return empty list (no detections, no tracks).
        # We keep the counter in case we want to create synthetic tracks for testing.
        return []


class DummyIdentityEngine(IdentityEngine):
    """
    Identity engine that always returns 'unknown' and no signals.

    Useful to check that the pipeline wiring is correct.
    """

    def update_signals(self, frame: Frame, tracks: List[Tracklet]) -> List[IdSignals]:
        # Just create empty IdSignals objects for each track, with default values.
        return [IdSignals(track_id=t.track_id) for t in tracks]

    def decide(self, signals: List[IdSignals]) -> List[IdentityDecision]:
        # Everyone is unknown with confidence 0.0
        return [
            IdentityDecision(
                track_id=s.track_id,
                identity_id=None,
                category="unknown",
                confidence=0.0,
                reason="dummy_identity_engine",
            )
            for s in signals
        ]


class DummyEventsEngine(EventsEngine):
    """
    Events engine that never fires any event.

    Later we will use real models for weapons / fights / falls.
    """

    def update(
        self,
        frame: Frame,
        tracks: List[Tracklet],
        decisions: List[IdentityDecision],
    ) -> List[EventFlags]:
        # Just return default EventFlags (all scores 0, flags False).
        return [EventFlags(track_id=t.track_id) for t in tracks]


class DummyAlertEngine(AlertEngine):
    """
    Alert engine that never raises any alert.

    This keeps the pipeline simple in Phase 0.
    """

    def __init__(self) -> None:
        self._next_alert_id = 0

    def update(
        self,
        frame: Frame,
        events: List[EventFlags],
        decisions: List[IdentityDecision],
    ) -> List[Alert]:
        # No alerts for now.
        # If you want to test alerts visually, you could create fake ones here.
        return []
