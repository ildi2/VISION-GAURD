
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

    def __init__(self) -> None:
        self._next_track_id = 0

    def process_frame(self, frame: Frame) -> List[Tracklet]:
        return []


class DummyIdentityEngine(IdentityEngine):

    def update_signals(self, frame: Frame, tracks: List[Tracklet]) -> List[IdSignals]:
        return [IdSignals(track_id=t.track_id) for t in tracks]

    def decide(self, signals: List[IdSignals]) -> List[IdentityDecision]:
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

    def update(
        self,
        frame: Frame,
        tracks: List[Tracklet],
        decisions: List[IdentityDecision],
    ) -> List[EventFlags]:
        return [EventFlags(track_id=t.track_id) for t in tracks]


class DummyAlertEngine(AlertEngine):

    def __init__(self) -> None:
        self._next_alert_id = 0

    def update(
        self,
        frame: Frame,
        events: List[EventFlags],
        decisions: List[IdentityDecision],
    ) -> List[Alert]:
        return []
