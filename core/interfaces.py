
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from schemas import (
    Frame,
    Tracklet,
    IdSignals,
    IdentityDecision,
    EventFlags,
    Alert,
)


class PerceptionEngine(ABC):

    @abstractmethod
    def process_frame(self, frame: Frame) -> List[Tracklet]:
        raise NotImplementedError


class IdentityEngine(ABC):

    @abstractmethod
    def update_signals(self, frame: Frame, tracks: List[Tracklet]) -> List[IdSignals]:
        raise NotImplementedError

    @abstractmethod
    def decide(self, signals: List[IdSignals]) -> List[IdentityDecision]:
        raise NotImplementedError


class EventsEngine(ABC):

    @abstractmethod
    def update(
        self,
        frame: Frame,
        tracks: List[Tracklet],
        decisions: List[IdentityDecision],
    ) -> List[EventFlags]:
        raise NotImplementedError


class AlertEngine(ABC):

    @abstractmethod
    def update(
        self,
        frame: Frame,
        events: List[EventFlags],
        decisions: List[IdentityDecision],
    ) -> List[Alert]:
        raise NotImplementedError
