
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger("privacy.policy_fsm")


class PolicyState(Enum):
    UNKNOWN_VISIBLE = auto()
    AUTHORIZED_LOCKED_REDACT = auto()
    REACQUIRE_REDACT = auto()
    ENDED_COOLDOWN = auto()


class PolicyAction(Enum):
    SHOW = auto()
    REDACT = auto()


@dataclass
class TrackPolicyState:
    track_id: int
    state: PolicyState = PolicyState.UNKNOWN_VISIBLE
    
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    locked_since_ts: Optional[float] = None
    last_authorized_ts: Optional[float] = None
    
    grace_until_ts: Optional[float] = None
    reacquire_since_ts: Optional[float] = None
    
    last_identity_id: Optional[str] = None
    last_category: Optional[str] = None
    last_binding_state: Optional[str] = None
    
    last_bbox: Optional[List[float]] = None
    
    transition_count: int = 0


@dataclass
class PolicyConfig:
    grace_sec: float = 5.0
    reacquire_sec: float = 10.0
    unlock_allowed: bool = False
    authorized_categories: List[str] = field(default_factory=lambda: ["resident"])
    require_confirmed_binding: bool = True


class PolicyFSM:
    
    def __init__(self, cfg: Any) -> None:
        self._grace_sec = getattr(cfg, "grace_sec", 5.0)
        self._reacquire_sec = getattr(cfg, "reacquire_sec", 10.0)
        self._unlock_allowed = getattr(cfg, "unlock_allowed", False)
        self._authorized_categories = getattr(cfg, "authorized_categories", ["resident"])
        self._require_confirmed_binding = getattr(cfg, "require_confirmed_binding", True)
        
        self._track_states: Dict[int, TrackPolicyState] = {}
        
        self._total_transitions = 0
        self._locks_created = 0
        self._false_unlock_prevented = 0
        
        log.info(
            "PolicyFSM initialized | grace_sec=%.1f, reacquire_sec=%.1f, unlock_allowed=%s, "
            "authorized_categories=%s, require_confirmed=%s",
            self._grace_sec,
            self._reacquire_sec,
            self._unlock_allowed,
            self._authorized_categories,
            self._require_confirmed_binding,
        )
    
    def update(
        self,
        frame_ts: float,
        track_ids_present: Set[int],
        decisions: List[Any],
    ) -> Dict[int, PolicyAction]:
        actions: Dict[int, PolicyAction] = {}
        
        decision_map: Dict[int, Any] = {}
        for dec in decisions:
            tid = getattr(dec, "track_id", None)
            if tid is not None:
                decision_map[tid] = dec
        
        for track_id in track_ids_present:
            dec = decision_map.get(track_id)
            action = self._process_present_track(track_id, frame_ts, dec)
            actions[track_id] = action
        
        missing_track_ids = set(self._track_states.keys()) - track_ids_present
        for track_id in missing_track_ids:
            action = self._process_missing_track(track_id, frame_ts)
            if action is not None:
                actions[track_id] = action
        
        self._cleanup_old_tracks(frame_ts)
        
        return actions
    
    def _process_present_track(
        self,
        track_id: int,
        frame_ts: float,
        decision: Optional[Any],
    ) -> PolicyAction:
        if track_id not in self._track_states:
            self._track_states[track_id] = TrackPolicyState(
                track_id=track_id,
                first_seen_ts=frame_ts,
                last_seen_ts=frame_ts,
            )
        
        state_obj = self._track_states[track_id]
        state_obj.last_seen_ts = frame_ts
        
        if decision is not None:
            category = getattr(decision, "category", None)
            binding_state = getattr(decision, "binding_state", None)
            identity_id = getattr(decision, "identity_id", None)
            
            state_obj.last_category = category
            state_obj.last_binding_state = binding_state
            state_obj.last_identity_id = identity_id
        else:
            category = None
            binding_state = None
            identity_id = None
        
        authorized_signal = self._is_authorized(category, binding_state)
        
        if authorized_signal:
            state_obj.last_authorized_ts = frame_ts
        
        current_state = state_obj.state
        new_state = current_state
        
        if current_state == PolicyState.UNKNOWN_VISIBLE:
            if authorized_signal:
                new_state = PolicyState.AUTHORIZED_LOCKED_REDACT
                state_obj.locked_since_ts = frame_ts
                self._locks_created += 1
                log.info(
                    "PolicyFSM LOCK: track_id=%d -> AUTHORIZED_LOCKED_REDACT | "
                    "identity=%s, category=%s, binding=%s",
                    track_id, identity_id, category, binding_state,
                )
        
        elif current_state == PolicyState.AUTHORIZED_LOCKED_REDACT:
            if not authorized_signal:
                if state_obj.grace_until_ts is None:
                    state_obj.grace_until_ts = frame_ts + self._grace_sec
                log.debug(
                    "PolicyFSM: track_id=%d lost authorized signal, staying locked (grace)",
                    track_id,
                )
            else:
                state_obj.grace_until_ts = None
        
        elif current_state == PolicyState.REACQUIRE_REDACT:
            new_state = PolicyState.AUTHORIZED_LOCKED_REDACT
            state_obj.reacquire_since_ts = None
            log.info(
                "PolicyFSM REACQUIRE: track_id=%d reappeared -> AUTHORIZED_LOCKED_REDACT",
                track_id,
            )
        
        elif current_state == PolicyState.ENDED_COOLDOWN:
            if self._unlock_allowed and not authorized_signal:
                new_state = PolicyState.UNKNOWN_VISIBLE
                state_obj.locked_since_ts = None
                state_obj.grace_until_ts = None
                log.info(
                    "PolicyFSM UNLOCK: track_id=%d -> UNKNOWN_VISIBLE (unlock_allowed=True)",
                    track_id,
                )
            else:
                new_state = PolicyState.AUTHORIZED_LOCKED_REDACT
                self._false_unlock_prevented += 1
        
        if new_state != current_state:
            state_obj.state = new_state
            state_obj.transition_count += 1
            self._total_transitions += 1
        
        return self._state_to_action(new_state)
    
    def _process_missing_track(
        self,
        track_id: int,
        frame_ts: float,
    ) -> Optional[PolicyAction]:
        if track_id not in self._track_states:
            return None
        
        state_obj = self._track_states[track_id]
        current_state = state_obj.state
        new_state = current_state
        
        time_since_seen = frame_ts - state_obj.last_seen_ts
        
        if current_state == PolicyState.UNKNOWN_VISIBLE:
            return None
        
        elif current_state == PolicyState.AUTHORIZED_LOCKED_REDACT:
            new_state = PolicyState.REACQUIRE_REDACT
            state_obj.reacquire_since_ts = frame_ts
            log.info(
                "PolicyFSM: track_id=%d disappeared -> REACQUIRE_REDACT",
                track_id,
            )
        
        elif current_state == PolicyState.REACQUIRE_REDACT:
            if state_obj.reacquire_since_ts:
                reacquire_duration = frame_ts - state_obj.reacquire_since_ts
                if reacquire_duration >= self._reacquire_sec:
                    new_state = PolicyState.ENDED_COOLDOWN
                    log.info(
                        "PolicyFSM: track_id=%d reacquire timeout -> ENDED_COOLDOWN",
                        track_id,
                    )
        
        elif current_state == PolicyState.ENDED_COOLDOWN:
            pass
        
        if new_state != current_state:
            state_obj.state = new_state
            state_obj.transition_count += 1
            self._total_transitions += 1
        
        return self._state_to_action(new_state)
    
    def _is_authorized(
        self,
        category: Optional[str],
        binding_state: Optional[str],
    ) -> bool:
        if category is None:
            return False
        
        if category not in self._authorized_categories:
            return False
        
        if self._require_confirmed_binding:
            if binding_state is None:
                return False
            valid_states = ("CONFIRMED_WEAK", "CONFIRMED_STRONG", "GPS_CARRY")
            if binding_state not in valid_states:
                return False
        
        return True
    
    def _state_to_action(self, state: PolicyState) -> PolicyAction:
        if state == PolicyState.UNKNOWN_VISIBLE:
            return PolicyAction.SHOW
        else:
            return PolicyAction.REDACT
    
    def _cleanup_old_tracks(self, frame_ts: float, max_age_sec: float = 60.0) -> None:
        to_remove = []
        for track_id, state_obj in self._track_states.items():
            age = frame_ts - state_obj.last_seen_ts
            if age > max_age_sec and state_obj.state == PolicyState.UNKNOWN_VISIBLE:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self._track_states[track_id]
        
        if to_remove:
            log.debug("PolicyFSM: cleaned up %d old UNKNOWN_VISIBLE tracks", len(to_remove))
    
    def get_track_state(self, track_id: int) -> Optional[TrackPolicyState]:
        return self._track_states.get(track_id)
    
    def get_track_policy_info(self, track_id: int, frame_ts: float) -> Dict[str, Any]:
        state_obj = self._track_states.get(track_id)
        
        if state_obj is None:
            return {
                "policy_state": "UNKNOWN_VISIBLE",
                "is_redacted": False,
                "redaction_method": "none",
                "lock_age_sec": None,
                "grace_remaining_sec": None,
                "first_seen_ts": None,
                "last_seen_ts": None,
            }
        
        is_redacted = state_obj.state != PolicyState.UNKNOWN_VISIBLE
        
        lock_age_sec = None
        if state_obj.locked_since_ts is not None:
            lock_age_sec = round(frame_ts - state_obj.locked_since_ts, 2)
        
        grace_remaining_sec = None
        if state_obj.grace_until_ts is not None:
            remaining = state_obj.grace_until_ts - frame_ts
            if remaining > 0:
                grace_remaining_sec = round(remaining, 2)
        
        return {
            "policy_state": state_obj.state.name,
            "is_redacted": is_redacted,
            "redaction_method": "bbox_blur" if is_redacted else "none",
            "lock_age_sec": lock_age_sec,
            "grace_remaining_sec": grace_remaining_sec,
            "transition_count": state_obj.transition_count,
            "first_seen_ts": state_obj.first_seen_ts,
            "last_seen_ts": state_obj.last_seen_ts,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        state_counts = {s.name: 0 for s in PolicyState}
        for state_obj in self._track_states.values():
            state_counts[state_obj.state.name] += 1
        
        return {
            "total_tracks_managed": len(self._track_states),
            "total_transitions": self._total_transitions,
            "locks_created": self._locks_created,
            "false_unlock_prevented": self._false_unlock_prevented,
            "state_counts": state_counts,
            "config": {
                "grace_sec": self._grace_sec,
                "reacquire_sec": self._reacquire_sec,
                "unlock_allowed": self._unlock_allowed,
            },
        }
