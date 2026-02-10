
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from collections import deque
from core.governance_metrics import get_metrics_collector, MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    enabled: bool = True
    budget_policy: str = "adaptive"
    fixed_budget_per_frame: int = 10
    min_check_interval_sec: float = 0.5
    max_check_interval_sec: float = 5.0
    
    priority_weight_unknown: float = 50.0
    priority_weight_pending: float = 80.0
    priority_weight_confirmed_weak: float = 20.0
    priority_weight_confirmed_strong: float = 10.0
    priority_weight_stale: float = 5.0
    
    time_decay_rate: float = 20.0
    
    fps_high: float = 15.0
    fps_medium: float = 5.0
    fps_low: float = 3.0


@dataclass
class TrackScheduleState:
    track_id: int
    last_scheduled_ts: float = 0.0
    last_processed_ts: float = 0.0
    schedule_count: int = 0
    skip_count: int = 0
    priority_history: deque = field(default_factory=lambda: deque(maxlen=10))


@dataclass
class ScheduleContext:
    scheduled_track_ids: Set[int]
    total_budget: int
    frame_index: int
    actual_fps: float
    timestamp: float

    def should_process(self, track_id: int) -> bool:
        return track_id in self.scheduled_track_ids


class FaceScheduler:
    
    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        self.config = config or SchedulerConfig()
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.track_states: Dict[int, TrackScheduleState] = {}
        self.frame_index = 0
        self.fps_samples: deque = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        logger.info(
            f"FaceScheduler initialized | "
            f"enabled={self.config.enabled} | "
            f"policy={self.config.budget_policy} | "
            f"fixed_budget={self.config.fixed_budget_per_frame}"
        )
    
    
    def compute_schedule(
        self,
        track_ids: List[int],
        binding_states: Dict[int, str],
        current_ts: float,
        actual_fps: float,
        id_sources: Optional[Dict[int, str]] = None,
    ) -> ScheduleContext:
        if not self.config.enabled:
            return ScheduleContext(
                scheduled_track_ids=set(track_ids),
                total_budget=len(track_ids),
                frame_index=self.frame_index,
                actual_fps=actual_fps,
                timestamp=current_ts,
            )
        
        self._update_fps_estimate(current_ts)
        
        for track_id in track_ids:
            if track_id not in self.track_states:
                self.track_states[track_id] = TrackScheduleState(track_id=track_id)
        
        self._cleanup_old_tracks(track_ids, current_ts)
        
        budget = self._compute_budget(len(track_ids), actual_fps)
        
        priority_scores = self._compute_priority_scores(
            track_ids, binding_states, current_ts, id_sources
        )
        
        scheduled_ids = self._select_top_k(priority_scores, budget)
        
        self._record_schedule(scheduled_ids, current_ts)
        
        self._record_metrics(scheduled_ids, track_ids, budget)
        
        logger.debug(
            f"Frame {self.frame_index}: FPS={actual_fps:.1f}, "
            f"tracks={len(track_ids)}, budget={budget}, "
            f"scheduled={len(scheduled_ids)}, "
            f"pending_priority={priority_scores.get('pending_count', 0)}"
        )
        
        self.frame_index += 1
        
        return ScheduleContext(
            scheduled_track_ids=set(scheduled_ids),
            total_budget=budget,
            frame_index=self.frame_index - 1,
            actual_fps=actual_fps,
            timestamp=current_ts,
        )
    
    def get_track_schedule_state(self, track_id: int) -> Optional[TrackScheduleState]:
        return self.track_states.get(track_id)
    
    def reset(self):
        self.track_states.clear()
        self.fps_samples.clear()
        self.frame_index = 0
        logger.info("FaceScheduler reset")
    
    
    def _compute_budget(self, num_tracks: int, actual_fps: float) -> int:
        if self.config.budget_policy == "fixed":
            return min(self.config.fixed_budget_per_frame, num_tracks)
        
        if actual_fps >= self.config.fps_high:
            fraction = 1.0
        elif actual_fps >= self.config.fps_medium:
            fraction = 0.5
        elif actual_fps >= self.config.fps_low:
            fraction = 0.2
        else:
            fraction = 0.1
        
        budget = max(1, int(num_tracks * fraction))
        return min(budget, num_tracks)
    
    
    def _compute_priority_scores(
        self,
        track_ids: List[int],
        binding_states: Dict[int, str],
        current_ts: float,
        id_sources: Optional[Dict[int, str]] = None,
    ) -> Dict:
        scores = {}
        state_counts = {}
        
        for track_id in track_ids:
            binding_state = binding_states.get(track_id, "UNKNOWN")
            id_source = id_sources.get(track_id, "U") if id_sources else "U"
            track_state = self.track_states[track_id]
            
            if binding_state == "UNKNOWN":
                if id_source == "G":
                    state_score = self.config.priority_weight_confirmed_weak
                    logger.debug(
                        f"Track {track_id}: GPS-carried (UNKNOWN+G) → priority {state_score}"
                    )
                else:
                    state_score = self.config.priority_weight_unknown
            elif binding_state == "PENDING":
                state_score = self.config.priority_weight_pending
            elif binding_state == "CONFIRMED_WEAK":
                state_score = self.config.priority_weight_confirmed_weak
            elif binding_state == "CONFIRMED_STRONG":
                state_score = self.config.priority_weight_confirmed_strong
            elif binding_state == "STALE":
                state_score = self.config.priority_weight_stale
            else:
                state_score = self.config.priority_weight_unknown
            
            time_since_last_check = current_ts - track_state.last_processed_ts
            
            if time_since_last_check < self.config.min_check_interval_sec:
                time_decay = 0.0
            else:
                time_decay = min(
                    100.0,
                    (time_since_last_check - self.config.min_check_interval_sec)
                    * self.config.time_decay_rate
                )
            
            total_score = state_score + time_decay
            scores[track_id] = total_score
            
            state_counts[f"{binding_state}_count"] = \
                state_counts.get(f"{binding_state}_count", 0) + 1
        
        scores.update(state_counts)
        return scores
    
    
    def _select_top_k(self, scores: Dict, k: int) -> List[int]:
        track_scores = {
            tid: score for tid, score in scores.items()
            if isinstance(tid, int)
        }
        
        if not track_scores:
            return []
        
        k = max(1, min(k, len(track_scores)))
        
        sorted_tracks = sorted(
            track_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [tid for tid, _ in sorted_tracks[:k]]
    
    
    def _record_schedule(self, scheduled_ids: List[int], current_ts: float) -> None:
        for track_id in scheduled_ids:
            if track_id in self.track_states:
                state = self.track_states[track_id]
                state.last_scheduled_ts = current_ts
                state.last_processed_ts = current_ts
                state.schedule_count += 1
                state.priority_history.append(current_ts)
    
    def _update_fps_estimate(self, current_ts: float) -> None:
        if self.last_frame_time > 0:
            frame_delta = current_ts - self.last_frame_time
            if frame_delta > 0:
                fps = 1.0 / frame_delta
                self.fps_samples.append(fps)
        self.last_frame_time = current_ts
    
    def _cleanup_old_tracks(
        self,
        active_track_ids: List[int],
        current_ts: float,
        max_age_sec: float = 120.0,
    ) -> None:
        active_set = set(active_track_ids)
        to_remove = []
        
        for track_id, state in self.track_states.items():
            if track_id not in active_set:
                age = current_ts - state.last_processed_ts
                if age > max_age_sec:
                    to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.track_states[track_id]
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old track states")
    
    def _record_metrics(
        self,
        scheduled_ids: List[int],
        track_ids: List[int],
        budget: int,
    ) -> None:
        try:
            scheduled_count = len(scheduled_ids)
            skipped_count = len(track_ids) - scheduled_count
            
            self.metrics_collector.metrics.scheduler_budget_available = budget
            self.metrics_collector.metrics.scheduler_selected = scheduled_count
            self.metrics_collector.metrics.scheduler_skipped = skipped_count
            
            logger.debug(
                f"Scheduler metrics: "
                f"scheduled={scheduled_count}, "
                f"skipped={skipped_count}, "
                f"budget={budget}"
            )
        except Exception as e:
            logger.warning(f"Failed to record scheduler metrics: {e}")


def create_scheduler_from_config(config_dict: dict, metrics_collector: Optional[MetricsCollector] = None) -> FaceScheduler:
    try:
        cfg = SchedulerConfig(
            enabled=config_dict.get("enabled", True),
            budget_policy=config_dict.get("budget_policy", "adaptive"),
            fixed_budget_per_frame=config_dict.get("fixed_budget_per_frame", 10),
            min_check_interval_sec=config_dict.get("min_check_interval_sec", 0.5),
            max_check_interval_sec=config_dict.get("max_check_interval_sec", 5.0),
            priority_weight_unknown=config_dict.get("priority_weights", {}).get("unknown", 50.0),
            priority_weight_pending=config_dict.get("priority_weights", {}).get("pending", 80.0),
            priority_weight_confirmed_weak=config_dict.get("priority_weights", {}).get("confirmed_weak", 20.0),
            priority_weight_confirmed_strong=config_dict.get("priority_weights", {}).get("confirmed_strong", 10.0),
            priority_weight_stale=config_dict.get("priority_weights", {}).get("stale", 5.0),
            time_decay_rate=config_dict.get("time_decay_rate", 20.0),
            fps_high=config_dict.get("fps_thresholds", {}).get("high", 15.0),
            fps_medium=config_dict.get("fps_thresholds", {}).get("medium", 5.0),
            fps_low=config_dict.get("fps_thresholds", {}).get("low", 3.0),
        )
        return FaceScheduler(config=cfg, metrics_collector=metrics_collector)
    except Exception as e:
        logger.error(f"Failed to create scheduler from config: {e}")
        return FaceScheduler(metrics_collector=metrics_collector)
