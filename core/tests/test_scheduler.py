
import pytest
import time
from core.scheduler import FaceScheduler, SchedulerConfig, ScheduleContext


class TestBudgetComputation:
    
    def test_budget_at_high_fps(self):
        cfg = SchedulerConfig(budget_policy="adaptive", fps_high=15.0)
        scheduler = FaceScheduler(cfg)
        
        budget = scheduler._compute_budget(num_tracks=100, actual_fps=30.0)
        assert budget == 100
    
    def test_budget_at_medium_fps(self):
        cfg = SchedulerConfig(budget_policy="adaptive", fps_medium=5.0)
        scheduler = FaceScheduler(cfg)
        
        budget = scheduler._compute_budget(num_tracks=100, actual_fps=10.0)
        assert 40 <= budget <= 60
    
    def test_budget_at_low_fps(self):
        cfg = SchedulerConfig(budget_policy="adaptive", fps_low=3.0)
        scheduler = FaceScheduler(cfg)
        
        budget = scheduler._compute_budget(num_tracks=100, actual_fps=4.0)
        assert 10 <= budget <= 30
    
    def test_budget_minimum_one(self):
        cfg = SchedulerConfig(budget_policy="adaptive")
        scheduler = FaceScheduler(cfg)
        
        budget = scheduler._compute_budget(num_tracks=1, actual_fps=1.0)
        assert budget >= 1
    
    def test_fixed_budget_policy(self):
        cfg = SchedulerConfig(budget_policy="fixed", fixed_budget_per_frame=5)
        scheduler = FaceScheduler(cfg)
        
        budget_at_30fps = scheduler._compute_budget(100, 30.0)
        budget_at_3fps = scheduler._compute_budget(100, 3.0)
        
        assert budget_at_30fps == 5
        assert budget_at_3fps == 5


class TestPriorityScoring:
    
    def test_pending_higher_priority_than_confirmed(self):
        cfg = SchedulerConfig()
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1, 2]
        binding_states = {1: "PENDING", 2: "CONFIRMED_STRONG"}
        ts = 1000.0
        
        scores = scheduler._compute_priority_scores(track_ids, binding_states, ts)
        
        assert scores[1] > scores[2]
    
    def test_unknown_higher_than_weak(self):
        cfg = SchedulerConfig()
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1, 2]
        binding_states = {1: "UNKNOWN", 2: "CONFIRMED_WEAK"}
        ts = 1000.0
        
        scores = scheduler._compute_priority_scores(track_ids, binding_states, ts)
        
        assert scores[1] > scores[2]
    
    def test_time_decay_older_higher(self):
        cfg = SchedulerConfig(min_check_interval_sec=0.5)
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1, 2]
        binding_states = {1: "UNKNOWN", 2: "UNKNOWN"}
        ts_start = 1000.0
        
        scheduler._record_schedule([1, 2], ts_start)
        
        scheduler.track_states[1].last_processed_ts = ts_start + 0.1
        
        scheduler.track_states[2].last_processed_ts = ts_start + 2.0
        
        ts_current = ts_start + 3.0
        scores = scheduler._compute_priority_scores(track_ids, binding_states, ts_current)
        
        assert scores[2] > scores[1]


class TestFairScheduling:
    
    def test_all_tracks_eventually_scheduled(self):
        cfg = SchedulerConfig(budget_policy="fixed", fixed_budget_per_frame=2)
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1, 2, 3, 4, 5]
        binding_states = {tid: "UNKNOWN" for tid in track_ids}
        
        scheduled_ever = set()
        
        for frame in range(50):
            context = scheduler.compute_schedule(
                track_ids=track_ids,
                binding_states=binding_states,
                current_ts=1000.0 + frame * 0.033,
                actual_fps=30.0,
            )
            scheduled_ever.update(context.scheduled_track_ids)
        
        assert scheduled_ever == set(track_ids)
    
    def test_pending_tracks_scheduled_frequently(self):
        cfg = SchedulerConfig(budget_policy="fixed", fixed_budget_per_frame=3)
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1, 2, 3, 4, 5]
        
        pending_count = {1: 0, 2: 0}
        confirmed_count = {3: 0, 4: 0, 5: 0}
        
        for frame in range(30):
            binding_states = {
                1: "PENDING", 2: "PENDING",
                3: "CONFIRMED_STRONG", 4: "CONFIRMED_STRONG", 5: "CONFIRMED_STRONG"
            }
            
            context = scheduler.compute_schedule(
                track_ids=track_ids,
                binding_states=binding_states,
                current_ts=1000.0 + frame * 0.033,
                actual_fps=30.0,
            )
            
            for tid in context.scheduled_track_ids:
                if tid in pending_count:
                    pending_count[tid] += 1
                else:
                    confirmed_count[tid] += 1
        
        avg_pending = sum(pending_count.values()) / len(pending_count)
        avg_confirmed = sum(confirmed_count.values()) / len(confirmed_count)
        
        assert avg_pending > avg_confirmed


class TestMinimumInterval:
    
    def test_min_interval_respected(self):
        cfg = SchedulerConfig(
            budget_policy="fixed",
            fixed_budget_per_frame=10,
            min_check_interval_sec=1.0,
        )
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1, 2, 3]
        binding_states = {1: "UNKNOWN", 2: "UNKNOWN", 3: "UNKNOWN"}
        ts = 1000.0
        
        context1 = scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=ts,
            actual_fps=30.0,
        )
        scheduled_frame_1 = context1.scheduled_track_ids
        assert len(scheduled_frame_1) == 3
        
        context2 = scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=ts + 0.01,
            actual_fps=30.0,
        )
        scheduled_frame_2 = context2.scheduled_track_ids
        
        context3 = scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=ts + 1.5,
            actual_fps=30.0,
        )
        scheduled_frame_3 = context3.scheduled_track_ids
        assert len(scheduled_frame_3) >= 1


class TestBypass:
    
    def test_disabled_schedules_all(self):
        cfg = SchedulerConfig(enabled=False)
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1, 2, 3, 4, 5]
        binding_states = {tid: "CONFIRMED_STRONG" for tid in track_ids}
        
        context = scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=1000.0,
            actual_fps=3.0,
        )
        
        assert context.scheduled_track_ids == set(track_ids)


class TestEdgeCases:
    
    def test_empty_track_list(self):
        cfg = SchedulerConfig()
        scheduler = FaceScheduler(cfg)
        
        context = scheduler.compute_schedule(
            track_ids=[],
            binding_states={},
            current_ts=1000.0,
            actual_fps=30.0,
        )
        
        assert context.scheduled_track_ids == set()
    
    def test_single_track(self):
        cfg = SchedulerConfig()
        scheduler = FaceScheduler(cfg)
        
        context = scheduler.compute_schedule(
            track_ids=[1],
            binding_states={1: "UNKNOWN"},
            current_ts=1000.0,
            actual_fps=30.0,
        )
        
        assert 1 in context.scheduled_track_ids
    
    def test_unknown_binding_state(self):
        cfg = SchedulerConfig()
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1]
        binding_states = {1: "UNKNOWN_STATE"}
        
        context = scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=1000.0,
            actual_fps=30.0,
        )
        
        assert 1 in context.scheduled_track_ids
    
    def test_missing_binding_state(self):
        cfg = SchedulerConfig()
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1, 2]
        binding_states = {1: "PENDING"}
        
        context = scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=1000.0,
            actual_fps=30.0,
        )
        
        assert len(context.scheduled_track_ids) >= 1


class TestConfiguration:
    
    def test_config_from_dict(self):
        from core.scheduler import create_scheduler_from_config
        
        config_dict = {
            "enabled": True,
            "budget_policy": "fixed",
            "fixed_budget_per_frame": 5,
            "priority_weights": {
                "unknown": 60,
                "pending": 90,
            },
        }
        
        scheduler = create_scheduler_from_config(config_dict)
        
        assert scheduler.config.enabled == True
        assert scheduler.config.budget_policy == "fixed"
        assert scheduler.config.fixed_budget_per_frame == 5
        assert scheduler.config.priority_weight_unknown == 60
    
    def test_invalid_config_fallback(self):
        from core.scheduler import create_scheduler_from_config
        
        config_dict = {"invalid": "config"}
        
        scheduler = create_scheduler_from_config(config_dict)
        
        assert scheduler.config is not None
        assert scheduler.config.enabled == True


class TestMetrics:
    
    def test_schedule_state_tracking(self):
        cfg = SchedulerConfig()
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1]
        binding_states = {1: "UNKNOWN"}
        ts = 1000.0
        
        context = scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=ts,
            actual_fps=30.0,
        )
        
        track_state = scheduler.get_track_schedule_state(1)
        assert track_state is not None
        assert track_state.schedule_count >= 1
        assert track_state.last_scheduled_ts == ts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
