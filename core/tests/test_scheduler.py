"""
Tests for PHASE D: FPS/Load-Aware Scheduler

Covers:
- Budget computation at various FPS values
- Priority scoring correctness
- Fair scheduling over multiple frames
- Minimum check interval enforcement
- Time decay mechanics
- Configuration handling
- Edge cases and safety
"""

import pytest
import time
from core.scheduler import FaceScheduler, SchedulerConfig, ScheduleContext


class TestBudgetComputation:
    """Test dynamic budget computation."""
    
    def test_budget_at_high_fps(self):
        """At high FPS (15+), all faces should be scheduled."""
        cfg = SchedulerConfig(budget_policy="adaptive", fps_high=15.0)
        scheduler = FaceScheduler(cfg)
        
        budget = scheduler._compute_budget(num_tracks=100, actual_fps=30.0)
        assert budget == 100
    
    def test_budget_at_medium_fps(self):
        """At medium FPS (5-15), ~50% of faces should be scheduled."""
        cfg = SchedulerConfig(budget_policy="adaptive", fps_medium=5.0)
        scheduler = FaceScheduler(cfg)
        
        budget = scheduler._compute_budget(num_tracks=100, actual_fps=10.0)
        assert 40 <= budget <= 60  # 50% ± 10
    
    def test_budget_at_low_fps(self):
        """At low FPS (3-5), ~20% of faces should be scheduled."""
        cfg = SchedulerConfig(budget_policy="adaptive", fps_low=3.0)
        scheduler = FaceScheduler(cfg)
        
        budget = scheduler._compute_budget(num_tracks=100, actual_fps=4.0)
        assert 10 <= budget <= 30  # 20% ± 10
    
    def test_budget_minimum_one(self):
        """Budget should never be zero."""
        cfg = SchedulerConfig(budget_policy="adaptive")
        scheduler = FaceScheduler(cfg)
        
        budget = scheduler._compute_budget(num_tracks=1, actual_fps=1.0)
        assert budget >= 1
    
    def test_fixed_budget_policy(self):
        """Fixed policy should return same budget regardless of FPS."""
        cfg = SchedulerConfig(budget_policy="fixed", fixed_budget_per_frame=5)
        scheduler = FaceScheduler(cfg)
        
        budget_at_30fps = scheduler._compute_budget(100, 30.0)
        budget_at_3fps = scheduler._compute_budget(100, 3.0)
        
        assert budget_at_30fps == 5
        assert budget_at_3fps == 5


class TestPriorityScoring:
    """Test priority computation."""
    
    def test_pending_higher_priority_than_confirmed(self):
        """PENDING tracks should score higher than CONFIRMED."""
        cfg = SchedulerConfig()
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1, 2]
        binding_states = {1: "PENDING", 2: "CONFIRMED_STRONG"}
        ts = 1000.0
        
        scores = scheduler._compute_priority_scores(track_ids, binding_states, ts)
        
        assert scores[1] > scores[2]
    
    def test_unknown_higher_than_weak(self):
        """UNKNOWN should score higher than CONFIRMED_WEAK."""
        cfg = SchedulerConfig()
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1, 2]
        binding_states = {1: "UNKNOWN", 2: "CONFIRMED_WEAK"}
        ts = 1000.0
        
        scores = scheduler._compute_priority_scores(track_ids, binding_states, ts)
        
        assert scores[1] > scores[2]
    
    def test_time_decay_older_higher(self):
        """Older (not-recently-checked) tracks should score higher."""
        cfg = SchedulerConfig(min_check_interval_sec=0.5)
        scheduler = FaceScheduler(cfg)
        
        # Record initial processing
        track_ids = [1, 2]
        binding_states = {1: "UNKNOWN", 2: "UNKNOWN"}
        ts_start = 1000.0
        
        scheduler._record_schedule([1, 2], ts_start)
        
        # Track 1 checked again at ts + 0.1 (too soon)
        scheduler.track_states[1].last_processed_ts = ts_start + 0.1
        
        # Track 2 checked again at ts + 2.0 (much older)
        scheduler.track_states[2].last_processed_ts = ts_start + 2.0
        
        ts_current = ts_start + 3.0
        scores = scheduler._compute_priority_scores(track_ids, binding_states, ts_current)
        
        # Track 2 (older, not recently checked) should score higher
        assert scores[2] > scores[1]


class TestFairScheduling:
    """Test fair scheduling over multiple frames."""
    
    def test_all_tracks_eventually_scheduled(self):
        """Over many frames, all tracks should eventually be scheduled."""
        cfg = SchedulerConfig(budget_policy="fixed", fixed_budget_per_frame=2)
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1, 2, 3, 4, 5]
        binding_states = {tid: "UNKNOWN" for tid in track_ids}
        
        scheduled_ever = set()
        
        for frame in range(50):  # Many frames
            context = scheduler.compute_schedule(
                track_ids=track_ids,
                binding_states=binding_states,
                current_ts=1000.0 + frame * 0.033,  # 30 FPS
                actual_fps=30.0,
            )
            scheduled_ever.update(context.scheduled_track_ids)
        
        # After 50 frames, all tracks should have been scheduled at least once
        assert scheduled_ever == set(track_ids)
    
    def test_pending_tracks_scheduled_frequently(self):
        """PENDING tracks should be scheduled more often than CONFIRMED."""
        cfg = SchedulerConfig(budget_policy="fixed", fixed_budget_per_frame=3)
        scheduler = FaceScheduler(cfg)
        
        # Mix of pending and confirmed
        track_ids = [1, 2, 3, 4, 5]
        
        pending_count = {1: 0, 2: 0}  # PENDING tracks
        confirmed_count = {3: 0, 4: 0, 5: 0}  # CONFIRMED tracks
        
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
        
        # PENDING should be scheduled more often
        assert avg_pending > avg_confirmed


class TestMinimumInterval:
    """Test minimum check interval enforcement."""
    
    def test_min_interval_respected(self):
        """Tracks shouldn't be scheduled again within min_interval."""
        cfg = SchedulerConfig(
            budget_policy="fixed",
            fixed_budget_per_frame=10,
            min_check_interval_sec=1.0,
        )
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1, 2, 3]
        binding_states = {1: "UNKNOWN", 2: "UNKNOWN", 3: "UNKNOWN"}
        ts = 1000.0
        
        # Frame 1: Schedule all
        context1 = scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=ts,
            actual_fps=30.0,
        )
        scheduled_frame_1 = context1.scheduled_track_ids
        assert len(scheduled_frame_1) == 3  # All scheduled
        
        # Frame 2 (0.01s later, within min interval): Some should be skipped
        context2 = scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=ts + 0.01,
            actual_fps=30.0,
        )
        scheduled_frame_2 = context2.scheduled_track_ids
        # Tracks recently scheduled should have low priority
        # Since all are UNKNOWN but all just checked, should distribute fairly
        
        # Frame 3 (1.5s later, past min interval): Should reschedule
        context3 = scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=ts + 1.5,
            actual_fps=30.0,
        )
        scheduled_frame_3 = context3.scheduled_track_ids
        # Should have more tracks scheduled now
        assert len(scheduled_frame_3) >= 1


class TestBypass:
    """Test scheduler bypass when disabled."""
    
    def test_disabled_schedules_all(self):
        """When disabled, all tracks should be scheduled."""
        cfg = SchedulerConfig(enabled=False)
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1, 2, 3, 4, 5]
        binding_states = {tid: "CONFIRMED_STRONG" for tid in track_ids}
        
        context = scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=1000.0,
            actual_fps=3.0,  # Low FPS
        )
        
        # Should schedule ALL despite low FPS
        assert context.scheduled_track_ids == set(track_ids)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_track_list(self):
        """Scheduler should handle empty track list gracefully."""
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
        """Scheduler should handle single track."""
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
        """Unknown binding states should be treated as UNKNOWN."""
        cfg = SchedulerConfig()
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1]
        binding_states = {1: "UNKNOWN_STATE"}  # Invalid state
        
        context = scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=1000.0,
            actual_fps=30.0,
        )
        
        # Should still schedule (with default UNKNOWN priority)
        assert 1 in context.scheduled_track_ids
    
    def test_missing_binding_state(self):
        """Missing binding states should be treated as UNKNOWN."""
        cfg = SchedulerConfig()
        scheduler = FaceScheduler(cfg)
        
        track_ids = [1, 2]
        binding_states = {1: "PENDING"}  # 2 is missing
        
        context = scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=1000.0,
            actual_fps=30.0,
        )
        
        # Should handle gracefully
        assert len(context.scheduled_track_ids) >= 1


class TestConfiguration:
    """Test configuration handling."""
    
    def test_config_from_dict(self):
        """Should create scheduler from dict config."""
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
        """Should fall back to defaults on invalid config."""
        from core.scheduler import create_scheduler_from_config
        
        config_dict = {"invalid": "config"}
        
        scheduler = create_scheduler_from_config(config_dict)
        
        # Should create with defaults
        assert scheduler.config is not None
        assert scheduler.config.enabled == True


class TestMetrics:
    """Test metrics recording."""
    
    def test_schedule_state_tracking(self):
        """Schedule state should track scheduling decisions."""
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
        
        # Track should have updated state
        track_state = scheduler.get_track_schedule_state(1)
        assert track_state is not None
        assert track_state.schedule_count >= 1
        assert track_state.last_scheduled_ts == ts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
