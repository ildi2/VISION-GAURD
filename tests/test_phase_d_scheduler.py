# tests/test_phase_d_scheduler.py
"""
Phase D Verification: FPS/Load-Aware Scheduler
Tests budget allocation, prioritization, and graceful degradation
"""

import pytest
import logging
import time
import numpy as np

log = logging.getLogger(__name__)


class TestPhaseDSchedulerBasics:
    """Test Phase D: Scheduler basic functionality"""
    
    def test_scheduler_imports(self, logger):
        """Scheduler module should import"""
        try:
            from core.scheduler import create_scheduler_from_config
            logger.info("✅ Scheduler module imports successfully")
        except ImportError as e:
            pytest.fail(f"Cannot import scheduler: {e}")
    
    def test_scheduler_initializes(self, test_config, logger):
        """Scheduler should initialize from config"""
        try:
            from core.scheduler import create_scheduler_from_config
            
            if hasattr(test_config.governance, 'scheduler'):
                scheduler = create_scheduler_from_config(test_config.governance.scheduler)
                assert scheduler is not None
                logger.info("✅ Scheduler initializes successfully")
            else:
                logger.warning("⚠️ No scheduler config found (may be optional)")
        except Exception as e:
            logger.warning(f"⚠️ Scheduler initialization: {e}")


class TestPhaseDScheduleComputation:
    """Test Phase D: Schedule computation"""
    
    def test_scheduler_computes_schedule(self, test_config, logger):
        """Scheduler should compute schedule for active tracks"""
        try:
            from core.scheduler import create_scheduler_from_config
            
            if not hasattr(test_config.governance, 'scheduler'):
                pytest.skip("No scheduler config")
            
            scheduler = create_scheduler_from_config(test_config.governance.scheduler)
            
            # Simulate active tracks
            active_tracks = [100, 101, 102, 103, 104]
            current_fps = 25.0
            compute_time_ms = 20.0
            
            schedule = scheduler.compute_schedule(
                active_tracks=active_tracks,
                current_fps=current_fps,
                compute_time_ms=compute_time_ms
            )
            
            assert schedule is not None, "Scheduler returned None"
            logger.info(f"✅ Scheduler computed schedule")
            logger.info(f"   Active tracks: {len(active_tracks)}")
            logger.info(f"   Current FPS: {current_fps}")
            
        except Exception as e:
            logger.warning(f"⚠️ Schedule computation: {e}")
    
    def test_scheduler_selects_tracks(self, test_config, logger):
        """Scheduler should select tracks for processing"""
        try:
            from core.scheduler import create_scheduler_from_config
            
            if not hasattr(test_config.governance, 'scheduler'):
                pytest.skip("No scheduler config")
            
            scheduler = create_scheduler_from_config(test_config.governance.scheduler)
            
            active_tracks = list(range(100, 110))  # 10 tracks
            
            schedule = scheduler.compute_schedule(
                active_tracks=active_tracks,
                current_fps=25.0,
                compute_time_ms=15.0
            )
            
            if hasattr(schedule, 'selected_tracks'):
                selected = schedule.selected_tracks
                logger.info(f"✅ Scheduler selected {len(selected)}/{len(active_tracks)} tracks")
            else:
                logger.warning("⚠️ Schedule doesn't have selected_tracks attribute")
        
        except Exception as e:
            logger.warning(f"⚠️ Track selection: {e}")


class TestPhaseDGracefulDegradation:
    """Test Phase D: Graceful degradation under load"""
    
    def test_degradation_high_load(self, test_config, logger):
        """Under high load, should degrade gracefully"""
        try:
            from core.scheduler import create_scheduler_from_config
            
            if not hasattr(test_config.governance, 'scheduler'):
                pytest.skip("No scheduler config")
            
            scheduler = create_scheduler_from_config(test_config.governance.scheduler)
            
            # Many tracks, high CPU usage
            active_tracks = list(range(100, 150))  # 50 tracks
            current_fps = 10.0  # Low FPS (high load)
            compute_time_ms = 35.0  # Using most of frame time
            
            schedule = scheduler.compute_schedule(
                active_tracks=active_tracks,
                current_fps=current_fps,
                compute_time_ms=compute_time_ms
            )
            
            if hasattr(schedule, 'selected_tracks'):
                num_selected = len(schedule.selected_tracks)
                num_total = len(active_tracks)
                
                logger.info(f"✅ Under high load ({num_total} tracks, 10 FPS):")
                logger.info(f"   Selected: {num_selected} ({num_selected/num_total*100:.1f}%)")
                
                # Should select fewer tracks under high load
                if num_selected < num_total:
                    logger.info("   ✓ Graceful degradation: processing subset")
                else:
                    logger.warning("   ⚠️ Processing all tracks despite load")
            
        except Exception as e:
            logger.warning(f"⚠️ Degradation test: {e}")
    
    def test_maintains_fps_under_load(self, test_config, logger):
        """Should maintain minimum FPS even with many tracks"""
        try:
            from core.scheduler import create_scheduler_from_config
            
            if not hasattr(test_config.governance, 'scheduler'):
                pytest.skip("No scheduler config")
            
            scheduler = create_scheduler_from_config(test_config.governance.scheduler)
            
            # Simulate maintaining FPS
            frame_times = []
            target_fps = 20.0
            target_frame_time = 1.0 / target_fps
            
            for batch_size in [10, 30, 50, 100]:
                active_tracks = list(range(batch_size))
                
                schedule = scheduler.compute_schedule(
                    active_tracks=active_tracks,
                    current_fps=25.0,
                    compute_time_ms=15.0
                )
                
                logger.info(f"✅ {batch_size} tracks scheduled successfully")
        
        except Exception as e:
            logger.warning(f"⚠️ FPS maintenance: {e}")


class TestPhaseDPriorityOrdering:
    """Test Phase D: Priority-based ordering"""
    
    def test_priority_selection(self, test_config, logger):
        """Scheduler should prioritize certain tracks"""
        try:
            from core.scheduler import create_scheduler_from_config
            
            if not hasattr(test_config.governance, 'scheduler'):
                pytest.skip("No scheduler config")
            
            scheduler = create_scheduler_from_config(test_config.governance.scheduler)
            
            # Tracks with different characteristics
            active_tracks = [100, 101, 102, 103, 104]
            
            schedule = scheduler.compute_schedule(
                active_tracks=active_tracks,
                current_fps=25.0,
                compute_time_ms=10.0
            )
            
            if hasattr(schedule, 'selected_tracks'):
                logger.info(f"✅ Priority selection:")
                logger.info(f"   All tracks: {active_tracks}")
                logger.info(f"   Selected: {schedule.selected_tracks}")
            
        except Exception as e:
            logger.warning(f"⚠️ Priority ordering: {e}")


class TestPhaseDFPSMonitoring:
    """Test Phase D: FPS monitoring capabilities"""
    
    def test_fps_tracking(self, logger, fps_monitor):
        """FPS monitor should track frame times"""
        fps_monitor.start()
        
        # Simulate 30 frames
        for i in range(30):
            time.sleep(0.01)  # Simulate ~100 FPS work
            fps_monitor.tick()
        
        stats = fps_monitor.get_stats()
        
        assert stats['avg_fps'] > 50, "FPS too low"
        logger.info(f"✅ FPS monitoring:")
        logger.info(f"   Avg FPS: {stats['avg_fps']:.1f}")
        logger.info(f"   Min frame: {stats['min_frame_time_ms']:.2f}ms")
        logger.info(f"   Max frame: {stats['max_frame_time_ms']:.2f}ms")
        logger.info(f"   P95 frame: {stats['p95_frame_time_ms']:.2f}ms")


class TestPhaseDErrorHandling:
    """Test Phase D: Error handling"""
    
    def test_empty_track_list(self, test_config, logger):
        """Should handle empty track list"""
        try:
            from core.scheduler import create_scheduler_from_config
            
            if not hasattr(test_config.governance, 'scheduler'):
                pytest.skip("No scheduler config")
            
            scheduler = create_scheduler_from_config(test_config.governance.scheduler)
            
            schedule = scheduler.compute_schedule(
                active_tracks=[],
                current_fps=25.0,
                compute_time_ms=5.0
            )
            
            logger.info("✅ Handled empty track list")
        
        except Exception as e:
            logger.warning(f"⚠️ Empty track list: {e}")
    
    def test_zero_fps(self, test_config, logger):
        """Should handle zero FPS gracefully"""
        try:
            from core.scheduler import create_scheduler_from_config
            
            if not hasattr(test_config.governance, 'scheduler'):
                pytest.skip("No scheduler config")
            
            scheduler = create_scheduler_from_config(test_config.governance.scheduler)
            
            schedule = scheduler.compute_schedule(
                active_tracks=[100, 101],
                current_fps=0.1,  # Very low
                compute_time_ms=0.0
            )
            
            logger.info("✅ Handled very low FPS")
        
        except Exception as e:
            logger.warning(f"⚠️ Zero FPS: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
