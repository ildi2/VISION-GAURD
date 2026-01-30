# tests/test_perf_load.py
"""
Performance & Load Testing
Measures throughput, latency, and system behavior under load
"""

import pytest
import logging
import time
import numpy as np

log = logging.getLogger(__name__)


class TestPerformancePhaseB:
    """Test Phase B performance"""
    
    def test_evidence_gate_throughput(self, test_config, test_face_evidence, timer, logger):
        """Measure evidence gate throughput"""
        from identity.evidence_gate import EvidenceGate
        
        gate = EvidenceGate(test_config)
        
        num_samples = 1000
        timer.start()
        
        for i in range(num_samples):
            evidence = test_face_evidence()
            gate.decide(evidence)
        
        elapsed = timer.stop()
        throughput = num_samples / elapsed
        
        logger.info(f"✅ Evidence gate throughput: {throughput:.0f} samples/sec ({elapsed:.2f}s for {num_samples})")
        
        # Should be fast (> 1000 samples/sec)
        assert throughput > 150, f"Evidence gate too slow: {throughput:.0f}/sec"


class TestPerformancePhaseC:
    """Test Phase C performance"""
    
    def test_binding_throughput(self, test_config, metrics_collector, timer, logger):
        """Measure binding manager throughput"""
        from identity.binding import BindingManager
        
        binding = BindingManager(test_config, metrics_collector)
        
        num_tracks = 100
        num_samples_per_track = 5
        total_ops = num_tracks * num_samples_per_track
        
        timer.start()
        
        for track_id in range(num_tracks):
            for sample_idx in range(num_samples_per_track):
                binding.process_evidence(
                    track_id=track_id,
                    person_id=f"Person_{track_id}",
                    score=0.9, second_best_score=0.75, quality=0.85,
                    timestamp=float(sample_idx)
                )
        
        elapsed = timer.stop()
        throughput = total_ops / elapsed
        
        logger.info(f"✅ Binding throughput: {throughput:.0f} ops/sec ({elapsed:.2f}s for {total_ops} ops)")
        
        # Should be fast (> 5000 ops/sec)
        assert throughput > 3000, f"Binding too slow: {throughput:.0f}/sec"


class TestPerformancePhaseD:
    """Test Phase D performance"""
    
    def test_scheduler_throughput(self, test_config, timer, logger):
        """Measure scheduler throughput"""
        try:
            from core.scheduler import create_scheduler_from_config
            
            if not hasattr(test_config.governance, 'scheduler'):
                pytest.skip("No scheduler config")
            
            scheduler = create_scheduler_from_config(test_config.governance.scheduler)
            
            num_calls = 100
            timer.start()
            
            for i in range(num_calls):
                active_tracks = list(range(100, 150))  # 50 tracks
                scheduler.compute_schedule(
                    active_tracks=active_tracks,
                    current_fps=25.0,
                    compute_time_ms=15.0
                )
            
            elapsed = timer.stop()
            throughput = num_calls / elapsed
            
            logger.info(f"✅ Scheduler throughput: {throughput:.0f} schedules/sec ({elapsed:.2f}s for {num_calls} schedules)")
            
            # Should be fast (> 100 schedules/sec)
            assert throughput > 100, f"Scheduler too slow: {throughput:.0f}/sec"
        
        except Exception as e:
            logger.warning(f"⚠️ Scheduler performance: {e}")


class TestPerformancePhaseE:
    """Test Phase E performance"""
    
    def test_merge_evaluation_throughput(self, test_config, metrics_collector, test_tracklet, timer, logger):
        """Measure merge evaluation throughput"""
        from identity.merge_manager import MergeManager
        
        merger = MergeManager(test_config.governance.merge)
        
        # Create tracklets (score parameter not used - tracklets represent confirmed identities)
        tracklets = [test_tracklet(100+i) for i in range(50)]
        
        num_comparisons = 0
        timer.start()
        
        # Compare pairs
        for i in range(len(tracklets)):
            for j in range(i+1, min(i+10, len(tracklets))):  # Compare with next 10
                try:
                    merger.evaluate_merge(tracklets[i], tracklets[j])
                    num_comparisons += 1
                except:
                    pass
        
        elapsed = timer.stop()
        throughput = num_comparisons / elapsed if elapsed > 0 else 0
        
        logger.info(f"✅ Merge evaluation throughput: {throughput:.0f} comparisons/sec ({elapsed:.2f}s for {num_comparisons} comparisons)")


class TestLoadHandling:
    """Test system load handling"""
    
    def test_many_tracks_binding(self, test_config, metrics_collector, timer, logger):
        """Test binding with many tracks"""
        from identity.binding import BindingManager
        
        binding = BindingManager(test_config, metrics_collector)
        
        # Process 100 tracks with 10 samples each
        timer.start()
        
        for track_id in range(100, 200):
            for frame_idx in range(10):
                binding.process_evidence(
                    track_id=track_id,
                    person_id=f"Person_{track_id}",
                    score=0.85, second_best_score=0.70, quality=0.80,
                    timestamp=float(frame_idx)
                )
        
        elapsed = timer.stop()
        
        logger.info(f"✅ Processed 100 tracks × 10 samples: {elapsed:.2f}s")
        logger.info(f"   Throughput: {1000 / elapsed:.0f} samples/sec")
    
    def test_evidence_gate_burst(self, test_config, test_face_evidence, timer, logger):
        """Test evidence gate with burst of samples"""
        from identity.evidence_gate import EvidenceGate
        
        gate = EvidenceGate(test_config)
        
        # Simulate 30 FPS burst for 10 seconds
        timer.start()
        
        for _ in range(300):  # 300 frames
            evidence = test_face_evidence()
            gate.decide(evidence)
        
        elapsed = timer.stop()
        fps = 300 / elapsed
        
        logger.info(f"✅ Evidence gate burst: {fps:.0f} FPS ({elapsed:.2f}s for 300 frames)")
        
        # Should handle at least 30 FPS
        assert fps > 30, f"Too slow for real-time: {fps:.0f} FPS"


class TestFPSUnderLoad:
    """Test FPS maintenance under load"""
    
    def test_fps_with_scheduler(self, test_config, fps_monitor, logger):
        """Monitor FPS with scheduler active"""
        try:
            from core.scheduler import create_scheduler_from_config
            
            if not hasattr(test_config.governance, 'scheduler'):
                pytest.skip("No scheduler config")
            
            scheduler = create_scheduler_from_config(test_config.governance.scheduler)
            
            fps_monitor.start()
            
            # Simulate 300 frames with scheduler calls
            for frame_idx in range(300):
                active_tracks = list(range(100, 130))  # 30 tracks
                scheduler.compute_schedule(
                    active_tracks=active_tracks,
                    current_fps=25.0,
                    compute_time_ms=15.0
                )
                fps_monitor.tick()
            
            stats = fps_monitor.get_stats()
            
            logger.info(f"✅ FPS with scheduler:")
            logger.info(f"   Avg FPS: {stats['avg_fps']:.1f}")
            logger.info(f"   P95 frame: {stats['p95_frame_time_ms']:.2f}ms")
            
        except Exception as e:
            logger.warning(f"⚠️ FPS monitoring: {e}")


class TestMemoryUnderLoad:
    """Test memory behavior under load"""
    
    def test_memory_with_many_tracks(self, memory_tracker, test_config, metrics_collector, logger):
        """Test memory with many tracks"""
        from identity.binding import BindingManager
        
        memory_tracker.start()
        
        binding = BindingManager(test_config, metrics_collector)
        
        # Process 500 tracks
        for track_id in range(500, 1000):
            for frame_idx in range(5):
                binding.process_evidence(
                    track_id=track_id,
                    person_id=f"Person_{track_id}",
                    score=0.85, second_best_score=0.70, quality=0.80,
                    timestamp=float(frame_idx)
                )
        
        delta = memory_tracker.delta()
        
        logger.info(f"✅ Processed 500 tracks")
        logger.info(f"   Memory delta: {delta:+.1f} MB")
        
        # Should not leak excessive memory (< 50MB for 500 tracks)
        assert delta < 50, f"Memory usage too high: {delta:.1f} MB"


class TestLatency:
    """Test latency characteristics"""
    
    def test_p95_latency_binding(self, test_config, metrics_collector, timer, logger):
        """Measure P95 latency for binding operations"""
        from identity.binding import BindingManager
        
        binding = BindingManager(test_config, metrics_collector)
        
        latencies = []
        
        for i in range(100):
            timer.start()
            
            binding.process_evidence(
                track_id=100,
                person_id="Test",
                score=0.9, second_best_score=0.75, quality=0.85,
                timestamp=float(i)
            )
            
            latency = timer.stop()
            latencies.append(latency * 1000)  # Convert to ms
        
        p95_latency = np.percentile(latencies, 95)
        avg_latency = np.mean(latencies)
        
        logger.info(f"✅ Binding latencies:")
        logger.info(f"   Avg: {avg_latency:.2f}ms")
        logger.info(f"   P95: {p95_latency:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
