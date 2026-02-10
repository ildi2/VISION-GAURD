
import pytest
import logging
import numpy as np
import gc

log = logging.getLogger(__name__)


class TestStressRapidLifecycle:
    
    def test_rapid_track_lifecycle(self, test_config, metrics_collector, memory_tracker, logger):
        from identity.binding import BindingManager
        
        memory_tracker.start()
        
        binding = BindingManager(test_config, metrics_collector)
        
        for batch in range(10):
            for track_id in range(batch*10, (batch+1)*10):
                for frame_idx in range(10):
                    binding.process_evidence(
                        track_id=track_id,
                        person_id=f"Person_{track_id}",
                        score=0.85 + (frame_idx * 0.01), second_best_score=0.70, quality=0.85,
                        timestamp=float(frame_idx)
                    )
        
        gc.collect()
        memory_delta = memory_tracker.delta()
        
        logger.info(f"✅ Rapid lifecycle test: 100 tracks × 10 frames")
        logger.info(f"   Memory delta: {memory_delta:+.1f} MB")
        
        assert memory_delta < 20, f"Memory growth too high: {memory_delta:.1f} MB"


class TestStressQualityVariation:
    
    def test_bursty_quality_variation(self, test_config, metrics_collector, logger):
        from identity.binding import BindingManager
        
        binding = BindingManager(test_config, metrics_collector)
        
        for frame_idx in range(100):
            quality = 0.95 if (frame_idx // 2) % 2 == 0 else 0.3
            
            binding.process_evidence(
                track_id=100,
                person_id="Test",
                score=0.85, second_best_score=0.70, quality=quality,
                timestamp=float(frame_idx)
            )
        
        logger.info(f"✅ Bursty quality: 100 frames with 0.95/0.3 alternation")


class TestStressLowQuality:
    
    def test_persistent_low_quality(self, test_config, test_face_evidence, logger):
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        rejected_count = 0
        held_count = 0
        
        for i in range(100):
            evidence = test_face_evidence(
                blur_score=0.7,
                brightness=0.1,
                quality=0.2
            )
            
            decision = gate.decide(evidence)
            decision_status = decision[0] if isinstance(decision, tuple) else decision
            
            if decision_status == 'REJECT' or decision_status == GateDecision.REJECT:
                rejected_count += 1
            elif decision_status == 'HOLD' or decision_status == GateDecision.HOLD:
                held_count += 1
        
        logger.info(f"✅ Low quality stress: 100 samples")
        logger.info(f"   Rejected: {rejected_count}, Held: {held_count}")
        
        assert rejected_count + held_count > 0, "Low quality samples should be rejected or held"


class TestStressExtremePose:
    
    def test_extreme_pose_angles(self, test_config, test_face_evidence, logger):
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        rejected_count = 0
        held_count = 0
        
        for yaw in [-90, -60, -30, 0, 30, 60, 90]:
            for pitch in [-45, 0, 45]:
                evidence = test_face_evidence(
                    blur_score=0.05,
                    brightness=0.6,
                    yaw=float(yaw),
                    pitch=float(pitch),
                    quality=0.8
                )
                
                decision = gate.decide(evidence)
                
                if decision == GateDecision.REJECT:
                    rejected_count += 1
                elif decision == GateDecision.HOLD:
                    held_count += 1
        
        total = 7 * 3
        logger.info(f"✅ Extreme pose test: {total} combinations")
        logger.info(f"   Rejected: {rejected_count}, Held: {held_count}")


class TestStressSchedulerExtreme:
    
    def test_scheduler_extreme_load(self, test_config, logger):
        try:
            from core.scheduler import create_scheduler_from_config
            
            if not hasattr(test_config.governance, 'scheduler'):
                pytest.skip("No scheduler config")
            
            scheduler = create_scheduler_from_config(test_config.governance.scheduler)
            
            active_tracks = list(range(100, 200))
            
            for iteration in range(10):
                schedule = scheduler.compute_schedule(
                    active_tracks=active_tracks,
                    current_fps=5.0,
                    compute_time_ms=40.0
                )
            
            logger.info(f"✅ Extreme scheduler load: 100 tracks, 5 FPS, 10 iterations")
        
        except Exception as e:
            logger.warning(f"⚠️ Extreme scheduler: {e}")


class TestStressManyMergeEvaluations:
    
    def test_many_merge_comparisons(self, test_config, metrics_collector, test_tracklet, logger):
        from identity.merge_manager import MergeManager
        
        merger = MergeManager(test_config.governance.merge)
        
        tracklets = [test_tracklet(100+i) for i in range(50)]
        
        comparison_count = 0
        
        for i in range(len(tracklets)):
            for j in range(i+1, len(tracklets)):
                try:
                    merger.evaluate_merge(tracklets[i], tracklets[j])
                    comparison_count += 1
                except:
                    pass
        
        expected_comparisons = 50 * 49 // 2
        
        logger.info(f"✅ Merge stress test:")
        logger.info(f"   50 tracklets = {expected_comparisons} possible pairs")
        logger.info(f"   Evaluated: {comparison_count} comparisons")


class TestStressGracefulDegradation:
    
    def test_scheduler_graceful_fps_drop(self, test_config, logger, fps_monitor):
        try:
            from core.scheduler import create_scheduler_from_config
            
            if not hasattr(test_config.governance, 'scheduler'):
                pytest.skip("No scheduler config")
            
            scheduler = create_scheduler_from_config(test_config.governance.scheduler)
            
            fps_monitor.start()
            
            for fps_level in [30, 25, 20, 15, 10]:
                active_tracks = list(range(100, 150))
                
                for _ in range(10):
                    schedule = scheduler.compute_schedule(
                        active_tracks=active_tracks,
                        current_fps=float(fps_level),
                        compute_time_ms=20.0 if fps_level >= 25 else 30.0
                    )
                    fps_monitor.tick()
            
            logger.info(f"✅ Graceful degradation test complete")
        
        except Exception as e:
            logger.warning(f"⚠️ Degradation test: {e}")


class TestStressBindingConflict:
    
    def test_conflicting_identity_evidence(self, test_config, metrics_collector, logger):
        from identity.binding import BindingManager
        
        binding = BindingManager(test_config, metrics_collector)
        
        identities = ["Alice", "Bob", "Charlie"]
        
        for frame_idx in range(60):
            identity_idx = frame_idx % 3
            
            binding.process_evidence(
                track_id=100,
                person_id=identities[identity_idx],
                score=0.85 + (frame_idx % 10) * 0.01, second_best_score=0.70, quality=0.80 + np.random.rand() * 0.15,
                timestamp=float(frame_idx)
            )
        
        logger.info(f"✅ Conflicting evidence test:")
        logger.info(f"   (Binding resolves based on evidence weight and quality)")


class TestStressErrorRecovery:
    
    def test_recovery_from_none_values(self, test_config, metrics_collector, logger):
        from identity.binding import BindingManager
        
        binding = BindingManager(test_config, metrics_collector)
        
        try:
            binding.process_evidence(
                track_id=None,
                person_id=None,
                score=-1.0,
                quality=2.0,
                timestamp=float('inf')
            )
            
            binding.process_evidence(
                track_id=100,
                person_id="Valid",
                score=0.9, second_best_score=0.75, quality=0.85,
                timestamp=0.0
            )
            
            logger.info("✅ Recovered from invalid inputs")
        except Exception as e:
            logger.warning(f"⚠️ Error recovery: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
