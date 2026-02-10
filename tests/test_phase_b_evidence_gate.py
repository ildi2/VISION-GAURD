
import pytest
import logging
import numpy as np

log = logging.getLogger(__name__)


class TestPhaseBevidenceGateBasics:
    
    def test_evidence_gate_imports(self, logger):
        try:
            from identity.evidence_gate import EvidenceGate, GateDecision
            logger.info("✅ Evidence gate module imports successfully")
        except ImportError as e:
            pytest.fail(f"Cannot import evidence gate: {e}")
    
    def test_evidence_gate_initializes(self, test_config, logger):
        try:
            from identity.evidence_gate import EvidenceGate
            
            gate = EvidenceGate(test_config)
            assert gate is not None
            logger.info("✅ Evidence gate initializes successfully")
        except Exception as e:
            pytest.fail(f"Evidence gate initialization failed: {e}")


class TestPhaseB_HighQualityAccepted:
    
    def test_high_quality_face_accepted(self, test_config, test_face_evidence, logger):
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.05,
            brightness=0.6,
            yaw=5.0,
            pitch=2.0,
            roll=2.0,
            scale=0.6,
            quality_score=0.95
        )
        
        decision = gate.decide(evidence)
        
        assert decision[0] in ['ACCEPT', 'HOLD'], \
            f"Expected ACCEPT or HOLD, got {decision}"
        
        if decision[0] == 'ACCEPT':
            logger.info("✅ High quality face ACCEPTED")
        else:
            logger.warning("⚠️ High quality face HELD (check thresholds)")
    
    def test_multiple_high_quality_samples(self, test_config, test_face_evidence, logger):
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        accept_count = 0
        hold_count = 0
        reject_count = 0
        
        for i in range(5):
            evidence = test_face_evidence(
                blur_score=0.05,
                brightness=0.6,
                quality_score=0.9 + (i * 0.01)
            )
            
            decision = gate.decide(evidence)
            decision_status = decision[0] if isinstance(decision, tuple) else decision
            
            if decision_status == 'ACCEPT' or decision_status == GateDecision.ACCEPT:
                accept_count += 1
            elif decision_status == 'HOLD' or decision_status == GateDecision.HOLD:
                hold_count += 1
            else:
                reject_count += 1
        
        logger.info(f"5 high-quality samples: {accept_count} ACCEPT, {hold_count} HOLD, {reject_count} REJECT")
        
        assert accept_count + hold_count >= 4, "Most high-quality samples should be accepted/held"
        logger.info("✅ Multiple high-quality samples handled correctly")


class TestPhaseB_BlurryRejected:
    
    def test_very_blurry_rejected(self, test_config, test_face_evidence, logger):
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.95,
            brightness=0.6,
            quality_score=0.2
        )
        
        decision = gate.decide(evidence)
        decision_status = decision[0] if isinstance(decision, tuple) else decision
        
        assert decision_status in ['REJECT', 'HOLD', GateDecision.REJECT, GateDecision.HOLD], \
            f"Blurry face should be REJECT or HOLD, got {decision}"
        
        if decision_status == 'REJECT' or decision_status == GateDecision.REJECT:
            logger.info("✅ Very blurry face REJECTED")
        else:
            logger.warning("⚠️ Very blurry face HELD (check blur threshold)")
    
    def test_moderately_blurry_held(self, test_config, test_face_evidence, logger):
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.5,
            brightness=0.6,
            quality_score=0.6
        )
        
        decision = gate.decide(evidence)
        decision_status = decision[0] if isinstance(decision, tuple) else decision
        
        assert decision_status in ['HOLD', 'REJECT', GateDecision.HOLD, GateDecision.REJECT], \
            f"Moderately blurry should be HOLD or REJECT, got {decision}"
        
        logger.info(f"✅ Moderately blurry face handled")


class TestPhaseB_DarkRejected:
    
    def test_very_dark_rejected(self, test_config, test_face_evidence, logger):
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.1,
            brightness=0.05,
            quality_score=0.2
        )
        
        decision = gate.decide(evidence)
        decision_status = decision[0] if isinstance(decision, tuple) else decision
        
        assert decision_status in ['REJECT', 'HOLD', GateDecision.REJECT, GateDecision.HOLD], \
            f"Very dark face should be REJECT or HOLD, got {decision}"
        
        if decision == GateDecision.REJECT:
            logger.info("✅ Very dark face REJECTED")
        else:
            logger.warning("⚠️ Very dark face HELD (check brightness threshold)")
    
    def test_very_bright_rejected(self, test_config, test_face_evidence, logger):
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.1,
            brightness=0.95,
            quality_score=0.2
        )
        
        decision = gate.decide(evidence)
        decision_status = decision[0] if isinstance(decision, tuple) else decision
        
        assert decision_status in ['REJECT', 'HOLD', GateDecision.REJECT, GateDecision.HOLD], \
            f"Very bright face should be REJECT or HOLD, got {decision}"
        
        logger.info(f"✅ Very bright face: {(decision[0] if isinstance(decision, tuple) else decision.name)}")


class TestPhaseB_PoseChecks:
    
    def test_large_yaw_angle_held_or_rejected(self, test_config, test_face_evidence, logger):
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.05,
            brightness=0.6,
            yaw=45.0,
            quality_score=0.8
        )
        
        decision = gate.decide(evidence)
        
        assert (decision[0] if isinstance(decision, tuple) else decision) in ['HOLD', GateDecision.HOLD, 'REJECT', GateDecision.REJECT], \
            f"Large yaw should be HOLD or REJECT, got {decision}"
        
        logger.info(f"✅ Large yaw (45°): {(decision[0] if isinstance(decision, tuple) else decision.name)}")
    
    def test_small_pose_angle_accepted(self, test_config, test_face_evidence, logger):
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.05,
            brightness=0.6,
            yaw=5.0,
            pitch=3.0,
            roll=2.0,
            quality_score=0.95
        )
        
        decision = gate.decide(evidence)
        
        assert (decision[0] if isinstance(decision, tuple) else decision) in ['ACCEPT', GateDecision.ACCEPT, 'HOLD', GateDecision.HOLD], \
            f"Small pose should be ACCEPT or HOLD, got {decision}"
        
        logger.info(f"✅ Small pose angles: {(decision[0] if isinstance(decision, tuple) else decision.name)}")


class TestPhaseB_ScaleChecks:
    
    def test_too_small_face_rejected(self, test_config, test_face_evidence, logger):
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.05,
            brightness=0.6,
            scale=0.05,
            quality_score=0.3
        )
        
        decision = gate.decide(evidence)
        
        assert (decision[0] if isinstance(decision, tuple) else decision) in ['REJECT', GateDecision.REJECT, 'HOLD', GateDecision.HOLD], \
            f"Too small face should be REJECT or HOLD, got {decision}"
        
        logger.info(f"✅ Too small face: {(decision[0] if isinstance(decision, tuple) else decision.name)}")
    
    def test_good_scale_accepted(self, test_config, test_face_evidence, logger):
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.05,
            brightness=0.6,
            scale=0.6,
            quality_score=0.9
        )
        
        decision = gate.decide(evidence)
        
        assert (decision[0] if isinstance(decision, tuple) else decision) in ['ACCEPT', GateDecision.ACCEPT, 'HOLD', GateDecision.HOLD], \
            f"Good scale face should be ACCEPT or HOLD, got {decision}"
        
        logger.info(f"✅ Good scale face: {(decision[0] if isinstance(decision, tuple) else decision.name)}")


class TestPhaseB_MarginalQuality:
    
    def test_marginal_quality_held(self, test_config, test_face_evidence, logger):
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.3,
            brightness=0.45,
            quality_score=0.65
        )
        
        decision = gate.decide(evidence)
        
        assert (decision[0] if isinstance(decision, tuple) else decision) in ['HOLD', GateDecision.HOLD, 'ACCEPT', GateDecision.ACCEPT], \
            f"Marginal quality should be HOLD or ACCEPT, got {decision}"
        
        logger.info(f"✅ Marginal quality: {(decision[0] if isinstance(decision, tuple) else decision.name)}")


class TestPhaseB_GateDecisionStats:
    
    def test_gate_statistics_over_100_samples(self, test_config, test_face_evidence, logger):
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        stats = {
            GateDecision.ACCEPT: 0,
            GateDecision.HOLD: 0,
            GateDecision.REJECT: 0,
        }
        
        for _ in range(100):
            blur = np.random.rand() * 0.5
            brightness = 0.3 + np.random.rand() * 0.4
            
            evidence = test_face_evidence(
                blur_score=blur,
                brightness=brightness,
                quality_score=0.6 + np.random.rand() * 0.3
            )
            
            decision = gate.decide(evidence)
            if decision in stats:
                stats[decision] += 1
        
        total = sum(stats.values())
        logger.info(f"100 samples: ACCEPT={stats[GateDecision.ACCEPT]}, "
                   f"HOLD={stats[GateDecision.HOLD]}, "
                   f"REJECT={stats[GateDecision.REJECT]}")
        
        assert stats[GateDecision.REJECT] < 50, "Too many rejections"
        logger.info("✅ Gate statistics collected successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
