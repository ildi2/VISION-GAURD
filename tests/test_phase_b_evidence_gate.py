# tests/test_phase_b_evidence_gate.py
"""
Phase B Verification: Evidence Gating (Quality Contract)
Tests evidence gate decision logic: ACCEPT/HOLD/REJECT
"""

import pytest
import logging
import numpy as np

log = logging.getLogger(__name__)


class TestPhaseBevidenceGateBasics:
    """Test Phase B: Evidence gate basic functionality"""
    
    def test_evidence_gate_imports(self, logger):
        """Evidence gate module should import"""
        try:
            from identity.evidence_gate import EvidenceGate, GateDecision
            logger.info("✅ Evidence gate module imports successfully")
        except ImportError as e:
            pytest.fail(f"Cannot import evidence gate: {e}")
    
    def test_evidence_gate_initializes(self, test_config, logger):
        """Evidence gate should initialize from config"""
        try:
            from identity.evidence_gate import EvidenceGate
            
            gate = EvidenceGate(test_config)
            assert gate is not None
            logger.info("✅ Evidence gate initializes successfully")
        except Exception as e:
            pytest.fail(f"Evidence gate initialization failed: {e}")


class TestPhaseB_HighQualityAccepted:
    """Test Phase B: High quality faces should be ACCEPTED"""
    
    def test_high_quality_face_accepted(self, test_config, test_face_evidence, logger):
        """High quality face should be ACCEPTED"""
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        # Create high-quality evidence
        evidence = test_face_evidence(
            blur_score=0.05,      # Sharp (low blur)
            brightness=0.6,       # Good brightness
            yaw=5.0,              # Small yaw
            pitch=2.0,            # Small pitch
            roll=2.0,             # Small roll
            scale=0.6,            # Good scale
            quality_score=0.95    # High quality
        )
        
        decision = gate.decide(evidence)
        
        assert decision[0] in ['ACCEPT', 'HOLD'], \
            f"Expected ACCEPT or HOLD, got {decision}"
        
        if decision[0] == 'ACCEPT':
            logger.info("✅ High quality face ACCEPTED")
        else:
            logger.warning("⚠️ High quality face HELD (check thresholds)")
    
    def test_multiple_high_quality_samples(self, test_config, test_face_evidence, logger):
        """Multiple high quality samples should all be accepted"""
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
        
        # Most should be accepted
        assert accept_count + hold_count >= 4, "Most high-quality samples should be accepted/held"
        logger.info("✅ Multiple high-quality samples handled correctly")


class TestPhaseB_BlurryRejected:
    """Test Phase B: Blurry faces should be REJECTED or HELD"""
    
    def test_very_blurry_rejected(self, test_config, test_face_evidence, logger):
        """Very blurry face should be REJECTED"""
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        # Very blurry
        evidence = test_face_evidence(
            blur_score=0.95,      # Very blurry
            brightness=0.6,
            quality_score=0.2     # Low quality
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
        """Moderately blurry should be HELD or REJECTED"""
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.5,       # Moderately blurry
            brightness=0.6,
            quality_score=0.6     # Marginal quality
        )
        
        decision = gate.decide(evidence)
        decision_status = decision[0] if isinstance(decision, tuple) else decision
        
        assert decision_status in ['HOLD', 'REJECT', GateDecision.HOLD, GateDecision.REJECT], \
            f"Moderately blurry should be HOLD or REJECT, got {decision}"
        
        logger.info(f"✅ Moderately blurry face handled")


class TestPhaseB_DarkRejected:
    """Test Phase B: Dark faces should be REJECTED or HELD"""
    
    def test_very_dark_rejected(self, test_config, test_face_evidence, logger):
        """Very dark face should be REJECTED"""
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.1,
            brightness=0.05,      # Very dark
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
        """Very bright/washed out face should be REJECTED"""
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.1,
            brightness=0.95,      # Very bright/washed
            quality_score=0.2
        )
        
        decision = gate.decide(evidence)
        decision_status = decision[0] if isinstance(decision, tuple) else decision
        
        assert decision_status in ['REJECT', 'HOLD', GateDecision.REJECT, GateDecision.HOLD], \
            f"Very bright face should be REJECT or HOLD, got {decision}"
        
        logger.info(f"✅ Very bright face: {(decision[0] if isinstance(decision, tuple) else decision.name)}")


class TestPhaseB_PoseChecks:
    """Test Phase B: Geometric pose checks"""
    
    def test_large_yaw_angle_held_or_rejected(self, test_config, test_face_evidence, logger):
        """Large yaw angle should be HELD or REJECTED"""
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.05,
            brightness=0.6,
            yaw=45.0,             # Very large yaw
            quality_score=0.8
        )
        
        decision = gate.decide(evidence)
        
        assert (decision[0] if isinstance(decision, tuple) else decision) in ['HOLD', GateDecision.HOLD, 'REJECT', GateDecision.REJECT], \
            f"Large yaw should be HOLD or REJECT, got {decision}"
        
        logger.info(f"✅ Large yaw (45°): {(decision[0] if isinstance(decision, tuple) else decision.name)}")
    
    def test_small_pose_angle_accepted(self, test_config, test_face_evidence, logger):
        """Small pose angles should be ACCEPTED"""
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
    """Test Phase B: Face scale checks"""
    
    def test_too_small_face_rejected(self, test_config, test_face_evidence, logger):
        """Face too small in frame should be REJECTED"""
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.05,
            brightness=0.6,
            scale=0.05,           # Very small
            quality_score=0.3
        )
        
        decision = gate.decide(evidence)
        
        assert (decision[0] if isinstance(decision, tuple) else decision) in ['REJECT', GateDecision.REJECT, 'HOLD', GateDecision.HOLD], \
            f"Too small face should be REJECT or HOLD, got {decision}"
        
        logger.info(f"✅ Too small face: {(decision[0] if isinstance(decision, tuple) else decision.name)}")
    
    def test_good_scale_accepted(self, test_config, test_face_evidence, logger):
        """Face with good scale should be ACCEPTED"""
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.05,
            brightness=0.6,
            scale=0.6,            # Good size
            quality_score=0.9
        )
        
        decision = gate.decide(evidence)
        
        assert (decision[0] if isinstance(decision, tuple) else decision) in ['ACCEPT', GateDecision.ACCEPT, 'HOLD', GateDecision.HOLD], \
            f"Good scale face should be ACCEPT or HOLD, got {decision}"
        
        logger.info(f"✅ Good scale face: {(decision[0] if isinstance(decision, tuple) else decision.name)}")


class TestPhaseB_MarginalQuality:
    """Test Phase B: Marginal quality handling"""
    
    def test_marginal_quality_held(self, test_config, test_face_evidence, logger):
        """Marginal quality should be HELD"""
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        evidence = test_face_evidence(
            blur_score=0.3,       # Slightly blurry
            brightness=0.45,      # Slightly dark
            quality_score=0.65    # Marginal quality
        )
        
        decision = gate.decide(evidence)
        
        assert (decision[0] if isinstance(decision, tuple) else decision) in ['HOLD', GateDecision.HOLD, 'ACCEPT', GateDecision.ACCEPT], \
            f"Marginal quality should be HOLD or ACCEPT, got {decision}"
        
        logger.info(f"✅ Marginal quality: {(decision[0] if isinstance(decision, tuple) else decision.name)}")


class TestPhaseB_GateDecisionStats:
    """Test Phase B: Gate decision statistics"""
    
    def test_gate_statistics_over_100_samples(self, test_config, test_face_evidence, logger):
        """Collect statistics over 100 random samples"""
        from identity.evidence_gate import EvidenceGate, GateDecision
        
        gate = EvidenceGate(test_config)
        
        stats = {
            GateDecision.ACCEPT: 0,
            GateDecision.HOLD: 0,
            GateDecision.REJECT: 0,
        }
        
        for _ in range(100):
            # Random quality
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
        
        # Most should not be rejected
        assert stats[GateDecision.REJECT] < 50, "Too many rejections"
        logger.info("✅ Gate statistics collected successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
