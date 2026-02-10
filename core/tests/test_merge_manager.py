
import pytest
import numpy as np
from identity.merge_manager import (
    MergeManager,
    MergeConfig,
    MergeCandidate,
    CanonicalMapping,
    MergeEvidence,
    MergeMetrics,
    MergeFailureReason,
    create_merge_config_from_dict,
)


@pytest.fixture
def default_merge_config():
    return MergeConfig(
        enabled=True,
        merge_confidence_min=60.0,
        tentative_threshold_min=40.0,
        min_gap_seconds=0.3,
        max_gap_seconds=5.0,
        max_distance_pixels=150.0,
        velocity_influence_factor=20.0,
        max_embedding_distance=0.35,
        quality_min_samples=2,
        velocity_similarity_min=0.6,
        allow_different_identities=False,
        confidence_required_for_diff_ids=0.95,
        max_merges_per_canonical=5,
        merge_reversal_window_seconds=5.0,
        min_time_between_merges_seconds=2.0,
        inactive_tracklet_retention_seconds=10.0,
        log_merge_reasons=True,
        log_reversal_reasons=True,
        debug_mode=False
    )


@pytest.fixture
def merge_manager(default_merge_config):
    return MergeManager(default_merge_config)


@pytest.fixture
def sample_tracklet_a():
    return MergeCandidate(
        tracklet_id="track_001",
        person_id="person_A",
        binding_state="CONFIRMED_STRONG",
        confidence=0.95,
        end_time=1000.0,
        start_time=990.0,
        last_position=np.array([100.0, 150.0], dtype=np.float32),
        motion_vector=np.array([1.0, 0.5], dtype=np.float32),
        appearance_features=np.random.rand(512).astype(np.float32),
        track_length=10,
        quality_samples=5
    )


@pytest.fixture
def sample_tracklet_b():
    return MergeCandidate(
        tracklet_id="track_002",
        person_id="person_A",
        binding_state="CONFIRMED_STRONG",
        confidence=0.90,
        start_time=1001.0,
        end_time=1015.0,
        first_position=np.array([105.0, 152.0], dtype=np.float32),
        motion_vector=np.array([1.0, 0.6], dtype=np.float32),
        appearance_features=np.random.rand(512).astype(np.float32),
        track_length=14,
        quality_samples=6
    )


class TestMergeManagerInitialization:
    
    def test_merge_manager_init(self, default_merge_config):
        mm = MergeManager(default_merge_config)
        assert mm is not None
        assert mm.config == default_merge_config
        assert len(mm.canonical_mappings) == 0
        assert len(mm.merge_history) == 0
        assert mm.current_time == 0.0
    
    def test_merge_manager_disabled(self):
        cfg = MergeConfig(enabled=False)
        mm = MergeManager(cfg)
        assert not mm.config.enabled
    
    def test_config_from_dict(self):
        config_dict = {
            'enabled': True,
            'thresholds': {'merge_confidence_min': 65, 'tentative_threshold': 45},
            'temporal': {'min_gap_seconds': 0.4, 'max_gap_seconds': 4.5},
            'spatial': {'max_distance_pixels': 200, 'velocity_influence': 25},
            'appearance': {'max_embedding_distance': 0.40, 'quality_min_samples': 3},
            'logging': {'log_merge_reasons': True, 'debug_mode': False}
        }
        cfg = create_merge_config_from_dict(config_dict)
        assert cfg.merge_confidence_min == 65
        assert cfg.tentative_threshold_min == 45
        assert cfg.min_gap_seconds == 0.4
        assert cfg.max_distance_pixels == 200
        assert cfg.quality_min_samples == 3


class TestCanonicalIdMapping:
    
    def test_get_canonical_id_new_tracklet(self, merge_manager):
        canonical_id = merge_manager.get_canonical_id("track_001")
        assert canonical_id == "track_001"
        assert "track_001" in merge_manager.canonical_mappings
    
    def test_canonical_id_persistence(self, merge_manager):
        id1 = merge_manager.get_canonical_id("track_001")
        id2 = merge_manager.get_canonical_id("track_001")
        assert id1 == id2 == "track_001"
    
    def test_get_all_tracklets_for_canonical_single(self, merge_manager):
        merge_manager.get_canonical_id("track_001")
        tracklets = merge_manager.get_all_tracklets_for_canonical("track_001")
        assert tracklets == ["track_001"]
    
    def test_get_all_tracklets_for_canonical_multiple(self, merge_manager, sample_tracklet_a, sample_tracklet_b):
        evidence = MergeEvidence(
            source_tracklet_id=sample_tracklet_a.tracklet_id,
            target_tracklet_id=sample_tracklet_b.tracklet_id,
            merge_time=1001.5,
            merge_reason="test_merge",
            confidence=75.0,
            details={}
        )
        
        merge_manager.current_time = 1001.5
        merge_manager.execute_merge(
            canonical_id=sample_tracklet_b.tracklet_id,
            tracklet_to_merge=sample_tracklet_a.tracklet_id,
            evidence=evidence
        )
        
        tracklets = merge_manager.get_all_tracklets_for_canonical(sample_tracklet_b.tracklet_id)
        assert set(tracklets) == {sample_tracklet_a.tracklet_id, sample_tracklet_b.tracklet_id}


class TestMergeScoring:
    
    def test_merge_scoring_high_confidence(self, merge_manager, sample_tracklet_a, sample_tracklet_b):
        sample_tracklet_b.start_time = 1000.5
        decision = merge_manager.get_merge_decision(sample_tracklet_a, sample_tracklet_b)
        assert decision.should_merge is True
        assert decision.score >= merge_manager.config.merge_confidence_min
    
    def test_time_gap_criterion_invalid_too_close(self, merge_manager, sample_tracklet_a, sample_tracklet_b):
        sample_tracklet_b.start_time = 1000.0 + 0.1
        decision = merge_manager.get_merge_decision(sample_tracklet_a, sample_tracklet_b)
        assert decision.should_merge is False
        assert decision.reason == "time_gap_invalid"
    
    def test_time_gap_criterion_invalid_too_far(self, merge_manager, sample_tracklet_a, sample_tracklet_b):
        sample_tracklet_b.start_time = 1000.0 + 10.0
        decision = merge_manager.get_merge_decision(sample_tracklet_a, sample_tracklet_b)
        assert decision.should_merge is False
        assert decision.reason == "time_gap_invalid"
    
    def test_spatial_criterion_distance_too_large(self, merge_manager, sample_tracklet_a, sample_tracklet_b):
        sample_tracklet_b.start_time = 1000.5
        sample_tracklet_b.first_position = np.array([500.0, 500.0], dtype=np.float32)
        decision = merge_manager.get_merge_decision(sample_tracklet_a, sample_tracklet_b)
        assert decision.should_merge is False
        assert decision.reason == "distance_too_large"
    
    def test_appearance_criterion_missing_features(self, merge_manager, sample_tracklet_a, sample_tracklet_b):
        sample_tracklet_b.start_time = 1000.5
        sample_tracklet_a.appearance_features = None
        decision = merge_manager.get_merge_decision(sample_tracklet_a, sample_tracklet_b)
        assert decision.should_merge is False
        assert decision.reason == "appearance_mismatch"
    
    def test_appearance_criterion_distance_too_large(self, merge_manager, sample_tracklet_a, sample_tracklet_b):
        sample_tracklet_b.start_time = 1000.5
        sample_tracklet_b.appearance_features = np.ones(512, dtype=np.float32)
        sample_tracklet_a.appearance_features = np.zeros(512, dtype=np.float32)
        decision = merge_manager.get_merge_decision(sample_tracklet_a, sample_tracklet_b)
        assert decision.should_merge is False
        assert decision.reason == "appearance_mismatch"
    
    def test_quality_criterion_insufficient(self, merge_manager, sample_tracklet_a, sample_tracklet_b):
        sample_tracklet_b.start_time = 1000.5
        sample_tracklet_a.quality_samples = 0
        sample_tracklet_b.quality_samples = 0
        sample_tracklet_a.binding_state = "UNKNOWN"
        decision = merge_manager.get_merge_decision(sample_tracklet_a, sample_tracklet_b)
        assert decision.should_merge is False
        assert decision.reason == "insufficient_quality"
    
    def test_binding_conflict_different_identities(self, merge_manager, sample_tracklet_a, sample_tracklet_b):
        sample_tracklet_b.start_time = 1000.5
        sample_tracklet_a.person_id = "person_A"
        sample_tracklet_b.person_id = "person_B"
        decision = merge_manager.get_merge_decision(sample_tracklet_a, sample_tracklet_b)
        assert decision.should_merge is False
        assert decision.reason == "conflicting_identities"


class TestMergeExecution:
    
    def test_execute_merge_confident(self, merge_manager, sample_tracklet_a, sample_tracklet_b):
        merge_manager.current_time = 1001.5
        evidence = MergeEvidence(
            source_tracklet_id=sample_tracklet_a.tracklet_id,
            target_tracklet_id=sample_tracklet_b.tracklet_id,
            merge_time=1001.5,
            merge_reason="test_merge",
            confidence=75.0,
            details={}
        )
        
        merge_manager.execute_merge(
            canonical_id=sample_tracklet_b.tracklet_id,
            tracklet_to_merge=sample_tracklet_a.tracklet_id,
            evidence=evidence,
            tentative=False
        )
        
        mapping = merge_manager.canonical_mappings[sample_tracklet_b.tracklet_id]
        assert sample_tracklet_a.tracklet_id in mapping.aliases
        assert len(mapping.merge_history) == 1
    
    def test_execute_merge_tentative(self, merge_manager, sample_tracklet_a, sample_tracklet_b):
        merge_manager.current_time = 1001.5
        evidence = MergeEvidence(
            source_tracklet_id=sample_tracklet_a.tracklet_id,
            target_tracklet_id=sample_tracklet_b.tracklet_id,
            merge_time=1001.5,
            merge_reason="test_merge",
            confidence=45.0,
            details={}
        )
        
        merge_manager.execute_merge(
            canonical_id=sample_tracklet_b.tracklet_id,
            tracklet_to_merge=sample_tracklet_a.tracklet_id,
            evidence=evidence,
            tentative=True,
            reversal_deadline=1006.5
        )
        
        assert sample_tracklet_a.tracklet_id in merge_manager.tentative_merges
        assert merge_manager.tentative_merges[sample_tracklet_a.tracklet_id] == 1006.5
    
    def test_reverse_merge(self, merge_manager, sample_tracklet_a, sample_tracklet_b):
        merge_manager.current_time = 1001.5
        evidence = MergeEvidence(
            source_tracklet_id=sample_tracklet_a.tracklet_id,
            target_tracklet_id=sample_tracklet_b.tracklet_id,
            merge_time=1001.5,
            merge_reason="test_merge",
            confidence=45.0,
            details={}
        )
        merge_manager.execute_merge(
            canonical_id=sample_tracklet_b.tracklet_id,
            tracklet_to_merge=sample_tracklet_a.tracklet_id,
            evidence=evidence
        )
        
        merge_manager.current_time = 1003.0
        success = merge_manager.reverse_merge(
            tracklet_id=sample_tracklet_a.tracklet_id,
            reason="contradiction"
        )
        
        assert success is True
        mapping = merge_manager.canonical_mappings[sample_tracklet_b.tracklet_id]
        assert sample_tracklet_a.tracklet_id not in mapping.aliases
        assert mapping.merge_history[0].reversed is True


class TestTrackletLifecycle:
    
    def test_on_tracklet_started(self, merge_manager):
        merge_manager.on_tracklet_started("track_001", (100.0, 150.0), 1000.0)
        assert "track_001" in merge_manager.canonical_mappings
    
    def test_on_tracklet_ended(self, merge_manager, sample_tracklet_a):
        merge_manager.on_tracklet_ended(
            tracklet_id=sample_tracklet_a.tracklet_id,
            binding_state={'person_id': 'person_A', 'status': 'CONFIRMED_STRONG', 'confidence': 0.95},
            appearance_features=sample_tracklet_a.appearance_features,
            last_position=(100.0, 150.0),
            end_time=1000.0,
            track_length=10,
            quality_samples=5
        )
        
        assert sample_tracklet_a.tracklet_id in merge_manager.inactive_tracklets
        candidate = merge_manager.inactive_tracklets[sample_tracklet_a.tracklet_id]
        assert candidate.person_id == 'person_A'
    
    def test_on_tracklet_updated(self, merge_manager):
        merge_manager.get_canonical_id("track_001")
        merge_manager.on_tracklet_updated(
            tracklet_id="track_001",
            binding_state={'person_id': 'person_A', 'status': 'CONFIRMED_STRONG'},
            appearance_features=np.random.rand(512),
            current_position=(100.0, 150.0),
            track_length=5,
            quality_samples=2,
            timestamp=1000.0
        )
        
        mapping = merge_manager.canonical_mappings["track_001"]
        assert mapping.primary_binding is not None


class TestMetricsAndCleanup:
    
    def test_get_metrics_empty(self, merge_manager):
        metrics = merge_manager.get_metrics()
        assert isinstance(metrics, MergeMetrics)
        assert metrics.merge_attempts == 0
        assert metrics.merges_executed == 0
    
    def test_cleanup_old_tracklets(self, merge_manager, sample_tracklet_a):
        merge_manager.current_time = 1000.0
        merge_manager.on_tracklet_ended(
            tracklet_id=sample_tracklet_a.tracklet_id,
            binding_state=None,
            appearance_features=sample_tracklet_a.appearance_features,
            last_position=(100.0, 150.0),
            end_time=1000.0,
            track_length=10,
            quality_samples=5
        )
        
        assert sample_tracklet_a.tracklet_id in merge_manager.inactive_tracklets
        
        merge_manager.current_time = 1020.0
        removed = merge_manager.cleanup_old_tracklets(threshold_seconds=10.0)
        
        assert removed > 0
        assert sample_tracklet_a.tracklet_id not in merge_manager.inactive_tracklets
    
    def test_get_state_summary(self, merge_manager):
        summary = merge_manager.get_state_summary()
        assert 'canonical_ids' in summary
        assert 'total_tracklets_aliased' in summary
        assert 'metrics' in summary


class TestEdgeCases:
    
    def test_disabled_manager(self):
        cfg = MergeConfig(enabled=False)
        mm = MergeManager(cfg)
        
        mm.on_tracklet_started("track_001", (100.0, 150.0), 1000.0)
        assert len(mm.canonical_mappings) == 0
    
    def test_null_motion_vector(self, merge_manager, sample_tracklet_a, sample_tracklet_b):
        sample_tracklet_b.start_time = 1000.5
        sample_tracklet_a.motion_vector = None
        sample_tracklet_b.motion_vector = None
        
        decision = merge_manager.get_merge_decision(sample_tracklet_a, sample_tracklet_b)
        assert decision is not None
    
    def test_cosine_similarity_zero_vector(self, merge_manager):
        zero_vec = np.zeros(3, dtype=np.float32)
        sim = merge_manager._cosine_similarity(zero_vec, zero_vec)
        assert sim == 0.0
    
    def test_empty_canonical_mapping(self, merge_manager):
        tracklets = merge_manager.get_all_tracklets_for_canonical("non_existent")
        assert tracklets == ["non_existent"]
    
    def test_merge_limit_per_canonical(self, merge_manager, sample_tracklet_a, sample_tracklet_b):
        merge_manager.config.max_merges_per_canonical = 1
        merge_manager.current_time = 1001.5
        
        evidence1 = MergeEvidence(
            source_tracklet_id="track_001",
            target_tracklet_id="track_002",
            merge_time=1001.5,
            merge_reason="merge1",
            confidence=75.0,
            details={}
        )
        merge_manager.execute_merge("track_002", "track_001", evidence1)
        
        evidence2 = MergeEvidence(
            source_tracklet_id="track_003",
            target_tracklet_id="track_002",
            merge_time=1003.5,
            merge_reason="merge2",
            confidence=75.0,
            details={}
        )
        merge_manager.current_time = 1003.5
        merge_manager.execute_merge("track_002", "track_003", evidence2)
        
        mapping = merge_manager.canonical_mappings["track_002"]
        assert len(mapping.aliases) <= merge_manager.config.max_merges_per_canonical


class TestIntegrationScenarios:
    
    def test_scenario_person_exits_and_reenters(self, merge_manager):
        merge_manager.on_tracklet_started("track_001", (100.0, 100.0), 1000.0)
        merge_manager.on_tracklet_ended(
            "track_001",
            binding_state={'person_id': 'person_A', 'status': 'CONFIRMED_STRONG', 'confidence': 0.95},
            appearance_features=np.random.rand(512).astype(np.float32),
            last_position=(150.0, 150.0),
            end_time=1010.0,
            track_length=100,
            quality_samples=8
        )
        
        merge_manager.on_tracklet_started("track_002", (155.0, 155.0), 1010.5)
        
        assert len(merge_manager.inactive_tracklets) == 1
        assert "track_001" in merge_manager.inactive_tracklets
    
    def test_scenario_two_people_crossing(self, merge_manager):
        merge_manager.on_tracklet_started("track_001", (100.0, 100.0), 1000.0)
        merge_manager.on_tracklet_ended(
            "track_001",
            binding_state={'person_id': 'person_A', 'status': 'CONFIRMED_STRONG', 'confidence': 0.95},
            appearance_features=np.random.rand(512).astype(np.float32),
            last_position=(110.0, 110.0),
            end_time=1005.0,
            track_length=50,
            quality_samples=5
        )
        
        merge_manager.on_tracklet_started("track_002", (105.0, 105.0), 1004.5)
        person_b_features = np.random.rand(512).astype(np.float32)
        person_b_features[:256] = 0.0
        
        merge_manager.on_tracklet_ended(
            "track_002",
            binding_state={'person_id': 'person_B', 'status': 'CONFIRMED_STRONG', 'confidence': 0.95},
            appearance_features=person_b_features,
            last_position=(120.0, 120.0),
            end_time=1010.0,
            track_length=50,
            quality_samples=5
        )
        
        tracklet_a = merge_manager.inactive_tracklets["track_001"]
        tracklet_b = merge_manager.inactive_tracklets["track_002"]
        decision = merge_manager.get_merge_decision(tracklet_a, tracklet_b)
        
        assert decision.should_merge is False


class TestConfigurationVariants:
    
    def test_conservative_mode(self):
        cfg = MergeConfig(
            merge_confidence_min=70.0,
            tentative_threshold_min=60.0
        )
        assert cfg.merge_confidence_min == 70.0
    
    def test_allow_different_identities_flag(self):
        cfg = MergeConfig(
            allow_different_identities=True,
            confidence_required_for_diff_ids=0.98
        )
        assert cfg.allow_different_identities is True
        assert cfg.confidence_required_for_diff_ids == 0.98
    
    def test_strict_spatial_constraint(self):
        cfg = MergeConfig(
            max_distance_pixels=50.0,
            velocity_influence_factor=5.0
        )
        assert cfg.max_distance_pixels == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
