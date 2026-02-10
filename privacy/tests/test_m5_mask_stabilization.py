
import time
import pytest
import numpy as np
from dataclasses import dataclass

from privacy.mask_stabilizer import (
    MaskStabilizer,
    StabilizationResult,
    StabilizationConfig,
    TrackMaskHistory,
    create_stabilizer,
)


@pytest.fixture
def default_config():
    return StabilizationConfig(
        enabled=True,
        method="union_only",
        history_size=5,
        mask_ttl_sec=0.75,
        max_shrink_ratio=0.85,
        morph_close_px=0,
        morph_open_px=0,
    )


@pytest.fixture
def majority_config():
    return StabilizationConfig(
        enabled=True,
        method="majority_vote",
        history_size=5,
        mask_ttl_sec=0.75,
        max_shrink_ratio=0.85,
        morph_close_px=0,
        morph_open_px=0,
    )


@pytest.fixture
def morph_config():
    return StabilizationConfig(
        enabled=True,
        method="union_only",
        history_size=5,
        mask_ttl_sec=0.75,
        max_shrink_ratio=0.85,
        morph_close_px=3,
        morph_open_px=2,
    )


@pytest.fixture
def stabilizer(default_config):
    return MaskStabilizer(default_config)


def make_mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


class TestBasicFunctionality:
    
    def test_create_stabilizer_enabled(self, default_config):
        stabilizer = create_stabilizer(default_config)
        assert stabilizer is not None
        assert isinstance(stabilizer, MaskStabilizer)
    
    def test_create_stabilizer_disabled(self):
        cfg = StabilizationConfig(enabled=False)
        stabilizer = create_stabilizer(cfg)
        assert stabilizer is not None
        assert not stabilizer._enabled
    
    def test_first_mask_passthrough(self, stabilizer):
        mask = make_mask(100, 100, 20, 20, 80, 80)
        result = stabilizer.update(track_id=1, mask=mask, ts=1.0)
        
        assert result.is_valid
        assert not result.shrink_detected
        assert not result.ttl_reuse
        assert mask_area(result.mask) == mask_area(mask)
    
    def test_identical_masks_passthrough(self, stabilizer):
        mask = make_mask(100, 100, 20, 20, 80, 80)
        
        r1 = stabilizer.update(track_id=1, mask=mask.copy(), ts=1.0)
        r2 = stabilizer.update(track_id=1, mask=mask.copy(), ts=1.033)
        
        assert r2.is_valid
        assert not r2.shrink_detected
        assert mask_area(r2.mask) == mask_area(mask)


class TestShrinkDetection:
    
    def test_shrink_detected_below_threshold(self, default_config):
        stabilizer = MaskStabilizer(default_config)
        
        mask1 = make_mask(200, 200, 50, 50, 150, 150)
        r1 = stabilizer.update(track_id=1, mask=mask1, ts=1.0)
        
        mask2 = make_mask(200, 200, 75, 75, 125, 125)
        r2 = stabilizer.update(track_id=1, mask=mask2, ts=1.033)
        
        assert r2.shrink_detected
        assert mask_area(r2.mask) > mask_area(mask2)
        assert mask_area(r2.mask) >= mask_area(mask2)
    
    def test_shrink_not_detected_above_threshold(self, default_config):
        stabilizer = MaskStabilizer(default_config)
        
        mask1 = make_mask(200, 200, 50, 50, 150, 150)
        stabilizer.update(track_id=1, mask=mask1, ts=1.0)
        
        mask2 = make_mask(200, 200, 52, 52, 147, 147)
        r2 = stabilizer.update(track_id=1, mask=mask2, ts=1.033)
        
        assert not r2.shrink_detected
    
    def test_fail_closed_invariant(self, default_config):
        stabilizer = MaskStabilizer(default_config)
        
        masks = [
            make_mask(200, 200, 50, 50, 150, 150),
            make_mask(200, 200, 30, 30, 170, 170),
            make_mask(200, 200, 80, 80, 120, 120),
            make_mask(200, 200, 60, 60, 140, 140),
            make_mask(200, 200, 90, 90, 110, 110),
        ]
        
        for i, mask in enumerate(masks):
            ts = 1.0 + i * 0.033
            result = stabilizer.update(track_id=1, mask=mask, ts=ts)
            
            assert mask_area(result.mask) >= mask_area(mask), \
                f"Frame {i}: stabilized ({mask_area(result.mask)}) < raw ({mask_area(mask)})"


class TestTTLReuse:
    
    def test_ttl_reuse_within_window(self, stabilizer):
        mask1 = make_mask(100, 100, 20, 20, 80, 80)
        r1 = stabilizer.update(track_id=1, mask=mask1, ts=1.0)
        
        r2 = stabilizer.update(track_id=1, mask=None, ts=1.5)
        
        assert r2.is_valid
        assert r2.ttl_reuse
        assert mask_area(r2.mask) == mask_area(mask1)
    
    def test_ttl_expired_no_reuse(self, stabilizer):
        mask1 = make_mask(100, 100, 20, 20, 80, 80)
        stabilizer.update(track_id=1, mask=mask1, ts=1.0)
        
        r2 = stabilizer.update(track_id=1, mask=None, ts=2.0)
        
        assert not r2.is_valid
        assert not r2.ttl_reuse
    
    def test_ttl_reuse_preserves_area(self, stabilizer):
        mask1 = make_mask(100, 100, 20, 20, 80, 80)
        mask2 = make_mask(100, 100, 25, 25, 75, 75)
        
        stabilizer.update(track_id=1, mask=mask1, ts=1.0)
        r2 = stabilizer.update(track_id=1, mask=mask2, ts=1.033)
        
        r3 = stabilizer.update(track_id=1, mask=None, ts=1.5)
        
        assert r3.ttl_reuse
        assert r3.mask is not None


class TestFlickerReduction:
    
    def test_alternating_masks_stabilized(self, default_config):
        stabilizer = MaskStabilizer(default_config)
        
        large = make_mask(100, 100, 20, 20, 80, 80)
        small = make_mask(100, 100, 35, 35, 65, 65)
        
        results = []
        for i in range(10):
            mask = large if i % 2 == 0 else small
            ts = 1.0 + i * 0.033
            r = stabilizer.update(track_id=1, mask=mask, ts=ts)
            results.append(r)
        
        assert all(r.is_valid for r in results)
        
        areas = [mask_area(r.mask) for r in results]
        
        input_areas = [3600 if i % 2 == 0 else 900 for i in range(10)]
        
        input_var = np.var(input_areas)
        output_var = np.var(areas)
        
        assert output_var < input_var, "Flicker not reduced"
    
    def test_majority_vote_smoothing(self, majority_config):
        stabilizer = MaskStabilizer(majority_config)
        
        base = make_mask(100, 100, 30, 30, 70, 70)
        
        for i in range(5):
            offset = (i % 3) - 1
            mask = make_mask(100, 100, 30 + offset, 30 + offset, 70 - offset, 70 - offset)
            r = stabilizer.update(track_id=1, mask=mask, ts=1.0 + i * 0.033)
            
            assert r.is_valid


class TestMultiTrackIsolation:
    
    def test_tracks_independent(self, stabilizer):
        mask1 = make_mask(100, 100, 10, 10, 90, 90)
        r1 = stabilizer.update(track_id=1, mask=mask1, ts=1.0)
        
        mask2 = make_mask(100, 100, 40, 40, 60, 60)
        r2 = stabilizer.update(track_id=2, mask=mask2, ts=1.0)
        
        assert mask_area(r1.mask) != mask_area(r2.mask)
        assert mask_area(r1.mask) == mask_area(mask1)
        assert mask_area(r2.mask) == mask_area(mask2)
    
    def test_track_cleanup(self, default_config):
        config = StabilizationConfig(
            enabled=True,
            method="union_only",
            history_size=3,
            mask_ttl_sec=0.1,
            max_shrink_ratio=0.85,
            morph_close_px=0,
            morph_open_px=0,
        )
        stabilizer = MaskStabilizer(config)
        
        for track_id in range(100):
            mask = make_mask(100, 100, 20, 20, 80, 80)
            stabilizer.update(track_id=track_id, mask=mask, ts=1.0)
        
        initial_count = len(stabilizer._track_histories)
        assert initial_count == 100
        
        time.sleep(0.15)
        mask = make_mask(100, 100, 20, 20, 80, 80)
        stabilizer.update(track_id=999, mask=mask, ts=time.time())
        
        assert 999 in stabilizer._track_histories


class TestMorphOperations:
    
    def test_morph_close_fills_gaps(self, morph_config):
        stabilizer = MaskStabilizer(morph_config)
        
        mask = make_mask(100, 100, 20, 20, 80, 80)
        mask[45:55, 45:55] = 0
        
        original_area = mask_area(mask)
        
        result = stabilizer.update(track_id=1, mask=mask, ts=1.0)
        
        assert mask_area(result.mask) >= original_area
    
    def test_morph_disabled_by_default(self, default_config):
        assert default_config.morph_close_px == 0
        assert default_config.morph_open_px == 0
        
        stabilizer = MaskStabilizer(default_config)
        
        mask = make_mask(100, 100, 20, 20, 80, 80)
        result = stabilizer.update(track_id=1, mask=mask, ts=1.0)
        
        assert mask_area(result.mask) == mask_area(mask)


class TestEdgeCases:
    
    def test_empty_mask(self, stabilizer):
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = stabilizer.update(track_id=1, mask=mask, ts=1.0)
        
        assert not result.is_valid or mask_area(result.mask) == 0
    
    def test_full_mask(self, stabilizer):
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        result = stabilizer.update(track_id=1, mask=mask, ts=1.0)
        
        assert result.is_valid
        assert mask_area(result.mask) == 100 * 100
    
    def test_none_mask_no_history(self, stabilizer):
        result = stabilizer.update(track_id=1, mask=None, ts=1.0)
        
        assert not result.is_valid
    
    def test_different_mask_sizes(self, stabilizer):
        mask1 = make_mask(100, 100, 20, 20, 80, 80)
        r1 = stabilizer.update(track_id=1, mask=mask1, ts=1.0)
        assert r1.is_valid
        
        mask2 = make_mask(200, 200, 40, 40, 160, 160)
        r2 = stabilizer.update(track_id=1, mask=mask2, ts=1.033)
        
        assert r2.is_valid
    
    def test_timestamp_ordering(self, stabilizer):
        mask = make_mask(100, 100, 20, 20, 80, 80)
        
        stabilizer.update(track_id=1, mask=mask, ts=1.0)
        stabilizer.update(track_id=1, mask=mask, ts=2.0)
        
        result = stabilizer.update(track_id=1, mask=mask, ts=1.5)
        
        assert result.is_valid


class TestConfigIntegration:
    
    def test_create_from_privacy_config(self):
        from core.config import PrivacyStabilizationConfig
        
        cfg = PrivacyStabilizationConfig(
            enabled=True,
            method="majority_vote",
            history_size=7,
            mask_ttl_sec=1.0,
            max_shrink_ratio=0.9,
            morph_close_px=5,
            morph_open_px=3,
        )
        
        stab_cfg = StabilizationConfig(
            enabled=cfg.enabled,
            method=cfg.method,
            history_size=cfg.history_size,
            mask_ttl_sec=cfg.mask_ttl_sec,
            max_shrink_ratio=cfg.max_shrink_ratio,
            morph_close_px=cfg.morph_close_px,
            morph_open_px=cfg.morph_open_px,
        )
        
        stabilizer = create_stabilizer(stab_cfg)
        assert stabilizer is not None
        assert stabilizer._cfg.method == "majority_vote"
        assert stabilizer._cfg.history_size == 7


class TestPerformance:
    
    def test_update_latency(self, stabilizer):
        mask = make_mask(480, 640, 100, 100, 500, 400)
        
        for i in range(5):
            stabilizer.update(track_id=1, mask=mask, ts=1.0 + i * 0.033)
        
        start = time.perf_counter()
        for i in range(100):
            stabilizer.update(track_id=1, mask=mask, ts=2.0 + i * 0.033)
        elapsed = time.perf_counter() - start
        
        avg_ms = (elapsed / 100) * 1000
        
        assert avg_ms < 10, f"Average latency {avg_ms:.2f}ms exceeds 10ms"
    
    def test_memory_bounded(self, default_config):
        stabilizer = MaskStabilizer(default_config)
        mask = make_mask(100, 100, 20, 20, 80, 80)
        
        for i in range(100):
            stabilizer.update(track_id=1, mask=mask, ts=1.0 + i * 0.033)
        
        history = stabilizer._track_histories.get(1)
        assert history is not None
        assert len(history.entries) <= default_config.history_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
