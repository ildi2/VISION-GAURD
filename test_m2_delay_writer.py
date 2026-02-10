#!/usr/bin/env python3

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from privacy.pipeline import PrivacyPipeline
from privacy.delay_buffer import DelayBuffer
from privacy.writer import PrivacyWriter
from core.config import load_config


def test_delay_buffer_standalone():
    print("\n=== TEST 1: DelayBuffer Standalone ===")
    
    delay_sec = 1.0
    buffer = DelayBuffer(delay_sec=delay_sec, max_frames=50)
    
    frames_pushed = 0
    start_ts = time.time()
    
    for i in range(20):
        ingest_ts = start_ts + (i * 0.1)
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        buffer.push(
            frame_id=i + 1,
            ingest_ts=ingest_ts,
            privacy_frame=fake_frame,
            audit_payload={"frame_id": i + 1},
            original_frame_ts=ingest_ts,
        )
        frames_pushed += 1
    
    print(f"  Pushed: {frames_pushed} frames")
    print(f"  Buffer depth: {buffer.get_buffer_depth()}")
    
    time.sleep(delay_sec + 0.5)
    
    current_ts = time.time()
    emitted = buffer.pop_eligible(current_ts)
    
    print(f"  Emitted: {len(emitted)} frames")
    print(f"  Buffer depth after pop: {buffer.get_buffer_depth()}")
    
    if emitted:
        first_item = emitted[0]
        lag = current_ts - first_item.ingest_ts
        print(f"  First frame lag: {lag:.3f}s (expected >= {delay_sec}s)")
        assert lag >= delay_sec, f"Lag {lag} < delay {delay_sec}"
    
    remaining = buffer.flush_all()
    print(f"  Flushed: {len(remaining)} remaining frames")
    
    stats = buffer.get_stats()
    print(f"  Stats: {stats}")
    
    assert stats["frames_pushed"] == 20
    assert stats["frames_emitted"] == 20
    assert stats["frames_dropped"] == 0
    
    print("  ✅ DelayBuffer test PASSED")
    return True


def test_privacy_writer_standalone():
    print("\n=== TEST 2: PrivacyWriter Standalone (Lazy Open) ===")
    
    test_dir = Path("privacy_output_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    writer = PrivacyWriter(
        output_dir=str(test_dir),
        basename="test_stream",
        fps=10.0,
        codec="mp4v",
        container="mp4",
        enabled=True,
    )
    
    print(f"  Writer created: is_open={writer.is_open}")
    assert not writer.is_open, "Writer should NOT be open before first write"
    
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    success = writer.write(fake_frame)
    
    print(f"  After first write: is_open={writer.is_open}, success={success}")
    assert writer.is_open, "Writer should be open after first write"
    assert success, "First write should succeed"
    
    for i in range(9):
        writer.write(fake_frame)
    
    print(f"  Frames written: {writer.frames_written}")
    assert writer.frames_written == 10
    
    writer.close()
    print(f"  After close: is_open={writer.is_open}")
    assert not writer.is_open
    
    assert writer.file_path is not None
    assert writer.file_path.exists(), f"Output file not found: {writer.file_path}"
    print(f"  Output file: {writer.file_path} ({writer.file_path.stat().st_size} bytes)")
    
    shutil.rmtree(test_dir)
    
    print("  ✅ PrivacyWriter test PASSED")
    return True


def test_m2_full_pipeline():
    print("\n=== TEST 3: Full M2 Pipeline (Synthetic Frames) ===")
    
    audit_path = Path("privacy_output/privacy_audit.jsonl")
    if audit_path.exists():
        audit_path.unlink()
    
    cfg = load_config("config/default.yaml")
    print(f"  Config loaded: privacy.delay_sec={cfg.privacy.delay_sec}")
    
    pipe = PrivacyPipeline(cfg.privacy)
    print(f"  Pipeline created: delay_sec={pipe._delay_sec}")
    
    class MockFrame:
        def __init__(self, frame_id):
            self.image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.ts = time.perf_counter()
            self.frame_id = frame_id
    
    class MockTrack:
        def __init__(self, tid):
            self.track_id = tid
            self.bbox = [100, 100, 200, 200]
    
    class MockDecision:
        def __init__(self, track_id):
            self.track_id = track_id
            self.identity_id = None
            self.category = "resident"
            self.confidence = 0.75
            self.binding_state = "CONFIRMED_STRONG"
            self.id_source = "F"
            self.extra = {}
    
    delay_sec = cfg.privacy.delay_sec
    test_duration = delay_sec + 2.0
    fps = 5.0
    frame_interval = 1.0 / fps
    
    print(f"  Running for {test_duration}s at {fps} FPS...")
    
    start_time = time.time()
    frame_id = 0
    
    while time.time() - start_time < test_duration:
        frame_id += 1
        frame = MockFrame(frame_id)
        tracks = [MockTrack(3)]
        decisions = [MockDecision(3)]
        
        pipe.ingest(frame, tracks, decisions)
        time.sleep(frame_interval)
    
    print(f"  Frames ingested: {pipe.frames_ingested}")
    print(f"  Frames emitted: {pipe.frames_emitted}")
    print(f"  Frames dropped: {pipe.frames_dropped}")
    
    assert pipe.frames_emitted > 0, "Should have emitted some frames after delay"
    
    expected_emitted = int((test_duration - delay_sec) * fps)
    print(f"  Expected emitted (approx): {expected_emitted}")
    
    pipe.shutdown()
    
    print(f"  After shutdown - frames_emitted: {pipe.frames_emitted}")
    
    if audit_path.exists():
        with open(audit_path, "r") as f:
            lines = f.readlines()
        
        print(f"  Audit entries: {len(lines)}")
        
        if lines:
            lag_values = []
            for line in lines[:10]:
                entry = json.loads(line)
                lag = entry.get("lag_sec", 0)
                lag_values.append(lag)
            
            avg_lag = sum(lag_values) / len(lag_values) if lag_values else 0
            print(f"  Average lag (first 10 entries): {avg_lag:.3f}s")
            print(f"  Expected lag: ~{delay_sec}s")
            
            tolerance = 0.5
            assert abs(avg_lag - delay_sec) < tolerance, \
                f"Lag {avg_lag:.3f}s too far from delay {delay_sec}s"
            
            print("  ✅ Delay correctness verified")
    
    video_files = list(Path("privacy_output").glob("privacy_stream_*.mp4"))
    print(f"  Video files: {video_files}")
    
    if video_files:
        video_size = video_files[0].stat().st_size
        print(f"  Video size: {video_size} bytes")
        assert video_size > 0, "Video file should not be empty"
        print("  ✅ Video file created")
    
    print("  ✅ Full M2 Pipeline test PASSED")
    return True


def test_writer_lazy_discipline():
    print("\n=== TEST 4: Writer Lazy Discipline ===")
    
    test_audit = Path("privacy_output/test_lazy_audit.jsonl")
    if test_audit.exists():
        test_audit.unlink()
    
    writer = PrivacyWriter(
        output_dir="privacy_output",
        basename="test_lazy",
        fps=10.0,
        codec="mp4v",
        container="mp4",
        enabled=True,
    )
    
    assert not writer.is_open, "Writer must NOT be open before first write"
    print("  ✅ Writer not open at creation")
    
    buffer = DelayBuffer(delay_sec=5.0, max_frames=100)
    
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    start_ts = time.time()
    
    for i in range(10):
        buffer.push(
            frame_id=i + 1,
            ingest_ts=start_ts + (i * 0.1),
            privacy_frame=fake_frame,
            audit_payload={},
            original_frame_ts=start_ts + (i * 0.1),
        )
    
    emitted = buffer.pop_eligible(time.time())
    print(f"  Emitted before delay: {len(emitted)}")
    assert len(emitted) == 0, "Should NOT emit frames before delay"
    
    assert not writer.is_open, "Writer must NOT be open until frames emitted"
    print("  ✅ Writer still not open (no frames emitted)")
    
    writer.close()
    print("  ✅ Writer Lazy Discipline test PASSED")
    return True


def main():
    print("=" * 60)
    print("M2 PRIVACY PIPELINE UNIT TESTS")
    print("=" * 60)
    
    all_passed = True
    
    try:
        if not test_delay_buffer_standalone():
            all_passed = False
    except Exception as e:
        print(f"  ❌ DelayBuffer test FAILED: {e}")
        all_passed = False
    
    try:
        if not test_privacy_writer_standalone():
            all_passed = False
    except Exception as e:
        print(f"  ❌ PrivacyWriter test FAILED: {e}")
        all_passed = False
    
    try:
        if not test_writer_lazy_discipline():
            all_passed = False
    except Exception as e:
        print(f"  ❌ Writer Lazy Discipline test FAILED: {e}")
        all_passed = False
    
    try:
        if not test_m2_full_pipeline():
            all_passed = False
    except Exception as e:
        print(f"  ❌ Full M2 Pipeline test FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("=== ALL M2 TESTS PASSED ===")
    else:
        print("=== SOME M2 TESTS FAILED ===")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
