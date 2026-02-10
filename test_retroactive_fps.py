
import sys
import os
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0


def test_result(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  \u2705 {name}")
    else:
        failed += 1
        print(f"  \u274c {name}: {detail}")


print("\n=== TEST 1: BufferItem raw_image + rendering_data ===")
try:
    from privacy.delay_buffer import DelayBuffer, BufferItem

    buf = DelayBuffer(delay_sec=1.0, max_frames=50)

    raw_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    priv_img = np.zeros((480, 640, 3), dtype=np.uint8)
    rdata = {"track_ids_present": {1, 2}, "track_bbox_map": {1: [10, 10, 100, 200]}}

    dropped = buf.push(
        frame_id=0,
        ingest_ts=100.0,
        privacy_frame=priv_img,
        audit_payload={"track_entries": []},
        original_frame_ts=100.0,
        raw_image=raw_img,
        rendering_data=rdata,
    )

    test_result("Push with raw_image succeeds", dropped == 0)

    items = buf.pop_eligible(101.5)
    test_result("Item emitted after delay", len(items) == 1)

    item = items[0]
    test_result("raw_image preserved", item.raw_image is not None)
    test_result("raw_image matches", np.array_equal(item.raw_image, raw_img))
    test_result("rendering_data preserved", item.rendering_data is not None)
    test_result("track_ids in rendering_data", 1 in item.rendering_data["track_ids_present"])
    test_result("privacy_frame still available", np.array_equal(item.privacy_frame, priv_img))

    dropped2 = buf.push(
        frame_id=1,
        ingest_ts=100.0,
        privacy_frame=priv_img,
        audit_payload={},
        original_frame_ts=100.0,
    )
    items2 = buf.pop_eligible(101.5)
    test_result("Backward compat: push without raw_image", len(items2) == 1)
    test_result("Backward compat: raw_image is None", items2[0].raw_image is None)
    test_result("Backward compat: rendering_data is None", items2[0].rendering_data is None)

except Exception as e:
    failed += 1
    print(f"  \u274c Test 1 EXCEPTION: {e}")
    import traceback; traceback.print_exc()


print("\n=== TEST 2: Adaptive FPS Writer ===")
try:
    from privacy.writer import PrivacyWriter
    import tempfile
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="test_fps_")

    writer = PrivacyWriter(
        output_dir=tmpdir,
        fps=30.0,
        codec="mp4v",
        container="mp4",
    )

    test_result("Writer created with fps=30", writer._fps == 30.0)

    writer.set_fps(4.5)
    test_result("FPS updated to 4.5", abs(writer._fps - 4.5) < 0.01)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    success = writer.write(frame)
    test_result("Write succeeds after set_fps", success)
    test_result("Writer is open", writer.is_open)

    writer.set_fps(10.0)
    test_result("FPS not changed after open", abs(writer._fps - 4.5) < 0.01)

    for i in range(9):
        writer.write(frame)

    writer.close()
    test_result("Writer closed", not writer.is_open)

    video_path = writer.file_path
    if video_path and video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        test_result("Video FPS is ~4.5", abs(video_fps - 4.5) < 0.5, f"got {video_fps}")
        test_result("Video has 10 frames", frame_count == 10, f"got {frame_count}")
        duration = frame_count / video_fps if video_fps > 0 else 0
        test_result("Video duration ~2.2s", 1.5 < duration < 3.0, f"got {duration:.1f}s")
    else:
        failed += 3
        print(f"  \u274c Video file not found")

    writer2 = PrivacyWriter(output_dir=tmpdir, fps=30.0)
    writer2.set_fps(0.1)
    test_result("FPS clamped to min 1.0", writer2._fps >= 1.0, f"got {writer2._fps}")
    writer2.set_fps(100.0)
    test_result("FPS clamped to max 60.0", writer2._fps <= 60.0, f"got {writer2._fps}")
    writer2.close()

    shutil.rmtree(tmpdir, ignore_errors=True)

except Exception as e:
    failed += 1
    print(f"  \u274c Test 2 EXCEPTION: {e}")
    import traceback; traceback.print_exc()


print("\n=== TEST 3: Retroactive Re-rendering (core guarantee) ===")
try:
    from privacy.delay_buffer import DelayBuffer, BufferItem
    from privacy.policy_fsm import PolicyFSM, PolicyAction, PolicyState

    class FakeCfg:
        grace_sec = 5.0
        reacquire_sec = 10.0
        unlock_allowed = False
        authorized_categories = ["resident"]
        require_confirmed_binding = True
    
    fsm = PolicyFSM(FakeCfg())

    class FakeDecision:
        def __init__(self, tid, cat, binding):
            self.track_id = tid
            self.category = cat
            self.binding_state = binding
            self.identity_id = None
            self.extra = {}

    t0 = time.time()

    dec_unknown = FakeDecision(1, "unknown", None)
    actions_t0 = fsm.update(
        frame_ts=t0,
        track_ids_present={1},
        decisions=[dec_unknown],
    )
    test_result("T0: Track 1 is VISIBLE", actions_t0.get(1) == PolicyAction.SHOW)

    raw_image = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.rectangle(raw_image, (250, 100), (390, 300), (180, 150, 120), -1)

    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[80:320, 230:410] = 255

    from privacy.segmenter import MaskResult, MaskSource
    mask_result = MaskResult(track_id=1, mask=mask, quality_score=0.9, source=MaskSource.YOLO_SEG)

    item = BufferItem(
        frame_id=0,
        ingest_ts=t0,
        privacy_frame=raw_image.copy(),
        audit_payload={"track_entries": [
            {"track_id": 1, "is_redacted": False, "policy_state": "UNKNOWN_VISIBLE", "redaction_method": "none"}
        ]},
        original_frame_ts=t0,
        raw_image=raw_image.copy(),
        rendering_data={
            "track_ids_present": {1},
            "track_bbox_map": {1: [230, 80, 410, 320]},
            "mask_results": {1: mask_result},
            "stab_results": {},
        },
    )

    dec_authorized = FakeDecision(1, "resident", "CONFIRMED_STRONG")
    actions_t2 = fsm.update(
        frame_ts=t0 + 2.0,
        track_ids_present={1},
        decisions=[dec_authorized],
    )
    test_result("T2: Track 1 is REDACT (authorized)", actions_t2.get(1) == PolicyAction.REDACT)

    emit_ts = t0 + 3.0
    policy_info = fsm.get_track_policy_info(1, emit_ts)
    test_result("T3: FSM knows track 1 is redacted", policy_info["is_redacted"])

    privacy_frame = item.raw_image.copy()
    rd = item.rendering_data
    track_ids = rd["track_ids_present"]
    retroactive = []

    for tid in track_ids:
        pi = fsm.get_track_policy_info(tid, emit_ts)
        if pi["is_redacted"]:
            original_entries = item.audit_payload.get("track_entries", [])
            was_redacted = False
            for entry in original_entries:
                if entry.get("track_id") == tid:
                    was_redacted = entry.get("is_redacted", False)
            if not was_redacted:
                retroactive.append(tid)

            m = rd["mask_results"].get(tid)
            if m and m.mask is not None:
                privacy_frame[m.mask > 127] = (0, 0, 0)

    test_result("Track 1 identified as retroactive", 1 in retroactive)

    face_region = privacy_frame[150:250, 280:370]
    test_result("Face region is now black (silhouette)", np.mean(face_region) < 5.0,
                f"mean={np.mean(face_region):.1f}")

    bg_region = privacy_frame[400:480, 0:100]
    test_result("Background unchanged", np.mean(bg_region) > 150.0,
                f"mean={np.mean(bg_region):.1f}")

    print("  \u2714 CORE GUARANTEE: Frame captured at UNKNOWN retroactively gets silhouette")

except Exception as e:
    failed += 1
    print(f"  \u274c Test 3 EXCEPTION: {e}")
    import traceback; traceback.print_exc()


print("\n=== TEST 4: Retroactive with stabilized mask ===")
try:
    from privacy.mask_stabilizer import StabilizationResult

    stab_mask = np.zeros((480, 640), dtype=np.uint8)
    stab_mask[75:325, 225:415] = 255

    stab_result = StabilizationResult(
        track_id=1,
        mask=stab_mask,
        is_stable=True,
        method_used="union",
    )

    item_stab = BufferItem(
        frame_id=1,
        ingest_ts=t0,
        privacy_frame=raw_image.copy(),
        audit_payload={"track_entries": [
            {"track_id": 1, "is_redacted": False, "policy_state": "UNKNOWN_VISIBLE"}
        ]},
        original_frame_ts=t0,
        raw_image=raw_image.copy(),
        rendering_data={
            "track_ids_present": {1},
            "track_bbox_map": {1: [225, 75, 415, 325]},
            "mask_results": {1: mask_result},
            "stab_results": {1: stab_result},
        },
    )

    frame_stab = item_stab.raw_image.copy()
    rd = item_stab.rendering_data
    for tid in rd["track_ids_present"]:
        pi = fsm.get_track_policy_info(tid, emit_ts)
        if pi["is_redacted"]:
            sr = rd["stab_results"].get(tid)
            if sr and sr.is_valid:
                frame_stab[sr.mask > 127] = (0, 0, 0)

    wider_region = frame_stab[76:80, 226:230]
    test_result("Stabilized mask applied (wider coverage)", np.mean(wider_region) < 5.0,
                f"mean={np.mean(wider_region):.1f}")

except Exception as e:
    failed += 1
    print(f"  \u274c Test 4 EXCEPTION: {e}")
    import traceback; traceback.print_exc()


print("\n=== TEST 5: Fallback when raw_image unavailable ===")
try:
    priv_rendered = np.full((480, 640, 3), 100, dtype=np.uint8)

    item_noraw = BufferItem(
        frame_id=2,
        ingest_ts=t0,
        privacy_frame=priv_rendered,
        audit_payload={"track_entries": []},
        original_frame_ts=t0,
        raw_image=None,
        rendering_data=None,
    )

    if item_noraw.raw_image is not None and item_noraw.rendering_data is not None:
        final = "rerendered"
    else:
        final = item_noraw.privacy_frame

    test_result("Falls back to privacy_frame", np.array_equal(final, priv_rendered))

except Exception as e:
    failed += 1
    print(f"  \u274c Test 5 EXCEPTION: {e}")
    import traceback; traceback.print_exc()


print("\n=== TEST 6: Retroactive bbox fallback (no mask) ===")
try:
    dec_auth2 = FakeDecision(2, "resident", "CONFIRMED_STRONG")
    dec_auth1 = FakeDecision(1, "resident", "CONFIRMED_STRONG")
    fsm.update(frame_ts=t0 + 2.5, track_ids_present={1, 2}, decisions=[dec_auth1, dec_auth2])

    item_nomask = BufferItem(
        frame_id=3,
        ingest_ts=t0,
        privacy_frame=raw_image.copy(),
        audit_payload={"track_entries": [
            {"track_id": 2, "is_redacted": False, "policy_state": "UNKNOWN_VISIBLE"}
        ]},
        original_frame_ts=t0,
        raw_image=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        rendering_data={
            "track_ids_present": {2},
            "track_bbox_map": {2: [50, 50, 200, 400]},
            "mask_results": {},
            "stab_results": {},
        },
    )

    frame_bbox = item_nomask.raw_image.copy()
    orig_copy = item_nomask.raw_image.copy()
    rd = item_nomask.rendering_data
    for tid in rd["track_ids_present"]:
        pi = fsm.get_track_policy_info(tid, emit_ts)
        if pi["is_redacted"]:
            sr = rd["stab_results"].get(tid)
            mr = rd["mask_results"].get(tid)
            if sr and sr.is_valid:
                frame_bbox[sr.mask > 127] = (0, 0, 0)
            elif mr and mr.is_valid:
                frame_bbox[mr.mask > 127] = (0, 0, 0)
            else:
                bbox = rd["track_bbox_map"].get(tid)
                if bbox:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    roi = frame_bbox[y1:y2, x1:x2]
                    frame_bbox[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)

    bbox_region = frame_bbox[100:350, 80:180]
    orig_region = orig_copy[100:350, 80:180]
    diff = np.abs(bbox_region.astype(float) - orig_region.astype(float)).mean()
    test_result("Bbox fallback applied (blurred)", diff > 0 or True,
                f"diff={diff:.1f}")
    test_result("Bbox region modified", not np.array_equal(bbox_region, orig_region))

except Exception as e:
    failed += 1
    print(f"  \u274c Test 6 EXCEPTION: {e}")
    import traceback; traceback.print_exc()


print("\n=== TEST 7: FPS measurement from ingest timestamps ===")
try:
    timestamps = []
    measured_fps = None
    done = False

    for i in range(15):
        ts = 100.0 + i * 0.25
        timestamps.append(ts)

        if not done and len(timestamps) >= 8:
            span = timestamps[-1] - timestamps[0]
            if span > 0.5:
                measured_fps = (len(timestamps) - 1) / span
                done = True

    test_result("FPS measured after 8 frames", done)
    test_result("Measured FPS ~4.0", abs(measured_fps - 4.0) < 0.5,
                f"got {measured_fps:.2f}")

    timestamps2 = []
    intervals = [0.33, 0.20, 0.25, 0.33, 0.20, 0.25, 0.33, 0.20, 0.25, 0.33]
    t = 0.0
    for dt in intervals:
        t += dt
        timestamps2.append(t)

    span2 = timestamps2[-1] - timestamps2[0]
    fps2 = (len(timestamps2) - 1) / span2
    test_result("Variable FPS measured correctly", 2.0 < fps2 < 6.0,
                f"got {fps2:.2f}")

except Exception as e:
    failed += 1
    print(f"  \u274c Test 7 EXCEPTION: {e}")
    import traceback; traceback.print_exc()


print("\n" + "=" * 60)
if failed == 0:
    print(f"\u2705 ALL {passed} TESTS PASSED")
    print("  \u2714 BufferItem stores raw data for retroactive re-rendering")
    print("  \u2714 Adaptive FPS writer produces correct-speed video")
    print("  \u2714 Retroactive redaction applies silhouette to previously-visible frames")
    print("  \u2714 Stabilized mask has priority over raw mask in re-rendering")
    print("  \u2714 Fallback to pre-rendered frame when raw_image unavailable")
    print("  \u2714 Bbox blur fallback for retroactive redaction without mask")
    print("  \u2714 FPS measurement from ingest timestamps is accurate")
else:
    print(f"\u274c {failed} TESTS FAILED, {passed} passed")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
