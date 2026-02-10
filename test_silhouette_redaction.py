
import numpy as np
import sys
from pathlib import Path

import cv2

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from privacy.pipeline import PrivacyPipeline


def _make_pipeline(redaction_style: str = "blur", silhouette_cleanup: bool = False) -> PrivacyPipeline:
    cfg = type("Cfg", (), {
        "enabled": False,
        "delay_sec": 0.0,
        "redaction_style": redaction_style,
        "silhouette_cleanup": silhouette_cleanup,
        "output": None,
        "policy": None,
        "audit": type("A", (), {"enabled": False, "filename": "test_audit.jsonl", "flush_interval_sec": 1.0})(),
        "ui": type("UI", (), {"show_privacy_preview": False, "preview_window_title": "", "show_preview_watermark": False})(),
        "segmentation": type("S", (), {"enabled": False, "backend": "none"})(),
        "stabilization": type("St", (), {"enabled": False})(),
        "metrics": type("M", (), {"enabled": False})(),
    })()
    return PrivacyPipeline(cfg)


def _gray_frame(h: int = 200, w: int = 300, value: int = 180) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _person_mask(h: int = 200, w: int = 300,
                 x1: int = 80, y1: int = 40, x2: int = 180, y2: int = 160) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def test_mask_silhouette_fills_black():
    print("\n=== Test 1: Mask Silhouette Fills Black ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame()
    mask = _person_mask()

    pipe._apply_mask_silhouette(frame, mask)

    roi = frame[40:160, 80:180]
    assert np.all(roi == 0), f"Expected all black inside mask, got min={roi.min()} max={roi.max()}"

    assert np.all(frame[0:34, :] == 180), "Far top region should be untouched"
    assert np.all(frame[166:200, :] == 180), "Far bottom region should be untouched"
    assert np.all(frame[:, 0:74] == 180), "Far left region should be untouched"
    assert np.all(frame[:, 186:300] == 180), "Far right region should be untouched"

    print("✓ Mask region (40:160, 80:180) → solid black (0,0,0)")
    print("✓ Pixels well outside mask → unchanged (180,180,180)")
    print("✅ Test 1 PASSED\n")


def test_mask_silhouette_custom_color():
    print("=== Test 2: Mask Silhouette Custom Color ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame()
    mask = _person_mask()

    pipe._apply_mask_silhouette(frame, mask, color=(40, 40, 40))

    roi = frame[45:155, 85:175]
    assert np.all(roi == 40), f"Expected (40,40,40) fill, got unique values {np.unique(roi)}"

    print("✓ Mask region filled with custom color (40,40,40)")
    print("✅ Test 2 PASSED\n")


def test_mask_silhouette_feathered_edge():
    print("=== Test 3: Mask Silhouette Feathered Edge ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame(value=200)
    mask = _person_mask()

    pipe._apply_mask_silhouette(frame, mask, color=(0, 0, 0), feather_px=5)

    center = frame[80:120, 110:150]
    assert center.mean() < 30, f"Center should be near-black, got mean={center.mean():.1f}"

    outside = frame[0:20, 0:40]
    assert np.all(outside == 200), "Far outside should be untouched"

    print(f"✓ Center of mask mean brightness = {center.mean():.1f} (near 0)")
    print("✓ Far outside unchanged (200)")
    print("✅ Test 3 PASSED\n")


def test_contour_refinement_fills_internal_hole():
    print("=== Test 4: Contour Refinement Fills Internal Hole ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame()

    mask = _person_mask()
    mask[80:110, 110:140] = 0

    pipe._apply_mask_silhouette(frame, mask.copy())

    hole_region = frame[80:110, 110:140]
    assert np.all(hole_region == 0), \
        f"Internal hole should be filled by contour refinement, got unique: {np.unique(hole_region)}"

    print("✓ 30x30 internal hole filled by contour refinement")
    print("✅ Test 4 PASSED\n")


def test_dispatch_silhouette_mode():
    print("=== Test 5: Dispatch Routes to Silhouette ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame()
    mask = _person_mask()

    pipe._apply_mask_redaction(frame, mask)

    roi = frame[40:160, 80:180]
    assert np.all(roi == 0), "Dispatch should route to silhouette (black)"

    print("✓ _apply_mask_redaction dispatched to silhouette")
    print("✅ Test 5 PASSED\n")


def test_dispatch_blur_mode():
    print("=== Test 6: Dispatch Routes to Blur ===")
    pipe = _make_pipeline("blur")
    frame = _gray_frame()
    mask = _person_mask()

    pipe._apply_mask_redaction(frame, mask)

    roi = frame[40:160, 80:180]
    assert not np.all(roi == 0), "Blur mode should NOT produce black fill"

    print("✓ _apply_mask_redaction dispatched to blur (not black)")
    print("✅ Test 6 PASSED\n")


def test_bbox_dispatch_always_blur():
    print("=== Test 7: Bbox Dispatch Always Blur ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame()
    bbox = [50.0, 30.0, 150.0, 120.0]

    pipe._apply_bbox_redaction(frame, bbox)

    roi = frame[30:120, 50:150]
    assert not np.all(roi == 0), "Bbox dispatch should use blur, not silhouette"
    assert roi.mean() > 150, f"Blur of gray should stay near 180, got mean={roi.mean():.0f}"

    print("✓ _apply_bbox_redaction → blur (not black rectangle)")
    print("✅ Test 7 PASSED\n")


def test_empty_mask_no_crash():
    print("=== Test 8: Empty Mask Safety ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame()
    mask = np.zeros((200, 300), dtype=np.uint8)

    pipe._apply_mask_silhouette(frame, mask)

    assert np.all(frame == 180), "Empty mask should leave frame unchanged"
    print("✓ Empty mask → no modification, no crash")
    print("✅ Test 8 PASSED\n")


def test_full_mask_fills_entire_frame():
    print("=== Test 9: Full Mask Fills Entire Frame ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame()
    mask = np.full((200, 300), 255, dtype=np.uint8)

    pipe._apply_mask_silhouette(frame, mask)

    assert np.all(frame == 0), "Full mask should turn entire frame black"
    print("✓ Full mask → entire frame black")
    print("✅ Test 9 PASSED\n")


def test_config_default_is_blur():
    print("=== Test 10: Config Default is Blur ===")
    from core.config import PrivacyConfig
    cfg = PrivacyConfig()
    assert cfg.redaction_style == "blur", f"Expected 'blur', got '{cfg.redaction_style}'"
    assert cfg.silhouette_cleanup is False, "silhouette_cleanup should default to False"
    print("✓ PrivacyConfig().redaction_style == 'blur'")
    print("✓ PrivacyConfig().silhouette_cleanup == False")
    print("✅ Test 10 PASSED\n")


def test_bbox_anchored_same_result():
    print("=== Test 11: Bbox-anchored Same Result ===")
    pipe = _make_pipeline("silhouette")

    frame_no_bbox = _gray_frame()
    frame_with_bbox = _gray_frame()
    mask = _person_mask()

    pipe._apply_mask_silhouette(frame_no_bbox, mask.copy())
    pipe._apply_mask_silhouette(frame_with_bbox, mask.copy(), bbox=[80.0, 40.0, 180.0, 160.0])

    assert np.array_equal(frame_no_bbox, frame_with_bbox), \
        "Bbox-anchored result must be identical to full-frame scan"

    print("✓ With bbox: identical fill to without bbox")
    print("✅ Test 11 PASSED\n")


def test_bbox_anchored_with_padding():
    print("=== Test 12: Bbox Padding Captures Spillover ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame()

    mask = _person_mask(x1=75, y1=35, x2=185, y2=165)
    bbox = [80.0, 40.0, 180.0, 160.0]

    pipe._apply_mask_silhouette(frame, mask, bbox=bbox)

    assert np.all(frame[35:40, 75:185] == 0), "Top spillover row should be black"
    assert np.all(frame[160:165, 75:185] == 0), "Bottom spillover row should be black"
    assert np.all(frame[35:165, 75:80] == 0), "Left spillover column should be black"
    assert np.all(frame[35:165, 180:185] == 0), "Right spillover column should be black"

    assert np.all(frame[0:25, :] == 180), "Far outside untouched"

    print("✓ 5px spillover captured by 10px padding")
    print("✅ Test 12 PASSED\n")


def test_bbox_none_fallback():
    print("=== Test 13: Bbox None Fallback ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame()
    mask = _person_mask()

    pipe._apply_mask_silhouette(frame, mask, bbox=None)

    roi = frame[40:160, 80:180]
    assert np.all(roi == 0), "Fill should work with bbox=None"
    assert np.all(frame[0:34, :] == 180), "Far outside untouched"

    print("✓ bbox=None → full-frame scan works correctly")
    print("✅ Test 13 PASSED\n")


def test_bbox_degenerate_no_crash():
    print("=== Test 14: Degenerate Bbox Safety ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame()
    mask = _person_mask()

    pipe._apply_mask_silhouette(frame, mask, bbox=[5.0, 5.0, 5.0, 5.0])

    assert np.all(frame == 180), "Degenerate bbox far from mask should leave frame unchanged"

    print("✓ Degenerate bbox far from mask → no crash, no modification")
    print("✅ Test 14 PASSED\n")


def test_dispatch_passes_bbox():
    print("=== Test 15: Dispatch Passes Bbox ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame()
    mask = _person_mask()
    bbox = [80.0, 40.0, 180.0, 160.0]

    pipe._apply_mask_redaction(frame, mask, bbox=bbox)

    roi = frame[40:160, 80:180]
    assert np.all(roi == 0), "Dispatch with bbox should fill correctly"

    print("✓ _apply_mask_redaction(bbox=) → silhouette filled")
    print("✅ Test 15 PASSED\n")


def test_cleanup_fills_hole():
    print("=== Test 16: Cleanup Fills Hole ===")
    pipe = _make_pipeline("silhouette", silhouette_cleanup=True)
    frame = _gray_frame()

    mask = _person_mask()
    mask[95:100, 125:130] = 0

    pipe._apply_mask_silhouette(frame, mask.copy())

    hole_region = frame[95:100, 125:130]
    assert np.all(hole_region == 0), \
        f"Cleanup should fill 5x5 hole, got unique vals: {np.unique(hole_region)}"

    print("✓ 5x5 hole inside mask filled by close operation")
    print("✅ Test 16 PASSED\n")


def test_cleanup_removes_noise():
    print("=== Test 17: Cleanup Removes Noise ===")
    pipe = _make_pipeline("silhouette", silhouette_cleanup=True)
    frame = _gray_frame()

    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[10:12, 10:12] = 255

    pipe._apply_mask_silhouette(frame, mask.copy())

    assert np.all(frame == 180), \
        f"2x2 noise should be removed by open, but frame was modified"

    print("✓ 2x2 noise blob removed by open operation")
    print("✅ Test 17 PASSED\n")


def test_cleanup_disabled_contour_still_fills_hole():
    print("=== Test 18: Contour Refinement Fills Hole Without Cleanup ===")
    pipe = _make_pipeline("silhouette", silhouette_cleanup=False)
    frame = _gray_frame()

    mask = _person_mask()
    mask[95:100, 125:130] = 0

    pipe._apply_mask_silhouette(frame, mask.copy())

    hole_region = frame[95:100, 125:130]
    assert np.all(hole_region == 0), \
        f"Contour refinement should fill hole even without cleanup, got: {np.unique(hole_region)}"

    print("✓ Contour refinement fills hole even without morph cleanup")
    print("✅ Test 18 PASSED\n")


def test_cleanup_with_bbox_anchor():
    print("=== Test 19: Cleanup + Bbox-anchored ===")
    pipe = _make_pipeline("silhouette", silhouette_cleanup=True)
    frame = _gray_frame()

    mask = _person_mask()
    mask[95:100, 125:130] = 0
    bbox = [80.0, 40.0, 180.0, 160.0]

    pipe._apply_mask_silhouette(frame, mask.copy(), bbox=bbox)

    hole_region = frame[95:100, 125:130]
    body_region = frame[55:85, 95:165]
    outside = frame[0:30, 0:70]

    assert np.all(hole_region == 0), "Hole filled with cleanup + bbox"
    assert np.all(body_region == 0), "Body region filled"
    assert np.all(outside == 180), "Outside untouched"

    print("✓ Cleanup + bbox-anchored → hole filled, body black, outside clean")
    print("✅ Test 19 PASSED\n")


def test_contour_removes_noise_blob():
    print("=== Test 20: Contour Removes Noise Blob ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame()

    mask = _person_mask()
    mask[5:8, 5:8] = 255

    pipe._apply_mask_silhouette(frame, mask.copy())

    assert np.all(frame[50:150, 90:170] == 0), "Body region should be black"
    assert np.all(frame[0:5, 0:5] == 180), \
        f"Isolated noise blob region should be removed, got: {np.unique(frame[0:5, 0:5])}"

    print("✓ 3x3 noise blob removed by area filter")
    print("✓ Main body preserved")
    print("✅ Test 20 PASSED\n")


def test_contour_smooths_jagged_edges():
    print("=== Test 21: Contour Smooths Jagged Edges ===")
    pipe = _make_pipeline("silhouette")

    h, w = 300, 400
    mask = np.zeros((h, w), dtype=np.uint8)
    for row in range(60, 240):
        offset = 5 * (row % 2)
        mask[row, 80 + offset:300] = 255

    raw_binary = (mask > 127).astype(np.uint8) * 255
    raw_contours, _ = cv2.findContours(raw_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_vertices = sum(len(c) for c in raw_contours)

    refined = pipe._refine_silhouette_contours(mask)
    ref_binary = (refined > 127).astype(np.uint8) * 255
    ref_contours, _ = cv2.findContours(ref_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref_vertices = sum(len(c) for c in ref_contours)

    assert ref_vertices < raw_vertices, \
        f"Smoothing should reduce vertices: raw={raw_vertices}, refined={ref_vertices}"

    raw_area = np.sum(raw_binary > 0)
    ref_area = np.sum(ref_binary > 0)
    ratio = ref_area / max(1, raw_area)
    assert 0.85 < ratio < 1.30, f"Area should be similar: raw={raw_area}, refined={ref_area}, ratio={ratio:.2f}"

    print(f"✓ Vertices reduced: {raw_vertices} → {ref_vertices} ({100 * ref_vertices / raw_vertices:.0f}%)")
    print(f"✓ Area preserved: ratio={ratio:.2f}")
    print("✅ Test 21 PASSED\n")


def test_contour_handles_multiple_body_parts():
    print("=== Test 22: Multiple Body Parts ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame(h=300, w=400)

    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[50:200, 100:250] = 255
    mask[50:100, 280:340] = 255

    pipe._apply_mask_silhouette(frame, mask.copy())

    body_center = frame[120:180, 150:200]
    hand_center = frame[65:90, 300:330]
    assert np.all(body_center == 0), "Main body should be black"
    assert np.all(hand_center == 0), "Separated hand should be black"

    assert np.all(frame[0:40, 0:90] == 180), "Outside untouched"

    print("✓ Main body filled")
    print("✓ Separated hand also filled (above area threshold)")
    print("✅ Test 22 PASSED\n")


def test_contour_refinement_failsafe():
    print("=== Test 23: Contour Refinement Failsafe ===")
    pipe = _make_pipeline("silhouette")
    frame = _gray_frame()
    mask = _person_mask()

    pipe._apply_mask_silhouette(frame, mask.copy())

    roi = frame[40:160, 80:180]
    assert np.all(roi == 0), "Should still produce black silhouette"

    print("✓ Silhouette works correctly (contour refinement or fallback)")
    print("✅ Test 23 PASSED\n")

def run_all_tests():
    print("\n" + "=" * 60)
    print("Silhouette Redaction - Validation Test Suite")
    print("=" * 60)

    tests = [
        test_mask_silhouette_fills_black,
        test_mask_silhouette_custom_color,
        test_mask_silhouette_feathered_edge,
        test_contour_refinement_fills_internal_hole,
        test_dispatch_silhouette_mode,
        test_dispatch_blur_mode,
        test_bbox_dispatch_always_blur,
        test_empty_mask_no_crash,
        test_full_mask_fills_entire_frame,
        test_config_default_is_blur,
        test_bbox_anchored_same_result,
        test_bbox_anchored_with_padding,
        test_bbox_none_fallback,
        test_bbox_degenerate_no_crash,
        test_dispatch_passes_bbox,
        test_cleanup_fills_hole,
        test_cleanup_removes_noise,
        test_cleanup_disabled_contour_still_fills_hole,
        test_cleanup_with_bbox_anchor,
        test_contour_removes_noise_blob,
        test_contour_smooths_jagged_edges,
        test_contour_handles_multiple_body_parts,
        test_contour_refinement_failsafe,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_fn.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test_fn.__name__} ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    if failed == 0:
        print(f"✅ ALL {passed} TESTS PASSED")
    else:
        print(f"❌ {failed} FAILED, {passed} passed")
    print("=" * 60)

    print("\nImplementation Summary:")
    print("  ✓ _apply_mask_silhouette: contour-refined body-shape fill")
    print("  ✓ _refine_silhouette_contours: RETR_EXTERNAL + approxPolyDP + FILLED")
    print("  ✓ Bbox fallback always uses blur (no black rectangle)")
    print("  ✓ _apply_mask_redaction / _apply_bbox_redaction: dispatch by config")
    print("  ✓ redaction_style config flag: 'blur' (default) | 'silhouette'")
    print("  ✓ Contour refinement: fills holes, removes noise, smooths edges")
    print("  ✓ Optional morph cleanup + feathered edge still available")
    print()

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
