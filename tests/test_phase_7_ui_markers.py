
import numpy as np
from typing import Any
from unittest import TestCase
from unittest.mock import Mock, MagicMock

from ui.overlay import _extract_id_source, _identity_label, draw_overlay
from schemas.identity_decision import IdentityDecision
from schemas import Frame, Tracklet


class TestPhase7IdSourceExtraction(TestCase):

    def test_extract_from_schema_field(self):
        decision_f = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
            id_source="F",
        )
        self.assertEqual(_extract_id_source(decision_f), "F")

        decision_g = IdentityDecision(
            track_id=2,
            identity_id="alice",
            binding_state="UNKNOWN",
            confidence=0.92,
            id_source="G",
        )
        self.assertEqual(_extract_id_source(decision_g), "G")

        decision_u = IdentityDecision(
            track_id=3,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0,
            id_source="U",
        )
        self.assertEqual(_extract_id_source(decision_u), "U")

    def test_extract_from_extra_dict(self):
        decision_g = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="UNKNOWN",
            confidence=0.92,
            extra={'id_source': 'G'},
        )
        self.assertEqual(_extract_id_source(decision_g), "G")

        decision_f = IdentityDecision(
            track_id=2,
            identity_id="bob",
            binding_state="CONFIRMED_WEAK",
            confidence=0.88,
            extra={'id_source': 'F'},
        )
        self.assertEqual(_extract_id_source(decision_f), "F")

    def test_extract_defaults_to_unknown(self):
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
        )
        self.assertEqual(_extract_id_source(decision), "U")

        decision_no_source = IdentityDecision(
            track_id=2,
            identity_id="bob",
            binding_state="CONFIRMED_WEAK",
            confidence=0.88,
            extra={'other_field': 'value'},
        )
        self.assertEqual(_extract_id_source(decision_no_source), "U")

        self.assertEqual(_extract_id_source(None), "U")

    def test_extract_invalid_id_source(self):
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
            id_source="INVALID",
        )
        self.assertEqual(_extract_id_source(decision), "U")

        decision_num = IdentityDecision(
            track_id=2,
            identity_id="bob",
            binding_state="CONFIRMED_WEAK",
            confidence=0.88,
            id_source=123,
        )
        self.assertEqual(_extract_id_source(decision_num), "U")


class TestPhase7IdentityLabelMarkers(TestCase):

    def test_label_with_face_marker(self):
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
            id_source="F",
        )

        label, debug, color = _identity_label(decision)

        self.assertIn("[F]", label)
        self.assertIn("alice", label)
        self.assertIn("0.85", label)

    def test_label_with_gps_marker(self):
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="UNKNOWN",
            confidence=0.92,
            id_source="G",
        )

        label, debug, color = _identity_label(decision)

        self.assertIn("[G]", label)
        self.assertIn("alice", label)

    def test_label_with_unknown_marker(self):
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0,
            id_source="U",
        )

        label, debug, color = _identity_label(decision)

        self.assertIn("[U]", label)

    def test_label_without_id_source(self):
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
        )

        label, debug, color = _identity_label(decision)

        self.assertIn("alice", label)
        self.assertIn("[U]", label)

    def test_label_marker_position(self):
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
            id_source="F",
        )

        label, debug, color = _identity_label(decision)

        self.assertIn("alice", label)
        self.assertIn("[F]", label)
        self.assertIn("0.85", label)

        f_pos = label.find("[F]")
        conf_pos = label.find("0.85")
        self.assertLess(f_pos, conf_pos, "Source marker should appear before confidence")

    def test_label_all_source_types(self):
        decisions = {
            "F": IdentityDecision(
                track_id=1,
                identity_id="alice",
                binding_state="CONFIRMED_WEAK",
                confidence=0.85,
                id_source="F",
            ),
            "G": IdentityDecision(
                track_id=2,
                identity_id="alice",
                binding_state="UNKNOWN",
                confidence=0.92,
                id_source="G",
            ),
            "U": IdentityDecision(
                track_id=3,
                identity_id=None,
                binding_state="UNKNOWN",
                confidence=0.0,
                id_source="U",
            ),
        }

        labels = {}
        for source, decision in decisions.items():
            label, debug, color = _identity_label(decision)
            labels[source] = label

        self.assertNotEqual(labels["F"], labels["G"])
        self.assertNotEqual(labels["F"], labels["U"])
        self.assertNotEqual(labels["G"], labels["U"])

        self.assertIn("[F]", labels["F"])
        self.assertIn("[G]", labels["G"])
        self.assertIn("[U]", labels["U"])


class TestPhase7DrawOverlayIntegration(TestCase):

    def test_draw_overlay_with_face_assigned(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(
            frame_id=1,
            ts=1.0,
            camera_id="cam0",
            size=(640, 480),
            image=img,
        )

        track = Mock(spec=Tracklet)
        track.track_id = 1
        track.tlbr = [100, 100, 200, 200]
        track.age_frames = 10
        track.lost_frames = 0
        track.confidence = 0.95

        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
            id_source="F",
            category="resident",
        )

        result = draw_overlay(
            frame=frame,
            tracks=[track],
            decisions=[decision],
            events=[],
            alerts=[],
            ui_cfg=None,
            fps=30.0,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, img.shape)

    def test_draw_overlay_with_gps_carried(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(
            frame_id=1,
            ts=1.0,
            camera_id="cam0",
            size=(640, 480),
            image=img,
        )

        track = Mock(spec=Tracklet)
        track.track_id=1
        track.tlbr = [100, 100, 200, 200]
        track.age_frames = 25
        track.lost_frames = 5
        track.confidence = 0.88

        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="UNKNOWN",
            confidence=0.92,
            id_source="G",
            category="resident",
        )

        result = draw_overlay(
            frame=frame,
            tracks=[track],
            decisions=[decision],
            events=[],
            alerts=[],
            ui_cfg=None,
            fps=30.0,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, img.shape)

    def test_draw_overlay_continuity_debug_disabled(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(
            frame_id=1,
            ts=1.0,
            camera_id="cam0",
            size=(640, 480),
            image=img,
        )

        track = Mock(spec=Tracklet)
        track.track_id = 1
        track.tlbr = [100, 100, 200, 200]
        track.age_frames = 25
        track.lost_frames = 5
        track.confidence = 0.88

        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="UNKNOWN",
            confidence=0.92,
            id_source="G",
        )

        result = draw_overlay(
            frame=frame,
            tracks=[track],
            decisions=[decision],
            events=[],
            alerts=[],
            ui_cfg=None,
            fps=30.0,
        )

        self.assertIsNotNone(result)

    def test_draw_overlay_continuity_debug_enabled(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(
            frame_id=1,
            ts=1.0,
            camera_id="cam0",
            size=(640, 480),
            image=img,
        )

        track = Mock(spec=Tracklet)
        track.track_id = 1
        track.tlbr = [100, 100, 200, 200]
        track.age_frames = 25
        track.lost_frames = 5
        track.confidence = 0.88

        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="UNKNOWN",
            confidence=0.92,
            id_source="G",
        )

        ui_cfg = {'show_continuity_debug': True}

        result = draw_overlay(
            frame=frame,
            tracks=[track],
            decisions=[decision],
            events=[],
            alerts=[],
            ui_cfg=ui_cfg,
            fps=30.0,
        )

        self.assertIsNotNone(result)

    def test_draw_overlay_mixed_sources(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(
            frame_id=1,
            ts=1.0,
            camera_id="cam0",
            size=(640, 480),
            image=img,
        )

        tracks = [
            Mock(spec=Tracklet, track_id=1, tlbr=[50, 50, 150, 150], age_frames=10, lost_frames=0, confidence=0.95),
            Mock(spec=Tracklet, track_id=2, tlbr=[200, 50, 300, 150], age_frames=25, lost_frames=5, confidence=0.88),
            Mock(spec=Tracklet, track_id=3, tlbr=[350, 50, 450, 150], age_frames=2, lost_frames=0, confidence=0.5),
        ]

        decisions = [
            IdentityDecision(track_id=1, identity_id="alice", binding_state="CONFIRMED_WEAK", confidence=0.85, id_source="F"),
            IdentityDecision(track_id=2, identity_id="alice", binding_state="UNKNOWN", confidence=0.92, id_source="G"),
            IdentityDecision(track_id=3, identity_id=None, binding_state="UNKNOWN", confidence=0.0, id_source="U"),
        ]

        result = draw_overlay(
            frame=frame,
            tracks=tracks,
            decisions=decisions,
            events=[],
            alerts=[],
            ui_cfg=None,
            fps=30.0,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, img.shape)


class TestPhase7BackwardsCompatibility(TestCase):

    def test_overlay_without_id_source_field(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(
            frame_id=1,
            ts=1.0,
            camera_id="cam0",
            size=(640, 480),
            image=img,
        )

        track = Mock(spec=Tracklet)
        track.track_id = 1
        track.tlbr = [100, 100, 200, 200]

        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
        )

        result = draw_overlay(
            frame=frame,
            tracks=[track],
            decisions=[decision],
            events=[],
            alerts=[],
            ui_cfg=None,
            fps=30.0,
        )

        self.assertIsNotNone(result)

    def test_identity_label_backwards_compatible(self):
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
        )

        label, debug, color = _identity_label(decision)

        self.assertIn("alice", label)
        self.assertIn("0.85", label)
        self.assertIsInstance(color, tuple)
        self.assertEqual(len(color), 3)


if __name__ == "__main__":
    import unittest
    unittest.main()
