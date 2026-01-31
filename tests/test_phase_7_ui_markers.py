"""
PHASE 7 TESTS: UI Overlay Updates (Visual Feedback)

Tests that UI overlay correctly displays [F]/[G]/[U] markers and optional
continuity debug information.

Test Coverage:
1. id_source extraction from decisions (schema field and extra dict)
2. [F]/[G]/[U] marker display logic
3. Source marker integration with existing labels
4. Optional continuity debug overlay
5. Backwards compatibility (no id_source field)

Phase 7 Goal: Visual feedback for operators showing identity source
"""

import numpy as np
from typing import Any
from unittest import TestCase
from unittest.mock import Mock, MagicMock

from ui.overlay import _extract_id_source, _identity_label, draw_overlay
from schemas.identity_decision import IdentityDecision
from schemas import Frame, Tracklet


class TestPhase7IdSourceExtraction(TestCase):
    """Test Phase 7 id_source extraction logic."""

    def test_extract_from_schema_field(self):
        """Test id_source extraction from decision.id_source field."""
        # Face-assigned
        decision_f = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
            id_source="F",
        )
        self.assertEqual(_extract_id_source(decision_f), "F")

        # GPS-carried
        decision_g = IdentityDecision(
            track_id=2,
            identity_id="alice",
            binding_state="UNKNOWN",
            confidence=0.92,
            id_source="G",
        )
        self.assertEqual(_extract_id_source(decision_g), "G")

        # Unknown
        decision_u = IdentityDecision(
            track_id=3,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0,
            id_source="U",
        )
        self.assertEqual(_extract_id_source(decision_u), "U")

    def test_extract_from_extra_dict(self):
        """Test id_source extraction from decision.extra fallback."""
        # GPS-carried in extra dict
        decision_g = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="UNKNOWN",
            confidence=0.92,
            extra={'id_source': 'G'},
        )
        self.assertEqual(_extract_id_source(decision_g), "G")

        # Face-assigned in extra dict
        decision_f = IdentityDecision(
            track_id=2,
            identity_id="bob",
            binding_state="CONFIRMED_WEAK",
            confidence=0.88,
            extra={'id_source': 'F'},
        )
        self.assertEqual(_extract_id_source(decision_f), "F")

    def test_extract_defaults_to_unknown(self):
        """Test id_source defaults to 'U' when missing."""
        # No id_source field at all
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
        )
        self.assertEqual(_extract_id_source(decision), "U")

        # extra dict without id_source
        decision_no_source = IdentityDecision(
            track_id=2,
            identity_id="bob",
            binding_state="CONFIRMED_WEAK",
            confidence=0.88,
            extra={'other_field': 'value'},
        )
        self.assertEqual(_extract_id_source(decision_no_source), "U")

        # None decision
        self.assertEqual(_extract_id_source(None), "U")

    def test_extract_invalid_id_source(self):
        """Test id_source extraction handles invalid values gracefully."""
        # Invalid source value (should default to U)
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
            id_source="INVALID",
        )
        self.assertEqual(_extract_id_source(decision), "U")

        # Numeric source (should default to U)
        decision_num = IdentityDecision(
            track_id=2,
            identity_id="bob",
            binding_state="CONFIRMED_WEAK",
            confidence=0.88,
            id_source=123,
        )
        # Will convert to string "123", which is not in valid set, so defaults to U
        self.assertEqual(_extract_id_source(decision_num), "U")


class TestPhase7IdentityLabelMarkers(TestCase):
    """Test Phase 7 identity label marker integration."""

    def test_label_with_face_marker(self):
        """Test identity label includes [F] marker for face-assigned."""
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
            id_source="F",
        )

        label, debug, color = _identity_label(decision)

        # Should contain [F] marker
        self.assertIn("[F]", label)
        # Should contain identity name
        self.assertIn("alice", label)
        # Should contain confidence
        self.assertIn("0.85", label)

    def test_label_with_gps_marker(self):
        """Test identity label includes [G] marker for GPS-carried."""
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="UNKNOWN",
            confidence=0.92,
            id_source="G",
        )

        label, debug, color = _identity_label(decision)

        # Should contain [G] marker
        self.assertIn("[G]", label)
        # Should contain identity name
        self.assertIn("alice", label)

    def test_label_with_unknown_marker(self):
        """Test identity label includes [U] marker for unknown."""
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0,
            id_source="U",
        )

        label, debug, color = _identity_label(decision)

        # Should contain [U] marker
        self.assertIn("[U]", label)

    def test_label_without_id_source(self):
        """Test identity label works without id_source (backwards compatibility)."""
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
        )

        label, debug, color = _identity_label(decision)

        # Should still work (defaults to [U])
        self.assertIn("alice", label)
        # Will have [U] as default
        self.assertIn("[U]", label)

    def test_label_marker_position(self):
        """Test source marker appears in correct position relative to other elements."""
        # With binding emoji and confidence
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
            id_source="F",
        )

        label, debug, color = _identity_label(decision)

        # Should have format: [emoji] name [source] (confidence)
        # Example: "✓ alice [F] (0.85)"
        self.assertIn("alice", label)
        self.assertIn("[F]", label)
        self.assertIn("0.85", label)

        # [F] should appear before confidence
        f_pos = label.find("[F]")
        conf_pos = label.find("0.85")
        self.assertLess(f_pos, conf_pos, "Source marker should appear before confidence")

    def test_label_all_source_types(self):
        """Test all three source types (F/G/U) produce distinct labels."""
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

        # All labels should be distinct
        self.assertNotEqual(labels["F"], labels["G"])
        self.assertNotEqual(labels["F"], labels["U"])
        self.assertNotEqual(labels["G"], labels["U"])

        # Each should contain its respective marker
        self.assertIn("[F]", labels["F"])
        self.assertIn("[G]", labels["G"])
        self.assertIn("[U]", labels["U"])


class TestPhase7DrawOverlayIntegration(TestCase):
    """Test Phase 7 draw_overlay integration with id_source markers."""

    def test_draw_overlay_with_face_assigned(self):
        """Test draw_overlay handles face-assigned tracks correctly."""
        # Create minimal frame
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(
            frame_id=1,
            ts=1.0,
            camera_id="cam0",
            size=(640, 480),
            image=img,
        )

        # Create track with bbox
        track = Mock(spec=Tracklet)
        track.track_id = 1
        track.tlbr = [100, 100, 200, 200]
        track.age_frames = 10
        track.lost_frames = 0
        track.confidence = 0.95

        # Create decision with face-assigned identity
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
            id_source="F",
            category="resident",
        )

        # Draw overlay
        result = draw_overlay(
            frame=frame,
            tracks=[track],
            decisions=[decision],
            events=[],
            alerts=[],
            ui_cfg=None,
            fps=30.0,
        )

        # Should return valid image
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, img.shape)

    def test_draw_overlay_with_gps_carried(self):
        """Test draw_overlay handles GPS-carried tracks correctly."""
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
        """Test continuity debug overlay is disabled by default."""
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

        # Draw with default ui_cfg (show_continuity_debug=False by default)
        result = draw_overlay(
            frame=frame,
            tracks=[track],
            decisions=[decision],
            events=[],
            alerts=[],
            ui_cfg=None,
            fps=30.0,
        )

        # Should succeed without errors
        self.assertIsNotNone(result)

    def test_draw_overlay_continuity_debug_enabled(self):
        """Test continuity debug overlay shows GPS diagnostics when enabled."""
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

        # Enable continuity debug overlay
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

        # Should succeed (debug overlay drawn)
        self.assertIsNotNone(result)

    def test_draw_overlay_mixed_sources(self):
        """Test draw_overlay handles mix of F/G/U sources correctly."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(
            frame_id=1,
            ts=1.0,
            camera_id="cam0",
            size=(640, 480),
            image=img,
        )

        # Three tracks with different sources
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
    """Test Phase 7 maintains backwards compatibility with existing code."""

    def test_overlay_without_id_source_field(self):
        """Test overlay works when decisions lack id_source field."""
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

        # Decision without id_source field
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
        )

        # Should work without errors (defaults to [U])
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
        """Test _identity_label maintains existing functionality."""
        # Old-style decision (no id_source)
        decision = IdentityDecision(
            track_id=1,
            identity_id="alice",
            binding_state="CONFIRMED_WEAK",
            confidence=0.85,
        )

        label, debug, color = _identity_label(decision)

        # Should still contain identity name
        self.assertIn("alice", label)
        # Should still contain confidence
        self.assertIn("0.85", label)
        # Should return valid color
        self.assertIsInstance(color, tuple)
        self.assertEqual(len(color), 3)


if __name__ == "__main__":
    import unittest
    unittest.main()
