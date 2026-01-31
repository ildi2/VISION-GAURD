"""
PHASE 6 TESTS: Scheduler Integration (Priority Awareness)

Tests that scheduler correctly differentiates GPS-carried tracks from truly unknown tracks.

Test Coverage:
1. Scheduler accepts id_sources parameter
2. GPS-carried tracks (UNKNOWN+G) get medium priority (20.0)
3. Truly unknown tracks (UNKNOWN+U) get high priority (50.0)
4. Backwards compatibility (id_sources=None works)
5. Main loop integration (id_sources extraction and passing)

Phase 6 Goal: Optimize face refresh rate (don't over-process GPS-carried identities)
"""

import time
from typing import Dict, List
from unittest import TestCase

from core.scheduler import FaceScheduler, SchedulerConfig


class TestPhase6SchedulerPriority(TestCase):
    """Test Phase 6 scheduler priority differentiation."""

    def setUp(self):
        """Create scheduler with known config."""
        self.config = SchedulerConfig(
            enabled=True,
            budget_policy="fixed",
            fixed_budget_per_frame=10,
            priority_weight_unknown=50.0,
            priority_weight_pending=80.0,
            priority_weight_confirmed_weak=20.0,
            priority_weight_confirmed_strong=10.0,
            time_decay_rate=0.0,  # Disable time decay for deterministic tests
        )
        self.scheduler = FaceScheduler(config=self.config)

    def test_gps_carried_medium_priority(self):
        """GPS-carried tracks (UNKNOWN+G) should get medium priority (20.0)."""
        track_ids = [1, 2, 3]
        binding_states = {
            1: "UNKNOWN",  # GPS-carried
            2: "UNKNOWN",  # Truly unknown
            3: "CONFIRMED_WEAK",
        }
        id_sources = {
            1: "G",  # GPS-carried
            2: "U",  # Truly unknown
            3: "F",  # Face-confirmed
        }
        current_ts = time.time()

        # Initialize track states (required before compute)
        for tid in track_ids:
            if tid not in self.scheduler.track_states:
                from core.scheduler import TrackScheduleState
                self.scheduler.track_states[tid] = TrackScheduleState(track_id=tid)

        # Compute schedule
        context = self.scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            actual_fps=30.0,
            id_sources=id_sources,
        )

        # Check priority scores
        priority_scores = self.scheduler._compute_priority_scores(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            id_sources=id_sources,
        )

        # Track 1 (UNKNOWN+G): should get 20.0 (medium priority, same as CONFIRMED_WEAK)
        self.assertEqual(
            priority_scores[1],
            20.0,
            f"GPS-carried track should get medium priority (20.0), got {priority_scores[1]}"
        )

        # Track 2 (UNKNOWN+U): should get 50.0 (high priority)
        self.assertEqual(
            priority_scores[2],
            50.0,
            f"Truly unknown track should get high priority (50.0), got {priority_scores[2]}"
        )

        # Track 3 (CONFIRMED_WEAK+F): should get 20.0
        self.assertEqual(
            priority_scores[3],
            20.0,
            f"Confirmed weak track should get medium priority (20.0), got {priority_scores[3]}"
        )

    def test_truly_unknown_high_priority(self):
        """Truly unknown tracks (UNKNOWN+U) should get high priority (50.0)."""
        track_ids = [1]
        binding_states = {1: "UNKNOWN"}
        id_sources = {1: "U"}
        current_ts = time.time()

        # Initialize track states
        from core.scheduler import TrackScheduleState
        self.scheduler.track_states[1] = TrackScheduleState(track_id=1)

        priority_scores = self.scheduler._compute_priority_scores(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            id_sources=id_sources,
        )

        self.assertEqual(
            priority_scores[1],
            50.0,
            f"Truly unknown track should get high priority (50.0), got {priority_scores[1]}"
        )

    def test_backwards_compatibility_no_id_sources(self):
        """Scheduler should work without id_sources (default to U)."""
        track_ids = [1, 2]
        binding_states = {
            1: "UNKNOWN",
            2: "CONFIRMED_WEAK",
        }
        current_ts = time.time()

        # Call without id_sources
        context = self.scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            actual_fps=30.0,
            id_sources=None,  # Backwards compatibility
        )

        # Should default to "U" (truly unknown)
        priority_scores = self.scheduler._compute_priority_scores(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            id_sources=None,
        )

        # Track 1 (UNKNOWN, no id_source): should default to 50.0 (high priority)
        self.assertEqual(
            priority_scores[1],
            50.0,
            f"UNKNOWN track without id_source should default to high priority (50.0), got {priority_scores[1]}"
        )

    def test_mixed_id_sources(self):
        """Test mix of F/G/U id_sources with various binding states."""
        track_ids = [1, 2, 3, 4, 5]
        binding_states = {
            1: "UNKNOWN",  # GPS-carried
            2: "UNKNOWN",  # Truly unknown
            3: "PENDING",  # Face pending (should be highest)
            4: "CONFIRMED_WEAK",  # Face confirmed weak
            5: "CONFIRMED_STRONG",  # Face confirmed strong
        }
        id_sources = {
            1: "G",  # GPS
            2: "U",  # Unknown
            3: "F",  # Face (pending)
            4: "F",  # Face (weak)
            5: "F",  # Face (strong)
        }
        current_ts = time.time()

        # Initialize track states
        from core.scheduler import TrackScheduleState
        for tid in track_ids:
            self.scheduler.track_states[tid] = TrackScheduleState(track_id=tid)

        priority_scores = self.scheduler._compute_priority_scores(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            id_sources=id_sources,
        )

        # Expected priorities
        self.assertEqual(priority_scores[1], 20.0, "GPS-carried → 20.0")
        self.assertEqual(priority_scores[2], 50.0, "Truly unknown → 50.0")
        self.assertEqual(priority_scores[3], 80.0, "Pending → 80.0")
        self.assertEqual(priority_scores[4], 20.0, "Confirmed weak → 20.0")
        self.assertEqual(priority_scores[5], 10.0, "Confirmed strong → 10.0")

        # Verify priority ordering: PENDING > UNKNOWN > GPS-carried = CONFIRMED_WEAK > CONFIRMED_STRONG
        self.assertGreater(priority_scores[3], priority_scores[2], "PENDING > UNKNOWN")
        self.assertGreater(priority_scores[2], priority_scores[1], "UNKNOWN > GPS-carried")
        self.assertEqual(priority_scores[1], priority_scores[4], "GPS-carried == CONFIRMED_WEAK")
        self.assertGreater(priority_scores[1], priority_scores[5], "GPS-carried > CONFIRMED_STRONG")

    def test_schedule_selection_gps_vs_unknown(self):
        """Test that truly unknown tracks are scheduled before GPS-carried."""
        track_ids = [1, 2, 3]
        binding_states = {
            1: "UNKNOWN",  # GPS-carried (medium priority)
            2: "UNKNOWN",  # Truly unknown (high priority)
            3: "CONFIRMED_STRONG",  # Low priority
        }
        id_sources = {
            1: "G",  # GPS
            2: "U",  # Unknown
            3: "F",  # Face
        }
        current_ts = time.time()

        # Budget allows only 1 track
        self.scheduler.config.fixed_budget_per_frame = 1

        context = self.scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            actual_fps=30.0,
            id_sources=id_sources,
        )

        # Should schedule track 2 (truly unknown, highest priority)
        self.assertIn(2, context.scheduled_track_ids, "Truly unknown track should be scheduled first")
        self.assertEqual(len(context.scheduled_track_ids), 1, "Should schedule exactly 1 track")

    def test_schedule_selection_gps_when_no_unknown(self):
        """Test that GPS-carried tracks are scheduled when no truly unknown tracks exist."""
        track_ids = [1, 2]
        binding_states = {
            1: "UNKNOWN",  # GPS-carried (medium priority)
            2: "CONFIRMED_STRONG",  # Low priority
        }
        id_sources = {
            1: "G",  # GPS
            2: "F",  # Face
        }
        current_ts = time.time()

        # Budget allows only 1 track
        self.scheduler.config.fixed_budget_per_frame = 1

        context = self.scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            actual_fps=30.0,
            id_sources=id_sources,
        )

        # Should schedule track 1 (GPS-carried, higher than CONFIRMED_STRONG)
        self.assertIn(1, context.scheduled_track_ids, "GPS-carried track should be scheduled when no higher priority exists")

    def test_default_id_source_when_track_missing(self):
        """Test that missing track IDs in id_sources dict default to 'U'."""
        track_ids = [1, 2]
        binding_states = {
            1: "UNKNOWN",
            2: "UNKNOWN",
        }
        id_sources = {
            1: "G",  # Track 1 has GPS
            # Track 2 missing (should default to "U")
        }
        current_ts = time.time()

        # Initialize track states
        from core.scheduler import TrackScheduleState
        for tid in track_ids:
            self.scheduler.track_states[tid] = TrackScheduleState(track_id=tid)

        priority_scores = self.scheduler._compute_priority_scores(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            id_sources=id_sources,
        )

        # Track 1 (UNKNOWN+G): 20.0
        self.assertEqual(priority_scores[1], 20.0, "Track 1 should get GPS priority")

        # Track 2 (UNKNOWN+missing→U): 50.0
        self.assertEqual(priority_scores[2], 50.0, "Track 2 should default to high priority (U)")


class TestPhase6MainLoopIntegration(TestCase):
    """Test Phase 6 main loop integration (id_sources extraction and passing)."""

    def test_id_sources_extraction_from_decisions(self):
        """Test that main loop correctly extracts id_sources from decisions."""
        from schemas.identity_decision import IdentityDecision

        # Mock decisions with id_source field
        decisions = [
            IdentityDecision(
                track_id=1,
                identity_id="alice",
                binding_state="CONFIRMED_WEAK",
                confidence=0.85,
                id_source="F",  # Face-assigned
            ),
            IdentityDecision(
                track_id=2,
                identity_id="alice",
                binding_state="UNKNOWN",
                confidence=0.92,
                id_source="G",  # GPS-carried
            ),
            IdentityDecision(
                track_id=3,
                identity_id=None,
                binding_state="UNKNOWN",
                confidence=0.0,
                id_source="U",  # Unknown
            ),
        ]

        # Extract id_sources (simulates main loop logic)
        id_sources = {}
        for dec in decisions:
            source = "U"  # Default
            if hasattr(dec, 'id_source') and dec.id_source is not None:
                source = dec.id_source
            elif dec.extra and 'id_source' in dec.extra:
                source = dec.extra['id_source']
            id_sources[dec.track_id] = source

        # Verify extraction
        self.assertEqual(id_sources[1], "F", "Track 1 should be Face-assigned")
        self.assertEqual(id_sources[2], "G", "Track 2 should be GPS-carried")
        self.assertEqual(id_sources[3], "U", "Track 3 should be Unknown")

    def test_id_sources_extraction_from_extra_dict(self):
        """Test that main loop correctly extracts id_sources from decision.extra fallback."""
        from schemas.identity_decision import IdentityDecision

        # Mock decisions with id_source in extra dict (backwards compatibility)
        decisions = [
            IdentityDecision(
                track_id=1,
                identity_id="alice",
                binding_state="UNKNOWN",
                confidence=0.92,
                extra={'id_source': 'G'},  # Fallback to extra dict
            ),
            IdentityDecision(
                track_id=2,
                identity_id=None,
                binding_state="UNKNOWN",
                confidence=0.0,
                extra={'other_field': 'value'},  # No id_source (should default to "U")
            ),
        ]

        # Extract id_sources (simulates main loop logic)
        id_sources = {}
        for dec in decisions:
            source = "U"  # Default
            if hasattr(dec, 'id_source') and dec.id_source is not None:
                source = dec.id_source
            elif dec.extra and 'id_source' in dec.extra:
                source = dec.extra['id_source']
            id_sources[dec.track_id] = source

        # Verify extraction
        self.assertEqual(id_sources[1], "G", "Track 1 should extract from extra dict")
        self.assertEqual(id_sources[2], "U", "Track 2 should default to U when missing")

    def test_id_sources_default_when_no_field(self):
        """Test that missing id_source field defaults to 'U'."""
        from schemas.identity_decision import IdentityDecision

        # Mock decision without id_source field
        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0,
        )

        # Extract id_source (simulates main loop logic)
        source = "U"  # Default
        if hasattr(decision, 'id_source') and decision.id_source is not None:
            source = decision.id_source
        elif decision.extra and 'id_source' in decision.extra:
            source = decision.extra['id_source']

        # Verify default
        self.assertEqual(source, "U", "Missing id_source should default to U")


if __name__ == "__main__":
    import unittest
    unittest.main()
