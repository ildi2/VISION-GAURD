
import time
from typing import Dict, List
from unittest import TestCase

from core.scheduler import FaceScheduler, SchedulerConfig


class TestPhase6SchedulerPriority(TestCase):

    def setUp(self):
        self.config = SchedulerConfig(
            enabled=True,
            budget_policy="fixed",
            fixed_budget_per_frame=10,
            priority_weight_unknown=50.0,
            priority_weight_pending=80.0,
            priority_weight_confirmed_weak=20.0,
            priority_weight_confirmed_strong=10.0,
            time_decay_rate=0.0,
        )
        self.scheduler = FaceScheduler(config=self.config)

    def test_gps_carried_medium_priority(self):
        track_ids = [1, 2, 3]
        binding_states = {
            1: "UNKNOWN",
            2: "UNKNOWN",
            3: "CONFIRMED_WEAK",
        }
        id_sources = {
            1: "G",
            2: "U",
            3: "F",
        }
        current_ts = time.time()

        for tid in track_ids:
            if tid not in self.scheduler.track_states:
                from core.scheduler import TrackScheduleState
                self.scheduler.track_states[tid] = TrackScheduleState(track_id=tid)

        context = self.scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            actual_fps=30.0,
            id_sources=id_sources,
        )

        priority_scores = self.scheduler._compute_priority_scores(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            id_sources=id_sources,
        )

        self.assertEqual(
            priority_scores[1],
            20.0,
            f"GPS-carried track should get medium priority (20.0), got {priority_scores[1]}"
        )

        self.assertEqual(
            priority_scores[2],
            50.0,
            f"Truly unknown track should get high priority (50.0), got {priority_scores[2]}"
        )

        self.assertEqual(
            priority_scores[3],
            20.0,
            f"Confirmed weak track should get medium priority (20.0), got {priority_scores[3]}"
        )

    def test_truly_unknown_high_priority(self):
        track_ids = [1]
        binding_states = {1: "UNKNOWN"}
        id_sources = {1: "U"}
        current_ts = time.time()

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
        track_ids = [1, 2]
        binding_states = {
            1: "UNKNOWN",
            2: "CONFIRMED_WEAK",
        }
        current_ts = time.time()

        context = self.scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            actual_fps=30.0,
            id_sources=None,
        )

        priority_scores = self.scheduler._compute_priority_scores(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            id_sources=None,
        )

        self.assertEqual(
            priority_scores[1],
            50.0,
            f"UNKNOWN track without id_source should default to high priority (50.0), got {priority_scores[1]}"
        )

    def test_mixed_id_sources(self):
        track_ids = [1, 2, 3, 4, 5]
        binding_states = {
            1: "UNKNOWN",
            2: "UNKNOWN",
            3: "PENDING",
            4: "CONFIRMED_WEAK",
            5: "CONFIRMED_STRONG",
        }
        id_sources = {
            1: "G",
            2: "U",
            3: "F",
            4: "F",
            5: "F",
        }
        current_ts = time.time()

        from core.scheduler import TrackScheduleState
        for tid in track_ids:
            self.scheduler.track_states[tid] = TrackScheduleState(track_id=tid)

        priority_scores = self.scheduler._compute_priority_scores(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            id_sources=id_sources,
        )

        self.assertEqual(priority_scores[1], 20.0, "GPS-carried → 20.0")
        self.assertEqual(priority_scores[2], 50.0, "Truly unknown → 50.0")
        self.assertEqual(priority_scores[3], 80.0, "Pending → 80.0")
        self.assertEqual(priority_scores[4], 20.0, "Confirmed weak → 20.0")
        self.assertEqual(priority_scores[5], 10.0, "Confirmed strong → 10.0")

        self.assertGreater(priority_scores[3], priority_scores[2], "PENDING > UNKNOWN")
        self.assertGreater(priority_scores[2], priority_scores[1], "UNKNOWN > GPS-carried")
        self.assertEqual(priority_scores[1], priority_scores[4], "GPS-carried == CONFIRMED_WEAK")
        self.assertGreater(priority_scores[1], priority_scores[5], "GPS-carried > CONFIRMED_STRONG")

    def test_schedule_selection_gps_vs_unknown(self):
        track_ids = [1, 2, 3]
        binding_states = {
            1: "UNKNOWN",
            2: "UNKNOWN",
            3: "CONFIRMED_STRONG",
        }
        id_sources = {
            1: "G",
            2: "U",
            3: "F",
        }
        current_ts = time.time()

        self.scheduler.config.fixed_budget_per_frame = 1

        context = self.scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            actual_fps=30.0,
            id_sources=id_sources,
        )

        self.assertIn(2, context.scheduled_track_ids, "Truly unknown track should be scheduled first")
        self.assertEqual(len(context.scheduled_track_ids), 1, "Should schedule exactly 1 track")

    def test_schedule_selection_gps_when_no_unknown(self):
        track_ids = [1, 2]
        binding_states = {
            1: "UNKNOWN",
            2: "CONFIRMED_STRONG",
        }
        id_sources = {
            1: "G",
            2: "F",
        }
        current_ts = time.time()

        self.scheduler.config.fixed_budget_per_frame = 1

        context = self.scheduler.compute_schedule(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            actual_fps=30.0,
            id_sources=id_sources,
        )

        self.assertIn(1, context.scheduled_track_ids, "GPS-carried track should be scheduled when no higher priority exists")

    def test_default_id_source_when_track_missing(self):
        track_ids = [1, 2]
        binding_states = {
            1: "UNKNOWN",
            2: "UNKNOWN",
        }
        id_sources = {
            1: "G",
        }
        current_ts = time.time()

        from core.scheduler import TrackScheduleState
        for tid in track_ids:
            self.scheduler.track_states[tid] = TrackScheduleState(track_id=tid)

        priority_scores = self.scheduler._compute_priority_scores(
            track_ids=track_ids,
            binding_states=binding_states,
            current_ts=current_ts,
            id_sources=id_sources,
        )

        self.assertEqual(priority_scores[1], 20.0, "Track 1 should get GPS priority")

        self.assertEqual(priority_scores[2], 50.0, "Track 2 should default to high priority (U)")


class TestPhase6MainLoopIntegration(TestCase):

    def test_id_sources_extraction_from_decisions(self):
        from schemas.identity_decision import IdentityDecision

        decisions = [
            IdentityDecision(
                track_id=1,
                identity_id="alice",
                binding_state="CONFIRMED_WEAK",
                confidence=0.85,
                id_source="F",
            ),
            IdentityDecision(
                track_id=2,
                identity_id="alice",
                binding_state="UNKNOWN",
                confidence=0.92,
                id_source="G",
            ),
            IdentityDecision(
                track_id=3,
                identity_id=None,
                binding_state="UNKNOWN",
                confidence=0.0,
                id_source="U",
            ),
        ]

        id_sources = {}
        for dec in decisions:
            source = "U"
            if hasattr(dec, 'id_source') and dec.id_source is not None:
                source = dec.id_source
            elif dec.extra and 'id_source' in dec.extra:
                source = dec.extra['id_source']
            id_sources[dec.track_id] = source

        self.assertEqual(id_sources[1], "F", "Track 1 should be Face-assigned")
        self.assertEqual(id_sources[2], "G", "Track 2 should be GPS-carried")
        self.assertEqual(id_sources[3], "U", "Track 3 should be Unknown")

    def test_id_sources_extraction_from_extra_dict(self):
        from schemas.identity_decision import IdentityDecision

        decisions = [
            IdentityDecision(
                track_id=1,
                identity_id="alice",
                binding_state="UNKNOWN",
                confidence=0.92,
                extra={'id_source': 'G'},
            ),
            IdentityDecision(
                track_id=2,
                identity_id=None,
                binding_state="UNKNOWN",
                confidence=0.0,
                extra={'other_field': 'value'},
            ),
        ]

        id_sources = {}
        for dec in decisions:
            source = "U"
            if hasattr(dec, 'id_source') and dec.id_source is not None:
                source = dec.id_source
            elif dec.extra and 'id_source' in dec.extra:
                source = dec.extra['id_source']
            id_sources[dec.track_id] = source

        self.assertEqual(id_sources[1], "G", "Track 1 should extract from extra dict")
        self.assertEqual(id_sources[2], "U", "Track 2 should default to U when missing")

    def test_id_sources_default_when_no_field(self):
        from schemas.identity_decision import IdentityDecision

        decision = IdentityDecision(
            track_id=1,
            identity_id=None,
            binding_state="UNKNOWN",
            confidence=0.0,
        )

        source = "U"
        if hasattr(decision, 'id_source') and decision.id_source is not None:
            source = decision.id_source
        elif decision.extra and 'id_source' in decision.extra:
            source = decision.extra['id_source']

        self.assertEqual(source, "U", "Missing id_source should default to U")


if __name__ == "__main__":
    import unittest
    unittest.main()
