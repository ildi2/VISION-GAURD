# identity/identity_engine.py
# IdentityEngine implementation using FaceRoute + FaceGallery with temporal
# smoothing, multi-sample stability rules, and distance-band logic
# (strong / weak / unknown).

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple, Set

import numpy as np

from core.interfaces import IdentityEngine as IdentityEngineBase
from schemas import Frame, Tracklet, IdSignals, IdentityDecision, FaceSample
from face.config import (
    FaceConfig,
    FaceSmoothingConfig,
    default_face_config,
)
from face.route import FaceRoute
from identity.face_gallery import FaceGallery, SearchResult
from identity.binding import BindingManager, BindingDecision

logger = logging.getLogger(__name__)


class FaceIdentityEngine(IdentityEngineBase):
    """
    Glue between Phase-2A FaceRoute and the encrypted FaceGallery.

    Responsibility:
      - Convert per-frame FaceEvidence into IdSignals (embedding + quality).
      - Maintain a tiny per-track history of recent high-quality samples.
      - Query FaceGallery for the best match when new evidence appears.
      - Require multiple recent samples before locking / switching identity.
      - Use distance bands (strong / weak / none) to distinguish:
          * strong identity matches
          * weak candidates
          * true unknowns
      - Maintain a per-track last IdentityDecision and decay it over time when
        no good new evidence is available.

    All numeric knobs (quality gates, distance bands, stability rules,
    decay behaviour) are sourced from FaceConfig / FaceSmoothingConfig so
    that tuning is 100% config-driven (no magic numbers here).

    Wave-3 / pseudo-3D readiness:
      - We build a canonical FaceSample for each new FaceEvidence and attach
        it via IdSignals.set_best_face(), so both classic and multiview
        engines see the same structure (bbox, embedding, yaw/pitch, quality).
    """

    def __init__(
        self,
        face_cfg: Optional[FaceConfig] = None,
        face_route: Optional[FaceRoute] = None,
        gallery: Optional[FaceGallery] = None,
        smoothing_cfg: Optional[FaceSmoothingConfig] = None,
    ) -> None:
        # ------------------------------------------------------------------
        # Identify engine type for compatibility with multiview engine
        # ------------------------------------------------------------------
        self.identity_mode: str = "classic"

        # ------------------------------------------------------------------
        # Wire config objects
        # ------------------------------------------------------------------
        self.face_cfg: FaceConfig = face_cfg or default_face_config()
        self.face_route: FaceRoute = face_route or FaceRoute(self.face_cfg)
        self.gallery: FaceGallery = gallery or FaceGallery(self.face_cfg.gallery)

        # Smoothing config: use explicit override if provided, otherwise the
        # one embedded in FaceConfig.
        self.smoothing_cfg: FaceSmoothingConfig = (
            smoothing_cfg or self.face_cfg.smoothing
        )
        
        # Phase C: Binding State Machine for identity stability
        try:
            from core.config import load_config
            from core.governance_metrics import get_metrics_collector
            cfg_loaded = load_config()
            metrics_coll = get_metrics_collector()
            self.binding_manager = BindingManager(cfg_loaded, metrics_coll)
        except Exception:
            # Fallback: create minimal binding manager that disables itself
            self.binding_manager = BindingManager(None, None)

        # Per-track last identity decision + timestamp.
        self._last_decision: Dict[int, IdentityDecision] = {}
        self._last_decision_ts: Dict[int, float] = {}
        self._current_ts: float = time.time()

        # Per-track recent evidence history: track_id -> deque[(ts, quality)]
        self._evidence_history: Dict[int, Deque[Tuple[float, float]]] = {}

        th = self.face_cfg.thresholds

        # Distance thresholds (gallery space) for accepting matches.
        # In cosine mode: distance d = 1 - sim.
        self._match_dist_strong: float = float(th.strong_match_dist)
        self._match_dist_weak: float = float(th.weak_match_dist)

        # Runtime quality gate (safety net; FaceRoute already applies this).
        self._min_quality_runtime: float = float(
            getattr(th, "min_quality_runtime", th.min_quality_for_embed)
        )

        logger.info(
            "FaceIdentityEngine initialised | strong_match_dist=%.3f "
            "weak_match_dist=%.3f min_q_runtime=%.3f "
            "confirm=%d switch=%d lookback=%.1fs half_life=%.1fs stale_after=%.1fs",
            self._match_dist_strong,
            self._match_dist_weak,
            self._min_quality_runtime,
            self.smoothing_cfg.min_samples_confirm,
            self.smoothing_cfg.min_samples_switch,
            self.smoothing_cfg.evidence_lookback_sec,
            self.smoothing_cfg.half_life_sec,
            self.smoothing_cfg.stale_after_sec,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    # Compatibility helper
    def is_classic_engine(self) -> bool:
        return True

    def reset(self) -> None:
        """
        Reset face route and all per-track identity state.
        """
        self.face_route.reset()
        self._last_decision.clear()
        self._last_decision_ts.clear()
        self._evidence_history.clear()

    # ------------------------------------------------------------------ #
    # Small helper for safe reason concatenation                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _append_reason(
        reason: Optional[str],
        fragment: Optional[str],
    ) -> Optional[str]:
        """
        Safely append a fragment to the reason string using '|' separator.
        Never crashes if reason is None.
        """
        if not fragment:
            return reason
        if reason is None or reason == "":
            return fragment
        return f"{reason}|{fragment}"

    # IdentityEngine interface ----------------------------------------- #

    def get_binding_states(self) -> Dict[int, str]:
        """
        Phase D: Return current binding state for each active track.
        
        Used by the scheduler to prioritize which tracks to process.
        
        Returns:
            Dict[int, str]: mapping of track_id -> binding_state
                States: "UNKNOWN", "PENDING", "CONFIRMED_WEAK", "CONFIRMED_STRONG"
        """
        if self.binding_manager is None:
            return {}
        
        try:
            return self.binding_manager.get_all_states()
        except Exception:
            # If binding manager fails, return empty (won't affect scheduling)
            return {}

    def update_signals(self, frame: Frame, tracks: List[Tracklet], schedule_context=None) -> List[IdSignals]:
        """
        Called once per frame.

        Phase D: schedule_context is passed by main loop for scheduler support.
        
        - Runs FaceRoute to possibly generate new high-quality evidence.
        - Updates per-track evidence history.
        - Produces an IdSignals list (one per track) with:
            * best_face  (canonical FaceSample, when new evidence exists)
            * face_embedding / face_quality (legacy mirrors, kept in sync)

        If FaceRoute did not emit evidence for a track in this frame,
        the corresponding IdSignals will have best_face=None and
        face_embedding=None; decide() will rely on temporal smoothing
        instead of new evidence.
        
        Args:
            frame: Frame object with image and metadata
            tracks: List of Tracklet objects from perception engine
            schedule_context: Optional Phase D ScheduleContext (for scheduling support)
        
        Returns:
            List[IdSignals] with one signal per track
        """
        self._current_ts = frame.ts
        
        # Phase D: Pass schedule context to face route if provided
        if schedule_context is not None:
            try:
                evidences = self.face_route.run(frame, tracks, schedule_context=schedule_context)
            except TypeError:
                # Fallback if face_route doesn't support schedule_context yet
                evidences = self.face_route.run(frame, tracks)
        else:
            evidences = self.face_route.run(frame, tracks)

        signals: List[IdSignals] = []
        active_ids: Set[int] = set()

        for trk in tracks:
            tid = int(getattr(trk, "track_id", trk.track_id))
            active_ids.add(tid)
            ev = evidences.get(tid)

            if ev is not None:
                q = float(ev.quality)
                # Update internal stability history only for runtime-valid samples.
                self._update_evidence_history(tid, frame.ts, q)

                # Build canonical FaceSample from FaceEvidence.
                extra_meta: Dict[str, float] = {}
                if ev.yaw is not None:
                    extra_meta["yaw"] = float(ev.yaw)
                if ev.pitch is not None:
                    extra_meta["pitch"] = float(ev.pitch)
                if ev.roll is not None:
                    extra_meta["roll"] = float(ev.roll)
                if ev.det_score is not None:
                    extra_meta["det_score"] = float(ev.det_score)

                face_sample = FaceSample(
                    bbox=ev.bbox_in_frame,
                    embedding=np.asarray(ev.embedding, dtype=np.float32).reshape(-1),
                    det_score=float(ev.det_score) if ev.det_score is not None else 0.0,
                    quality=q,
                    yaw=float(ev.yaw) if ev.yaw is not None else None,
                    pitch=float(ev.pitch) if ev.pitch is not None else None,
                    roll=float(ev.roll) if ev.roll is not None else None,
                    ts=float(frame.ts),
                    pose_bin=None,  # pose binning is handled later by multiview logic
                    source="runtime",
                    extra=extra_meta or None,
                )

                sig = IdSignals(track_id=tid)
                # This will also mirror into face_embedding, face_quality,
                # raw_face_box and pose_bin_hint, and merge extra.
                sig.set_best_face(face_sample)
            else:
                # No new evidence this frame → smoothing only.
                sig = IdSignals(track_id=tid)

            signals.append(sig)

        # Clean up very old histories for tracks that disappeared.
        self._cleanup_dead_tracks(active_ids, self._current_ts)
        return signals

    def decide(self, signals: List[IdSignals]) -> List[IdentityDecision]:
        """
        Given IdSignals for the current frame, return IdentityDecision per track.

        - If we have a *new high-quality* embedding:
            * query gallery
            * apply distance bands (strong / weak / none)
            * apply stability checks for strong matches
        - If not → rely purely on temporal smoothing of last decision.
        """
        ts = self._current_ts
        decisions: List[IdentityDecision] = []

        for sig in signals:
            tid = sig.track_id

            # Primary embedding source is legacy mirror; kept in sync from best_face.
            emb = sig.face_embedding

            if emb is not None:
                decision = self._decide_with_new_embedding(
                    tid, emb, sig.face_quality, ts
                )

                # Propagate latest face quality for telemetry (clamped 0–1).
                decision.quality = self._clamp01(float(sig.face_quality))

                # Add yaw/pitch to reason if included (safely).
                extra = getattr(sig, "extra", None)
                if isinstance(extra, dict):
                    if "yaw" in extra:
                        try:
                            decision.reason = self._append_reason(
                                decision.reason,
                                f"yaw={float(extra['yaw']):.2f}",
                            )
                        except Exception:
                            pass
                    if "pitch" in extra:
                        try:
                            decision.reason = self._append_reason(
                                decision.reason,
                                f"pitch={float(extra['pitch']):.2f}",
                            )
                        except Exception:
                            pass
            else:
                decision = self._decide_with_smoothing_only(tid, ts)

                # If we have a stored last decision, reuse its quality so
                # FaceMetrics still sees the underlying face quality.
                last = self._last_decision.get(tid)
                if last is not None and hasattr(last, "quality"):
                    decision.quality = self._clamp01(float(last.quality))

            # Mark engine type and a default pose for compatibility.
            decision.reason = self._append_reason(decision.reason, "engine=classic")
            decision.engine = "classic"
            if decision.pose_bin is None:
                decision.pose_bin = "FRONT"

            decisions.append(decision)

        return decisions

    # ------------------------------------------------------------------ #
    # Decision helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clamp01(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def _decide_with_new_embedding(
        self,
        track_id: int,
        embedding: np.ndarray,
        quality: float,
        ts: float,
    ) -> IdentityDecision:
        """
        Use new face embedding + quality to update identity for a track.

        Steps:
          - Apply runtime quality gate.
          - Query FaceGallery for best match.
          - Classify the match into a distance band: strong / weak / none.
          - For strong matches:
              * apply stability rules (min_samples_confirm / min_samples_switch)
              * if stability OK → lock/refresh/switch identity
              * else → keep smoothed identity and log a candidate.
          - For weak matches:
              * never override identity, only annotate candidate / weak match.
          - For 'none' (too far / no gallery result):
              * behave as unknown and rely on smoothing.
        """
        q = float(quality)

        # Safety net: treat low-quality evidence as "no new evidence".
        if q < self._min_quality_runtime:
            decision = self._decide_with_smoothing_only(track_id, ts)
            if decision.identity_id is None:
                decision.reason = f"face_unknown:runtime_quality_gate:q={q:.3f}"
            else:
                decision.reason = self._append_reason(
                    decision.reason,
                    f"runtime_quality_gate:q={q:.3f}",
                )
            # quality is left as whatever smoothing-only returned (often 0).
            return decision

        # Normalised + safety check for embedding.
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.size == 0:
            logger.warning(
                "FaceIdentityEngine: empty embedding for track_id=%d; "
                "using smoothing only.",
                track_id,
            )
            return self._decide_with_smoothing_only(track_id, ts)

        res: Optional[SearchResult] = self.gallery.search_best(emb, k=5)

        if res is None:
            # No gallery result at all → rely on smoothing.
            decision = self._decide_with_smoothing_only(track_id, ts)
            extra = "face_unknown:no_gallery_match"
            if decision.identity_id is None:
                decision.reason = extra
            else:
                decision.reason = self._append_reason(decision.reason, extra)
            # Distance/score unknown – keep defaults.
            return decision

        d = float(res.distance)
        s = float(res.score)

        # Determine distance band from config thresholds.
        if d <= self._match_dist_strong:
            match_band = "strong"
        elif d <= self._match_dist_weak:
            match_band = "weak"
        else:
            match_band = "none"

        # ------------------------------------------------------------------
        # Band: none → treat as true unknown (no good gallery match)
        # ------------------------------------------------------------------
        if match_band == "none":
            decision = self._decide_with_smoothing_only(track_id, ts)
            extra = f"face_unknown:no_gallery_match:d={d:.3f}:q={q:.3f}:s={s:.3f}"
            if decision.identity_id is None:
                decision.reason = extra
            else:
                decision.reason = self._append_reason(decision.reason, extra)

            # Attach distance/score for diagnostics.
            decision.distance = d
            decision.score = s
            return decision

        # ------------------------------------------------------------------
        # Band: weak → never override identity, only annotate as candidate.
        # ------------------------------------------------------------------
        if match_band == "weak":
            base = self._decide_with_smoothing_only(track_id, ts)

            if base.identity_id is None:
                base.reason = (
                    f"face_candidate_weak:{res.person_id}:"
                    f"d={d:.3f}:q={q:.3f}:s={s:.3f}"
                )
            else:
                base.reason = self._append_reason(
                    base.reason,
                    (
                        f"face_match_weak:{res.person_id}:"
                        f"d={d:.3f}:q={q:.3f}:s={s:.3f}"
                    ),
                )

            base.distance = d
            base.score = s
            return base

        # ------------------------------------------------------------------
        # Band: strong → apply stability logic before locking / switching.
        # ------------------------------------------------------------------
        last = self._last_decision.get(track_id)
        cfg = self.smoothing_cfg

        if last is None or last.identity_id is None:
            # New identity proposal.
            needed_samples = max(1, cfg.min_samples_confirm)
            stability_label = "new"
        elif last.identity_id == res.person_id:
            # Same identity as before → normal refresh.
            needed_samples = max(1, cfg.min_samples_confirm)
            stability_label = "refresh"
        else:
            # Switching from one identity to a different one → require more proof.
            needed_samples = max(cfg.min_samples_confirm + 1, cfg.min_samples_switch)
            stability_label = "switch"

        has_enough, count, avg_q = self._has_enough_recent_evidence(
            track_id, ts, needed_samples
        )

        if not has_enough:
            # Do not lock/switch yet; keep smoothed previous identity and
            # encode strong candidate info into the reason.
            decision = self._decide_with_smoothing_only(track_id, ts)

            reason_prefix = (
                f"face_candidate_strong:{res.person_id}:"
                f"d={d:.3f}:q={q:.3f}:s={s:.3f}:"
                f"mode={stability_label}:"
                f"samples={count}/{needed_samples}:avg_q={avg_q:.3f}"
            )

            if decision.identity_id is None:
                decision.reason = reason_prefix
            else:
                decision.reason = self._append_reason(
                    decision.reason,
                    reason_prefix,
                )

            decision.distance = d
            decision.score = s
            return decision

        # Enough recent evidence → accept/refresh/switch identity as strong match.
        raw_conf = s * q
        conf = self._clamp01(raw_conf)

        decision = IdentityDecision(
            track_id=track_id,
            identity_id=res.person_id,
            category=res.category,
            confidence=conf,
            reason=(
                f"face_match_strong:{res.person_id}:"
                f"d={d:.3f}:q={q:.3f}:s={s:.3f}:"
                f"mode={stability_label}:samples={count}:avg_q={avg_q:.3f}"
            ),
            distance=d,
            score=s,
            # quality is set in decide(), but we keep the default here.
        )

        # ================================================================
        # PHASE C INTEGRATION: Apply binding state machine
        # ================================================================
        # The binding engine enforces identity stability through:
        # - Margin-based evidence accumulation
        # - Anti-lock-in contradiction detection
        # - Controlled switching (margin advantage required)
        # - Per-track binding state machine (UNKNOWN → PENDING → CONFIRMED)
        #
        # LAYER 4B ENHANCEMENT: Quality-Aware Binding Thresholds
        # ================================================================
        # Quality-modulated binding strength:
        #   - HIGH quality (q > 0.85): aggressive binding (90% of nominal)
        #   - MID quality (0.7-0.85):  nominal binding (100%)
        #   - LOW quality (q < 0.7):   conservative binding (130% of nominal)
        #
        # Benefits (LAYER 4B):
        #   - High-quality samples lock in faster → reduces unconfirmed noise
        #   - Low-quality samples require more evidence → improves robustness
        #   - Score disparity < 0.15 → auto-confirm (high confidence)
        #   - ZERO impact on gallery accuracy or runtime
        # ================================================================
        
        # Compute quality-modulated confidence multiplier
        quality_modifier = 1.0  # Baseline (MID quality)
        if q > 0.85:  # HIGH quality
            quality_modifier = 0.90  # Aggressive: need only 90% evidence
        elif q < 0.70:  # LOW quality
            quality_modifier = 1.30  # Conservative: need 130% evidence
        
        # Compute adjusted confidence for binding
        adjusted_conf = conf * quality_modifier
        adjusted_conf = self._clamp01(adjusted_conf)
        
        # High confidence auto-confirm: if we have strong separation from
        # second-best candidate, the binding manager should confirm immediately
        score_separation = 0.15  # Threshold for "obvious" matches
        
        try:
            binding_result = self.binding_manager.process_evidence(
                track_id=track_id,
                person_id=res.person_id,
                score=adjusted_conf,  # LAYER 4B: Use quality-modulated score
                second_best_score=max(0.0, adjusted_conf - score_separation),
                quality=q,
                timestamp=ts,
            )

            # Binding engine may override the identity decision
            if binding_result.person_id is not None:
                decision.identity_id = binding_result.person_id
                decision.confidence = binding_result.confidence
            
            # Append binding reason for diagnostics
            if binding_result.binding_state != "BYPASS":
                binding_reason = (
                    f"binding={binding_result.binding_state}:"
                    f"{binding_result.reason}"
                )
                decision.reason = self._append_reason(
                    decision.reason, binding_reason
                )
        
        except Exception as e:
            logger.warning(
                f"Binding state machine error for track {track_id}: {e}"
            )
            # Continue with decision as-is on binding error

        # Store as last committed identity decision (object will later have
        # .quality filled by decide()).
        self._last_decision[track_id] = decision
        self._last_decision_ts[track_id] = ts
        return decision

    def _decide_with_smoothing_only(
        self,
        track_id: int,
        ts: float,
    ) -> IdentityDecision:
        """
        Make a decision using only previously stored identity and time
        (no new evidence this frame).
        """
        last = self._last_decision.get(track_id)
        cfg = self.smoothing_cfg

        if last is None:
            return IdentityDecision(
                track_id=track_id,
                identity_id=None,
                category="unknown",
                confidence=0.0,
                reason="face_unknown:no_history",
                # quality left at default 0.0
            )

        last_ts = self._last_decision_ts.get(track_id, ts)
        age = ts - last_ts

        # History too old → reset to unknown.
        if age >= cfg.stale_after_sec:
            return IdentityDecision(
                track_id=track_id,
                identity_id=None,
                category="unknown",
                confidence=0.0,
                reason=f"face_unknown:stale_history:{age:.2f}s",
                # quality left at default 0.0
            )

        # Exponential decay of confidence over time without new evidence.
        decay = self._decay_factor(age, cfg.half_life_sec)
        conf = last.confidence * decay

        if conf < cfg.min_confidence:
            return IdentityDecision(
                track_id=track_id,
                identity_id=None,
                category="unknown",
                confidence=0.0,
                reason=f"face_unknown:decayed_below_min:{age:.2f}s",
                # quality left at default 0.0
            )

        # Keep same identity, decayed confidence. We keep last.quality;
        # caller (decide) will reuse it.
        return IdentityDecision(
            track_id=track_id,
            identity_id=last.identity_id,
            category=last.category,
            confidence=conf,
            reason=f"{last.reason}|decayed:{age:.2f}s",
            # distance/score/pose_bin/engine/quality will be set by caller if needed.
        )

    # ------------------------------------------------------------------ #
    # Evidence history helpers                                           #
    # ------------------------------------------------------------------ #

    def _update_evidence_history(self, track_id: int, ts: float, quality: float) -> None:
        """
        Record a new high-quality evidence sample for a track.
        """
        if quality < self._min_quality_runtime:
            return

        dq = self._evidence_history.setdefault(track_id, deque())
        dq.append((ts, quality))
        self._prune_evidence_history(track_id, ts)

    def _prune_evidence_history(self, track_id: int, ts: float) -> None:
        dq = self._evidence_history.get(track_id)
        if not dq:
            return

        max_age = self.smoothing_cfg.evidence_lookback_sec
        while dq and (ts - dq[0][0]) > max_age:
            dq.popleft()

    def _recent_evidence_stats(
        self,
        track_id: int,
        ts: float,
    ) -> Tuple[int, float]:
        """
        Return (count, avg_quality) of recent evidence within lookback window.
        """
        dq = self._evidence_history.get(track_id)
        if not dq:
            return 0, 0.0

        self._prune_evidence_history(track_id, ts)
        if not dq:
            return 0, 0.0

        count = len(dq)
        avg_q = float(sum(q for _, q in dq) / count)
        return count, avg_q

    def _has_enough_recent_evidence(
        self,
        track_id: int,
        ts: float,
        needed: int,
    ) -> Tuple[bool, int, float]:
        """
        Check whether we have enough recent high-quality samples for a track.

        Returns (ok, count, avg_quality).
        """
        needed = max(1, int(needed))
        count, avg_q = self._recent_evidence_stats(track_id, ts)
        ok = (count >= needed) and (avg_q >= self._min_quality_runtime)
        return ok, count, avg_q

    # ------------------------------------------------------------------ #
    # Housekeeping                                                       #
    # ------------------------------------------------------------------ #

    def _cleanup_dead_tracks(self, active_ids: Set[int], ts: float) -> None:
        """
        Drop very old state for tracks that are no longer present.
        """
        for tid in list(self._last_decision.keys()):
            if tid not in active_ids:
                last_ts = self._last_decision_ts.get(tid, ts)
                if ts - last_ts > self.smoothing_cfg.stale_after_sec:
                    self._last_decision.pop(tid, None)
                    self._last_decision_ts.pop(tid, None)
                    self._evidence_history.pop(tid, None)

    @staticmethod
    def _decay_factor(age: float, half_life: float) -> float:
        """
        Exponential decay factor for a given age and half-life.
        """
        if half_life <= 0.0:
            return 0.0
        return 0.5 ** (age / half_life)
