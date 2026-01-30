# gait_subsystem/gait/gait_engine.py
"""
Gait Engine: Integrates GaitExtractor and GaitGallery to process Tracklets
with pose data and produce identity recognition decisions based on gait.

DEEP ROBUST DESIGN:
1. Clear separation: Extract → Match → Decide
2. Uses main system's schemas for interoperability
3. Quality-gated processing (skip low-quality sequences)
4. Detailed logging for debugging
"""
from __future__ import annotations
import logging
from typing import List, Optional, Dict
import numpy as np

# Use main system schemas
from schemas import Frame, Tracklet
from schemas.id_signals import IdSignal
from schemas.identity_decision import IdentityDecision

# Use local gait modules (relative imports within gait_subsystem)
from gait_subsystem.gait.config import GaitConfig, default_gait_config, GaitRobustConfig
from gait_subsystem.gait.gait_extractor import GaitExtractor
from gait_subsystem.gait.gait_gallery import GaitGallery
from gait_subsystem.gait.state import GaitTrackState, GaitState, GaitReason
import time

logger = logging.getLogger(__name__)


class GaitEngine:
    """
    Main orchestrator for gait-based identity recognition.
    
    DEEP ROBUST DESIGN:
    1. Lazy initialization of heavy models
    2. Per-track quality checks before processing
    3. Threshold-based accept/reject decisions
    4. Integration with main system's IdentityDecision format
    """
    
    def __init__(self, config: Optional[GaitConfig] = None):
        """
        Initialize the GaitEngine with configuration.
        
        Args:
            config: Optional custom configuration. Uses default if None.
        """
        self.config = config or default_gait_config()
        self.extractor = GaitExtractor(self.config)
        self.gallery = GaitGallery(self.config)
        # Per-track robustness state (Gold Spec D1)
        self._track_states: Dict[int, GaitTrackState] = {} 
        logger.info("✅ GaitEngine initialized with Deep Robust logic")

    def _get_or_create_state(self, track_id: int) -> GaitTrackState:
        if track_id not in self._track_states:
            self._track_states[track_id] = GaitTrackState(track_id=track_id)
        return self._track_states[track_id]

    def _compute_quality(self, track: Tracklet) -> float:
        """
        Compute robust sequence quality scalar (q_seq).
        Gold Spec D2: Visibility ratio + Core Anchors presence.
        """
        if not track.gait_sequence_data:
            return 0.0
            
        # Analyze last 30 frames max
        recent = track.gait_sequence_data[-30:]
        total_score = 0.0
        
        for kp in recent:
            # kp shape: (17, 3) -> [x, y, conf]
            # 1. Joint Visibility Ratio (how many joints > 0.4?)
            hits = (kp[:, 2] > 0.4).sum()
            vis_ratio = hits / 17.0
            
            # 2. Core Anchors (Shoulders: 5,6, Hips: 11,12) - Critical for gait
            core_conf = kp[[5,6,11,12], 2].mean()

            # 3. [DEEP ROBUST] Legs Required (Knees: 13,14, Ankles: 15,16)
            # Prevent "Upper Body Only" FPs (behind desk/wall)
            leg_conf = kp[[13,14,15,16], 2].mean()
            
            # Combined score for this frame
            # If legs are missing (conf < 0.35), penalty exists but not lethal
            # Changed 0.1 -> 0.5 to allow Eval but likely block Confirm (Q < 0.65)
            # Combined score for this frame
            base_score = 0.5 * vis_ratio + 0.3 * core_conf + 0.2 * leg_conf
            
            # If legs are missing (conf < 0.45), strictly penalize to prevent Desk/Wall FPs.
            # We want to force a toggle to UNSURE or prevent EVAL if legs aren't clear.
            
            if leg_conf < 0.45:
                frame_score = 0.1 # Hard Reject for this frame
            else:
                frame_score = base_score
            
            total_score += frame_score
            
        return total_score / len(recent)

    def _check_regime(self, track: Tracklet) -> bool:
        """
        Regime Gate: Check for Motion and Validity.
        Gold Spec D3.
        Returns True if regime is valid (moving), False if STILL/CHAOS.
        """
        if not track.gait_sequence_data or len(track.gait_sequence_data) < 5:
            return False
            
        # [DEEP ROBUST MOTION]
        # Use Limb Alternation + Energy (Frequency Domain Proxy)
        # Walking implies the left and right legs swap positions relative to the hip center.
        
        hist = track.gait_sequence_data[-15:] # Need slightly more history for alternation
        if len(hist) < 5: return False
        
        # 1. Compute Left-Right difference over time
        # (Ankle_L_y - Ankle_R_y) should oscillate around 0
        diffs = []
        
        # 2. Compute "Leg Energy" (Variance of relative height)
        # Existing logic: std(ankle - hip)
        # Improved: std(left - right) is cleaner signal of gait cycle
        
        for frame_kp in hist:
            # Hip Center Y
            hip_y = frame_kp[[11,12], 1].mean()
            
            # Ankle Y relative to Hip (positive = below hip)
            la_y = frame_kp[15, 1] - hip_y 
            ra_y = frame_kp[16, 1] - hip_y
            
            diffs.append(la_y - ra_y)

        diffs = np.array(diffs)
        
        # A. Energy Gate (Amplitude of stride)
        # Normalize by Torso Height
        curr = track.gait_sequence_data[-1]
        shoulders = curr[[5,6], :2].mean(axis=0)
        hips = curr[[11,12], :2].mean(axis=0)
        torso_h = np.linalg.norm(shoulders - hips)
        
        if torso_h < 10.0: return False
        
        motion_energy = np.std(diffs)
        min_energy = 0.02 * torso_h # 2% of torso height minimum variance
        
        # B. Alternation Gate (Zero Crossings / Sign Changes)
        # A real step involves potential sign change or at least significant variance
        # "Standing with jitter" has high frequency low amplitude noise.
        # "Walking" has lower frequency high amplitude.
        
        # Simple count of sign changes isn't perfect if noisy around 0, 
        # but combined with Energy it is robust.
        
        is_moving = motion_energy > min_energy
        
        # Optional: Require at least some trend change if sequence is long enough
        # But Energy is the primary "Are legs moving relative to each other" metric.
        
        return is_moving

    def update_signals(self, frame: Frame, tracks: List[Tracklet]) -> List[IdSignal]:
        """
        Execute Gold Spec Pipeline: Evidence -> Gate -> Evaluate -> Decide.
        """
        signals = []
        now = time.perf_counter()
        cfg_robust = self.config.robust
        
        for track in tracks:
            # --- 1. Evidence & State ---
            state = self._get_or_create_state(track.track_id)
            seq_len = len(track.gait_sequence_data)
            state.q_seq = self._compute_quality(track)
            
            # --- 2. Gating (Regime & Quality) ---
            motion_valid = self._check_regime(track)
            quality_ok = state.q_seq >= cfg_robust.quality_min
            len_ok = seq_len >= cfg_robust.min_seq_len
            
            # State Transitions (D1)
            if state.state == GaitState.COLLECTING:
                if len_ok and quality_ok and motion_valid:
                    state.state = GaitState.EVALUATING
                    state.set_reason(GaitReason.HOLD_NEUTRAL)
                    
            elif state.state == GaitState.EVALUATING:
                if not quality_ok:
                    state.set_reason(GaitReason.SKIP_LOW_QUALITY)
                    # We don't drop to COLLECTING immediately to avoid flicker, just pause eval
                elif not motion_valid:
                    state.set_reason(GaitReason.SKIP_STILL)

            # --- 3. Evaluate (Scheduling D4) ---
            should_eval = (
                state.state in [GaitState.EVALUATING, GaitState.CONFIRMED, GaitState.UNSURE] and
                state.can_evaluate(now, cfg_robust.eval_period) and
                quality_ok and
                motion_valid
            )
            
            match_id = None
            confidence = 0.0
            
            if should_eval:
                state.last_eval_ts = now
                
                # Extract & Search
                embedding, q = self.extractor.extract_gait_embedding_and_quality(track.gait_sequence_data)
                
                # [DEEP ROBUST] FIX: Unreachable Demotion Logic
                # Capture previous state to handle CONFIRMED -> UNSURE transition
                prev_state = state.state
                
                # Double check quality gate after extraction
                if embedding is not None and q >= cfg_robust.quality_min:
                    # [DEEP ROBUST] Extract Anthropometry
                    anthro_stats = self.extractor.extract_anthropometry(track.gait_sequence_data)
                    
                    match_id, confidence, details = self.gallery.search(embedding, anthro_query=anthro_stats)
                    
                    # Check Identity Swap Check
                    current_best = details.get("best_pid", "Unknown")
                    prev_best = state.best_id
                    
                    if prev_best and current_best != prev_best:
                         # Identity Swap! Reset streak immediately.
                         # If we were confirmed, this is a major conflict -> UNSURE
                         state.reset_streak()
                         if prev_state == GaitState.CONFIRMED:
                             state.state = GaitState.UNSURE
                             state.set_reason(GaitReason.UNSURE_CONFLICT)
                    
                    # Store Raw Match Data
                    state.update_match(
                        best_id=current_best, 
                        best_sim=details.get("best_sim", 0.0), 
                        best_dist=details.get("best_dist", 1.0)
                    )
                    
                    # --- 4. Decide (Policy D5) ---
                    margin = details.get("margin", 0.0) # Might be missing if 1 candidate
                    if "margin" not in details and confidence > 0:
                        margin = 1.0 # Only one candidate means huge margin
                        
                    sim = state.best_sim
                    
                    # CONFIRM LOGIC
                    is_confirm_candidate = (
                        sim >= cfg_robust.threshold_confirm and
                        margin >= cfg_robust.margin_confirm and
                        state.q_seq >= cfg_robust.quality_confirm
                    )
                    
                    if is_confirm_candidate:
                        state.increment_streak()
                        # Clear bad streak on success
                        state.bad_eval_streak = 0
                        
                        if state.confirm_streak >= cfg_robust.confirm_streak:
                            state.state = GaitState.CONFIRMED
                            state.set_reason(GaitReason.CONFIRM_STRONG)
                        else:
                            state.set_reason(GaitReason.HOLD_STREAK)
                    else:
                        # [FIX 2] Streak Pause / Demotion Logic
                        # If sim is decent (> T_candidate) but Margin/Quality low, just PAUSE or DEMOTE.
                        # DEEP ROBUST POLICY: High Similarity but Low Margin is NOT enough.
                        
                        if sim < cfg_robust.threshold_candidate:
                            # Verify REJECT logic
                            state.reset_streak()
                            state.set_reason(GaitReason.REJECT_LOW_SIM)
                            
                            if prev_state == GaitState.CONFIRMED:
                                # [DEEP ROBUST] Demotion
                                state.bad_eval_streak += 1
                                if state.bad_eval_streak >= 2:
                                    state.state = GaitState.UNSURE
                                    state.set_reason(GaitReason.REJECT_LOW_SIM)
                            else:
                                state.state = GaitState.EVALUATING 
                                
                        elif sim < cfg_robust.threshold_confirm:
                            # Candidate zone (-0.60..0.70). Reset streak.
                            state.reset_streak()
                            state.set_reason(GaitReason.HOLD_BORDERLINE)
                            # If confirmed, accumulate bad streak (weak signal)
                            if prev_state == GaitState.CONFIRMED:
                                state.bad_eval_streak += 1
                                if state.bad_eval_streak >= 4:
                                    state.state = GaitState.UNSURE
                                    
                        else:
                            # Sim is GOOD (>= T_confirm), but Margin/Quality failed.
                            # [DEEP ROBUST FIX] strict policy:
                            # If Margin is low, we DO NOT CONFIRM. We HOLD.
                            
                            state.set_reason(GaitReason.HOLD_LOW_MARGIN)
                            # Do NOT increment confirm streak.
                            # In fact, if we are currently confirmed, this IS a stability failure.
                            # "You look like Castle(0.9) but also Johny(0.85)" -> Unsafe to lock.
                            
                            if prev_state == GaitState.CONFIRMED:
                                # Check if margin is CRITICALLY low or just slightly low
                                if margin < (cfg_robust.margin_confirm * 0.5):
                                    # Very ambiguous -> Demote fast
                                    state.bad_eval_streak += 2 
                                else:
                                    # Slightly ambiguous -> Drift
                                    state.bad_eval_streak += 1
                                    
                                if state.bad_eval_streak >= 3: # Fast demotion on ambiguity
                                    state.state = GaitState.UNSURE
                                    state.set_reason(GaitReason.UNSURE_CONFLICT)
                            
            # --- 5. Robust Demotion Check (Gate Failures) ---
            # If we are CONFIRMED but currently failing GATES (Quality/Motion/Len),
            # we must count this against the track to prevent "Ghosting".
            
            # Note: If should_eval was False, we skipped the logic above.
            # We need to handle the "Not Evaluating" case for confirmed tracks.
             
            if state.state == GaitState.CONFIRMED and not should_eval:
                # Why are we not evaluating?
                if not quality_ok:
                    state.bad_eval_streak += 1
                    state.set_reason(GaitReason.UNSURE_QUALITY_DROP)
                elif not motion_valid:
                     # Stillness shouldn't immediately kill ID, but prolonged sitting should.
                     # Let's say 3-4 seconds of stillness -> Unsure.
                     # eval_period is 0.7s. So ~5 checks.
                     state.bad_eval_streak += 0.5 # Accumulate slower for stillness
                
                if state.bad_eval_streak >= 5:
                    state.state = GaitState.UNSURE
                    
            # Construct Final Signal
            # If CONFIRMED, we broadcast the ID - BUT only if currently valid-ish
            # To be "High Efficient", we don't want to hide ID instantly on one bad frame,
            # but we also don't want to show it if we are UNSURE.
            
            final_id = None
            final_conf = 0.0
            
            if state.state == GaitState.CONFIRMED:
                final_id = state.best_id
                final_conf = state.best_sim
            elif state.state == GaitState.EVALUATING:
                 # Optional: Show candidate if above weak threshold
                 if state.best_sim >= cfg_robust.threshold_candidate:
                     final_id = state.best_id # "Maybe"
                     final_conf = state.best_sim
            
            # [FIX A] Build Extra Metadata for Logs - FULL VISIBILITY
            # We need to retrieve the last computed details from somewhere, 
            # or just use state + defaults. state doesn't store margin/2nd yet.
            # Ideally we'd store these in state too, but for now let's rely on what we just computed 
            # if we ran eval, or defaults.
            
            # This 'details' var is only available inside the `if should_eval` block.
            # We need to persist these visibility metrics in State if we want them always visible.
            # For now, let's just output them if we have them.
            
            meta = {
                "gait_state": state.state.value,
                "reason": state.reason.value,
                "q_seq": state.q_seq,
                "best_pid": state.best_id,
                "best_sim": state.best_sim,
                "best_dist": state.best_dist,
                "streak": state.confirm_streak,
                # New Debug Fields
                "bad_streak": state.bad_eval_streak,
            }
            
            # Merge details if available (from recent eval)
            # Note: This limits visibility to frames where eval ran.
            if should_eval and 'details' in locals():
                 meta["second_pid"] = details.get("second_pid")
                 meta["second_sim"] = details.get("second_sim")
                 meta["margin"] = details.get("margin")
                 meta["status"] = details.get("status")
                 # [DEEP ROBUST] Log Geometry
                 if "match_anthro_dist" in details:
                     meta["anthro_dist"] = f"{details['match_anthro_dist']:.2f}"
            
            signal = IdSignal(
                track_id=track.track_id, 
                identity_id=final_id, 
                confidence=final_conf,
                method="gait",
                extra=meta
            )
            signals.append(signal)

        return signals

    def decide(self, signals: List[IdSignal]) -> List[IdentityDecision]:
        """
        Convert raw IdSignals into final IdentityDecisions.
        
        Enriches decisions with category information from gallery
        for display in UI/Overlay.
        
        Args:
            signals: List of gait identification signals
            
        Returns:
            List of IdentityDecision objects for overlay rendering
        """
        decisions = []
        
        for signal in signals:
            identity_id = None
            category = "unknown"
            confidence = 0.0
            
            # If we found a valid match
            if signal.identity_id:
                identity_id = signal.identity_id
                confidence = signal.confidence
                
                # Retrieve category from gallery
                category = self.gallery.get_category(identity_id)
            
            # Create final decision for overlay
            decision = IdentityDecision(
                track_id=signal.track_id,
                identity_id=identity_id,
                category=category, 
                confidence=confidence,
                reason=f"gait:{confidence:.2f}",
                extra=signal.extra  # Propagate metadata to decision for logging
            )
            decisions.append(decision)
            
        return decisions