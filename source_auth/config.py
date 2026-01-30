# source_auth/config.py
#
# Configuration objects for Source Authenticity (“SourceAuth”).
#
# Goal:
#   Decide whether a tracked face comes from a *real 3D head* in the scene
#   or from a *2D device* (phone, screen, printed photo).
#
# Design principles:
#   - Pure configuration: no OpenCV / NumPy / heavy deps here.
#   - All knobs are explicit, with safe defaults.
#   - Structurally similar to face.config: small focused dataclasses +
#     one top-level SourceAuthConfig with sub-configs:
#         * motion     → 3D vs planar motion
#         * screen     → screen / phone artifacts
#         * background → background consistency inside vs outside
#         * fusion     → how we combine sub-scores into a final score/state
#   - Backwards-safe: this module is new and does not depend on the rest
#     of the system, so importing it cannot break existing code.
#

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any


# ---------------------------------------------------------------------------
# 3D vs planar motion config (aligned with motion.py)
# ---------------------------------------------------------------------------


@dataclass
class SourceAuthMotionConfig:
    """
    Configuration for the 3D-vs-planar motion cue.

    Intuition:
      Over a short time window (e.g. 1–2 seconds), a real 3D head produces
      *parallax* between landmarks (nose vs ears vs cheeks), while a phone/photo
      behaves like a flat card (all landmarks move in sync under a single 2D
      transform).

    The engine will:
      - keep a short history of landmark positions per track,
      - compare multiple reference frames (oldest + mid) to later frames,
      - fit a global 2D affine model (planar card motion),
      - measure residual motion of each landmark vs that model,
      - aggregate residuals into a motion score ∈ [0,1].

    This config controls:
      - how long the window is,
      - how much temporal span we require,
      - how many landmarks we need,
      - how big motion must be (adaptive to face size),
      - how we map parallax → [0,1] score,
      - optional boost from yaw/pitch span.
    """

    # Length of the time window (in seconds) used to aggregate motion.
    # Too short → noisy; too long → mixes different poses.
    window_sec: float = 1.5

    # Minimum fraction of motion_window_sec that must be covered by history
    # before we trust the cue (e.g. 0.5 → at least 0.75s if window=1.5s).
    min_span_factor: float = 0.5

    # Minimum landmarks per frame required to consider a frame usable.
    # (We map this to engine-level motion_min_landmarks.)
    min_landmarks: int = 5

    # Base pixel motion threshold for "enough motion".
    # The engine computes:
    #   min_motion_pixels = max(min_motion_pixels_base,
    #                           face_size_px * min_motion_frac)
    # where face_size_px ≈ min(width, height) of the face bbox.
    min_motion_pixels_base: float = 2.0

    # Fraction of face size to demand as motion (e.g. 0.01 → 1% of bbox size).
    min_motion_frac: float = 0.01

    # Yaw / pitch based boost:
    #
    # If yaw/pitch span across the window exceeds yaw_deg_threshold and we
    # have non-trivial residuals, we can add a small positive bias to the
    # 3D motion score to reward true head rotations.
    yaw_deg_threshold: float = 8.0
    yaw_boost: float = 0.15

    # Mapping from parallax_ratio → [0,1] score:
    #
    #   parallax_ratio <= parallax_low   → score ≈ 0 (planar / card-like)
    #   parallax_ratio >= parallax_high  → score ≈ 1 (strong 3D parallax)
    #
    # In between we interpolate linearly.
    #
    # These can be tuned later based on telemetry, but defaults aim to
    # separate phones (<~0.1) from real heads (>~0.2) under normal motion.
    parallax_low: float = 0.08
    parallax_high: float = 0.22


# ---------------------------------------------------------------------------
# Screen / display artifact config (aligned with screen_artifacts.py)
# ---------------------------------------------------------------------------


@dataclass
class SourceAuthScreenConfig:
    """
    Configuration for the "screen / phone" cue.

    Intuition:
      A face shown on a phone or monitor tends to exhibit:
        - rectangular borders / bezels around the image,
        - high-contrast straight edges and right angles near the face,
        - subtle flicker / refresh patterns across frames,
        - moiré textures and pixel grid patterns.

      This cue focuses on visual evidence that the *object* around the face
      is a display, not a real head.

    The engine will:
      - look around the face bounding box for long straight edges and corners,
      - analyse temporal brightness variations for flicker-like behaviour,
      - inspect high-frequency textures for moiré patterns,
      - aggregate these into a screen_score ∈ [0,1].
    """

    enabled: bool = True

    # Region of interest around the face for edge / border detection, measured
    # as a fraction of face bbox size (e.g. 0.1 = 10% margin).
    #
    # The engine will expand the bbox by this factor in each direction and
    # search for straight edges / rectangles in that ring.
    border_margin_frac: float = 0.15

    # Minimum edge length relative to the expanded box diagonal to consider it
    # a candidate border segment (avoids tiny noisy edges). This is mainly
    # intended for a richer engine; screen_artifacts.py focuses on edge
    # density + Hough rectangles and does not directly consume this field yet.
    min_edge_length_frac: float = 0.35

    # Tolerance (in degrees) when checking if edges are approximately
    # horizontal/vertical and when evaluating corners. Used conceptually
    # by the screen cue and concretely mapped to screen_rectangularity_angle_tol.
    corner_angle_tolerance_deg: float = 15.0

    # How many strong border "votes" (edges + corners) we need before we
    # believe there is a rectangular phone / screen around the face.
    #
    # The current screen_artifacts implementation does not use discrete votes,
    # but a future SourceAuthEngine can read this field.
    border_votes_for_strong: int = 3

    # Temporal flicker analysis:
    #
    # Length of the sliding window (in seconds) of average brightness over
    # which we run a simple frequency / variance analysis.
    flicker_window_sec: float = 1.0

    # Minimum normalised variance in brightness over the window before flicker
    # is considered meaningful. If the scene is almost static lighting-wise,
    # flicker cue is down-weighted. Kept here for a future rich engine.
    min_brightness_variance: float = 0.002

    # Whether to explicitly look for periodicity near 50/60 Hz aliases in the
    # recorded frame-rate band. If False, only coarse variance is used.
    enable_flicker_frequency_hint: bool = True

    # Moiré / texture cue:
    #
    # Whether to enable a simple high-frequency texture score in the face
    # region (e.g. strong regular grids).
    enable_moire_hint: bool = True

    # Weight of border vs flicker vs moiré when aggregating into a single
    # screen_score ∈ [0,1]. These will be normalised internally by the engine
    # so they do not need to sum to 1 here, but relative magnitudes matter.
    #
    # Step-2 design: border is dominant (0.6), flicker/moiré are secondary.
    w_border: float = 0.6
    w_flicker: float = 0.2
    w_moire: float = 0.2

    # Mapping from aggregated "screen evidence" metric E_screen to
    # screen_score ∈ [0,1]:
    #
    #   E_screen <= evidence_low   → screen_score ≈ 0 (no screen)
    #   E_screen >= evidence_high  → screen_score ≈ 1 (definitely screen)
    #
    # In between we interpolate.
    evidence_low: float = 0.2
    evidence_high: float = 0.7

    # --- Parameters directly used by screen_artifacts.py -----------------

    # Minimum number of valid frames in the window before we trust the cue.
    min_frames: int = 4

    # Minimum face size (in pixels, min(width, height)) for screen analysis.
    min_face_px: int = 40

    # Normalisation for flicker variance → [0,1].
    # If rel_var == flicker_var_norm → var_score ≈ 0.5.
    flicker_var_norm: float = 0.01

    # Normalisation for moiré (Laplacian variance) → [0,1].
    # If mean_val ~ moire_norm → moire_score ≈ 0.5.
    moire_norm: float = 300.0

    # Rectangularity angle tolerance used by Hough-based border detection.
    rectangularity_angle_tol: float = 15.0

    # Normalisation for bezel intensity std-dev: lower std-dev → more uniform
    # bezel → higher bezel_uniformity_score.
    bezel_uniformity_norm: float = 40.0

    # Reliability thresholds for border temporal stability:
    #   - border_reliable_min: minimum average border_strength
    #   - border_reliable_std_max: maximum std-dev across window
    border_reliable_min: float = 0.50
    border_reliable_std_max: float = 0.15


# ---------------------------------------------------------------------------
# Background consistency config (aligned with background.py)
# ---------------------------------------------------------------------------


@dataclass
class SourceAuthBackgroundConfig:
    """
    Configuration for the background consistency cue.

    Intuition:
      For a real head:
        - the background directly behind the head (inside the bbox, non-face
          areas) belongs to the *same physical environment* as the strip
          just outside the face (walls, room, etc.).

      For a phone / screen:
        - inside the face patch you see a *different world* (other room),
        - outside you see your current room (desk, walls, etc.),
        - colour distributions, noise patterns, sharpness differ.

    The engine will:
      - define two regions:
          * inner_bg: parts of the face bbox likely to be "background",
          * outer_bg: a ring around the bbox (same scene).
      - compute simple statistics / histograms for both,
      - measure their difference,
      - map difference → background_consistency ∈ [0,1]
            1 = consistent same-world background
            0 = strong mismatch / different world.
    """

    enabled: bool = True

    # Fraction of the face bbox to treat as "inner background" margin from
    # the border (to avoid hair/face too much). Example:
    #   inner_margin_frac = 0.15 means we shrink the bbox by 15% from each
    #   side and then take top/bottom strips etc. as candidate background.
    inner_margin_frac: float = 0.15

    # Fraction to expand the bbox for "outer background" (a ring around
    # the face). Similar meaning as in screen config, but used here to
    # compare overall scene statistics.
    outer_margin_frac: float = 0.25

    # Minimum number of pixels for each region (inner + outer) before we
    # consider this cue valid. If regions are too small (tiny face, big zoom),
    # background statistics become noisy.
    min_region_pixels: int = 500

    # Minimum face size (in pixels, min(width, height)) for background analysis.
    # This avoids extremely small faces where background vs foreground is noisy.
    min_face_px: int = 40

    # Histogram / colour stats configuration.
    #
    # For simplicity we can use fixed-size histograms in HSV or Lab space.
    # This parameter controls the number of bins per channel.
    color_hist_bins: int = 32

    # Optional Gaussian blur sigma used to smooth images before computing
    # noise / sharpness measures; helps reduce high-frequency sensor noise.
    blur_sigma_for_noise: float = 0.5

    # Mapping from background difference D_bg to score:
    #
    # We expect D_bg ≈ 0 for same-environment (real head) and D_bg large
    # for different-environment (phone/screens).
    #
    #   D_bg <= diff_low   → background_consistency ≈ 1 (consistent background)
    #   D_bg >= diff_high  → background_consistency ≈ 0 (inconsistent background)
    #
    # In between we interpolate 1 → 0.
    diff_low: float = 0.15
    diff_high: float = 0.6

    # If True, the engine may incorporate simple sharpness / noise statistics
    # (e.g. Laplacian variance) to refine the decision, since phone/screens
    # often have different sharpness / compression compared to the live scene.
    enable_noise_sharpness_hint: bool = True

    # Weight of colour vs noise/texture in the background consistency score.
    # These are used inside background.py as:
    #   bg_mismatch = w_color * color_delta_norm + w_texture * texture_delta_norm
    #   background_consistency = 1 - bg_mismatch
    w_color: float = 0.7
    w_texture: float = 0.3

    # Normalisation constants for mapping raw deltas → [0,1] before weighting.
    #
    # After computing color_delta / texture_delta, background.py will do
    # something like:
    #   color_delta_norm   = clamp(color_delta   / color_norm,   0, 1)
    #   texture_delta_norm = clamp(texture_delta / texture_norm, 0, 1)
    #
    # These defaults are conservative and can be tuned from telemetry.
    color_norm: float = 0.5
    texture_norm: float = 50.0


# ---------------------------------------------------------------------------
# Fusion config: combining cues into final score/state
# ---------------------------------------------------------------------------


@dataclass
class SourceAuthFusionConfig:
    """
    Configuration for fusing motion, screen, and background cues into:

        - source_auth_score ∈ [0,1]
        - source_auth_state ∈ {REAL, LIKELY_REAL, UNCERTAIN, LIKELY_SPOOF, SPOOF}

    Core idea:
      - Each cue produces a score in [0,1] with semantics:
          * motion_score      → 1 = strong 3D, 0 = planar
          * screen_score      → 1 = screen-like, 0 = non-screen
          * background_score  → 1 = same-world, 0 = different-world
      - The engine then maps them to a *realness* score and runs a
        temporal state machine to obtain a discrete label.
    """

    # Weights for each cue in the fused "realness" decision.
    #
    # The fusion module will internally compute something like:
    #
    #   M_real = planar_3d
    #   S_real = 1 - screen_artifacts
    #   B_real = background_consistency
    #
    # then build:
    #
    #   real_evidence  = w_motion * M_real + w_background * B_real
    #   spoof_evidence = w_screen * screen_artifacts
    #                    + w_bg_inverse * (1 - background_consistency)
    #
    # where w_bg_inverse can be tied to w_background by default.
    w_motion: float = 0.5
    w_screen: float = 0.3
    w_background: float = 0.2

    # Weight for background mismatch in spoof_evidence. By default we re-use
    # the same magnitude as w_background, but this can be tuned independently.
    w_bg_inverse: float = 0.2

    # Non-linearity strength for mapping real_evidence - spoof_evidence into
    # [0,1]. Higher alpha → sharper transitions, lower alpha → smoother.
    # Used as "fusion_alpha" in the flat config.
    alpha: float = 4.0

    # Temporal smoothing window (seconds) for the *fused_realness* score at
    # the track level. This is separate from the raw cue windows; here we
    # control how fast the state (REAL/SPOOF/...) is allowed to flip.
    fused_window_sec: float = 1.0

    # Minimum number of fused samples required before we consider the
    # state reliable at all. This avoids making hard decisions for tracks
    # that appeared only for a few frames.
    min_fused_samples: int = 5

    # Thresholds on fused_realness for mapping to discrete states, using
    # hysteresis-style bands to avoid flip-flopping.
    #
    # Example mapping:
    #
    #   fused_realness ≥ real_strong          → REAL
    #   fused_realness ≥ real_weak           → LIKELY_REAL
    #   fused_realness ≤ spoof_strong        → SPOOF
    #   fused_realness ≤ spoof_weak          → LIKELY_SPOOF
    #   otherwise                            → UNCERTAIN
    #
    # Ensure:
    #   0.0 ≤ spoof_strong ≤ spoof_weak ≤ real_weak ≤ real_strong ≤ 1.0
    # CRITICAL FIX: Lowered thresholds to reduce UNCERTAIN states
    # Original: 0.80/0.60/0.40/0.20 caused too many UNC states
    real_strong: float = 0.70   # CHANGED: 0.80 → 0.70
    real_weak: float = 0.50     # CHANGED: 0.60 → 0.50
    spoof_weak: float = 0.40
    spoof_strong: float = 0.20

    # If True, the fusion logic is allowed to mark a track as "SPOOF" only if
    # the fused_realness has been below spoof_strong for at least
    # min_spoof_persistence_sec continuously. This prevents brief glitches
    # from causing strong spoof labels.
    enable_spoof_persistence: bool = True
    min_spoof_persistence_sec: float = 0.5

    # Similarly, if True, the fusion logic is allowed to promote a track to
    # "REAL" only after real_strong has been sustained for this long.
    enable_real_persistence: bool = True
    min_real_persistence_sec: float = 0.5

    # Optional: whether "UNCERTAIN" is allowed as a stable state or we want
    # the fusion logic to resolve towards REAL / SPOOF whenever possible using
    # slight bias towards REAL (for operator friendliness).
    allow_uncertain_state: bool = True

    # If allow_uncertain_state is False, we can bias ambiguous values:
    #
    #   fused_realness in (spoof_weak, real_weak) → treat as REAL if
    #   fused_realness ≥ bias_towards_real_threshold, otherwise SPOOF.
    bias_towards_real_threshold: float = 0.55


# ---------------------------------------------------------------------------
# Top-level SourceAuthConfig
# ---------------------------------------------------------------------------


@dataclass
class SourceAuthConfig:
    """
    Top-level configuration for the Source Authenticity module.

    This is the single object that SourceAuthEngine receives.

    It contains:
      - global toggles and timing parameters,
      - sub-configs for each cue (motion/screen/background),
      - fusion parameters for combining cues into final score/state.
    """

    # Master switch for SourceAuth. If False, the engine should behave as if
    # all tracks have "unknown" source authenticity and avoid heavy
    # computation (no landmark history, no flicker analysis).
    enabled: bool = True

    # Whether to require a minimum track age before we produce *strong* states
    # (REAL / SPOOF). Very fresh tracks may stay UNCERTAIN / LIKELY_* only.
    min_track_age_sec_for_strong_state: float = 0.8

    # Hard idle timeout for discarding SourceAuth state for tracks that
    # disappeared. Typically should be ≥ fusion.fused_window_sec, but still
    # small so we do not accumulate stale state.
    max_idle_sec: float = 3.0

    # If True, and if multiview / pose information is available from the
    # face route (yaw/pitch), SourceAuth can reuse those to:
    #   - ignore extremely rotated frames in motion cue,
    #   - optionally condition motion/background analysis on pose.
    #
    # If False, SourceAuth will rely only on its own estimates / raw landmarks.
    use_face_route_pose: bool = True

    # Sub-configs for each cue.
    motion: SourceAuthMotionConfig = field(default_factory=SourceAuthMotionConfig)
    screen: SourceAuthScreenConfig = field(default_factory=SourceAuthScreenConfig)
    background: SourceAuthBackgroundConfig = field(
        default_factory=SourceAuthBackgroundConfig
    )
    fusion: SourceAuthFusionConfig = field(default_factory=SourceAuthFusionConfig)

    # Fusion mode:
    #   "v2"    → use source_auth.fusion.fuse_source_auth (new path)
    #   "legacy"→ keep using older engine-internal state machine (if wired)
    fusion_mode: str = "v2"

    # Optional reserved field for future per-environment overrides, e.g.
    # different presets for "indoor", "outdoor", "low-light" etc. This is
    # kept simple for now; a later revision can expand it without breaking
    # existing code.
    preset_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Factory for default config
# ---------------------------------------------------------------------------


def default_source_auth_config(
    face_cfg: Optional[Any] = None,
    **kwargs: Any,
) -> SourceAuthConfig:
    """
    Construct a SourceAuthConfig with tuned but conservative defaults.

    Parameters
    ----------
    face_cfg:
        Optional face configuration object (e.g. FaceConfig). If provided,
        we MAY derive a few SourceAuth thresholds from it (duck-typed via
        getattr; we never import heavy face modules here).

    **kwargs:
        Reserved for future overrides. Any extra keyword will be applied
        as a top-level attribute on the returned config, so that call sites
        can pass extra tuning knobs without breaking.

    Notes
    -----
    This helper mirrors the pattern used elsewhere in the project
    (e.g. face.config.default_face_config) and allows other modules
    to obtain a ready-to-use configuration without touching dataclass
    details.

    The defaults are intended to:
      - make real heads tend to "REAL / LIKELY_REAL",
      - make obvious phones/screens tend to "SPOOF / LIKELY_SPOOF",
      - avoid hard decisions for ambiguous cases (stay in UNCERTAIN).
    """
    cfg = SourceAuthConfig()

    # ------------------------------------------------------------------
    # Global knobs expected by SourceAuthEngine / fusion via getattr(...)
    # ------------------------------------------------------------------

    # Neutral score for "no information yet".
    cfg.neutral_score = 0.5

    # Base window used by the engine when no per-cue window is specified.
    cfg.window_sec = float(cfg.motion.window_sec)

    # Per-cue windows used in engine history pruning.
    cfg.motion_window_sec = float(cfg.motion.window_sec)
    # For screen cue we can reasonably reuse flicker_window_sec as its
    # characteristic temporal scale.
    cfg.screen_window_sec = float(
        getattr(cfg.screen, "flicker_window_sec", cfg.motion.window_sec)
    )
    # Background cue can share the motion window for now.
    cfg.background_window_sec = float(cfg.motion.window_sec)

    # Reliability softening factor when cues are unreliable (used by legacy
    # fusion path; kept for backwards compatibility).
    cfg.unreliable_cue_weight_factor = 0.3

    # Fusion weights interpreted by the legacy engine path
    # (align with fusion sub-config). These are also convenient aliases.
    cfg.fusion_weight_motion = float(cfg.fusion.w_motion)
    cfg.fusion_weight_screen = float(cfg.fusion.w_screen)
    cfg.fusion_weight_background = float(cfg.fusion.w_background)

    # New fusion-v2 weights: real/spoof evidence components.
    cfg.fusion_w_motion = float(cfg.fusion.w_motion)
    cfg.fusion_w_background = float(cfg.fusion.w_background)
    cfg.fusion_w_screen = float(cfg.fusion.w_screen)
    cfg.fusion_w_bg_inverse = float(
        getattr(cfg.fusion, "w_bg_inverse", cfg.fusion.w_background)
    )

    # Non-linearity strength for mapping evidence-difference → [0,1].
    cfg.fusion_alpha = float(getattr(cfg.fusion, "alpha", 4.0))

    # Minimum track age before allowing non-UNCERTAIN states in fusion.
    cfg.fusion_min_track_age_sec = float(
        getattr(cfg, "min_track_age_sec_for_strong_state", 0.8)
    )

    # Fusion mode default.
    cfg.fusion_mode = getattr(cfg, "fusion_mode", "v2")

    # EMA coefficient used by the legacy temporal smoother in the engine.
    # CRITICAL FIX: Reduced alpha for smoother state transitions
    # Original 0.6 was too reactive, causing oscillation
    cfg.ema_alpha = 0.35  # CHANGED: 0.6 → 0.35 (70% history)

    # State machine thresholds (REAL / LIKELY_REAL / LIKELY_SPOOF / SPOOF)
    # used by legacy path; we also expose fusion_* aliases for fusion-v2.
    cfg.real_min_score = float(cfg.fusion.real_strong)
    cfg.likely_real_min_score = float(cfg.fusion.real_weak)
    cfg.spoof_max_score = float(cfg.fusion.spoof_strong)
    cfg.likely_spoof_max_score = float(cfg.fusion.spoof_weak)

    # Aliases for fusion-v2 thresholds.
    cfg.fusion_real_high = float(cfg.fusion.real_strong)
    cfg.fusion_real_low = float(cfg.fusion.real_weak)
    cfg.fusion_spoof_low = float(cfg.fusion.spoof_strong)
    cfg.fusion_spoof_high = float(cfg.fusion.spoof_weak)

    # Hysteresis frame counts for confirming strong REAL/SPOOF
    # (legacy; fusion-v2 may use time-based persistence instead).
    cfg.frames_to_confirm_real = 5
    cfg.frames_to_confirm_spoof = 5

    # ------------------------------------------------------------------
    # Motion cue thresholds (aligned with motion.py)
    # ------------------------------------------------------------------
    cfg.motion_min_span_factor = float(
        getattr(cfg.motion, "min_span_factor", 0.5)
    )
    cfg.motion_min_landmarks = int(
        getattr(cfg.motion, "min_landmarks", 3)
    )
    cfg.motion_min_motion_pixels_base = float(
        getattr(cfg.motion, "min_motion_pixels_base", 2.0)
    )
    cfg.motion_min_motion_frac = float(
        getattr(cfg.motion, "min_motion_frac", 0.01)
    )
    cfg.motion_yaw_deg_threshold = float(
        getattr(cfg.motion, "yaw_deg_threshold", 8.0)
    )
    cfg.motion_yaw_boost = float(
        getattr(cfg.motion, "yaw_boost", 0.15)
    )

    # Legacy compatibility: some places may still read cfg.min_motion_pixels.
    cfg.min_motion_pixels = float(cfg.motion_min_motion_pixels_base)

    # Parallax mapping bounds (used by motion.py).
    cfg.parallax_low = float(
        getattr(cfg.motion, "parallax_low", 0.08)
    )
    cfg.parallax_high = float(
        getattr(cfg.motion, "parallax_high", 0.22)
    )

    # Minimum face quality for motion cue – can be tied to face_cfg if present.
    if face_cfg is not None:
        # Duck-typed: we only rely on attributes, no imports.
        q_runtime = getattr(face_cfg, "q_runtime", None)
        if q_runtime is not None:
            try:
                cfg.motion_min_quality = float(q_runtime)
            except Exception:
                cfg.motion_min_quality = 0.55
        else:
            cfg.motion_min_quality = 0.55
    else:
        cfg.motion_min_quality = 0.55

    # ------------------------------------------------------------------
    # Screen cue: map structured config → flat attributes used by
    # screen_artifacts.py (and future SourceAuthEngine).
    # ------------------------------------------------------------------
    cfg.screen_min_frames = int(getattr(cfg.screen, "min_frames", 4))
    cfg.screen_min_face_px = int(getattr(cfg.screen, "min_face_px", 40))

    cfg.screen_border_margin_frac = float(
        getattr(cfg.screen, "border_margin_frac", 0.15)
    )

    cfg.screen_weight_border = float(getattr(cfg.screen, "w_border", 0.6))
    cfg.screen_weight_flicker = float(getattr(cfg.screen, "w_flicker", 0.2))
    cfg.screen_weight_moire = float(getattr(cfg.screen, "w_moire", 0.2))

    cfg.screen_flicker_var_norm = float(
        getattr(cfg.screen, "flicker_var_norm", 0.01)
    )
    cfg.screen_moire_norm = float(
        getattr(cfg.screen, "moire_norm", 300.0)
    )

    cfg.screen_rectangularity_angle_tol = float(
        getattr(cfg.screen, "rectangularity_angle_tol", 15.0)
    )
    cfg.screen_bezel_uniformity_norm = float(
        getattr(cfg.screen, "bezel_uniformity_norm", 40.0)
    )

    cfg.screen_border_reliable_min = float(
        getattr(cfg.screen, "border_reliable_min", 0.50)
    )
    cfg.screen_border_reliable_std_max = float(
        getattr(cfg.screen, "border_reliable_std_max", 0.15)
    )

    # ------------------------------------------------------------------
    # Background cue: map structured config → flat attributes used by
    # background.py and the engine.
    # ------------------------------------------------------------------
    cfg.background_min_face_px = int(
        getattr(cfg.background, "min_face_px", 40)
    )
    cfg.background_min_region_pixels = int(
        getattr(cfg.background, "min_region_pixels", 500)
    )
    cfg.background_color_hist_bins = int(
        getattr(cfg.background, "color_hist_bins", 32)
    )
    cfg.background_blur_sigma_for_noise = float(
        getattr(cfg.background, "blur_sigma_for_noise", 0.5)
    )

    cfg.background_diff_low = float(
        getattr(cfg.background, "diff_low", 0.15)
    )
    cfg.background_diff_high = float(
        getattr(cfg.background, "diff_high", 0.6)
    )

    cfg.background_enable_noise_sharpness_hint = bool(
        getattr(cfg.background, "enable_noise_sharpness_hint", True)
    )

    cfg.background_weight_color = float(
        getattr(cfg.background, "w_color", 0.7)
    )
    cfg.background_weight_texture = float(
        getattr(cfg.background, "w_texture", 0.3)
    )

    cfg.background_color_norm = float(
        getattr(cfg.background, "color_norm", 0.5)
    )
    cfg.background_texture_norm = float(
        getattr(cfg.background, "texture_norm", 50.0)
    )

    # ------------------------------------------------------------------
    # Apply any extra overrides passed via **kwargs (future-proofing).
    # This allows higher-level code to tweak top-level knobs without
    # changing this factory.
    # ------------------------------------------------------------------
    for key, value in kwargs.items():
        setattr(cfg, key, value)

    return cfg
