"""
face/config.py

Central configuration for the Phase-2A Face Route (2D + pseudo-3D).

This module does **not** run any heavy models; it only defines
typed configuration objects and helpers to build a sane default
FaceConfig using the existing device-selection logic.

Other modules (detector_align, route, gallery, identity, multiview, etc.)
should **only** depend on these dataclasses instead of hard-coding
paths, thresholds, or device strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Mapping, Any

import logging

from core.device import select_device

logger = logging.getLogger(__name__)

# Allowed identity engine modes for safety.
IDENTITY_MODES = ("classic", "multiview", "hybrid")


# ---------------------------------------------------------------------------
# Low-level configs
# ---------------------------------------------------------------------------


@dataclass
class FaceDeviceConfig:
    """
    Device configuration for all face models.

    Attributes
    ----------
    device : str
        Device string understood by PyTorch / InsightFace
        ("cuda", "cpu", "mps", "0", "0,1", ...).
    use_half : bool
        If True, models that support it may use FP16 on GPU.
        For safety we keep embeddings in FP32 and usually allow only
        detector / backbone parts to run in FP16 where safe.
    """

    device: str = "cpu"
    use_half: bool = False


@dataclass
class FaceModelConfig:
    """
    Model identifiers / paths for the face route.

    These are **names or paths** understood by InsightFace / your
    chosen backend. We keep them as strings so they can be either:

      - built-in InsightFace model packs (e.g. "buffalo_l"), or
      - explicit filesystem paths to ONNX / PyTorch weights.

    In the current design, `retinaface_name` is used as the *pack name*
    for InsightFace's FaceAnalysis (e.g. "buffalo_l"), which already
    includes detection + embedding. `arcface_name` is kept for future
    use if you ever want a separate embedding model.
    """

    # For the buffalo_l pipeline, this acts as the FaceAnalysis pack name.
    retinaface_name: str = "buffalo_l"

    # Kept for future extensibility; not used by buffalo_l-based flow.
    arcface_name: str = "arcface_r100_v1"


@dataclass
class FaceThresholdConfig:
    """
    All numeric thresholds used in the face pipeline.

    You can tune these later without touching model code.

    NOTE:
    - min_quality_enroll
        Quality gate used by the *enrollment* CLI. Samples below this
        are rejected and not added to the gallery.
    - min_quality_for_embed
        Quality gate for what enters the per-track buffer / gallery
        search during normal runtime.
    - min_quality_runtime
        Strict runtime evidence gate. Candidates below this are treated
        as "no evidence" by FaceRoute + IdentityEngine.
    - strong_match_dist / weak_match_dist
        Distance bands in gallery space for strong / weak / unknown.
      (We keep compatibility aliases max_match_distance / max_weak_match_distance.)

    These are global to both the classic 2D gallery and the 3D/multiview
    logic (the 3D side may add extra constraints on top of these).
    """

    # ----- geometric / size thresholds ----- #
    # Minimum detected face height (in pixels) for a candidate face.
    min_face_height_px: int = 40  # discard very tiny faces

    # Person bbox (from Phase-1) must be at least this tall before we even
    # try to run face detection to avoid wasting compute on tiny people.
    min_box_height_for_face_px: int = 80

    # ----- detector / quality thresholds ----- #
    # Raw InsightFace detection confidence.
    min_det_score: float = 0.6

    # Quality gate specifically for enrollment via CLI.
    min_quality_enroll: float = 0.60

    # Final q_face in [0,1] required before we accept a face embedding
    # into the per-track buffer / gallery search during runtime.
    min_quality_for_embed: float = 0.60

    # Runtime quality gate for identity updates.
    min_quality_runtime: float = 0.55

    # Pose limits (beyond these angles, quality is strongly penalised).
    max_yaw_deg: float = 40.0   # left/right turn
    max_pitch_deg: float = 30.0 # up/down tilt

    # Blur / sharpness heuristic (Laplacian-of-variance).
    min_sharpness: float = 0.25

    # ----- gallery / matching thresholds ----- #
    # Distances here refer to **cosine distance** if you use cos similarity,
    # or L2 distances if you choose that metric.
    #
    # In the default setup we use cosine similarity via FAISS IndexFlatIP,
    # and convert it to a "distance" for thresholds: d = 1 - sim.
    strong_match_dist: float = 0.85       # below → strong match
    weak_match_dist: float = 0.93         # reserved band for weak candidates
    # everything above weak_match_dist is treated as UNKNOWN

    # ---- stability parameters (top-level, YAML compatible) ----
    # These are duplicated in FaceSmoothingConfig but kept here to allow
    # simple wiring from YAML where they live under `face.*`.
    min_samples_confirm: int = 3
    min_samples_switch: int = 4
    evidence_lookback_sec: float = 2.0

    # Backwards-compatibility aliases for older code that still expects
    # max_match_distance / max_weak_match_distance.
    @property
    def max_match_distance(self) -> float:  # pragma: no cover - simple alias
        return self.strong_match_dist

    @property
    def max_weak_match_distance(self) -> float:  # pragma: no cover
        return self.weak_match_dist


@dataclass
class FaceRouteConfig:
    """
    Behaviour of the FaceRoute (how often and how far back we look).

    Attributes
    ----------
    lookback_seconds : float
        Time window (in seconds) to search for the best recent face
        per track. Typical value: 1–2 seconds.
    max_entries_per_track : int
        Hard cap on how many face entries we keep for one track.
    process_every_n_frames : int
        Run heavy face detection at most once every N frames per track.
    min_interval_ms : int
        Minimum time between heavy face passes for the same track.
        This protects the GPU when FPS is high.
    """

    lookback_seconds: float = 2.0
    max_entries_per_track: int = 30
    process_every_n_frames: int = 5
    min_interval_ms: int = 150

    # Backwards-compat alias for older field name used in some drafts.
    @property
    def max_seconds_lookback(self) -> float:  # pragma: no cover
        return self.lookback_seconds


@dataclass
class FaceSmoothingConfig:
    """
    Temporal smoothing / decay behaviour for identity decisions.

    These parameters control how IdentityEngine treats evidence over time.
    """

    # Exponential decay half-life (seconds).
    half_life_sec: float = 2.0

    # After this idle time (no good evidence) identity is considered stale.
    stale_after_sec: float = 6.0

    # Below this confidence, we treat the track as effectively unknown.
    min_confidence: float = 0.05

    # Evidence requirements in a temporal window.
    min_samples_confirm: int = 3
    min_samples_switch: int = 4
    evidence_lookback_sec: float = 2.0


@dataclass
class FaceGalleryConfig:
    """
    Configuration for the face gallery (FAISS + encrypted storage).

    Attributes
    ----------
    dim : int
        Embedding dimensionality (ArcFace / buffalo_l is usually 512).
    metric : str
        Similarity metric for FAISS ("cosine" or "l2").
    gallery_path : Path
        Path to the encrypted gallery file on disk.
    encryption_key_env : str
        Name of the environment variable containing the AES key
        used by crypto.py.
    """

    dim: int = 512
    metric: str = "cosine"
    gallery_path: Path = Path("data/face_gallery.enc")
    encryption_key_env: str = "GAITGUARD_FACE_KEY"


# ---------------------------------------------------------------------------
# Multi-view / pseudo-3D config (for head logic & coverage)
# ---------------------------------------------------------------------------


@dataclass
class FaceMultiViewConfig:
    """
    Configuration for the pseudo-3D / multi-view head representation.

    These parameters do **not** change how embeddings are computed.
    They only control how we:
      - assign pose bins (FRONT/LEFT/RIGHT/UP/DOWN/OCCLUDED/UNKNOWN),
      - limit per-bin templates,
      - build centroids and coverage metrics.

    All of this is used by multiview_builder + multiview_gallery_view +
    multiview_matcher, and should be independent from the classic 2D
    gallery logic.
    """

    # Pose bin boundaries (in degrees).
    # FRONT: |yaw| <= yaw_front_deg
    yaw_front_deg: float = 15.0

    # SIDE: (yaw_front_deg, yaw_side_max_deg] maps to LEFT/RIGHT by sign.
    yaw_side_max_deg: float = 45.0

    # UP/DOWN bins based on pitch.
    pitch_up_deg: float = 20.0     # above this → UP bin
    pitch_down_deg: float = -20.0  # below this → DOWN bin

    # Beyond this roll, we may consider the sample unstable / occluded.
    max_roll_deg: float = 35.0

    # Max number of raw templates we keep per bin per person before
    # we start pruning by quality / recency.
    max_templates_per_bin: int = 10

    # Max number of canonical bins we consider "core" for coverage.
    # Usually FRONT, LEFT, RIGHT, UP, DOWN, OCCLUDED (6 total).
    num_canonical_bins: int = 6

    # If true, multiview diagnostics will treat UNKNOWN-only persons
    # as "coverage=0", even if they have many samples.
    unknown_only_is_zero_coverage: bool = True


# ---------------------------------------------------------------------------
# Aggregate top-level config
# ---------------------------------------------------------------------------


@dataclass
class FaceConfig:
    """
    Aggregated configuration object for the entire face route.

    This is what higher-level components (FaceRoute, FaceGallery,
    FaceIdentityEngine, Multiview* modules) should receive in
    their constructors.

    identity_mode determines which identity engine variant is active:
      - "classic"   → classic 2D gallery / engine only
      - "multiview" → pose-aware pseudo-3D engine only
      - "hybrid"    → both in parallel (for comparison / diagnostics)
    """

    device: FaceDeviceConfig
    models: FaceModelConfig
    thresholds: FaceThresholdConfig
    route: FaceRouteConfig
    smoothing: FaceSmoothingConfig
    gallery: FaceGalleryConfig
    # Configuration dedicated to multi-view / pseudo-3D logic.
    multiview: FaceMultiViewConfig = field(default_factory=FaceMultiViewConfig)
    # High-level switch for identity engine wiring.
    identity_mode: str = "classic"


# ---------------------------------------------------------------------------
# Factory for a sane default configuration
# ---------------------------------------------------------------------------


def default_face_config(
    prefer_gpu: bool = True,
    base_dir: Optional[Path] = None,
    face_section: Optional[Mapping[str, Any]] = None,
) -> FaceConfig:
    """
    Build a default FaceConfig.

    Parameters
    ----------
    prefer_gpu : bool
        If True, try to use CUDA (or other accelerator) when available.
        Falls back to CPU automatically.
    base_dir : Optional[Path]
        Base directory for derived paths (e.g. gallery file). If None,
        the current working directory is used.
    face_section : Optional[Mapping[str, Any]]
        Optional dictionary coming from config/default.yaml under the
        `face:` key. If provided, values here override the default
        dataclass values (thresholds, route, smoothing, gallery dims,
        multiview binning, etc.).

        Example (YAML → dict):

            face:
              identity_mode: "multiview"
              min_quality_enroll: 0.60
              min_quality_for_embed: 0.60
              min_quality_runtime: 0.55
              strong_match_dist: 0.85
              weak_match_dist: 0.93
              route:
                lookback_seconds: 2.0
                max_entries_per_track: 30
                process_every_n_frames: 5
                min_interval_ms: 150
              smoothing:
                half_life_sec: 2.0
                stale_after_sec: 6.0
              multiview:
                yaw_front_deg: 15.0
                yaw_side_max_deg: 45.0

    Returns
    -------
    FaceConfig
        Ready-to-use configuration object.
    """
    # Decide on device (reuse the same logic as the rest of the system).
    device_str, use_half = select_device(prefer_gpu=prefer_gpu)
    device_cfg = FaceDeviceConfig(device=device_str, use_half=use_half)

    # Model configuration – single InsightFace pack (buffalo_l).
    model_cfg = FaceModelConfig(
        retinaface_name="buffalo_l",
        arcface_name="arcface_r100_v1",
    )

    thresholds_cfg = FaceThresholdConfig()
    route_cfg = FaceRouteConfig()
    smoothing_cfg = FaceSmoothingConfig()
    multiview_cfg = FaceMultiViewConfig()

    # Identity mode default (safe).
    identity_mode: str = "classic"

    # Resolve gallery path relative to base_dir (or CWD).
    if base_dir is None:
        base_dir = Path.cwd()
    else:
        base_dir = Path(base_dir).resolve()

    gallery_path = (base_dir / "data" / "face_gallery.enc").resolve()
    gallery_cfg = FaceGalleryConfig(
        dim=512,
        metric="cosine",
        gallery_path=gallery_path,
        encryption_key_env="GAITGUARD_FACE_KEY",
    )

    # ------------------------------------------------------------------
    # Optional: override defaults from YAML `face:` section.
    # ------------------------------------------------------------------
    if face_section is not None:
        # identity_mode (classic / multiview / hybrid)
        if "identity_mode" in face_section:
            raw_mode = str(face_section["identity_mode"]).strip().lower()
            if raw_mode in IDENTITY_MODES:
                identity_mode = raw_mode
            else:
                logger.warning(
                    "FaceConfig: invalid identity_mode '%s' in config; "
                    "falling back to 'classic'. Allowed: %s",
                    raw_mode,
                    ", ".join(IDENTITY_MODES),
                )

        # Flat threshold values.
        if "min_quality_enroll" in face_section:
            thresholds_cfg.min_quality_enroll = float(face_section["min_quality_enroll"])
        if "min_quality_for_embed" in face_section:
            thresholds_cfg.min_quality_for_embed = float(face_section["min_quality_for_embed"])
        if "min_quality_runtime" in face_section:
            thresholds_cfg.min_quality_runtime = float(face_section["min_quality_runtime"])

        if "strong_match_dist" in face_section:
            thresholds_cfg.strong_match_dist = float(face_section["strong_match_dist"])
        if "weak_match_dist" in face_section:
            thresholds_cfg.weak_match_dist = float(face_section["weak_match_dist"])

        if "min_samples_confirm" in face_section:
            thresholds_cfg.min_samples_confirm = int(face_section["min_samples_confirm"])
        if "min_samples_switch" in face_section:
            thresholds_cfg.min_samples_switch = int(face_section["min_samples_switch"])
        if "evidence_lookback_sec" in face_section:
            thresholds_cfg.evidence_lookback_sec = float(face_section["evidence_lookback_sec"])

        # Gallery dims / metric (optional).
        if "dim" in face_section:
            gallery_cfg.dim = int(face_section["dim"])
        if "metric" in face_section:
            gallery_cfg.metric = str(face_section["metric"])

        # Route sub-section.
        route_sec = face_section.get("route")
        if isinstance(route_sec, Mapping):
            if "lookback_seconds" in route_sec:
                route_cfg.lookback_seconds = float(route_sec["lookback_seconds"])
            if "max_entries_per_track" in route_sec:
                route_cfg.max_entries_per_track = int(route_sec["max_entries_per_track"])
            if "process_every_n_frames" in route_sec:
                route_cfg.process_every_n_frames = int(route_sec["process_every_n_frames"])
            if "min_interval_ms" in route_sec:
                route_cfg.min_interval_ms = int(route_sec["min_interval_ms"])

        # Smoothing sub-section.
        smooth_sec = face_section.get("smoothing")
        if isinstance(smooth_sec, Mapping):
            if "half_life_sec" in smooth_sec:
                smoothing_cfg.half_life_sec = float(smooth_sec["half_life_sec"])
            if "stale_after_sec" in smooth_sec:
                smoothing_cfg.stale_after_sec = float(smooth_sec["stale_after_sec"])
            if "min_confidence" in smooth_sec:
                smoothing_cfg.min_confidence = float(smooth_sec["min_confidence"])
            if "min_samples_confirm" in smooth_sec:
                smoothing_cfg.min_samples_confirm = int(smooth_sec["min_samples_confirm"])
            if "min_samples_switch" in smooth_sec:
                smoothing_cfg.min_samples_switch = int(smooth_sec["min_samples_switch"])
            if "evidence_lookback_sec" in smooth_sec:
                smoothing_cfg.evidence_lookback_sec = float(smooth_sec["evidence_lookback_sec"])

        # Multiview sub-section (for 3D head).
        mv_sec = face_section.get("multiview")
        if isinstance(mv_sec, Mapping):
            if "yaw_front_deg" in mv_sec:
                multiview_cfg.yaw_front_deg = float(mv_sec["yaw_front_deg"])
            if "yaw_side_max_deg" in mv_sec:
                multiview_cfg.yaw_side_max_deg = float(mv_sec["yaw_side_max_deg"])
            if "pitch_up_deg" in mv_sec:
                multiview_cfg.pitch_up_deg = float(mv_sec["pitch_up_deg"])
            if "pitch_down_deg" in mv_sec:
                multiview_cfg.pitch_down_deg = float(mv_sec["pitch_down_deg"])
            if "max_roll_deg" in mv_sec:
                multiview_cfg.max_roll_deg = float(mv_sec["max_roll_deg"])
            if "max_templates_per_bin" in mv_sec:
                multiview_cfg.max_templates_per_bin = int(mv_sec["max_templates_per_bin"])
            if "num_canonical_bins" in mv_sec:
                multiview_cfg.num_canonical_bins = int(mv_sec["num_canonical_bins"])
            if "unknown_only_is_zero_coverage" in mv_sec:
                multiview_cfg.unknown_only_is_zero_coverage = bool(
                    mv_sec["unknown_only_is_zero_coverage"]
                )

    cfg = FaceConfig(
        device=device_cfg,
        models=model_cfg,
        thresholds=thresholds_cfg,
        route=route_cfg,
        smoothing=smoothing_cfg,
        gallery=gallery_cfg,
        multiview=multiview_cfg,
        identity_mode=identity_mode,
    )

    logger.info(
        "FaceConfig initialised | device=%s half=%s | gallery=%s | dim=%d metric=%s | "
        "q_enroll=%.2f q_embed=%.2f q_runtime=%.2f strong_dist=%.3f weak_dist=%.3f | "
        "route(lookback=%.1fs, max_entries=%d, every_n=%d, min_interval_ms=%d) | "
        "multiview(yaw_front=%g, yaw_side_max=%g, pitch_up=%g, pitch_down=%g) | "
        "identity_mode=%s",
        cfg.device.device,
        cfg.device.use_half,
        cfg.gallery.gallery_path,
        cfg.gallery.dim,
        cfg.gallery.metric,
        cfg.thresholds.min_quality_enroll,
        cfg.thresholds.min_quality_for_embed,
        cfg.thresholds.min_quality_runtime,
        cfg.thresholds.strong_match_dist,
        cfg.thresholds.weak_match_dist,
        cfg.route.lookback_seconds,
        cfg.route.max_entries_per_track,
        cfg.route.process_every_n_frames,
        cfg.route.min_interval_ms,
        cfg.multiview.yaw_front_deg,
        cfg.multiview.yaw_side_max_deg,
        cfg.multiview.pitch_up_deg,
        cfg.multiview.pitch_down_deg,
        cfg.identity_mode,
    )

    return cfg
