
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Mapping, Any

import logging

from core.device import select_device

logger = logging.getLogger(__name__)

IDENTITY_MODES = ("classic", "multiview", "hybrid")


@dataclass
class FaceDeviceConfig:

    device: str = "cpu"
    use_half: bool = False


@dataclass
class FaceModelConfig:

    retinaface_name: str = "buffalo_l"

    arcface_name: str = "arcface_r100_v1"


@dataclass
class FaceThresholdConfig:

    min_face_height_px: int = 40

    min_box_height_for_face_px: int = 80

    min_det_score: float = 0.6

    min_quality_enroll: float = 0.60

    min_quality_for_embed: float = 0.60

    min_quality_runtime: float = 0.55

    max_yaw_deg: float = 40.0
    max_pitch_deg: float = 30.0

    min_sharpness: float = 0.25

    strong_match_dist: float = 0.85
    weak_match_dist: float = 0.93

    min_samples_confirm: int = 3
    min_samples_switch: int = 4
    evidence_lookback_sec: float = 2.0

    @property
    def max_match_distance(self) -> float:
        return self.strong_match_dist

    @property
    def max_weak_match_distance(self) -> float:
        return self.weak_match_dist


@dataclass
class FaceRouteConfig:

    lookback_seconds: float = 2.0
    max_entries_per_track: int = 30
    process_every_n_frames: int = 5
    min_interval_ms: int = 150

    @property
    def max_seconds_lookback(self) -> float:
        return self.lookback_seconds


@dataclass
class FaceSmoothingConfig:

    half_life_sec: float = 2.0

    stale_after_sec: float = 6.0

    min_confidence: float = 0.05

    min_samples_confirm: int = 3
    min_samples_switch: int = 4
    evidence_lookback_sec: float = 2.0


@dataclass
class FaceGalleryConfig:

    dim: int = 512
    metric: str = "cosine"
    gallery_path: Path = Path("data/face_gallery.enc")
    encryption_key_env: str = "GAITGUARD_FACE_KEY"


@dataclass
class FaceMultiViewConfig:

    yaw_front_deg: float = 15.0

    yaw_side_max_deg: float = 45.0

    pitch_up_deg: float = 20.0
    pitch_down_deg: float = -20.0

    max_roll_deg: float = 35.0

    max_templates_per_bin: int = 10

    num_canonical_bins: int = 6

    unknown_only_is_zero_coverage: bool = True


@dataclass
class FaceConfig:

    device: FaceDeviceConfig
    models: FaceModelConfig
    thresholds: FaceThresholdConfig
    route: FaceRouteConfig
    smoothing: FaceSmoothingConfig
    gallery: FaceGalleryConfig
    multiview: FaceMultiViewConfig = field(default_factory=FaceMultiViewConfig)
    identity_mode: str = "classic"


def default_face_config(
    prefer_gpu: bool = True,
    base_dir: Optional[Path] = None,
    face_section: Optional[Mapping[str, Any]] = None,
) -> FaceConfig:
    device_str, use_half = select_device(prefer_gpu=prefer_gpu)
    device_cfg = FaceDeviceConfig(device=device_str, use_half=use_half)

    model_cfg = FaceModelConfig(
        retinaface_name="buffalo_l",
        arcface_name="arcface_r100_v1",
    )

    thresholds_cfg = FaceThresholdConfig()
    route_cfg = FaceRouteConfig()
    smoothing_cfg = FaceSmoothingConfig()
    multiview_cfg = FaceMultiViewConfig()

    identity_mode: str = "classic"

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

    if face_section is not None:
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

        if "dim" in face_section:
            gallery_cfg.dim = int(face_section["dim"])
        if "metric" in face_section:
            gallery_cfg.metric = str(face_section["metric"])

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
