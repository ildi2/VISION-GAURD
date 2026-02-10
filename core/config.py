from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class PathsConfig:
    models_dir: str = "models"
    logs_dir: str = "logs"
    evidence_dir: str = "evidence"


@dataclass
class RuntimeConfig:
    use_gpu: bool = True
    save_evidence: bool = False
    log_face_metrics: bool = False
    metrics_window_sec: float = 5.0

    use_multiview_engine: bool = False


@dataclass
class UiConfig:
    show_identity_labels: bool = True
    show_debug_face_hud: bool = True
    show_fps: bool = True

    show_engine_tag: bool = False
    show_pose_tag: bool = False

    show_source_auth_tag: bool = True
    show_source_auth_border: bool = True


@dataclass
class IdentityRuntimeConfig:
    mode: str = "classic"


@dataclass
class EvidenceGateThresholds:
    unknown_min_quality: float = 0.55
    unknown_min_size_px: float = 80.0
    unknown_min_margin: float = 0.15
    confirmed_min_quality: float = 0.55
    confirmed_min_margin: float = 0.08
    max_yaw_unknown: float = 40.0
    max_yaw_confirmed: float = 60.0
    min_brightness_normalized: float = 0.2
    max_brightness_normalized: float = 0.9
    min_blur_score: float = 200.0


@dataclass
class EvidenceGateConfig:
    enabled: bool = True
    description: str = ""
    thresholds: EvidenceGateThresholds = field(default_factory=EvidenceGateThresholds)
    accept_ratio_target: float = 0.85
    hold_ratio_target: float = 0.10
    reject_ratio_target: float = 0.05
    log_reasons: bool = True
    log_level: str = "DEBUG"


@dataclass
class BindingConfirmationRules:
    min_samples_strong: int = 1
    min_samples_weak: int = 5
    window_seconds: float = 3.0
    min_avg_score: float = 0.75


@dataclass
class BindingSwitchingRules:
    min_sustained_samples: int = 4
    margin_advantage: float = 0.12
    window_seconds: float = 2.0


@dataclass
class BindingContradictionRules:
    threshold: float = 0.15
    counter_max: int = 5
    downgrade_factor: float = 0.8
    window_seconds: float = 5.0


@dataclass
class BindingStateConfig:
    enabled: bool = True
    description: str = ""
    confirmation: BindingConfirmationRules = field(default_factory=BindingConfirmationRules)
    switching: BindingSwitchingRules = field(default_factory=BindingSwitchingRules)
    contradiction: BindingContradictionRules = field(default_factory=BindingContradictionRules)
    stale_threshold_sec: float = 8.0
    stale_recovery_samples_needed: int = 2


@dataclass
class SchedulerBudget:
    max_faces_per_frame: int = 10
    max_faces_per_second: int = 30
    budget_mode: str = "time"


@dataclass
class SchedulerPriorityRules:
    unknown_pending_weight: float = 1.0
    expiring_weight: float = 0.9
    watchlist_weight: float = 1.2
    confirmed_refresh_weight: float = 0.3
    confirmed_strong_weight: float = 0.1


@dataclass
class SchedulerFairness:
    starvation_threshold_sec: float = 15.0
    starved_priority_boost: float = 2.0


@dataclass
class SchedulerConfig:
    enabled: bool = True
    description: str = ""
    budget: SchedulerBudget = field(default_factory=SchedulerBudget)
    priority_rules: SchedulerPriorityRules = field(default_factory=SchedulerPriorityRules)
    fairness: SchedulerFairness = field(default_factory=SchedulerFairness)


@dataclass
class MergeThresholds:
    max_spatial_distance_px: float = 100.0
    min_appearance_sim: float = 0.7
    min_embedding_sim: float = 0.80
    handoff_window_sec: float = 3.0


@dataclass
class MergeModes:
    conservative: bool = True
    loose_pending: bool = False


@dataclass
class MergeConfig:
    enabled: bool = True
    description: str = ""
    handoff_merge_enabled: bool = True
    simul_merge_enabled: bool = False
    thresholds: MergeThresholds = field(default_factory=MergeThresholds)
    merge_modes: MergeModes = field(default_factory=MergeModes)


@dataclass
class DebugUIToggles:
    show_binding_state: bool = True
    show_evidence_gate_reason: bool = True
    show_merge_alias: bool = True
    show_scheduler_budget: bool = False
    show_contradiction_counter: bool = False


@dataclass
class DebugConfig:
    evidence_gate_decisions: bool = True
    binding_state_transitions: bool = True
    merge_attempts: bool = True
    scheduler_selections: bool = True
    ui: DebugUIToggles = field(default_factory=DebugUIToggles)
    emit_metrics_every_sec: float = 1.0
    log_level: str = "INFO"


@dataclass
class GovernanceConfig:
    enabled: bool = True
    evidence_gate: EvidenceGateConfig = field(default_factory=EvidenceGateConfig)
    binding: BindingStateConfig = field(default_factory=BindingStateConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


@dataclass
class ContinuityConfig:
    min_track_age_frames: int = 10
    min_track_confidence: float = 0.3
    max_lost_frames: int = 5
    appearance_distance_threshold: float = 0.35
    appearance_safe_zone_frames: int = 30
    appearance_ema_alpha: float = 0.1
    max_bbox_displacement_fraction: float = 0.25
    max_bbox_displacement_px: int = 600
    bbox_iou_threshold: float = 0.3
    max_face_contradiction_count: int = 3
    grace_window_seconds: float = 1.0
    grace_bbox_displacement_fraction: float = 0.25
    shadow_mode: bool = False
    shadow_metrics_log_interval_sec: float = 30.0


@dataclass
class PrivacyOutputConfig:
    dir: str = "privacy_output"
    basename: str = "privacy_stream"
    codec: str = "mp4v"
    container: str = "mp4"
    fps: int = 30


@dataclass
class PrivacyPolicyConfig:
    grace_sec: float = 5.0
    reacquire_sec: float = 10.0
    unlock_allowed: bool = False
    
    authorized_categories: List[str] = field(default_factory=lambda: ["resident"])
    require_confirmed_binding: bool = True
    
    grace_reacquire_sec: float = 2.0
    cooldown_after_end_sec: float = 1.0


@dataclass
class PrivacyAuditConfig:
    enabled: bool = True
    filename: str = "privacy_audit.jsonl"
    flush_interval_sec: float = 1.0


@dataclass
class PrivacyUiConfig:
    show_privacy_preview: bool = True
    preview_window_title: str = "Privacy Output (Delayed)"
    show_preview_watermark: bool = True


@dataclass
class PrivacySegmentationConfig:
    enabled: bool = False
    backend: str = "none"
    model_path: str = ""
    imgsz: int = 640
    conf: float = 0.35
    iou_threshold: float = 0.5
    iou_assoc_min: float = 0.25
    run_every_n_frames: int = 1
    
    min_mask_area_ratio: float = 0.01
    max_mask_area_ratio: float = 5.0
    min_mask_quality: float = 0.3
    
    dilate_mask_px: int = 3
    
    device: str = "auto"


@dataclass
class PrivacyStabilizationConfig:
    enabled: bool = False
    method: str = "union_only"
    history_size: int = 5
    mask_ttl_sec: float = 0.75
    max_shrink_ratio: float = 0.85
    morph_close_px: int = 3
    morph_open_px: int = 0


@dataclass
class PrivacyMetricsConfig:
    enabled: bool = True
    log_path: str = "privacy_output/privacy_metrics.jsonl"
    flush_interval_sec: float = 1.0
    
    leakage_enabled: bool = True
    leakage_backend: str = "opencv_haar"
    haar_scale_factor: float = 1.1
    haar_min_neighbors: int = 5
    overlap_threshold: float = 0.2
    
    flicker_enabled: bool = True
    timing_enabled: bool = True
    utility_enabled: bool = True


@dataclass
class PrivacyConfig:
    enabled: bool = False
    delay_sec: float = 3.0
    redaction_style: str = "blur"
    silhouette_cleanup: bool = False
    output: PrivacyOutputConfig = field(default_factory=PrivacyOutputConfig)
    policy: PrivacyPolicyConfig = field(default_factory=PrivacyPolicyConfig)
    audit: PrivacyAuditConfig = field(default_factory=PrivacyAuditConfig)
    ui: PrivacyUiConfig = field(default_factory=PrivacyUiConfig)
    segmentation: PrivacySegmentationConfig = field(default_factory=PrivacySegmentationConfig)
    stabilization: PrivacyStabilizationConfig = field(default_factory=PrivacyStabilizationConfig)
    metrics: PrivacyMetricsConfig = field(default_factory=PrivacyMetricsConfig)


@dataclass
class VisionIdentityConfig:
    mode: str = "classic"
    continuity: ContinuityConfig = field(default_factory=ContinuityConfig)


@dataclass
class Config:
    camera: CameraConfig
    paths: PathsConfig
    runtime: RuntimeConfig
    ui: UiConfig
    identity: IdentityRuntimeConfig = field(default_factory=IdentityRuntimeConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    vision_identity: VisionIdentityConfig = field(default_factory=VisionIdentityConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)


def _update_dataclass_from_dict(obj: Any, data: Dict[str, Any]) -> Any:
    for k, v in data.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj


def load_config(path: str | Path = "config/default.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a dict, got: {type(raw)}")

    cam_data = raw.get("camera", {}) or {}
    paths_data = raw.get("paths", {}) or {}
    runtime_data = raw.get("runtime", {}) or {}
    ui_data = raw.get("ui", {}) or {}
    identity_data = raw.get("identity", {}) or {}
    governance_data = raw.get("governance", {}) or {}
    vision_identity_data = raw.get("vision_identity", {}) or {}
    privacy_data = raw.get("privacy", {}) or {}

    camera = _update_dataclass_from_dict(CameraConfig(), cam_data)
    paths = _update_dataclass_from_dict(PathsConfig(), paths_data)
    runtime = _update_dataclass_from_dict(RuntimeConfig(), runtime_data)
    ui = _update_dataclass_from_dict(UiConfig(), ui_data)
    identity = _update_dataclass_from_dict(IdentityRuntimeConfig(), identity_data)
    
    governance = _parse_governance_config(governance_data)
    
    vision_identity = _parse_vision_identity_config(vision_identity_data)
    
    privacy = _parse_privacy_config(privacy_data)

    cfg = Config(
        camera=camera,
        paths=paths,
        runtime=runtime,
        ui=ui,
        identity=identity,
        governance=governance,
        vision_identity=vision_identity,
        privacy=privacy,
    )

    logger.info(
        "Config loaded from %s | camera index=%d, runtime.use_gpu=%s, "
        "runtime.use_multiview_engine=%s, identity.mode=%s, governance.enabled=%s, privacy.enabled=%s",
        path,
        cfg.camera.index,
        cfg.runtime.use_gpu,
        cfg.runtime.use_multiview_engine,
        cfg.identity.mode,
        cfg.governance.enabled,
        cfg.privacy.enabled,
    )
    
    if cfg.governance.enabled:
        logger.info(
            "Governance layers: evidence_gate=%s, binding=%s, scheduler=%s, merge=%s",
            cfg.governance.evidence_gate.enabled,
            cfg.governance.binding.enabled,
            cfg.governance.scheduler.enabled,
            cfg.governance.merge.enabled,
        )

    return cfg


def _parse_governance_config(data: Dict[str, Any]) -> GovernanceConfig:
    def safe_update(obj: Any, data: Dict[str, Any]) -> Any:
        for k, v in data.items():
            if hasattr(obj, k):
                attr_type = type(getattr(obj, k))
                if v is not None and hasattr(attr_type, '__dataclass_fields__'):
                    nested_obj = getattr(obj, k)
                    safe_update(nested_obj, v)
                else:
                    setattr(obj, k, v)
        return obj
    
    gov = GovernanceConfig()
    
    if "evidence_gate" in data:
        gate_data = data["evidence_gate"] or {}
        gov.evidence_gate = GovernanceConfig().evidence_gate
        gate = gov.evidence_gate
        gate.enabled = gate_data.get("enabled", gate.enabled)
        gate.description = gate_data.get("description", gate.description)
        gate.accept_ratio_target = gate_data.get("accept_ratio_target", gate.accept_ratio_target)
        gate.hold_ratio_target = gate_data.get("hold_ratio_target", gate.hold_ratio_target)
        gate.reject_ratio_target = gate_data.get("reject_ratio_target", gate.reject_ratio_target)
        gate.log_reasons = gate_data.get("log_reasons", gate.log_reasons)
        gate.log_level = gate_data.get("log_level", gate.log_level)
        
        if "thresholds" in gate_data:
            thresholds_data = gate_data["thresholds"] or {}
            gate.thresholds = safe_update(gate.thresholds, thresholds_data)
    
    if "binding" in data:
        binding_data = data["binding"] or {}
        gov.binding = GovernanceConfig().binding
        binding = gov.binding
        binding.enabled = binding_data.get("enabled", binding.enabled)
        if "confirmation" in binding_data:
            binding.confirmation = safe_update(binding.confirmation, binding_data["confirmation"] or {})
        if "switching" in binding_data:
            binding.switching = safe_update(binding.switching, binding_data["switching"] or {})
        if "contradiction" in binding_data:
            binding.contradiction = safe_update(binding.contradiction, binding_data["contradiction"] or {})
        binding.stale_threshold_sec = binding_data.get("stale_threshold_sec", binding.stale_threshold_sec)
        binding.stale_recovery_samples_needed = binding_data.get("stale_recovery_samples_needed", 
                                                                  binding.stale_recovery_samples_needed)
    
    if "scheduler" in data:
        scheduler_data = data["scheduler"] or {}
        gov.scheduler = GovernanceConfig().scheduler
        scheduler = gov.scheduler
        scheduler.enabled = scheduler_data.get("enabled", scheduler.enabled)
        if "budget" in scheduler_data:
            scheduler.budget = safe_update(scheduler.budget, scheduler_data["budget"] or {})
        if "priority_rules" in scheduler_data:
            scheduler.priority_rules = safe_update(scheduler.priority_rules, scheduler_data["priority_rules"] or {})
        if "fairness" in scheduler_data:
            scheduler.fairness = safe_update(scheduler.fairness, scheduler_data["fairness"] or {})
    
    if "merge" in data:
        merge_data = data["merge"] or {}
        gov.merge = GovernanceConfig().merge
        merge = gov.merge
        merge.enabled = merge_data.get("enabled", merge.enabled)
        merge.handoff_merge_enabled = merge_data.get("handoff_merge_enabled", merge.handoff_merge_enabled)
        merge.simul_merge_enabled = merge_data.get("simul_merge_enabled", merge.simul_merge_enabled)
        if "thresholds" in merge_data:
            merge.thresholds = safe_update(merge.thresholds, merge_data["thresholds"] or {})
        if "merge_modes" in merge_data:
            merge.merge_modes = safe_update(merge.merge_modes, merge_data["merge_modes"] or {})
    
    if "debug" in data:
        debug_data = data["debug"] or {}
        gov.debug = GovernanceConfig().debug
        debug = gov.debug
        debug.evidence_gate_decisions = debug_data.get("evidence_gate_decisions", debug.evidence_gate_decisions)
        debug.binding_state_transitions = debug_data.get("binding_state_transitions", debug.binding_state_transitions)
        debug.merge_attempts = debug_data.get("merge_attempts", debug.merge_attempts)
        debug.scheduler_selections = debug_data.get("scheduler_selections", debug.scheduler_selections)
        debug.emit_metrics_every_sec = debug_data.get("emit_metrics_every_sec", debug.emit_metrics_every_sec)
        debug.log_level = debug_data.get("log_level", debug.log_level)
        if "ui" in debug_data:
            debug.ui = safe_update(debug.ui, debug_data["ui"] or {})
    
    gov.enabled = data.get("enabled", gov.enabled)
    return gov

def _parse_vision_identity_config(data: Dict[str, Any]) -> VisionIdentityConfig:
    vision_identity = VisionIdentityConfig()
    
    vision_identity.mode = data.get("mode", vision_identity.mode)
    
    if "continuity" in data:
        continuity_data = data["continuity"] or {}
        cont = vision_identity.continuity
        
        cont.min_track_age_frames = continuity_data.get("min_track_age_frames", cont.min_track_age_frames)
        
        cont.min_track_confidence = continuity_data.get("min_track_confidence", cont.min_track_confidence)
        cont.max_lost_frames = continuity_data.get("max_lost_frames", cont.max_lost_frames)
        
        cont.appearance_distance_threshold = continuity_data.get("appearance_distance_threshold", 
                                                                   cont.appearance_distance_threshold)
        cont.appearance_safe_zone_frames = continuity_data.get("appearance_safe_zone_frames", 
                                                                 cont.appearance_safe_zone_frames)
        cont.appearance_ema_alpha = continuity_data.get("appearance_ema_alpha", cont.appearance_ema_alpha)
        
        cont.max_bbox_displacement_fraction = continuity_data.get("max_bbox_displacement_fraction", 
                                                                    cont.max_bbox_displacement_fraction)
        cont.max_bbox_displacement_px = continuity_data.get("max_bbox_displacement_px", 
                                                              cont.max_bbox_displacement_px)
        cont.bbox_iou_threshold = continuity_data.get("bbox_iou_threshold", cont.bbox_iou_threshold)
        
        cont.max_face_contradiction_count = continuity_data.get("max_face_contradiction_count", 
                                                                  cont.max_face_contradiction_count)
        
        cont.grace_window_seconds = continuity_data.get("grace_window_seconds", cont.grace_window_seconds)
        cont.grace_bbox_displacement_fraction = continuity_data.get("grace_bbox_displacement_fraction", 
                                                                      cont.grace_bbox_displacement_fraction)
        
        cont.shadow_mode = continuity_data.get("shadow_mode", cont.shadow_mode)
        cont.shadow_metrics_log_interval_sec = continuity_data.get("shadow_metrics_log_interval_sec", 
                                                                     cont.shadow_metrics_log_interval_sec)
    
    return vision_identity


def _parse_privacy_config(data: Dict[str, Any]) -> PrivacyConfig:
    privacy = PrivacyConfig()
    
    privacy.enabled = data.get("enabled", privacy.enabled)
    privacy.delay_sec = data.get("delay_sec", privacy.delay_sec)
    privacy.redaction_style = data.get("redaction_style", privacy.redaction_style)
    privacy.silhouette_cleanup = data.get("silhouette_cleanup", privacy.silhouette_cleanup)
    
    if "output" in data:
        output_data = data["output"] or {}
        out = privacy.output
        out.dir = output_data.get("dir", out.dir)
        out.basename = output_data.get("basename", out.basename)
        out.codec = output_data.get("codec", out.codec)
        out.container = output_data.get("container", out.container)
        out.fps = output_data.get("fps", out.fps)
    
    if "policy" in data:
        policy_data = data["policy"] or {}
        pol = privacy.policy
        pol.grace_reacquire_sec = policy_data.get("grace_reacquire_sec", pol.grace_reacquire_sec)
        pol.cooldown_after_end_sec = policy_data.get("cooldown_after_end_sec", pol.cooldown_after_end_sec)
        pol.authorized_categories = policy_data.get("authorized_categories", pol.authorized_categories)
        pol.require_confirmed_binding = policy_data.get("require_confirmed_binding", pol.require_confirmed_binding)
    
    if "audit" in data:
        audit_data = data["audit"] or {}
        aud = privacy.audit
        aud.enabled = audit_data.get("enabled", aud.enabled)
        aud.filename = audit_data.get("filename", aud.filename)
        aud.flush_interval_sec = audit_data.get("flush_interval_sec", aud.flush_interval_sec)
    
    if "ui" in data:
        ui_data = data["ui"] or {}
        ui = privacy.ui
        ui.show_privacy_preview = ui_data.get("show_privacy_preview", ui.show_privacy_preview)
        ui.preview_window_title = ui_data.get("preview_window_title", ui.preview_window_title)
        ui.show_preview_watermark = ui_data.get("show_preview_watermark", ui.show_preview_watermark)
    
    if "segmentation" in data:
        seg_data = data["segmentation"] or {}
        seg = privacy.segmentation
        seg.enabled = seg_data.get("enabled", seg.enabled)
        seg.backend = seg_data.get("backend", seg.backend)
        seg.model_path = seg_data.get("model_path", seg.model_path)
        seg.imgsz = seg_data.get("imgsz", seg.imgsz)
        seg.conf = seg_data.get("conf", seg.conf)
        seg.iou_threshold = seg_data.get("iou_threshold", seg.iou_threshold)
        seg.iou_assoc_min = seg_data.get("iou_assoc_min", seg.iou_assoc_min)
        seg.run_every_n_frames = seg_data.get("run_every_n_frames", seg.run_every_n_frames)
        seg.min_mask_area_ratio = seg_data.get("min_mask_area_ratio", seg.min_mask_area_ratio)
        seg.max_mask_area_ratio = seg_data.get("max_mask_area_ratio", seg.max_mask_area_ratio)
        seg.min_mask_quality = seg_data.get("min_mask_quality", seg.min_mask_quality)
        seg.dilate_mask_px = seg_data.get("dilate_mask_px", seg.dilate_mask_px)
        seg.device = seg_data.get("device", seg.device)
    
    if "stabilization" in data:
        stab_data = data["stabilization"] or {}
        stab = privacy.stabilization
        stab.enabled = stab_data.get("enabled", stab.enabled)
        stab.method = stab_data.get("method", stab.method)
        stab.history_size = stab_data.get("history_size", stab.history_size)
        stab.mask_ttl_sec = stab_data.get("mask_ttl_sec", stab.mask_ttl_sec)
        stab.max_shrink_ratio = stab_data.get("max_shrink_ratio", stab.max_shrink_ratio)
        stab.morph_close_px = stab_data.get("morph_close_px", stab.morph_close_px)
        stab.morph_open_px = stab_data.get("morph_open_px", stab.morph_open_px)
    
    if "metrics" in data:
        metrics_data = data["metrics"] or {}
        met = privacy.metrics
        met.enabled = metrics_data.get("enabled", met.enabled)
        met.log_path = metrics_data.get("log_path", met.log_path)
        met.flush_interval_sec = metrics_data.get("flush_interval_sec", met.flush_interval_sec)
        met.leakage_enabled = metrics_data.get("leakage_enabled", met.leakage_enabled)
        met.leakage_backend = metrics_data.get("leakage_backend", met.leakage_backend)
        met.haar_scale_factor = metrics_data.get("haar_scale_factor", met.haar_scale_factor)
        met.haar_min_neighbors = metrics_data.get("haar_min_neighbors", met.haar_min_neighbors)
        met.overlap_threshold = metrics_data.get("overlap_threshold", met.overlap_threshold)
        met.flicker_enabled = metrics_data.get("flicker_enabled", met.flicker_enabled)
        met.timing_enabled = metrics_data.get("timing_enabled", met.timing_enabled)
        met.utility_enabled = metrics_data.get("utility_enabled", met.utility_enabled)
    
    return privacy