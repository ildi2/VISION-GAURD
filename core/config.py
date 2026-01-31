# core/config.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses matching config/default.yaml
# ---------------------------------------------------------------------------


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
    # existing fields
    use_gpu: bool = True
    save_evidence: bool = False
    log_face_metrics: bool = False
    metrics_window_sec: float = 5.0

    # NEW: flag to request multiview engine (Wave-3)
    use_multiview_engine: bool = False


@dataclass
class UiConfig:
    # Phase-1 + Face 2A overlay controls
    show_identity_labels: bool = True
    show_debug_face_hud: bool = True
    show_fps: bool = True

    # Wave-3 visual tags (optional)
    show_engine_tag: bool = False
    show_pose_tag: bool = False

    # SourceAuth visual controls (optional)
    show_source_auth_tag: bool = True
    show_source_auth_border: bool = True


@dataclass
class IdentityRuntimeConfig:
    """
    Small wrapper for identity section.

    Only one field for now, but ready to grow if we add more knobs later.
    """
    mode: str = "classic"  # "classic" or "multiview"


# ============================================================================
# GOVERNANCE CONFIGURATION (Phase A-E: Robustness Architecture)
# ============================================================================

@dataclass
class EvidenceGateThresholds:
    """Quality thresholds for evidence gating (Phase B)"""
    # CRITICAL FIX: Aligned to 0.55 to match system threshold hierarchy
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
    """Evidence gating configuration (Phase B)"""
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
    """Rules for confirming identity (Phase C)"""
    min_samples_strong: int = 1
    min_samples_weak: int = 5
    window_seconds: float = 3.0
    min_avg_score: float = 0.75


@dataclass
class BindingSwitchingRules:
    """Rules for switching to a different person (Phase C)"""
    min_sustained_samples: int = 4
    margin_advantage: float = 0.12
    window_seconds: float = 2.0


@dataclass
class BindingContradictionRules:
    """Anti-lock-in: allow downgrade if contradictions accumulate (Phase C)"""
    threshold: float = 0.15
    counter_max: int = 5
    downgrade_factor: float = 0.8
    window_seconds: float = 5.0


@dataclass
class BindingStateConfig:
    """Binding state machine configuration (Phase C)"""
    enabled: bool = True
    description: str = ""
    confirmation: BindingConfirmationRules = field(default_factory=BindingConfirmationRules)
    switching: BindingSwitchingRules = field(default_factory=BindingSwitchingRules)
    contradiction: BindingContradictionRules = field(default_factory=BindingContradictionRules)
    stale_threshold_sec: float = 8.0
    stale_recovery_samples_needed: int = 2


@dataclass
class SchedulerBudget:
    """GPU budget configuration (Phase D)"""
    max_faces_per_frame: int = 10
    max_faces_per_second: int = 30
    budget_mode: str = "time"  # "frame" or "time"


@dataclass
class SchedulerPriorityRules:
    """Priority weighting for track selection (Phase D)"""
    unknown_pending_weight: float = 1.0
    expiring_weight: float = 0.9
    watchlist_weight: float = 1.2
    confirmed_refresh_weight: float = 0.3
    confirmed_strong_weight: float = 0.1


@dataclass
class SchedulerFairness:
    """Prevent track starvation (Phase D)"""
    starvation_threshold_sec: float = 15.0
    starved_priority_boost: float = 2.0


@dataclass
class SchedulerConfig:
    """Load-aware scheduling configuration (Phase D)"""
    enabled: bool = True
    description: str = ""
    budget: SchedulerBudget = field(default_factory=SchedulerBudget)
    priority_rules: SchedulerPriorityRules = field(default_factory=SchedulerPriorityRules)
    fairness: SchedulerFairness = field(default_factory=SchedulerFairness)


@dataclass
class MergeThresholds:
    """Thresholds for track merging (Phase E)"""
    max_spatial_distance_px: float = 100.0
    min_appearance_sim: float = 0.7
    min_embedding_sim: float = 0.80
    handoff_window_sec: float = 3.0


@dataclass
class MergeModes:
    """Which merge types to enable (Phase E)"""
    conservative: bool = True
    loose_pending: bool = False


@dataclass
class MergeConfig:
    """Handoff merge manager configuration (Phase E)"""
    enabled: bool = True
    description: str = ""
    handoff_merge_enabled: bool = True
    simul_merge_enabled: bool = False
    thresholds: MergeThresholds = field(default_factory=MergeThresholds)
    merge_modes: MergeModes = field(default_factory=MergeModes)


@dataclass
class DebugUIToggles:
    """UI display toggles (Phase A debug)"""
    show_binding_state: bool = True
    show_evidence_gate_reason: bool = True
    show_merge_alias: bool = True
    show_scheduler_budget: bool = False
    show_contradiction_counter: bool = False


@dataclass
class DebugConfig:
    """Debug and telemetry configuration (Phase A)"""
    evidence_gate_decisions: bool = True
    binding_state_transitions: bool = True
    merge_attempts: bool = True
    scheduler_selections: bool = True
    ui: DebugUIToggles = field(default_factory=DebugUIToggles)
    emit_metrics_every_sec: float = 1.0
    log_level: str = "INFO"


@dataclass
class GovernanceConfig:
    """Master governance configuration (Phases A-E)"""
    enabled: bool = True
    evidence_gate: EvidenceGateConfig = field(default_factory=EvidenceGateConfig)
    binding: BindingStateConfig = field(default_factory=BindingStateConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


@dataclass
class ContinuityConfig:
    """Chimeric continuity mode configuration (GPS-like identity persistence)"""
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
class ChimericConfig:
    """Chimeric identity mode configuration (classic vs continuity)"""
    mode: str = "classic"  # Options: "classic", "shadow_continuity", "continuity"
    continuity: ContinuityConfig = field(default_factory=ContinuityConfig)


@dataclass
class Config:
    camera: CameraConfig
    paths: PathsConfig
    runtime: RuntimeConfig
    ui: UiConfig
    # NEW: identity section
    identity: IdentityRuntimeConfig = field(default_factory=IdentityRuntimeConfig)
    # NEW: governance section (Phases A-E)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    # NEW: chimeric section (continuity mode)
    chimeric: ChimericConfig = field(default_factory=ChimericConfig)


# ---------------------------------------------------------------------------
# Helpers to convert plain dicts into dataclasses safely
# ---------------------------------------------------------------------------


def _update_dataclass_from_dict(obj: Any, data: Dict[str, Any]) -> Any:
    """
    Assign only known fields from dict into dataclass instance.
    Unknown keys in YAML are ignored (backwards-compatible).
    """
    for k, v in data.items():
        if hasattr(obj, k):
            setattr(obj, k, v)
    return obj


def load_config(path: str | Path = "config/default.yaml") -> Config:
    """
    Load YAML config and map it to our dataclasses.

    This function is the single source of truth for all configuration sections:
      - cfg.camera
      - cfg.paths
      - cfg.runtime
      - cfg.ui
      - cfg.identity
      - cfg.governance (NEW: Phases A-E)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a dict, got: {type(raw)}")

    # Top-level sections
    cam_data = raw.get("camera", {}) or {}
    paths_data = raw.get("paths", {}) or {}
    runtime_data = raw.get("runtime", {}) or {}
    ui_data = raw.get("ui", {}) or {}
    identity_data = raw.get("identity", {}) or {}
    governance_data = raw.get("governance", {}) or {}
    chimeric_data = raw.get("chimeric", {}) or {}

    camera = _update_dataclass_from_dict(CameraConfig(), cam_data)
    paths = _update_dataclass_from_dict(PathsConfig(), paths_data)
    runtime = _update_dataclass_from_dict(RuntimeConfig(), runtime_data)
    ui = _update_dataclass_from_dict(UiConfig(), ui_data)
    identity = _update_dataclass_from_dict(IdentityRuntimeConfig(), identity_data)
    
    # Parse governance nested sections
    governance = _parse_governance_config(governance_data)
    
    # Parse chimeric nested sections
    chimeric = _parse_chimeric_config(chimeric_data)

    cfg = Config(
        camera=camera,
        paths=paths,
        runtime=runtime,
        ui=ui,
        identity=identity,
        governance=governance,
        chimeric=chimeric,
    )

    logger.info(
        "Config loaded from %s | camera index=%d, runtime.use_gpu=%s, "
        "runtime.use_multiview_engine=%s, identity.mode=%s, governance.enabled=%s",
        path,
        cfg.camera.index,
        cfg.runtime.use_gpu,
        cfg.runtime.use_multiview_engine,
        cfg.identity.mode,
        cfg.governance.enabled,
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
    """
    Recursively parse nested governance configuration from YAML dict.
    
    Handles deep nesting with safe defaults.
    """
    def safe_update(obj: Any, data: Dict[str, Any]) -> Any:
        """Recursively update dataclass from dict, handling nested structures"""
        for k, v in data.items():
            if hasattr(obj, k):
                attr_type = type(getattr(obj, k))
                if v is not None and hasattr(attr_type, '__dataclass_fields__'):
                    # Recursively update nested dataclass
                    nested_obj = getattr(obj, k)
                    safe_update(nested_obj, v)
                else:
                    setattr(obj, k, v)
        return obj
    
    # Build governance config
    gov = GovernanceConfig()
    
    # Parse each governance section
    if "evidence_gate" in data:
        gate_data = data["evidence_gate"] or {}
        gov.evidence_gate = GovernanceConfig().evidence_gate  # Get default
        gate = gov.evidence_gate
        gate.enabled = gate_data.get("enabled", gate.enabled)
        gate.description = gate_data.get("description", gate.description)
        gate.accept_ratio_target = gate_data.get("accept_ratio_target", gate.accept_ratio_target)
        gate.hold_ratio_target = gate_data.get("hold_ratio_target", gate.hold_ratio_target)
        gate.reject_ratio_target = gate_data.get("reject_ratio_target", gate.reject_ratio_target)
        gate.log_reasons = gate_data.get("log_reasons", gate.log_reasons)
        gate.log_level = gate_data.get("log_level", gate.log_level)
        
        # Parse thresholds
        if "thresholds" in gate_data:
            thresholds_data = gate_data["thresholds"] or {}
            gate.thresholds = safe_update(gate.thresholds, thresholds_data)
    
    # Similar parsing for binding, scheduler, merge, debug
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

def _parse_chimeric_config(data: Dict[str, Any]) -> ChimericConfig:
    """
    Parse chimeric continuity mode configuration from YAML dict.
    
    Handles nested continuity section with safe defaults.
    """
    chimeric = ChimericConfig()
    
    # Top-level mode
    chimeric.mode = data.get("mode", chimeric.mode)
    
    # Parse continuity sub-section
    if "continuity" in data:
        continuity_data = data["continuity"] or {}
        cont = chimeric.continuity
        
        # Guard 1: Track stability
        cont.min_track_age_frames = continuity_data.get("min_track_age_frames", cont.min_track_age_frames)
        
        # Guard 2c: Track health
        cont.min_track_confidence = continuity_data.get("min_track_confidence", cont.min_track_confidence)
        cont.max_lost_frames = continuity_data.get("max_lost_frames", cont.max_lost_frames)
        
        # Guard 2: Appearance consistency
        cont.appearance_distance_threshold = continuity_data.get("appearance_distance_threshold", 
                                                                   cont.appearance_distance_threshold)
        cont.appearance_safe_zone_frames = continuity_data.get("appearance_safe_zone_frames", 
                                                                 cont.appearance_safe_zone_frames)
        cont.appearance_ema_alpha = continuity_data.get("appearance_ema_alpha", cont.appearance_ema_alpha)
        
        # Guard 2b: BBox stability
        cont.max_bbox_displacement_fraction = continuity_data.get("max_bbox_displacement_fraction", 
                                                                    cont.max_bbox_displacement_fraction)
        cont.max_bbox_displacement_px = continuity_data.get("max_bbox_displacement_px", 
                                                              cont.max_bbox_displacement_px)
        cont.bbox_iou_threshold = continuity_data.get("bbox_iou_threshold", cont.bbox_iou_threshold)
        
        # Guard 3: Face consistency
        cont.max_face_contradiction_count = continuity_data.get("max_face_contradiction_count", 
                                                                  cont.max_face_contradiction_count)
        
        # Grace reattachment
        cont.grace_window_seconds = continuity_data.get("grace_window_seconds", cont.grace_window_seconds)
        cont.grace_bbox_displacement_fraction = continuity_data.get("grace_bbox_displacement_fraction", 
                                                                      cont.grace_bbox_displacement_fraction)
        
        # Shadow mode
        cont.shadow_mode = continuity_data.get("shadow_mode", cont.shadow_mode)
        cont.shadow_metrics_log_interval_sec = continuity_data.get("shadow_metrics_log_interval_sec", 
                                                                     cont.shadow_metrics_log_interval_sec)
    
    return chimeric