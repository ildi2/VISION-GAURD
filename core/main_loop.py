# core/main_loop.py
from __future__ import annotations

import time
import logging

import cv2
import numpy as np
from dataclasses import asdict
from core.interfaces import IdentityEngine as IdentityEngineBase

from schemas import Frame
from .camera import CameraSource
from .dummies import (
    DummyEventsEngine,
    DummyAlertEngine,
)
from .config import load_config
from .logging_setup import setup_logging
from .device import select_device, get_gpu_memory_stats
from ui.overlay import draw_overlay

# Phase-1 perception engine (YOLO + OC-SORT + appearance + ring buffer)
from perception.perception_engine import Phase1PerceptionEngine

# Phase-2A: real face-based IdentityEngine (classic 2D route)
from identity.identity_engine import FaceIdentityEngine

# Wave-3 telemetry (face metrics)
from .metrics import FaceMetrics

# Phase A: Governance metrics collection (robustness monitoring)
from .governance_metrics import MetricsCollector

# Wave-3: optional multiview / pseudo-3D identity engine
try:
    from identity.identity_engine_multiview import IdentityEngineMultiView
except Exception:  # pragma: no cover - safe fallback if module missing
    IdentityEngineMultiView = None  # type: ignore[assignment]

# SourceAuth (Source Authenticity: real head vs phone / screen / photo)
try:
    from source_auth.config import default_source_auth_config
    from source_auth.engine import SourceAuthEngine
    from source_auth.diagnostics import format_source_auth_reason
except Exception:  # pragma: no cover - safe fallback if module missing
    default_source_auth_config = None  # type: ignore[assignment]
    SourceAuthEngine = None  # type: ignore[assignment]
    format_source_auth_reason = None  # type: ignore[assignment]


def run() -> None:
    """
    Run the GaitGuard pipeline (Phase-1 + Phase-2A Face).

    Pipeline:
        Frame (camera) ->
        Perception (detect + track + ring buffers) ->
        Identity (classic 2D gallery OR multiview OR hybrid) ->
        SourceAuth (real head vs phone/screen/photo) ->
        Events (dummy) ->
        Alerts (dummy) ->
        UI overlay (+ Wave-3 telemetry)

    Press ESC in the window to exit.
    """
    # ---- load config & logging ----
    cfg = load_config()
    setup_logging(cfg.paths.logs_dir)
    log = logging.getLogger("gaitguard.main")

    # ---- decide device (GPU/CPU + FP16) ----
    device, use_half = select_device(prefer_gpu=cfg.runtime.use_gpu)
    log.info("Runtime device=%s | half=%s", device, use_half)

    # ---- instantiate Phase-1 perception engine ----
    perception = Phase1PerceptionEngine()

    # ------------------------------------------------------------------
    # Identity engine selection (classic vs multiview vs hybrid)
    # ------------------------------------------------------------------
    # 1) Start with default.
    identity_mode_requested: str = "classic"
    identity_mode_source: str = "default"

    # 2) Prefer FaceConfig.identity_mode if available (new 3D-aware switch).
    face_cfg = getattr(cfg, "face", None)
    raw_mode = None
    if face_cfg is not None and hasattr(face_cfg, "identity_mode"):
        raw_mode = getattr(face_cfg, "identity_mode", None)
        if isinstance(raw_mode, str) and raw_mode.strip():
            identity_mode_requested = raw_mode.strip().lower()
            identity_mode_source = "face.identity_mode"

    # 3) Fallback: cfg.identity.mode (older config style).
    if identity_mode_source == "default":
        identity_cfg = getattr(cfg, "identity", None)
        raw_mode = None
        if identity_cfg is not None:
            if hasattr(identity_cfg, "mode"):
                raw_mode = getattr(identity_cfg, "mode")
            elif isinstance(identity_cfg, dict):
                raw_mode = identity_cfg.get("mode")

        if isinstance(raw_mode, str) and raw_mode.strip():
            identity_mode_requested = raw_mode.strip().lower()
            identity_mode_source = "identity.mode"

    # 4) Final fallback: runtime.use_multiview_engine (legacy bool flag).
    if identity_mode_source == "default":
        use_mv_flag = bool(getattr(cfg.runtime, "use_multiview_engine", False))
        if use_mv_flag:
            identity_mode_requested = "multiview"
            identity_mode_source = "runtime.use_multiview_engine"

    # Normalise invalid values to safe default.
    if identity_mode_requested not in ("classic", "multiview", "hybrid"):
        log.warning(
            "Invalid identity_mode '%s' (source=%s); falling back to 'classic'. "
            "Allowed: classic, multiview, hybrid.",
            identity_mode_requested,
            identity_mode_source,
        )
        identity_mode_requested = "classic"

    multiview_available = IdentityEngineMultiView is not None

    # 5) Instantiate the appropriate engine(s) and track the *active* mode.
    identity_primary: IdentityEngineBase
    identity_secondary: IdentityEngineBase | None = None
    identity_mode_active: str

    if identity_mode_requested == "multiview" and multiview_available:
        # Pure 3D / multiview mode.
        # FIX: Pass governance config for binding manager initialization
        identity_primary = IdentityEngineMultiView(governance_cfg=cfg.governance)  # type: ignore[call-arg]
        identity_mode_active = "multiview"
        log.info("Identity engine: multiview only (pseudo-3D, Wave-3)")
    elif identity_mode_requested == "hybrid" and multiview_available:
        # Hybrid: run multiview as the primary engine and classic in parallel
        # for comparison / safety. Overlay and alerts use primary decisions.
        # FIX: Pass governance config for binding manager initialization
        identity_primary = IdentityEngineMultiView(governance_cfg=cfg.governance)  # type: ignore[call-arg]
        identity_secondary = FaceIdentityEngine()
        identity_mode_active = "hybrid"
        log.info(
            "Identity engine: hybrid (primary=multiview, secondary=classic for comparison)"
        )
    else:
        # Fallback to classic if:
        #   - mode is 'classic', OR
        #   - multiview was requested but not available.
        if identity_mode_requested in ("multiview", "hybrid") and not multiview_available:
            log.warning(
                "Multiview requested (mode=%s, source=%s) but "
                "IdentityEngineMultiView is not available. "
                "Falling back to classic FaceIdentityEngine.",
                identity_mode_requested,
                identity_mode_source,
            )
        identity_primary = FaceIdentityEngine()
        identity_mode_active = "classic"
        log.info("Identity engine: classic only (Phase-2A face route)")

    log.info(
        "Identity selection: requested=%s (source=%s, runtime.use_multiview_engine=%s), "
        "active=%s",
        identity_mode_requested,
        identity_mode_source,
        getattr(cfg.runtime, "use_multiview_engine", None),
        identity_mode_active,
    )

    # Decide window title based on the *active* mode (purely UI).
    if identity_mode_active == "multiview":
        window_title = "GaitGuard 1.0 - Phase 1 + Face 2A (3D multiview)"
    elif identity_mode_active == "hybrid":
        window_title = "GaitGuard 1.0 - Phase 1 + Face 2A (hybrid classic+3D)"
    else:
        window_title = "GaitGuard 1.0 - Phase 1 + Face 2A"

    # Events / alerts still dummy for now.
    events_engine = DummyEventsEngine()
    alert_engine = DummyAlertEngine()

    # ---- Wave-3 telemetry (optional) ----
    metrics: FaceMetrics | None = None
    log_face_metrics = bool(getattr(cfg.runtime, "log_face_metrics", False))
    if log_face_metrics:
        # Use configurable metrics window if present, else default to 5.0
        window_sec = float(getattr(cfg.runtime, "metrics_window_sec", 5.0))
        metrics = FaceMetrics(window_sec=window_sec, log_every_sec=5.0)
        log.info(
            "FaceMetrics telemetry enabled (window=%.1fs, log_every=5s)",
            window_sec,
        )

    # ---- SourceAuth engine (Source Authenticity: real head vs phone / screen / photo) ----
    source_auth_engine = None
    # FIX 4: Check if source auth should be enabled via config
    # CRITICAL: Check governance section exists first to avoid AttributeError
    source_auth_enabled = True
    try:
        if hasattr(cfg, 'governance') and hasattr(cfg.governance, 'source_auth'):
            sa_config = cfg.governance.source_auth
            source_auth_enabled = bool(getattr(sa_config, 'enabled', True))
    except (AttributeError, TypeError, ValueError) as e:
        # If config reading fails, default to enabled for backward compatibility
        log.warning(f"Could not read source_auth config: {e}; defaulting to enabled")
        source_auth_enabled = True
    
    if source_auth_enabled and SourceAuthEngine is not None and default_source_auth_config is not None:
        try:
            # Reuse face_cfg when available so SourceAuth thresholds are aligned
            # with the same camera / runtime environment as face identity.
            sa_cfg = default_source_auth_config(face_cfg=face_cfg)
            source_auth_engine = SourceAuthEngine(sa_cfg)  # type: ignore[call-arg]
            log.info(
                "SourceAuth engine initialised "
                "(motion + screen + background cues; annotating IdentityDecision)."
            )
        except Exception:
            log.exception(
                "Failed to initialise SourceAuth engine; "
                "continuing without source authenticity checks."
            )
            source_auth_engine = None
    elif not source_auth_enabled:
        log.info("SourceAuth engine disabled via governance.source_auth.enabled=false")

    # ---- Chimeric Continuity Binder (GPS-like identity persistence) ----
    continuity_binder = None
    continuity_mode = "classic"
    
    if hasattr(cfg, "chimeric"):
        try:
            chimeric_mode = getattr(cfg.chimeric, "mode", "classic")
            if isinstance(chimeric_mode, str):
                chimeric_mode = chimeric_mode.strip().lower()
            
            if chimeric_mode in ("continuity", "shadow_continuity"):
                from identity.continuity_binder import ContinuityBinder
                
                # Instantiate binder with chimeric config
                continuity_binder = ContinuityBinder(cfg.chimeric)
                continuity_mode = chimeric_mode
                
                # CRITICAL: Set frame dimensions for resolution-aware thresholds
                # Extract from camera config (fallback to defaults if missing)
                frame_width = getattr(cfg.camera, 'width', 640)
                frame_height = getattr(cfg.camera, 'height', 480)
                continuity_binder.set_frame_dimensions(frame_width, frame_height)
                
                # Determine shadow mode status
                shadow_status = "shadow mode (observe-only)" if chimeric_mode == "shadow_continuity" else "real mode (GPS carry)"
                
                log.info(
                    "Continuity binder initialized | mode=%s | %s | frame=%dx%d | "
                    "min_age=%d | appearance_thresh=%.2f | bbox_frac=%.2f",
                    chimeric_mode,
                    shadow_status,
                    frame_width,
                    frame_height,
                    continuity_binder.min_track_age_frames,
                    continuity_binder.appearance_distance_threshold,
                    continuity_binder.max_bbox_displacement_frac  # FIX: Correct attribute name
                )
        except Exception:
            log.exception(
                "Failed to initialize continuity binder; continuing without continuity mode"
            )
            continuity_binder = None
    else:
        log.debug("Chimeric continuity mode disabled (no chimeric config section)")

    # ---- Phase D: FPS/Load-Aware Scheduler ----
    scheduler = None
    scheduler_enabled = False
    if hasattr(cfg, "governance") and hasattr(cfg.governance, "scheduler"):
        try:
            from core.scheduler import create_scheduler_from_config
            scheduler_cfg_dict = cfg.governance.scheduler
            # FIX: Handle Dataclass configuration correctly
            if hasattr(scheduler_cfg_dict, "enabled"):
                # It's a Dataclass (SchedulerConfig)
                scheduler_cfg_dict = asdict(scheduler_cfg_dict)
            
            if isinstance(scheduler_cfg_dict, dict):
                scheduler = create_scheduler_from_config(scheduler_cfg_dict)
                scheduler_enabled = scheduler_cfg_dict.get("enabled", True)
                log.info(
                    "Phase D Scheduler initialised "
                    "(budget_policy=%s, enabled=%s)",
                    scheduler_cfg_dict.get("budget_policy", "adaptive"),
                    scheduler_enabled,
                )
            else:
                log.warning(f"governance.scheduler config is {type(scheduler_cfg_dict)}, expected dict or dataclass; scheduler disabled")
        except Exception:
            log.exception("Failed to initialise Phase D scheduler; continuing without scheduler")
            scheduler = None
            scheduler_enabled = False

    # ---- Phase E: Handoff Merge Manager ----
    merge_manager = None
    merge_enabled = False
    if hasattr(cfg, "governance") and hasattr(cfg.governance, "merge"):
        try:
            from identity.merge_manager import create_merge_config_from_dict, MergeManager
            merge_cfg_dict = cfg.governance.merge
            # FIX: Handle Dataclass configuration correctly
            if hasattr(merge_cfg_dict, "enabled"):
                # It's a Dataclass (MergeConfig)
                merge_cfg_dict = asdict(merge_cfg_dict)
            
            if isinstance(merge_cfg_dict, dict):
                merge_config = create_merge_config_from_dict(merge_cfg_dict)
                merge_manager = MergeManager(merge_config)
                merge_enabled = merge_cfg_dict.get("enabled", True)
                log.info(
                    "Phase E Merge Manager initialised "
                    "(strategy=%s, enabled=%s)",
                    merge_cfg_dict.get("merge_strategy", {}).get("mode", "conservative"),
                    merge_enabled,
                )
            else:
                log.warning(f"governance.merge config is {type(merge_cfg_dict)}, expected dict or dataclass; merge manager disabled")
        except Exception:
            log.exception("Failed to initialise Phase E merge manager; continuing without merge manager")
            merge_manager = None
            merge_enabled = False

    # Optional warmup hooks (if implemented on these classes).
    if hasattr(perception, "warmup"):
        try:
            log.info("Warming up perception engine (if supported)...")
            perception.warmup()  # type: ignore[call-arg]
        except Exception:
            log.exception("Perception warmup failed")

    if hasattr(identity_primary, "warmup"):
        try:
            log.info("Warming up primary identity engine (if supported)...")
            identity_primary.warmup()  # type: ignore[call-arg]
        except Exception:
            log.exception("Primary identity warmup failed")

    if identity_secondary is not None and hasattr(identity_secondary, "warmup"):
        try:
            log.info("Warming up secondary identity engine (hybrid mode)...")
            identity_secondary.warmup()  # type: ignore[call-arg]
        except Exception:
            log.exception("Secondary identity warmup failed")

    if source_auth_engine is not None and hasattr(source_auth_engine, "warmup"):
        try:
            log.info("Warming up SourceAuth engine (if supported)...")
            source_auth_engine.warmup()  # type: ignore[call-arg]
        except Exception:
            log.exception("SourceAuth warmup failed")

    # ---- camera source ----
    src = CameraSource(
        cam_index=cfg.camera.index,
        w=cfg.camera.width,
        h=cfg.camera.height,
        fps=cfg.camera.fps,
        buffersize=1,
    )
    src.start()

    log.info(
        "GaitGuard pipeline started "
        "(Phase-1 + Face identity, mode=%s). Press ESC to exit.",
        identity_mode_active,
    )

    frame_id = 0
    camera_id = "cam0"

    # Use monotonic clock for FPS / latency measurements.
    t0 = time.perf_counter()
    frames = 0
    last_fps = 0.0
    prev_frame_ts = -1.0  # Phase D: FPS tracking

    # Simple counter for hybrid agreement diagnostics.
    hybrid_agree = 0
    hybrid_total = 0
    
    # Phase 6: Track id_sources from previous frame for scheduler priority awareness
    # Scheduler uses prev frame's id_sources to optimize current frame's face processing
    prev_frame_id_sources = {}
    
    # NEW (Phase A): Governance metrics collection
    governance_enabled = bool(getattr(cfg, "governance", {}).enabled if hasattr(cfg, "governance") else False)
    metrics_collector = None
    if governance_enabled:
        from core.governance_metrics import get_metrics_collector
        metrics_interval = float(getattr(cfg.governance.debug, "emit_metrics_every_sec", 1.0))
        # FIX: Use global singleton so we report the same metrics that EvidenceGate records to
        metrics_collector = get_metrics_collector()
        metrics_collector.interval_sec = metrics_interval
        log.info("Governance metrics collection enabled (emit every %.1f sec)", metrics_interval)

    try:
        while True:
            img = src.read_latest(timeout=1.0)
            if img is None:
                continue

            # Monotonic timestamp for internal timing (stable even if system time changes).
            ts = time.perf_counter()
            h, w = img.shape[:2]

            # Phase D: Measure actual FPS and compute schedule
            actual_fps = last_fps if last_fps > 0 else 30.0
            if prev_frame_ts > 0:
                frame_time = ts - prev_frame_ts
                if frame_time > 0:
                    actual_fps = 1.0 / frame_time
            prev_frame_ts = ts



            # Wrap raw image into our Frame schema.
            frame = Frame(
                frame_id=frame_id,
                ts=ts,
                camera_id=camera_id,
                size=(w, h),
                image=img,
            )
            frame_id += 1

            # ---- full pipeline ----
            # 1) Perception: detect + track
            tracks = perception.process_frame(frame)

            # Phase D: Scheduler (Moved here to have access to tracks)
            schedule_context = None
            if scheduler_enabled and scheduler is not None:
                try:
                    # Get binding states for all tracks
                    binding_states = {}
                    if identity_secondary is None:
                        # Single engine: use active engine for binding state
                        binding_states = getattr(identity_primary, "get_binding_states", lambda: {})()
                    else:
                        # Hybrid: use primary engine for binding state
                        binding_states = getattr(identity_primary, "get_binding_states", lambda: {})()
                    
                    # Compute schedule (Phase 6: use prev frame's id_sources)
                    schedule_context = scheduler.compute_schedule(
                        track_ids=list(t.track_id for t in tracks) if tracks else [],
                        binding_states=binding_states,
                        current_ts=ts,
                        actual_fps=actual_fps,
                        id_sources=prev_frame_id_sources,
                    )
                except Exception:
                    log.exception("Failed to compute scheduler context; continuing without scheduling")
                    schedule_context = None

            # 2) Identity: classic / multiview / hybrid
            #    Keep a reference to the IdSignals used by the active engine,
            #    so SourceAuth sees exactly the same face evidence.
            signals_active = None

            if identity_secondary is None:
                # Single engine (classic or pure multiview).
                # Try to pass schedule_context if engine supports it
                try:
                    signals = identity_primary.update_signals(frame, tracks, schedule_context=schedule_context)
                except TypeError:
                    # Fallback for engines that don't support schedule_context yet
                    signals = identity_primary.update_signals(frame, tracks)
                
                decisions = identity_primary.decide(signals)
                signals_active = signals
            else:
                # Hybrid: run both engines in parallel.
                # Primary (multiview) drives UI / events.
                try:
                    signals_primary = identity_primary.update_signals(frame, tracks, schedule_context=schedule_context)
                except TypeError:
                    signals_primary = identity_primary.update_signals(frame, tracks)
                decisions_primary = identity_primary.decide(signals_primary)

                # Secondary (classic) for comparison only.
                try:
                    signals_secondary = identity_secondary.update_signals(frame, tracks, schedule_context=schedule_context)
                except TypeError:
                    signals_secondary = identity_secondary.update_signals(frame, tracks)
                decisions_secondary = identity_secondary.decide(signals_secondary)

                decisions = decisions_primary
                signals_active = signals_primary

                # Lightweight agreement diagnostics (no heavy logic).
                try:
                    primary_map = {
                        d.track_id: (d.identity_id, getattr(d, "category", None))
                        for d in decisions_primary
                    }
                    secondary_map = {
                        d.track_id: (d.identity_id, getattr(d, "category", None))
                        for d in decisions_secondary
                    }
                    common_ids = set(primary_map.keys()) & set(secondary_map.keys())
                    if common_ids:
                        for tid in common_ids:
                            if primary_map[tid] == secondary_map[tid]:
                                hybrid_agree += 1
                            hybrid_total += 1
                        if hybrid_total > 0 and (frame_id % 60 == 0):
                            agree_ratio = hybrid_agree / max(hybrid_total, 1)
                            log.info(
                                "Hybrid diagnostics: agree=%d / %d (%.1f%%)",
                                hybrid_agree,
                                hybrid_total,
                                agree_ratio * 100.0,
                            )
                except Exception:
                    # Never let diagnostics crash the main loop.
                    log.exception("Hybrid diagnostics failed")

            # 2a) CONTINUITY BINDER (Chimeric Mode - GPS-like identity persistence)
            # Placed after identity engine, before SourceAuth (policy layer position)
            if continuity_binder is not None:
                try:
                    # CRITICAL FIX: Populate track.embedding AND track.has_face_this_frame
                    # The continuity binder needs:
                    # 1. embedding: For appearance consistency guard (EMA comparison)
                    # 2. has_face_this_frame: To know if face is actively visible THIS frame
                    #
                    # IMPORTANT: has_face_this_frame must be set EVERY frame, defaulting to False.
                    # This is the KEY signal for distinguishing [F] from [G].
                    
                    if signals_active is not None:
                        signals_by_track = {int(s.track_id): s for s in signals_active if s is not None}
                    else:
                        signals_by_track = {}
                    
                    for trk in tracks:
                        try:
                            sig = signals_by_track.get(trk.track_id)
                            
                            # CRITICAL: Default to no face this frame
                            trk.has_face_this_frame = False
                            
                            if sig is not None and sig.best_face is not None:
                                # Face detected this frame
                                trk.has_face_this_frame = True
                                
                                # Also populate embedding for appearance guard
                                if sig.best_face.embedding is not None:
                                    trk.embedding = np.asarray(sig.best_face.embedding, dtype=np.float32).reshape(-1)
                        except Exception:
                            # Never crash on signal processing failure
                            trk.has_face_this_frame = False
                    
                    decisions = continuity_binder.apply(ts, tracks, decisions)
                    # In shadow mode: decisions annotated with extra['would_carry'] (no mutations)
                    # In real mode: decisions.identity_id carried from memory (GPS-like continuity)
                except Exception:
                    # Continuity must never crash pipeline (fail-closed: skip carry on error)
                    log.exception("Continuity binder failed; continuing with original decisions")
            
            # Phase 6: Extract id_sources for scheduler priority awareness
            # Build dict mapping track_id → id_source ("F"=face, "G"=GPS, "U"=unknown)
            id_sources = {}
            for dec in decisions:
                # Get id_source (schema field or extra dict fallback)
                source = "U"  # Default
                if hasattr(dec, 'id_source') and dec.id_source is not None:
                    source = dec.id_source
                elif dec.extra and 'id_source' in dec.extra:
                    source = dec.extra['id_source']
                id_sources[dec.track_id] = source
            
            # Store for next frame's scheduler
            prev_frame_id_sources = id_sources

            # 2b) SourceAuth: annotate decisions with source authenticity
            if source_auth_engine is not None and signals_active is not None:
                try:
                    # SourceAuth works per-tracklet, over a short time window,
                    # using the same Frame + Tracklets + IdSignals used by the
                    # active identity engine.
                    sa_results = source_auth_engine.update(
                        frame,
                        tracks,
                        signals_active,
                    )

                    if sa_results:
                        for dec in decisions:
                            tid = dec.track_id
                            sa = sa_results.get(tid)  # per-track SourceAuthScores

                            if sa is None:
                                continue

                            # Attach source_auth_score (robust to missing/invalid).
                            try:
                                sa_score = float(getattr(sa, "source_auth_score", 0.0))
                            except Exception:
                                sa_score = 0.0

                            try:
                                dec.source_auth_score = sa_score
                            except Exception:
                                # If decision object does not expose this field,
                                # we silently skip – SourceAuth must never break identity.
                                pass

                            # Attach source_auth_state (string label).
                            try:
                                state = getattr(sa, "state", None)
                                sa_state = str(state) if state is not None else "UNCERTAIN"
                            except Exception:
                                sa_state = "UNCERTAIN"

                            try:
                                dec.source_auth_state = sa_state
                            except Exception:
                                pass

                            # Optional compact reason fragment for logs / overlay.
                            reason_fragment = None
                            if format_source_auth_reason is not None:
                                try:
                                    reason_fragment = format_source_auth_reason(sa, debug=None)
                                except Exception:
                                    reason_fragment = None

                            if reason_fragment:
                                # Be robust even if the decision did not predefine `reason`.
                                try:
                                    existing_reason = getattr(dec, "reason", "") or ""
                                    if existing_reason:
                                        dec.reason = f"{existing_reason}|{reason_fragment}"
                                    else:
                                        dec.reason = reason_fragment
                                except Exception:
                                    # If reason cannot be set, we just skip; no crash.
                                    pass

                            # Optional: compute a combined risk label if the decision
                            # object exposes a `risk_label` attribute. This is a pure
                            # annotation and does not change any downstream logic.
                            if hasattr(dec, "risk_label"):
                                try:
                                    # Identity "strength": prefer `score`, fall back to `confidence`.
                                    identity_score = getattr(dec, "score", None)
                                    if identity_score is None:
                                        identity_score = getattr(dec, "confidence", None)

                                    strong_identity = False
                                    if identity_score is not None:
                                        try:
                                            strong_identity = float(identity_score) >= 0.80
                                        except Exception:
                                            strong_identity = False

                                    sa_state_upper = sa_state.upper()
                                    sa_real = sa_state_upper in ("REAL", "LIKELY_REAL")
                                    sa_spoof = sa_state_upper in ("SPOOF", "LIKELY_SPOOF")

                                    if strong_identity and sa_real:
                                        dec.risk_label = "ID_STRONG_SA_REAL"
                                    elif strong_identity and sa_spoof:
                                        dec.risk_label = "ID_STRONG_SA_SPOOF"
                                    elif (not strong_identity) and sa_real:
                                        dec.risk_label = "ID_WEAK_SA_REAL"
                                    elif (not strong_identity) and sa_spoof:
                                        dec.risk_label = "ID_WEAK_SA_SPOOF"
                                    else:
                                        dec.risk_label = "ID_SA_UNCERTAIN"
                                except Exception:
                                    # Never let annotation logic break the frame loop.
                                    pass

                except Exception:
                    # SourceAuth must never crash the main loop.
                    log.exception(
                        "SourceAuth update failed; "
                        "continuing without source authenticity for this frame."
                    )

            # 2c) Phase E: Update merge manager with tracklet events
            if merge_manager is not None and merge_enabled:
                try:
                    # Report current frame time to merge manager
                    merge_manager.current_time = ts
                    
                    # Process each track for merge manager
                    for track in tracks:
                        track_id = track.track_id
                        
                        # Extract binding state for this track (if available)
                        binding_state = None
                        for dec in decisions:
                            if dec.track_id == track_id:
                                binding_state = {
                                    'person_id': getattr(dec, 'identity_id', None),
                                    'status': getattr(dec, 'binding_state', 'UNKNOWN'),
                                    'confidence': getattr(dec, 'score', getattr(dec, 'confidence', 0.0))
                                }
                                break
                        
                        # Get track appearance features if available
                        appearance_features = None
                        if hasattr(track, 'embedding') and track.embedding is not None:
                            appearance_features = track.embedding
                        elif hasattr(track, 'appearance_features') and track.appearance_features is not None:
                            appearance_features = track.appearance_features
                        
                        # Count quality samples (faces accepted for this track)
                        quality_samples = 0
                        if hasattr(track, 'face_hits'):
                            quality_samples = track.face_hits
                        elif hasattr(track, 'quality_samples'):
                            quality_samples = track.quality_samples
                        
                        # Report updated tracklet
                        merge_manager.on_tracklet_updated(
                            tracklet_id=track_id,
                            binding_state=binding_state,
                            appearance_features=appearance_features,
                            current_position=(float(track.x), float(track.y)) if hasattr(track, 'x') else (0.0, 0.0),
                            track_length=getattr(track, 'frame_hits', 0),
                            quality_samples=quality_samples,
                            timestamp=ts
                        )
                    
                    # Periodically check for merges (every 10 frames to avoid overhead)
                    if frame_id % 10 == 0:
                        # Build active tracklets dict for merge candidate search
                        active_tracklets_dict = {}
                        for track in tracks:
                            from identity.merge_manager import MergeCandidate
                            candidate = MergeCandidate(
                                tracklet_id=track.track_id,
                                start_time=ts - (getattr(track, 'frame_hits', 0) / max(last_fps, 1.0)),
                                track_length=getattr(track, 'frame_hits', 0),
                                first_position=np.array([getattr(track, 'x', 0.0), getattr(track, 'y', 0.0)], dtype=np.float32),
                                quality_samples=quality_samples
                            )
                            active_tracklets_dict[track.track_id] = candidate
                        
                        # Execute merge checks
                        merges_executed = merge_manager.check_and_execute_merges(active_tracklets_dict)
                        
                        if merges_executed > 0 and (frame_id % 30 == 0):  # Log periodically
                            metrics = merge_manager.get_metrics()
                            log.debug(
                                "Phase E Merge: %d executions this frame, "
                                "total=%d merges, reversals=%d, avg_score=%.1f",
                                merges_executed,
                                metrics.merges_executed,
                                metrics.merge_reversals,
                                metrics.average_merge_score
                            )
                    
                    # Cleanup old inactive tracklets periodically
                    if frame_id % 100 == 0:
                        merge_manager.cleanup_old_tracklets()
                
                except Exception:
                    # Merge manager must never crash the main loop
                    log.exception("Phase E merge manager update failed; continuing without merge manager")

            # 3) Events / alerts (still dummy)
            events = events_engine.update(frame, tracks, decisions)
            alerts = alert_engine.update(frame, events, decisions)

            # 4) Wave-3 telemetry update
            if metrics is not None:
                try:
                    metrics.update(decisions, tracks, ts)
                    metrics.maybe_log(ts)
                except Exception:
                    # Telemetry must never crash the pipeline.
                    log.exception("FaceMetrics update/maybe_log failed")
            
            # NEW (Phase A): Governance metrics collection & emission
            if metrics_collector is not None:
                try:
                    # Update system health metrics
                    metrics_collector.metrics.fps_estimate = last_fps
                    metrics_collector.metrics.track_count = len(tracks)
                    
                    # Compute binding state distribution
                    binding_states = {}
                    for dec in decisions:
                        # Extract binding state if available (will be populated in Phase C)
                        state = getattr(dec, "binding_state", "UNKNOWN")
                        binding_states[state] = binding_states.get(state, 0) + 1
                    metrics_collector.metrics.binding_state_counts = binding_states
                    
                    # Compute rate metrics
                    if len(tracks) > 0:
                        unknown_cnt = binding_states.get("UNKNOWN", 0)
                        pending_cnt = binding_states.get("PENDING", 0)
                        confirmed_cnt = len(tracks) - unknown_cnt - pending_cnt
                        metrics_collector.metrics.unknown_rate = unknown_cnt / len(tracks)
                        metrics_collector.metrics.pending_rate = pending_cnt / len(tracks)
                        metrics_collector.metrics.confirmed_rate = max(0.0, confirmed_cnt / len(tracks))
                    
                    # Emit metrics if threshold reached
                    metrics_collector.maybe_emit()
                except Exception:
                    # Governance metrics must never crash the pipeline
                    log.exception("Governance metrics collection failed")

            # ---- FPS tracking ----
            frames += 1
            if frames % 30 == 0:
                elapsed = ts - t0
                last_fps = frames / max(elapsed, 1e-6)
                
                # Get GPU memory stats
                gpu_stats = get_gpu_memory_stats()
                gpu_info = ""
                if gpu_stats["status"] != "CPU_MODE":
                    gpu_info = f" | GPU: {gpu_stats['allocated_gb']:.2f}GB/{gpu_stats['total_gb']:.2f}GB ({gpu_stats['percent_used']:.1f}%)"
                    if gpu_stats["status"] == "CRITICAL":
                        log.warning(
                            "🚨 GPU CRITICAL: FPS=%.1f | tracks=%d | alerts=%d%s",
                            last_fps,
                            len(tracks),
                            len(alerts),
                            gpu_info,
                        )
                    elif gpu_stats["status"] == "WARNING":
                        log.warning(
                            "⚠️  GPU WARNING: FPS=%.1f | tracks=%d | alerts=%d%s",
                            last_fps,
                            len(tracks),
                            len(alerts),
                            gpu_info,
                        )
                    else:
                        log.info(
                            "FPS=%.1f | tracks=%d | alerts=%d%s",
                            last_fps,
                            len(tracks),
                            len(alerts),
                            gpu_info,
                        )
                else:
                    log.info(
                        "FPS=%.1f | tracks=%d | alerts=%d",
                        last_fps,
                        len(tracks),
                        len(alerts),
                    )

            # ---- UI overlay ----
            # Pass cfg.ui (for flags) and last_fps (for HUD).
            try:
                display_img = draw_overlay(
                    frame,
                    tracks,
                    decisions,
                    events,
                    alerts,
                    ui_cfg=getattr(cfg, "ui", None),
                    fps=last_fps,
                )
            except TypeError:
                # Backwards-compat: if overlay doesn't accept ui_cfg/fps yet,
                # fall back to the legacy signature.
                display_img = draw_overlay(frame, tracks, decisions, events, alerts)

            cv2.imshow(window_title, display_img)

            # ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        try:
            if src is not None and hasattr(src, "stop"):
                src.stop()
        except Exception:
            # Never let cleanup crash the process.
            log.exception("Error while stopping camera source")
        cv2.destroyAllWindows()
        log.info("GaitGuard pipeline stopped.")


if __name__ == "__main__":
    run()
