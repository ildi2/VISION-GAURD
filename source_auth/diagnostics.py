
from __future__ import annotations

from typing import Optional, List, Iterable

from source_auth.types import (
    SourceAuthScores,
    SourceAuthDebug,
)


def _format_reliability_suffix(scores: SourceAuthScores) -> Optional[str]:
    rel = scores.reliability
    if rel is None:
        return None

    flags: List[str] = []
    if getattr(rel, "enough_motion", False):
        flags.append("motion")
    if getattr(rel, "enough_landmarks", False):
        flags.append("landmarks")
    if getattr(rel, "enough_background", False):
        flags.append("bg")

    if not flags:
        return None

    return "rel=" + ",".join(flags)


def _format_phase_suffix(debug: Optional[SourceAuthDebug]) -> Optional[str]:
    if not debug or not isinstance(debug, dict):
        return None

    phase = debug.get("phase")
    if not phase:
        return None

    return f"phase={phase}"


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _append_scalar(
    parts: List[str],
    label: str,
    value: object,
) -> None:
    v = _safe_float(value)
    if v is None:
        return
    parts.append(f"{label}={v:.2f}")


def _first_debug_scalar(
    debug: Optional[SourceAuthDebug],
    candidate_keys: Iterable[str],
) -> Optional[float]:
    if not debug or not isinstance(debug, dict):
        return None

    for key in candidate_keys:
        if key in debug:
            v = _safe_float(debug[key])
            if v is not None:
                return v
    return None


def format_source_auth_reason(
    scores: SourceAuthScores,
    debug: Optional[SourceAuthDebug] = None,
) -> str:
    raw_score = float(getattr(scores, "source_auth_score", 0.0))
    s = max(0.0, min(1.0, raw_score))

    state = getattr(scores, "state", None) or "UNCERTAIN"

    parts: List[str] = [
        f"src_auth:score={s:.2f}",
        f"state={state}",
    ]

    comps = getattr(scores, "components", None)
    if comps is not None:
        _append_scalar(parts, "3d", getattr(comps, "planar_3d", None))
        _append_scalar(parts, "screen", getattr(comps, "screen_artifacts", None))
        _append_scalar(
            parts,
            "bg",
            getattr(comps, "background_consistency", None),
        )

    parallax_val = _first_debug_scalar(
        debug,
        candidate_keys=[
            "motion_parallax_ratio",
            "motion_parallax",
            "motion_depth_cue",
        ],
    )
    if parallax_val is not None:
        _append_scalar(parts, "parallax", parallax_val)

    border_val = _first_debug_scalar(
        debug,
        candidate_keys=[
            "screen_border_strength",
            "screen_border_score",
            "screen_border_ratio",
        ],
    )
    if border_val is not None:
        _append_scalar(parts, "border", border_val)

    flicker_val = _first_debug_scalar(
        debug,
        candidate_keys=[
            "screen_flicker_score",
            "screen_temporal_inconsistency",
        ],
    )
    if flicker_val is not None:
        _append_scalar(parts, "flicker", flicker_val)

    bg_color_val = _first_debug_scalar(
        debug,
        candidate_keys=[
            "background_color_delta_norm",
            "background_color_mismatch",
        ],
    )
    if bg_color_val is not None:
        _append_scalar(parts, "bg_color", bg_color_val)

    bg_tex_val = _first_debug_scalar(
        debug,
        candidate_keys=[
            "background_texture_delta_norm",
            "background_texture_mismatch",
        ],
    )
    if bg_tex_val is not None:
        _append_scalar(parts, "bg_tex", bg_tex_val)

    rel_suffix = _format_reliability_suffix(scores)
    if rel_suffix:
        parts.append(rel_suffix)

    phase_suffix = _format_phase_suffix(debug)
    if phase_suffix:
        parts.append(phase_suffix)

    return ":".join(parts)
