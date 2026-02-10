
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np

from schemas import Frame, Tracklet, IdentityDecision, EventFlags, Alert


def _get_ui_flag(ui_cfg: Any, name: str, default: bool) -> bool:
    if ui_cfg is None:
        return default

    if isinstance(ui_cfg, dict):
        val = ui_cfg.get(name, default)
    else:
        val = getattr(ui_cfg, name, default)

    try:
        return bool(val)
    except Exception:
        return default


def _norm_xyxy(arr: Any) -> Optional[Tuple[int, int, int, int]]:
    if arr is None:
        return None

    a = np.asarray(arr).reshape(-1)
    if a.size != 4:
        return None

    x_candidates = [a[0], a[2]]
    y_candidates = [a[1], a[3]]

    x1 = int(min(x_candidates))
    y1 = int(min(y_candidates))
    x2 = int(max(x_candidates))
    y2 = int(max(y_candidates))

    return x1, y1, x2, y2


def _extract_bbox(trk: Tracklet) -> Optional[Tuple[int, int, int, int]]:
    for attr in ("tlbr", "bbox_xyxy", "bbox", "xyxy"):
        if hasattr(trk, attr):
            return _norm_xyxy(getattr(trk, attr))

    if hasattr(trk, "xywh"):
        x, y, w, h = np.asarray(getattr(trk, "xywh")).reshape(-1).astype(float)
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        return _norm_xyxy([x1, y1, x2, y2])

    try:
        for name, value in vars(trk).items():
            if isinstance(value, (list, tuple, np.ndarray)):
                a = np.asarray(value).reshape(-1)
                if a.size == 4:
                    return _norm_xyxy(a)
    except Exception:
        pass

    return None


def _category_color(category: str) -> Tuple[int, int, int]:
    c = (category or "").lower()
    if c == "resident":
        return (0, 255, 0)
    if c == "visitor":
        return (255, 200, 0)
    if c == "watchlist":
        return (0, 0, 255)
    return (200, 200, 200)


def _build_decision_map(
    decisions: List[IdentityDecision],
) -> Dict[int, IdentityDecision]:
    by_track: Dict[int, IdentityDecision] = {}
    for d in decisions:
        try:
            tid = int(d.track_id)
        except Exception:
            continue
        by_track[tid] = d
    return by_track


def _extract_id_source(decision: Optional[IdentityDecision]) -> str:
    if decision is None:
        return "U"
    
    if hasattr(decision, 'id_source') and decision.id_source is not None:
        source = str(decision.id_source)
        if source in ("F", "G", "U"):
            return source
    
    if hasattr(decision, 'extra') and decision.extra:
        try:
            source = decision.extra.get('id_source')
            if source in ("F", "G", "U"):
                return str(source)
        except (AttributeError, TypeError):
            pass
    
    return "U"


def _compute_identity_consensus(
    decisions: List[IdentityDecision],
) -> Optional[str]:
    if not decisions:
        return None
    
    person_counts: Dict[Optional[str], int] = {}
    for decision in decisions:
        try:
            pid = decision.identity_id
            if pid and pid != "unknown":
                person_counts[pid] = person_counts.get(pid, 0) + 1
        except Exception:
            pass
    
    if not person_counts:
        return None
    
    total_tracks = len(decisions)
    majority_threshold = total_tracks * 0.5
    
    for person_id, count in person_counts.items():
        if count >= majority_threshold:
            return person_id
    
    return max(person_counts.items(), key=lambda x: x[1])[0]


def _identity_label(
    decision: Optional[IdentityDecision],
    ui_cfg: Any = None,
    consensus_person: Optional[str] = None,
) -> Tuple[str, str, Tuple[int, int, int]]:
    if decision is None:
        if consensus_person:
            name = consensus_person
            color = _category_color("resident")
            return f"{name} (consensus)", "", color
        else:
            return "unknown", "", _category_color("unknown")

    if hasattr(decision, "display_name") and getattr(decision, "display_name"):
        name = str(getattr(decision, "display_name"))
    elif hasattr(decision, "name") and getattr(decision, "name"):
        name = str(getattr(decision, "name"))
    elif getattr(decision, "identity_id", None):
        name = str(decision.identity_id)
    else:
        if consensus_person:
            name = consensus_person
        else:
            name = "unknown"

    binding_state = getattr(decision, "binding_state", "UNKNOWN") or "UNKNOWN"
    if not isinstance(binding_state, str):
        binding_state = str(binding_state)
    binding_confidence = getattr(decision, "binding_confidence", 0.0)
    try:
        binding_confidence = float(binding_confidence)
    except (ValueError, TypeError):
        binding_confidence = 0.0
    
    binding_emoji = ""

    id_source = _extract_id_source(decision)
    
    source_marker = ""
    if id_source == "F":
        source_marker = "[F]"
    elif id_source == "G":
        source_marker = "[G]"
    elif id_source == "U":
        source_marker = "[U]"
    
    conf = getattr(decision, "confidence", None)
    if conf is None:
        main_label = name
        if binding_emoji:
            main_label = f"{binding_emoji} {name}"
        if source_marker:
            main_label = f"{main_label} {source_marker}"
    else:
        try:
            c = max(0.0, min(float(conf), 1.0))
        except Exception:
            c = 0.0
        if binding_emoji:
            if source_marker:
                main_label = f"{binding_emoji} {name} {source_marker} ({c:.2f})"
            else:
                main_label = f"{binding_emoji} {name} ({c:.2f})"
        else:
            if source_marker:
                main_label = f"{name} {source_marker} ({c:.2f})"
            else:
                main_label = f"{name} ({c:.2f})"

    reason = getattr(decision, "reason", "") or ""
    if len(reason) > 48:
        reason = reason[:45] + "..."

    dbg = reason

    show_engine_tag = _get_ui_flag(ui_cfg, "show_engine_tag", False)
    show_pose_tag = _get_ui_flag(ui_cfg, "show_pose_tag", False)
    show_binding_state = _get_ui_flag(ui_cfg, "show_binding_state", True)

    tags: List[str] = []
    if show_engine_tag and hasattr(decision, "engine"):
        try:
            eng = str(getattr(decision, "engine") or "").strip()
        except Exception:
            eng = ""
        if eng:
            tags.append(eng)

    if show_pose_tag and hasattr(decision, "pose_bin"):
        try:
            pb = str(getattr(decision, "pose_bin") or "").strip()
        except Exception:
            pb = ""
        if pb and pb.lower() != "none":
            tags.append(pb)
    
    if show_binding_state and binding_state:
        try:
            bs = str(binding_state).upper()
            if bs != "BYPASS":
                tags.append(f"binding:{bs}")
        except Exception:
            pass

    if tags:
        tag_str = "[" + ",".join(tags) + "]"
        dbg = (dbg + " " + tag_str).strip() if dbg else tag_str

    base_color = _category_color(getattr(decision, "category", "unknown") or "unknown")
    
    if binding_state in ["CONFIRMED_STRONG", "CONFIRMED_WEAK"]:
        color = (0, 255, 0)
    elif binding_state == "GPS_CARRY":
        color = (255, 255, 0)
    elif binding_state in ["PENDING", "SWITCH_PENDING"]:
        color = (0, 165, 255)
    elif binding_state in ["UNKNOWN", "BYPASS", "STALE", "ERROR"] or binding_state is None:
        color = (128, 128, 128)
    else:
        color = base_color
    
    return main_label, dbg, color


def _get_source_auth_state_and_score(
    decision: Optional[IdentityDecision],
) -> Tuple[Optional[str], Optional[float]]:
    if decision is None:
        return None, None

    state = getattr(decision, "source_auth_state", None)
    if state is not None:
        try:
            state = str(state).upper()
        except Exception:
            state = None

    score_raw = getattr(decision, "source_auth_score", None)
    score: Optional[float]
    try:
        score = float(score_raw) if score_raw is not None else None
    except Exception:
        score = None

    return state, score


def _apply_source_auth_color(
    decision: Optional[IdentityDecision],
    base_color: Tuple[int, int, int],
    ui_cfg: Any = None,
) -> Tuple[int, int, int]:
    show_border_hint = _get_ui_flag(ui_cfg, "show_source_auth_border", True)
    if not show_border_hint or decision is None:
        return base_color

    state, _ = _get_source_auth_state_and_score(decision)
    if not state:
        return base_color

    if state in ("SPOOF", "LIKELY_SPOOF"):
        return (0, 0, 255)

    return base_color


def _source_auth_badge(
    decision: Optional[IdentityDecision],
    ui_cfg: Any = None,
) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    if not _get_ui_flag(ui_cfg, "show_source_auth_tag", True):
        return None

    if decision is None:
        return None

    state, score = _get_source_auth_state_and_score(decision)
    if state is None and score is None:
        return None

    s = (state or "").upper()

    if s in ("SPOOF", "LIKELY_SPOOF"):
        return "SPOOF", (0, 0, 255)

    ok_threshold = 0.55
    if s in ("REAL", "LIKELY_REAL") and score is not None and score >= ok_threshold:
        return "REAL", (0, 200, 0)

    neutral = 0.5

    if score is None:
        return "UNC", (0, 165, 255)

    eps = 1e-3

    if abs(score - neutral) <= eps:
        return "UNC", (0, 165, 255)

    if score < neutral:
        return "SAS", (0, 0, 255)

    return "SAR", (0, 200, 0)


def _draw_source_auth_badge(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    badge: Tuple[str, Tuple[int, int, int]],
) -> None:
    if img is None or badge is None:
        return

    h, w = img.shape[:2]
    text, color = badge
    if not text:
        return

    x1, y1, x2, y2 = bbox

    margin_x = 4
    margin_y = 4

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    br_x2 = min(x2 - margin_x, w - 5)
    br_x1 = max(br_x2 - tw - 6, 5)
    br_y1 = max(y1 + margin_y, th + baseline + 5)
    br_y2 = br_y1 + th + baseline

    cv2.rectangle(
        img,
        (br_x1 - 1, br_y1 - 1),
        (br_x2 + 1, br_y2 + 1),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.rectangle(
        img,
        (br_x1, br_y1),
        (br_x2, br_y2),
        color,
        thickness=-1,
    )

    cv2.putText(
        img,
        text,
        (br_x1 + 3, br_y2 - baseline),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_overlay(
    frame: Frame,
    tracks: List[Tracklet],
    decisions: List[IdentityDecision],
    events: List[EventFlags],
    alerts: List[Alert],
    ui_cfg: Any = None,
    fps: Optional[float] = None,
) -> np.ndarray:
    if frame.image is None:
        raise ValueError("Frame.image is None inside draw_overlay")

    img = frame.image.copy()
    h, w = img.shape[:2]

    show_identity_labels = _get_ui_flag(ui_cfg, "show_identity_labels", True)
    show_debug_face_hud = _get_ui_flag(ui_cfg, "show_debug_face_hud", False)
    show_fps = _get_ui_flag(ui_cfg, "show_fps", True)

    if show_fps and fps is not None and fps > 0.0:
        status = f"FPS: {fps:4.1f} | tracks: {len(tracks)} | alerts: {len(alerts)}"
    else:
        status = f"tracks: {len(tracks)} | alerts: {len(alerts)}"

    cv2.putText(
        img,
        status,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    dec_by_track = _build_decision_map(decisions)

    consensus_person = _compute_identity_consensus(decisions)
    
    for trk in tracks:
        bbox = _extract_bbox(trk)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        tid = int(getattr(trk, "track_id", -1))

        decision = dec_by_track.get(tid)
        main_label, dbg_line, color = _identity_label(
            decision, ui_cfg=ui_cfg, consensus_person=consensus_person
        )

        color = _apply_source_auth_color(decision, color, ui_cfg=ui_cfg)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        sa_badge = _source_auth_badge(decision, ui_cfg=ui_cfg)
        if sa_badge is not None:
            _draw_source_auth_badge(img, bbox, sa_badge)

        if not show_identity_labels:
            continue

        label = f"ID {tid}: {main_label}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_x = x1
        label_y = max(y1 - 10, th + 5)

        cv2.rectangle(
            img,
            (label_x - 2, label_y - th - baseline),
            (label_x + tw + 2, label_y + baseline),
            (0, 0, 0),
            thickness=-1,
        )

        cv2.putText(
            img,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        if show_debug_face_hud and dbg_line:
            dbg_y = label_y + th + 6
            if dbg_y < h - 5:
                (dw, dh), db = cv2.getTextSize(
                    dbg_line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
                )
                cv2.rectangle(
                    img,
                    (label_x - 2, dbg_y - dh - db),
                    (label_x + dw + 2, dbg_y + db),
                    (0, 0, 0),
                    thickness=-1,
                )
                cv2.putText(
                    img,
                    dbg_line,
                    (label_x, dbg_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 200),
                    1,
                )
        
        show_continuity_debug = _get_ui_flag(ui_cfg, "show_continuity_debug", False)
        if show_continuity_debug and decision is not None:
            id_source = _extract_id_source(decision)
            if id_source == "G":
                age_frames = getattr(trk, "age_frames", 0)
                lost_frames = getattr(trk, "lost_frames", 0)
                track_conf = getattr(trk, "confidence", 0.0)
                
                try:
                    age_frames = int(age_frames)
                    lost_frames = int(lost_frames)
                    track_conf = float(track_conf)
                except (ValueError, TypeError):
                    age_frames = 0
                    lost_frames = 0
                    track_conf = 0.0
                
                cont_debug = f"GPS: age={age_frames} lost={lost_frames} conf={track_conf:.2f}"
                
                if show_debug_face_hud and dbg_line:
                    cont_y = dbg_y + dh + 6
                else:
                    cont_y = label_y + th + 6
                
                if cont_y < h - 5:
                    (cw, ch), cb = cv2.getTextSize(
                        cont_debug, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                    )
                    cv2.rectangle(
                        img,
                        (label_x - 2, cont_y - ch - cb),
                        (label_x + cw + 2, cont_y + cb),
                        (0, 0, 0),
                        thickness=-1,
                    )
                    cv2.putText(
                        img,
                        cont_debug,
                        (label_x, cont_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 255),
                        1,
                    )

    return img
