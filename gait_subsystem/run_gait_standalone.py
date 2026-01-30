#!/usr/bin/env python
# gait_subsystem/run_gait_standalone.py
"""
Standalone Gait Recognition Test Runner - DEEP ROBUST VERSION

This script runs the gait recognition system independently to verify
it works correctly before integration with the main face system.

USAGE:
    cd c:\\Users\\ildi\\Desktop\\GaitGuard - 2o
    python -m gait_subsystem.run_gait_standalone

WHAT IT DOES:
1. Opens camera
2. Runs YOLO-pose for person detection + skeleton extraction
3. Tracks persons with IOU-based tracking (Robust fallback to Mask/Keypoint IOU)
4. Collects pose sequences per track
5. Runs gait recognition when sequence is ready
6. Displays overlay with skeleton and gait recognition results

Press ESC to exit.
"""

from __future__ import annotations

import time
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from main system (shared components)
from core.camera import CameraSource
from core.config import load_config
from core.device import select_device
from schemas import Frame, Tracklet
from schemas.identity_decision import IdentityDecision

# Import gait-specific modules
from gait_subsystem.gait.config import default_gait_config
from gait_subsystem.gait.gait_engine import GaitEngine

# YOLO pose for skeleton extraction
from ultralytics import YOLO


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("gait_standalone")


# ============================================================================
# SKELETON DRAWING - COCO 17-KEYPOINT FORMAT
# ============================================================================
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Upper body
    (5, 11), (6, 12), (11, 12),          # Torso
    (11, 13), (13, 15), (12, 14), (14, 16) # Lower body
]

# Robust Colors
COLOR_SKELETON = {
    'face': (255, 200, 100), 'upper': (0, 255, 0),
    'torso': (255, 165, 0), 'lower': (255, 0, 255)
}

def draw_skeleton(img: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int], conf_threshold: float = 0.4) -> np.ndarray:
    """Draw skeleton on image with specific color."""
    if keypoints is None or len(keypoints) == 0:
        return img
    
    kp = keypoints
    
    # Draw connections
    for i, (start, end) in enumerate(SKELETON_CONNECTIONS):
        if start >= len(kp) or end >= len(kp): continue
        if kp[start, 2] < conf_threshold or kp[end, 2] < conf_threshold: continue
        
        pt1 = (int(kp[start, 0]), int(kp[start, 1]))
        pt2 = (int(kp[end, 0]), int(kp[end, 1]))
        
        cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)
    
    # Draw keypoints
    for i, point in enumerate(kp):
        if point[2] < conf_threshold: continue
        x, y = int(point[0]), int(point[1])
        cv2.circle(img, (x, y), 3, color, -1, cv2.LINE_AA)
    
    return img


class GaitPerceptionEngine:
    """
    Robust perception engine for gait standalone testing.
    
    FEATURES:
    - Prefer GPU (CUDA) for high efficiency
    - Robust box extraction (fallback to keypoint bounds)
    - EMA smoothing for stable keypoints
    """
    
    def __init__(self, gait_config=None):
        self.config = gait_config or default_gait_config()
        
        # 1. Determine Device - PREFER GPU (CUDA)
        # Even if OpenVINO model exists, if we have CUDA, use it with PyTorch model
        # because OpenVINO on CPU is slow (7 FPS) vs CUDA (30+ FPS).
        main_device, _ = select_device(prefer_gpu=True)
        
        self.use_gpu = False
        pose_model_path = "yolov8n-pose.pt" # Default PyTorch
        
        if main_device == "cuda":
            logger.info("🚀 CUDA GPU Detected - Using PyTorch model for maximum efficiency")
            self.use_gpu = True
            pose_model_path = "yolov8n-pose.pt" # Ultralytics will auto-download
        else:
            # Fallback to OpenVINO if available and NO GPU
            ov_path = self.config.models.pose_model_name
            if Path(ov_path).exists() and Path(ov_path).is_dir():
                logger.info(f"ℹ️ GPU not found, using OpenVINO model: {ov_path}")
                pose_model_path = ov_path
                self.use_gpu = False
            else:
                logger.info("ℹ️ Using CPU with PyTorch model")
                pose_model_path = "yolov8n-pose.pt"
                
        # Load Model
        logger.info(f"Loading Pose Model: {pose_model_path}")
        self.yolo = YOLO(pose_model_path)
        if self.use_gpu:
            self.yolo.to("cuda")
        
        # Detection Config
        self.det_conf = 0.25
        
        # Tracker State
        self._track_states: Dict[int, dict] = {}
        self._next_track_id = 1
        
        # Config
        self.ema_alpha = self.config.route.keypoint_ema_alpha
        self.max_history = self.config.route.keypoint_history_length
        
        logger.info(f"✅ GaitPerceptionEngine Ready (GPU: {self.use_gpu})")
    
    def process_frame(self, frame: Frame) -> Tuple[List[Tracklet], List[np.ndarray]]:
        """Process frame -> Tracklets + Display Keypoints"""
        if frame.image is None: return [], []
        
        # Predict
        device = "cuda" if self.use_gpu else "cpu"
        results = self.yolo.predict(
            source=frame.image, device=device, imgsz=640,
            conf=self.det_conf, classes=[0], verbose=False, task="pose"
        )
        
        if not results: return [], []
        
        result = results[0]
        
        # ROBUST EXTRACTION: Ensure we always get boxes if we have keypoints
        boxes, scores, keypoints = self._extract_robust(result, frame.image.shape)
        
        # Tracking
        matched, new_dets = self._match_tracks(boxes)
        
        # Update existing
        for track_id, det_idx in matched.items():
            self._update_track(track_id, det_idx, boxes, scores, keypoints, frame.frame_id)
            
        # Create new
        for det_idx in new_dets:
            self._create_track(det_idx, boxes, scores, keypoints, frame.frame_id)
            
        # Cleanup lost
        self._cleanup_tracks(matched)
        
        # Build Tracklets
        tracklets = []
        for track_id, state in self._track_states.items():
            if state["lost_frames"] > 0: continue
            
            t = Tracklet(
                track_id=track_id,
                camera_id="cam0",
                last_frame_id=state["last_frame_id"],
                last_box=tuple(state["last_box"]),
                confidence=state["confidence"],
                age_frames=state["age_frames"],
                lost_frames=state["lost_frames"]
            )
            t.gait_sequence_data = list(state["kp_history"])
            tracklets.append(t)
            
        return tracklets, keypoints # Return raw keypoints for display
    
    def _extract_robust(self, result, img_shape) -> Tuple[List, List, List]:
        """Robustly extract boxes and keypoints, inferring boxes from KPs if missing."""
        boxes, scores, keypoints = [], [], []
        h, w = img_shape[:2]
        
        try:
            # 1. Extract Keypoints (CRITICAL)
            if hasattr(result, "keypoints") and result.keypoints is not None and hasattr(result.keypoints, "xy"):
                kp_data = result.keypoints.xy.cpu().numpy()
                kp_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
                
                for i in range(kp_data.shape[0]):
                    if kp_conf is not None and len(kp_conf) > i:
                        kp = np.hstack([kp_data[i], kp_conf[i][:, None]])
                    else:
                        kp = np.hstack([kp_data[i], np.ones((kp_data[i].shape[0], 1))])
                    keypoints.append(kp)
            
            # 2. Extract Boxes (Standard)
            if hasattr(result, "boxes") and result.boxes is not None:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for i in range(xyxy.shape[0]):
                    boxes.append([float(xyxy[i, j]) for j in range(4)])
                    scores.append(float(confs[i]))
            
            # 3. ROBUST FIX: Mismatch handling
            # If we have keypoints but NO boxes, compute bounding box from keypoints
            if len(keypoints) > len(boxes):
                for i in range(len(boxes), len(keypoints)):
                    kp = keypoints[i]
                    # Filter valid keypoints (conf > 0.1)
                    valid_kp = kp[kp[:, 2] > 0.1]
                    if len(valid_kp) > 0:
                        min_x, min_y = np.min(valid_kp[:, :2], axis=0)
                        max_x, max_y = np.max(valid_kp[:, :2], axis=0)
                        # Add padding
                        pad = 10
                        boxes.append([max(0, min_x-pad), max(0, min_y-pad), 
                                      min(w, max_x+pad), min(h, max_y+pad)])
                        scores.append(0.5) # Default score
            
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            
        return boxes, scores, keypoints

    def _match_tracks(self, boxes: List) -> Tuple[Dict, List]:
        """IOU Matching"""
        matched = {}
        used_dets = set()
        
        for track_id, state in self._track_states.items():
            if state["lost_frames"] > 5: continue # Only match if recently seen
            
            best_iou = 0.2 # Loose threshold
            best_det = -1
            
            for i, box in enumerate(boxes):
                if i in used_dets: continue
                iou = self._iou(state["last_box"], box)
                if iou > best_iou:
                    best_iou = iou
                    best_det = i
            
            if best_det >= 0:
                matched[track_id] = best_det
                used_dets.add(best_det)
                
        new_dets = [i for i in range(len(boxes)) if i not in used_dets]
        return matched, new_dets

    def _iou(self, boxA, boxB) -> float:
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0

    def _update_track(self, track_id, det_idx, boxes, scores, keypoints, frame_id):
        state = self._track_states[track_id]
        state["last_box"] = boxes[det_idx]
        state["confidence"] = scores[det_idx]
        state["age_frames"] += 1
        state["lost_frames"] = 0
        state["last_frame_id"] = frame_id
        
        kp = keypoints[det_idx] if det_idx < len(keypoints) else None
        if kp is not None:
            if state["kp_ema"] is None: state["kp_ema"] = kp.copy()
            else: state["kp_ema"] = self.ema_alpha * kp + (1 - self.ema_alpha) * state["kp_ema"]
            state["kp_history"].append(state["kp_ema"].copy())

    def _create_track(self, det_idx, boxes, scores, keypoints, frame_id):
        track_id = self._next_track_id
        self._next_track_id += 1
        kp = keypoints[det_idx] if det_idx < len(keypoints) else None
        
        self._track_states[track_id] = {
            "last_box": boxes[det_idx],
            "confidence": scores[det_idx],
            "age_frames": 1,
            "lost_frames": 0,
            "last_frame_id": frame_id,
            "kp_ema": kp.copy() if kp is not None else None,
            "kp_history": deque(maxlen=self.max_history),
        }
        if kp is not None:
             self._track_states[track_id]["kp_history"].append(kp.copy())

    def _cleanup_tracks(self, matched):
        matched_ids = set(matched.keys())
        to_remove = []
        for track_id, state in self._track_states.items():
            if track_id not in matched_ids:
                state["lost_frames"] += 1
                if state["lost_frames"] > 60: # Keep state longer
                    to_remove.append(track_id)
        for t in to_remove: del self._track_states[t]


def draw_gait_overlay(frame: Frame, tracks: List[Tracklet], decisions: List[IdentityDecision]) -> np.ndarray:
    """
    Draw Deep Robust Overlay.
    
    Logic:
    - SKELETON Only (No Bounding Box)
    - GREEN Skeleton = Known (Resident)
    - WHITE Skeleton = Unknown
    """
    img = frame.image.copy()
    h, w = img.shape[:2]

    dec_map = {d.track_id: d for d in decisions}
    
    for track in tracks:
        # Determine Status & Color
        dec = dec_map.get(track.track_id)
        
        # [DEEP ROBUST UI]
        # Only show identity name if CONFIRMED.
        # If evaluating, show "Scanning..." to avoid false positive perception.
        gait_state = "UNKNOWN"
        if dec and dec.extra:
            gait_state = dec.extra.get("gait_state", "UNKNOWN")
            
        if dec and dec.identity_id and gait_state == "CONFIRMED":
            # KNOWN / MATCHED (Deep Confirmed)
            color = (0, 255, 0) # Green
            label = f"{dec.identity_id} ({dec.confidence:.2f})"
        elif dec and dec.identity_id and gait_state == "EVALUATING":
            # EVALUATING (Candidate Found but verifying)
            color = (0, 255, 255) # Yellow
            label = "Scanning..." # Hide name to prevent FP
        else:
            # UNKNOWN or COLLECTING
            color = (255, 255, 255) # White
            seq_len = len(track.gait_sequence_data)
            label = f"Unknown"
            
            # Show progress bar for gathering gait
            min_req = 30
            if seq_len < min_req:
                label += f" [{seq_len}/{min_req}]"
        
        # Draw Skeleton associated with this track
        if track.gait_sequence_data:
            latest_kp = track.gait_sequence_data[-1]
            draw_skeleton(img, latest_kp, color=color)
        
        # Position Label relative to head (Nose: index 0, or Eyes: 1,2)
        # Fallback to bounding box top if keypoints missing/low conf
        label_pos = None
        if track.gait_sequence_data:
            kp = track.gait_sequence_data[-1]
            # Try nose (0) -> Left Eye (1) -> Right Eye (2) -> Shoulders (5,6)
            for idx in [0, 1, 2, 5, 6]: 
                if idx < len(kp) and kp[idx, 2] > 0.5:
                    label_pos = (int(kp[idx, 0]), int(kp[idx, 1] - 20))
                    break
        
        if label_pos is None:
            # Fallback to Box Top
            x1, y1, x2, y2 = [int(v) for v in track.last_box]
            label_pos = (x1, y1 - 10)
            
        lx, ly = label_pos
        
        # Label Background
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
        
        # Ensure label fits in image
        lx = max(0, min(lx, w - tw))
        ly = max(th + 5, min(ly, h))
        
        cv2.rectangle(img, (lx - 5, ly - th - 5), (lx + tw + 5, ly + 5), color, -1)
        
        # Text Color: Black text on the colored background looks clear
        text_color = (0, 0, 0) 
        cv2.putText(img, label, (lx, ly), font, 0.6, text_color, 2)

    return img


def run_gait_standalone():
    logger.info("="*60)
    logger.info("🚀 GAIT RECOGNITION STANDALONE - DEEP ROBUST")
    logger.info("="*60)
    
    cfg = None
    try: cfg = load_config()
    except: pass
    
    gait_config = default_gait_config()
    perception = GaitPerceptionEngine(gait_config)
    gait_engine = GaitEngine(gait_config)
    
    # Camera
    cam_idx = getattr(cfg.camera, "index", 0) if cfg else 0
    camera = CameraSource(cam_index=cam_idx, w=1280, h=720, fps=30)
    camera.start()
    
    logger.info("📷 Camera active. Press ESC to exit.")
    
    frame_id = 0
    t0 = time.perf_counter()
    frames = 0
    
    try:
        while True:
            img = camera.read_latest(timeout=1.0)
            if img is None: continue
            
            ts = time.perf_counter()
            frame = Frame(frame_id=frame_id, ts=ts, camera_id="cam0", size=(img.shape[1], img.shape[0]), image=img)
            frame_id += 1
            
            # 1. Perception
            tracks, kps_display = perception.process_frame(frame)
            
            # 2. Recognition
            signals = gait_engine.update_signals(frame, tracks)
            decisions = gait_engine.decide(signals)
            
            # 3. Overlay
            display_img = draw_gait_overlay(frame, tracks, decisions)
            
            frames += 1
            if frames % 15 == 0:
                elapsed = ts - t0
                fps = frames / max(elapsed, 1e-6)
                logger.info(f"FPS={fps:.1f} | Tracks={len(tracks)}")
                
                # Deterministic Debug Log [DEEP ROBUST]
                for dec in decisions:
                    if not dec.extra: continue
                    
                    state = dec.extra.get("gait_state", "UNKNOWN")
                    reason = dec.extra.get("reason", "NONE")
                    
                    # Top 1
                    best_pid = dec.extra.get("best_pid", "N/A")
                    best_sim = dec.extra.get("best_sim", 0.0)
                    
                    # Top 2 & Margin (Critical for Hold diagnosis)
                    sec_pid = dec.extra.get("second_pid", "None")
                    sec_sim = dec.extra.get("second_sim", 0.0)
                    margin = dec.extra.get("margin", 0.0)
                    
                    streak = dec.extra.get("streak", 0)
                    q_seq = dec.extra.get("q_seq", 0.0)
                    
                    # Icons
                    icon = "👀 COLL"
                    if state == "CONFIRMED": icon = "✅ MATCH"
                    elif state == "EVALUATING": icon = "⏳ EVAL"
                    
                    # One line per track
                    logger.info(
                        f"   > {icon}: {best_pid}({best_sim:.2f}) "
                        f"| 2nd: {sec_pid}({sec_sim:.2f}) M:{margin:.2f} "
                        f"| {reason} | S:{streak} Q:{q_seq:.2f}"
                    )
            
            cv2.imshow("Gait Robust Test", display_img)
            if cv2.waitKey(1) & 0xFF == 27: break
            
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gait_standalone()
