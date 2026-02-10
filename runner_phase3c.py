#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.camera import CameraSource
from core.config import load_config
from core.device import select_device
from schemas import Frame, Tracklet
from schemas.identity_decision import IdentityDecision

from face.route import FaceRoute
from motion_analysis.gait.gait_engine import GaitEngine
from source_auth.engine import SourceAuthEngine

from vision_identity.simple_fusion import SimpleFusionEngine, FusionResult, FusionState, FaceInput, GaitInput

from perception.tracker_ocsort import OCSortTracker
from perception.detector import Detector, Detection


def setup_logging(log_file: Optional[str] = None, verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers
    )
    
    logging.getLogger('ultralytics').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


logger = logging.getLogger('PHASE3C')


@dataclass
class Phase3CMetrics:
    
    frames_processed: int = 0
    frames_per_second: float = 0.0
    total_runtime_sec: float = 0.0
    
    total_tracks_created: int = 0
    concurrent_tracks_peak: int = 0
    
    total_decisions: int = 0
    confirmed_decisions: int = 0
    tentative_decisions: int = 0
    conflict_resolutions: int = 0
    
    learning_gates_triggered: int = 0
    face_templates_learned: int = 0
    gait_templates_learned: int = 0
    
    bindings_created: int = 0
    bindings_strengthened: int = 0
    bindings_invalidated_by_conflict: int = 0
    binding_boosts_applied: int = 0
    
    memory_peak_mb: float = 0.0
    memory_current_mb: float = 0.0
    inference_latency_ms: Dict[str, float] = field(default_factory=dict)
    
    camera_reconnects: int = 0
    inference_errors: int = 0
    
    def print_summary(self):
        logger.info("\n" + "="*70)
        logger.info("PHASE 3C VALIDATION SUMMARY")
        logger.info("="*70)
        logger.info(f"Runtime: {self.total_runtime_sec:.1f}s | Frames: {self.frames_processed}")
        logger.info(f"FPS: {self.frames_per_second:.1f} | Tracks Created: {self.total_tracks_created}")
        logger.info(f"Peak Concurrent Tracks: {self.concurrent_tracks_peak}")
        logger.info(f"\nDecisions:")
        logger.info(f"  Total: {self.total_decisions} | Confirmed: {self.confirmed_decisions} | "
                   f"Tentative: {self.tentative_decisions}")
        logger.info(f"  Conflicts Resolved: {self.conflict_resolutions}")
        logger.info(f"\nLearning:")
        logger.info(f"  Gates Triggered: {self.learning_gates_triggered}")
        logger.info(f"  Face Templates Learned: {self.face_templates_learned}")
        logger.info(f"  Gait Templates Learned: {self.gait_templates_learned}")
        logger.info(f"\nBindings (Phase 3B):")
        logger.info(f"  Created: {self.bindings_created}")
        logger.info(f"  Strengthened: {self.bindings_strengthened}")
        logger.info(f"  Invalidated: {self.bindings_invalidated_by_conflict}")
        logger.info(f"  Boosts Applied: {self.binding_boosts_applied}")
        logger.info(f"\nMemory/Performance:")
        logger.info(f"  Peak: {self.memory_peak_mb:.1f}MB | Current: {self.memory_current_mb:.1f}MB")
        logger.info(f"  Errors: {self.inference_errors}")
        logger.info("="*70 + "\n")


class Phase3CRunner:
    
    def __init__(self, camera_mode: str = 'webcam', camera_source: Optional[str] = None):
        logger.info("[PHASE3C] Initializing deep robust system...")
        
        self.camera_mode = camera_mode
        self.camera_source = camera_source or 0
        self.metrics = Phase3CMetrics()
        
        logger.info(f"[CAMERA] Initializing {camera_mode} input...")
        if camera_mode == 'video':
            self.camera = self._init_video_source(str(self.camera_source))
        else:
            self.camera = self._init_webcam(int(self.camera_source) if isinstance(self.camera_source, str) else 0)
        
        if not self.camera:
            raise RuntimeError("[CAMERA] Failed to initialize camera/video")
        
        logger.info("[CONFIG] Loading system configuration...")
        self.config = load_config()
        self.device = select_device()
        logger.info(f"[DEVICE] Using: {self.device}")
        
        logger.info("[PERCEPTION] Initializing YOLO detector...")
        self.detector = Detector()
        
        logger.info("[PERCEPTION] Initializing OC-SORT tracker...")
        self.tracker = OCSortTracker()
        self.active_tracks: Dict[int, Dict] = {}
        
        logger.info("[FACE] Initializing face engine...")
        self.face_engine = FaceRoute()
        
        logger.info("[GAIT] Initializing gait engine...")
        self.gait_engine = GaitEngine()
        
        logger.info("[SOURCEAUTH] Initializing source auth engine...")
        self.sourceauth_engine = SourceAuthEngine()
        
        logger.info("[VISION-ID] Initializing fusion engine (Phase 3A + 3B)...")
        self.vision_identity = SimpleFusionEngine()
        
        logger.info("[PHASE3C] ✓ All systems initialized successfully")
    
    def _init_webcam(self, camera_id: int = 0) -> Optional[cv2.VideoCapture]:
        try:
            cap = cv2.VideoCapture(camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            ret, frame = cap.read()
            if not ret:
                logger.error(f"[CAMERA] Could not read from webcam {camera_id}")
                return None
            
            logger.info(f"[CAMERA] Webcam initialized: {frame.shape}")
            return cap
        except Exception as e:
            logger.error(f"[CAMERA] Webcam init error: {e}")
            return None
    
    def _init_video_source(self, video_path: str) -> Optional[cv2.VideoCapture]:
        try:
            if not Path(video_path).exists():
                logger.error(f"[CAMERA] Video file not found: {video_path}")
                return None
            
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if not ret:
                logger.error(f"[CAMERA] Could not read from video: {video_path}")
                return None
            
            logger.info(f"[CAMERA] Video initialized: {frame.shape}")
            return cap
        except Exception as e:
            logger.error(f"[CAMERA] Video init error: {e}")
            return None
    
    def run(self, max_frames: Optional[int] = None, show_output: bool = False):
        logger.info("[PHASE3C] Starting validation loop...")
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    logger.info("[PHASE3C] End of input or camera disconnected")
                    break
                
                frame_count += 1
                frame_time = time.time()
                
                if max_frames and frame_count > max_frames:
                    break
                
                if show_output:
                    should_continue = self._visualize_frame(frame, self.active_tracks)
                    if not should_continue:
                        logger.info("[PHASE3C] User closed window")
                        break
                
                try:
                    detections = self._detect_persons(frame)
                    if detections is None or len(detections) == 0:
                        if frame_count % 30 == 0:
                            logger.debug(f"[DETECTION] No persons detected in frame {frame_count}")
                        continue
                    
                    tracked = self.tracker.update(detections)
                    
                    active_track_ids = set()
                    for detection in tracked:
                        track_id = int(detection[4])
                        bbox = detection[:4]
                        active_track_ids.add(track_id)
                        
                        if track_id not in self.active_tracks:
                            self.active_tracks[track_id] = {
                                'created_at': frame_time,
                                'last_seen': frame_time,
                                'frame_count': 0,
                                'face_results': [],
                                'gait_results': [],
                                'vision_decisions': [],
                            }
                            self.metrics.total_tracks_created += 1
                        
                        self.active_tracks[track_id]['last_seen'] = frame_time
                        self.active_tracks[track_id]['frame_count'] += 1
                        self.active_tracks[track_id]['bbox'] = bbox
                    
                    self.metrics.concurrent_tracks_peak = max(
                        self.metrics.concurrent_tracks_peak,
                        len(active_track_ids)
                    )
                    
                except Exception as e:
                    logger.error(f"[PERCEPTION] Error: {e}")
                    self.metrics.inference_errors += 1
                    continue
                
                for track_id, track_data in list(self.active_tracks.items()):
                    if track_id not in active_track_ids:
                        continue
                    
                    bbox = track_data.get('bbox')
                    if bbox is None:
                        continue
                    
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                        person_region = frame[y1:y2, x1:x2]
                        
                        if person_region.shape[0] < 50 or person_region.shape[1] < 50:
                            continue
                        
                        face_decision = self.face_engine.recognize(person_region)
                        if face_decision:
                            track_data['face_results'].append({
                                'time': frame_time,
                                'decision': face_decision
                            })
                        
                        gait_decision = self.gait_engine.recognize(person_region)
                        if gait_decision:
                            track_data['gait_results'].append({
                                'time': frame_time,
                                'decision': gait_decision
                            })
                        
                        evidence = self._build_evidence(track_id, track_data, frame_time)
                        
                        face_input = None
                        if 'face' in evidence:
                            face_dec = evidence['face']
                            face_input = FaceInput(
                                identity_id=getattr(face_dec, 'identity_id', None),
                                confidence=getattr(face_dec, 'confidence', 0.0),
                                quality=getattr(face_dec, 'quality', 0.0),
                                bbox=bbox if bbox else None
                            )
                        
                        gait_input = None
                        if 'gait' in evidence:
                            gait_dec = evidence['gait']
                            gait_input = GaitInput(
                                identity_id=getattr(gait_dec, 'identity_id', None),
                                confidence=getattr(gait_dec, 'confidence', 0.0),
                                quality=getattr(gait_dec, 'quality', 0.0),
                                confirm_streak=getattr(gait_dec, 'confirm_streak', 0)
                            )
                        
                        vision_decision = self.vision_identity.fuse(
                            track_id=track_id,
                            face_input=face_input,
                            gait_input=gait_input,
                            timestamp=frame_time
                        )
                        
                        if vision_decision:
                            track_data['vision_decisions'].append({
                                'time': frame_time,
                                'decision': vision_decision
                            })
                            
                            self._update_metrics(vision_decision, track_id)
                    
                    except Exception as e:
                        logger.error(f"[FUSION] Track {track_id} error: {e}")
                        self.metrics.inference_errors += 1
                
                self._cleanup_stale_tracks(frame_time)
                
                self.metrics.frames_processed = frame_count
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    self.metrics.total_runtime_sec = elapsed
                    self.metrics.frames_per_second = frame_count / elapsed if elapsed > 0 else 0
                
                if frame_count % 100 == 0:
                    logger.info(f"[PROGRESS] Frame {frame_count} | Tracks: {len(self.active_tracks)} | "
                               f"Decisions: {self.metrics.total_decisions}")
        
        except KeyboardInterrupt:
            logger.info("[PHASE3C] User interrupted")
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
            self.metrics.total_runtime_sec = time.time() - start_time
            self.metrics.print_summary()
    
    def _detect_persons(self, frame: np.ndarray) -> Optional[np.ndarray]:
        try:
            detections = self.detector.detect(frame)
            if not detections:
                return None
            
            det_array = np.array([
                [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence]
                for d in detections
            ])
            return det_array
        except Exception as e:
            logger.error(f"[DETECTION] Error: {e}")
            return None
    
    def _build_evidence(self, track_id: int, track_data: Dict, frame_time: float) -> Dict:
        evidence = {}
        
        if track_data.get('face_results'):
            latest_face = track_data['face_results'][-1]
            evidence['face'] = latest_face['decision']
        
        if track_data.get('gait_results'):
            latest_gait = track_data['gait_results'][-1]
            evidence['gait'] = latest_gait['decision']
        
        return evidence
    
    def _update_metrics(self, decision: FusionResult, track_id: int):
        self.metrics.total_decisions += 1
        
        if decision.state == FusionState.FUSED:
            self.metrics.confirmed_decisions += 1
        elif decision.state == FusionState.FACE_ONLY or decision.state == FusionState.GAIT_ONLY:
            self.metrics.tentative_decisions += 1
    
    def _cleanup_stale_tracks(self, frame_time: float, timeout_sec: float = 10.0):
        stale_ids = [
            tid for tid, data in self.active_tracks.items()
            if frame_time - data['last_seen'] > timeout_sec
        ]
        
        for track_id in stale_ids:
            del self.active_tracks[track_id]
    
    def _visualize_frame(self, frame: np.ndarray, tracks: Dict):
        display = frame.copy()
        
        for track_id, data in tracks.items():
            bbox = data.get('bbox')
            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, f"T{track_id}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Phase 3C - Vision Identity Fusion", display)
        if cv2.waitKey(1) & 0xFF == 27:
            return False
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3C - End-to-End Vision Identity Biometric Validation"
    )
    parser.add_argument('--mode', choices=['webcam', 'video'], default='webcam',
                       help='Input mode: webcam or video file')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID for webcam (default: 0)')
    parser.add_argument('--video', type=str,
                       help='Path to video file (required if mode=video)')
    parser.add_argument('--max-frames', type=int,
                       help='Stop after N frames (for testing)')
    parser.add_argument('--show', action='store_true',
                       help='Display visualization')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--log', type=str,
                       help='Log file path')
    
    args = parser.parse_args()
    
    setup_logging(log_file=args.log, verbose=args.verbose)
    
    if args.mode == 'video' and not args.video:
        parser.error('--video is required when mode=video')
    
    camera_source = args.video if args.mode == 'video' else args.camera
    runner = Phase3CRunner(camera_mode=args.mode, camera_source=camera_source)
    runner.run(max_frames=args.max_frames, show_output=args.show)


if __name__ == '__main__':
    main()
