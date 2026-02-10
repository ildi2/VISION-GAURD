
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import logging
import traceback
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from schemas import Frame, Tracklet

from vision_identity.simple_fusion import (
    SimpleFusionEngine,
    FusionResult,
    FusionState,
    FusionWeights,
    FaceInput,
    GaitInput,
    create_face_input_from_decision,
    create_gait_input_from_signal,
    format_fusion_result,
    get_fusion_engine,
)
from vision_identity.identity_registry import get_identity_registry

logger = logging.getLogger(__name__)


@dataclass
class SimpleRunnerConfig:
    
    camera_device_id: int = 0
    video_file_path: Optional[str] = None
    
    face_weight: float = 0.75
    gait_weight: float = 0.25
    
    frame_skip: int = 0
    max_frames: Optional[int] = None
    
    display_results: bool = True
    display_fps: bool = True
    window_name: str = "Vision Identity"
    
    log_level: str = "INFO"
    console_output: bool = True


class SimpleVisionRunner:
    
    def __init__(self, config: SimpleRunnerConfig):
        self.config = config
        self._setup_logging()
        
        weights = FusionWeights(
            face_weight=config.face_weight,
            gait_weight=config.gait_weight
        )
        self.fusion_engine = SimpleFusionEngine(weights=weights)
        self.registry = get_identity_registry()
        
        self.perception_engine = None
        
        self.face_engine = None
        self._init_face_engine()
        
        self.gait_engine = None
        self._init_gait_engine()
        
        self._running = False
        self._frame_count = 0
        self._fps_counter = FPSCounter()
        
        self._track_results: Dict[int, FusionResult] = {}
        
        self._current_tracklets: List[Tracklet] = []
        
        self._sticky_identities: Dict[int, dict] = {}
        self._sticky_duration: float = 3.0
        
        logger.info(
            f"[SIMPLE-RUNNER] Initialized: "
            f"face_weight={config.face_weight}, gait_weight={config.gait_weight}"
        )
    
    def _setup_logging(self):
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def _init_face_engine(self):
        try:
            from identity.identity_engine import FaceIdentityEngine
            from face.config import default_face_config
            
            face_cfg = default_face_config()
            self.face_engine = FaceIdentityEngine(face_cfg=face_cfg)
            logger.info("[SIMPLE-RUNNER] ✓ Face engine initialized")
            
        except ImportError as e:
            logger.warning(f"[SIMPLE-RUNNER] Face engine not available: {e}")
        except Exception as e:
            logger.error(f"[SIMPLE-RUNNER] Face engine init failed: {e}")
    
    def _init_gait_engine(self):
        try:
            from motion_analysis.gait.gait_engine import GaitEngine
            from motion_analysis.gait.config import default_gait_config
            
            gait_cfg = default_gait_config()
            self.gait_engine = GaitEngine(config=gait_cfg)
            logger.info("[SIMPLE-RUNNER] ✓ Gait engine initialized")
            
        except ImportError as e:
            logger.warning(f"[SIMPLE-RUNNER] Gait engine not available: {e}")
        except Exception as e:
            logger.error(f"[SIMPLE-RUNNER] Gait engine init failed: {e}")
    
    def _init_perception(self):
        if self.perception_engine is not None:
            return
        
        try:
            from motion_analysis.run_gait_standalone import GaitPerceptionEngine
            from motion_analysis.gait.config import default_gait_config
            
            gait_cfg = default_gait_config()
            self.perception_engine = GaitPerceptionEngine(gait_cfg)
            logger.info("[SIMPLE-RUNNER] ✓ Perception engine initialized")
            
        except ImportError as e:
            logger.warning(f"[SIMPLE-RUNNER] Perception engine not available: {e}")
        except Exception as e:
            logger.error(f"[SIMPLE-RUNNER] Perception init failed: {e}")
    
    
    def run(self):
        import cv2
        
        logger.info("[SIMPLE-RUNNER] Starting main loop...")
        
        cap = self._open_video_source()
        if cap is None:
            logger.error("[SIMPLE-RUNNER] Failed to open video source")
            return
        
        self._init_perception()
        
        self._running = True
        self._frame_count = 0
        
        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    if self.config.video_file_path:
                        logger.info("[SIMPLE-RUNNER] End of video")
                        break
                    continue
                
                self._frame_count += 1
                
                if self.config.frame_skip > 0:
                    if self._frame_count % (self.config.frame_skip + 1) != 0:
                        continue
                
                if self.config.max_frames and self._frame_count >= self.config.max_frames:
                    logger.info(f"[SIMPLE-RUNNER] Max frames reached: {self.config.max_frames}")
                    break
                
                results = self._process_frame(frame)
                
                if self.config.display_results:
                    self._visualize(frame, results)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        logger.info("[SIMPLE-RUNNER] ESC pressed, stopping")
                        break
                    elif key == ord('q'):
                        break
                
                if self.config.console_output and results:
                    for result in results:
                        if result.state != FusionState.UNKNOWN:
                            print(format_fusion_result(result))
        
        except KeyboardInterrupt:
            logger.info("[SIMPLE-RUNNER] Interrupted")
        except Exception as e:
            logger.error(f"[SIMPLE-RUNNER] Error: {e}")
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info(f"[SIMPLE-RUNNER] Stopped after {self._frame_count} frames")
    
    def _open_video_source(self):
        import cv2
        
        if self.config.video_file_path:
            cap = cv2.VideoCapture(self.config.video_file_path)
            if cap.isOpened():
                logger.info(f"[SIMPLE-RUNNER] Opened video: {self.config.video_file_path}")
                return cap
        else:
            cap = cv2.VideoCapture(self.config.camera_device_id)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                logger.info(f"[SIMPLE-RUNNER] Opened camera: {self.config.camera_device_id}")
                return cap
        
        return None
    
    
    def _process_frame(self, frame) -> List[FusionResult]:
        results = []
        timestamp = time.time()
        
        frame_obj = Frame(
            frame_id=self._frame_count,
            ts=timestamp,
            camera_id="cam0",
            size=(frame.shape[1], frame.shape[0]),
            image=frame
        )
        
        tracklets = self._run_perception(frame_obj)
        
        self._current_tracklets = tracklets
        
        for tracklet in tracklets:
            try:
                result = self._process_tracklet(tracklet, frame_obj, timestamp)
                if result:
                    results.append(result)
                    self._track_results[tracklet.track_id] = result
            except Exception as e:
                logger.error(f"[SIMPLE-RUNNER] Error processing track {tracklet.track_id}: {e}")
        
        return results
    
    def _run_perception(self, frame_obj: Frame) -> List[Tracklet]:
        if self.perception_engine is None:
            return []
        
        try:
            tracklets, _ = self.perception_engine.process_frame(frame_obj)
            return tracklets
        except Exception as e:
            logger.error(f"[SIMPLE-RUNNER] Perception error: {e}")
            return []
    
    def _process_tracklet(
        self,
        tracklet: Tracklet,
        frame_obj: Frame,
        timestamp: float
    ) -> Optional[FusionResult]:
        track_id = tracklet.track_id
        
        face_input = None
        if self.face_engine:
            try:
                face_signals = self.face_engine.update_signals(
                    frame=frame_obj,
                    tracks=[tracklet]
                )
                
                if face_signals:
                    face_decisions = self.face_engine.decide(face_signals)
                    if face_decisions:
                        face_decision = face_decisions[0]
                        face_input = create_face_input_from_decision(face_decision)
                        
                        if face_input and hasattr(tracklet, 'last_box') and tracklet.last_box:
                            face_input.bbox = tuple(int(v) for v in tracklet.last_box)
            
            except Exception as e:
                logger.debug(f"[SIMPLE-RUNNER] Face engine error: {e}")
        
        gait_input = None
        if self.gait_engine:
            try:
                gait_signals = self.gait_engine.update_signals(
                    frame=frame_obj,
                    tracks=[tracklet]
                )
                
                gait_state = None
                if track_id in self.gait_engine._track_states:
                    gait_state = self.gait_engine._track_states[track_id]
                
                if gait_signals:
                    gait_signal = gait_signals[0]
                    gait_input = create_gait_input_from_signal(gait_signal, gait_state)
            
            except Exception as e:
                logger.debug(f"[SIMPLE-RUNNER] Gait engine error: {e}")
        
        result = self.fusion_engine.fuse(
            track_id=track_id,
            face_input=face_input,
            gait_input=gait_input,
            timestamp=timestamp
        )
        
        return result
    
    
    def _update_sticky_identity(self, track_id: int, result: FusionResult, current_time: float):
        is_recognized = (
            result.display_name is not None and 
            result.combined_confidence >= 0.40
        )
        
        if is_recognized:
            self._sticky_identities[track_id] = {
                'name': result.display_name,
                'confidence': result.combined_confidence,
                'last_seen': current_time,
                'color_rgb': result.color_rgb,
                'state': result.state,
                'face_id': result.face_id,
                'gait_id': result.gait_id,
            }
        
        if track_id in self._sticky_identities:
            sticky = self._sticky_identities[track_id]
            age = current_time - sticky['last_seen']
            if age <= self._sticky_duration:
                return sticky
            else:
                del self._sticky_identities[track_id]
        
        return None
    
    def _cleanup_stale_sticky(self, current_time: float, active_track_ids: set):
        stale_tracks = []
        for track_id, sticky in self._sticky_identities.items():
            if track_id not in active_track_ids:
                age = current_time - sticky['last_seen']
                if age > self._sticky_duration:
                    stale_tracks.append(track_id)
        
        for track_id in stale_tracks:
            del self._sticky_identities[track_id]
    
    
    def _visualize(self, frame, results: List[FusionResult]):
        import cv2
        
        current_time = time.time()
        
        SKELETON = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        result_map = {r.track_id: r for r in results}
        
        tracklets = self._current_tracklets
        
        active_track_ids = {t.track_id for t in tracklets}
        self._cleanup_stale_sticky(current_time, active_track_ids)
        
        for tracklet in tracklets:
            track_id = tracklet.track_id
            
            result = result_map.get(track_id) or self._track_results.get(track_id)
            
            sticky_info = None
            if result:
                sticky_info = self._update_sticky_identity(track_id, result, current_time)
            else:
                if track_id in self._sticky_identities:
                    sticky = self._sticky_identities[track_id]
                    age = current_time - sticky['last_seen']
                    if age <= self._sticky_duration:
                        sticky_info = sticky
            
            if sticky_info:
                color = sticky_info['color_rgb'][::-1]
                display_name = sticky_info['name']
                display_conf = sticky_info['confidence']
            elif result and result.display_name:
                color = result.color_rgb[::-1]
                display_name = result.display_name
                display_conf = result.combined_confidence
            else:
                color = (0, 200, 255)
                display_name = None
                display_conf = 0.0
            
            if hasattr(tracklet, 'last_box') and tracklet.last_box:
                x1, y1, x2, y2 = [int(v) for v in tracklet.last_box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            if hasattr(tracklet, 'gait_sequence_data') and tracklet.gait_sequence_data:
                kp = tracklet.gait_sequence_data[-1]
                
                for (start, end) in SKELETON:
                    if start >= len(kp) or end >= len(kp):
                        continue
                    if kp[start, 2] < 0.3 or kp[end, 2] < 0.3:
                        continue
                    
                    pt1 = (int(kp[start, 0]), int(kp[start, 1]))
                    pt2 = (int(kp[end, 0]), int(kp[end, 1]))
                    cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
                
                for i, point in enumerate(kp):
                    if point[2] < 0.3:
                        continue
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)
            
            if sticky_info:
                name = sticky_info['name']
                if len(name) > 15:
                    name = name[:13] + ".."
                conf_pct = int(sticky_info['confidence'] * 100)
                state = sticky_info.get('state', FusionState.FACE_ONLY)
                state_marker = {
                    FusionState.FUSED: "F+G",
                    FusionState.FACE_ONLY: "F",
                    FusionState.GAIT_ONLY: "G",
                    FusionState.CONFLICT: "!",
                    FusionState.UNKNOWN: "?"
                }
                marker = state_marker.get(state, "F")
                label = f"{name} ({conf_pct}%) [{marker}]"
            elif display_name:
                if len(display_name) > 15:
                    display_name = display_name[:13] + ".."
                conf_pct = int(display_conf * 100)
                label = f"{display_name} ({conf_pct}%) [F]"
            else:
                label = f"Track-{track_id}: Scanning..."
            
            label_pos = self._get_label_position(tracklet, result)
            
            if label_pos:
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
                lx, ly = label_pos
                lx = max(0, min(lx, frame.shape[1] - tw - 10))
                ly = max(th + 10, min(ly, frame.shape[0] - 10))
                
                cv2.rectangle(frame, (lx - 5, ly - th - 5), (lx + tw + 5, ly + 5), color, -1)
                cv2.putText(frame, label, (lx, ly), font, 0.6, (0, 0, 0), 2)
        
        if self.config.display_fps:
            self._fps_counter.tick()
            fps_text = f"FPS: {self._fps_counter.fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow(self.config.window_name, frame)
    
    def _format_label_for_track(self, track_id: int, result: Optional[FusionResult]) -> str:
        if result is None:
            return f"Track-{track_id}: Detecting..."
        
        if not result.is_recognized():
            return f"Track-{track_id}: Scanning..."
        
        conf_pct = int(result.combined_confidence * 100)
        state_marker = {
            FusionState.FUSED: "F+G",
            FusionState.FACE_ONLY: "F",
            FusionState.GAIT_ONLY: "G",
            FusionState.CONFLICT: "!",
            FusionState.UNKNOWN: "?"
        }
        marker = state_marker.get(result.state, "")
        
        name = result.display_name or "Unknown"
        if len(name) > 15:
            name = name[:13] + ".."
        
        return f"{name} ({conf_pct}%) [{marker}]"
    
    def _get_label_position(self, tracklet, result: Optional[FusionResult]) -> Optional[Tuple[int, int]]:
        if tracklet is None:
            return None
        
        if hasattr(tracklet, 'gait_sequence_data') and tracklet.gait_sequence_data:
            kp = tracklet.gait_sequence_data[-1]
            for idx in [0, 1, 2, 5, 6]:
                if idx < len(kp) and kp[idx, 2] > 0.5:
                    return (int(kp[idx, 0]) - 50, int(kp[idx, 1]) - 40)
        
        if hasattr(tracklet, 'last_box') and tracklet.last_box:
            x1, y1, _, _ = tracklet.last_box
            return (int(x1), int(y1) - 10)
        
        return None
    
    def stop(self):
        self._running = False


class FPSCounter:
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = []
        self.fps = 0.0
    
    def tick(self):
        now = time.time()
        self.timestamps.append(now)
        
        while len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        
        if len(self.timestamps) >= 2:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed > 0:
                self.fps = (len(self.timestamps) - 1) / elapsed


def run_vision_identity(
    camera_id: int = 0,
    video_path: Optional[str] = None,
    face_weight: float = 0.75,
    gait_weight: float = 0.25,
    max_frames: Optional[int] = None,
    display: bool = True
):
    config = SimpleRunnerConfig(
        camera_device_id=camera_id,
        video_file_path=video_path,
        face_weight=face_weight,
        gait_weight=gait_weight,
        max_frames=max_frames,
        display_results=display
    )
    
    runner = SimpleVisionRunner(config)
    runner.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Vision Identity Runner")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--video", type=str, default=None, help="Video file path")
    parser.add_argument("--face-weight", type=float, default=0.75, help="Face weight (0-1)")
    parser.add_argument("--gait-weight", type=float, default=0.25, help="Gait weight (0-1)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    
    args = parser.parse_args()
    
    run_vision_identity(
        camera_id=args.camera,
        video_path=args.video,
        face_weight=args.face_weight,
        gait_weight=args.gait_weight,
        max_frames=args.max_frames,
        display=not args.no_display
    )
