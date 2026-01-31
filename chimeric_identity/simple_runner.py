# chimeric_identity/simple_runner.py
# ============================================================================
# SIMPLE CHIMERIC RUNNER - Inheritance-Based Architecture
# ============================================================================
#
# Purpose:
#   Run face and gait recognition TOGETHER with proper fusion.
#   This is what you described: camera shows bbox AND skeleton,
#   both turn green when recognized, confidence combines face + gait.
#
# Key Design:
#   - Face engine runs its logic (unchanged) → we read output
#   - Gait engine runs its logic (unchanged) → we read output
#   - Simple fusion combines them with weights
#   - Visualization shows both modalities unified
#
# NO complex state machines, NO re-implementation of recognition logic!

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports to work
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import logging
import traceback
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Core system imports
from schemas import Frame, Tracklet

# Chimeric fusion (simple version)
from chimeric_identity.simple_fusion import (
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
from chimeric_identity.identity_registry import get_identity_registry

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SimpleRunnerConfig:
    """Configuration for simple chimeric runner."""
    
    # Camera/video source
    camera_device_id: int = 0
    video_file_path: Optional[str] = None
    
    # Fusion weights
    face_weight: float = 0.75
    gait_weight: float = 0.25
    
    # Processing
    frame_skip: int = 0
    max_frames: Optional[int] = None
    
    # Display
    display_results: bool = True
    display_fps: bool = True
    window_name: str = "Chimeric Identity"
    
    # Logging
    log_level: str = "INFO"
    console_output: bool = True


# ============================================================================
# SIMPLE CHIMERIC RUNNER
# ============================================================================

class SimpleChimericRunner:
    """
    Simple runner for chimeric identity fusion.
    
    This is the CORRECT architecture:
    1. Initialize face engine (unchanged)
    2. Initialize gait engine (unchanged)
    3. For each frame:
       a. Run perception (get tracklets with bbox + keypoints)
       b. Call face_engine.update_signals() + decide() → face output
       c. Call gait_engine.update_signals() → gait output
       d. Fuse outputs with SimpleFusionEngine
       e. Visualize bbox + skeleton with unified colors
    
    The face and gait engines run their own logic - we just read their outputs!
    """
    
    def __init__(self, config: SimpleRunnerConfig):
        """Initialize simple chimeric runner."""
        self.config = config
        self._setup_logging()
        
        # Initialize fusion engine with weights
        weights = FusionWeights(
            face_weight=config.face_weight,
            gait_weight=config.gait_weight
        )
        self.fusion_engine = SimpleFusionEngine(weights=weights)
        self.registry = get_identity_registry()
        
        # Initialize perception (for tracking + pose)
        self.perception_engine = None
        
        # Initialize face engine (unchanged, we just read its output)
        self.face_engine = None
        self._init_face_engine()
        
        # Initialize gait engine (unchanged, we just read its output)
        self.gait_engine = None
        self._init_gait_engine()
        
        # Runtime state
        self._running = False
        self._frame_count = 0
        self._fps_counter = FPSCounter()
        
        # Per-track last results (for display)
        self._track_results: Dict[int, FusionResult] = {}
        
        # Store tracklets for visualization (CRITICAL for drawing bbox + skeleton)
        self._current_tracklets: List[Tracklet] = []
        
        # ====================================================================
        # STICKY IDENTITY SYSTEM (Hysteresis)
        # ====================================================================
        # Keeps identity displayed for N seconds after last recognition
        # Prevents flickering when face detection misses a few frames
        self._sticky_identities: Dict[int, dict] = {}  # track_id -> {name, confidence, last_seen, color}
        self._sticky_duration: float = 3.0  # Keep identity for 3 seconds after last recognition
        
        logger.info(
            f"[SIMPLE-RUNNER] Initialized: "
            f"face_weight={config.face_weight}, gait_weight={config.gait_weight}"
        )
    
    def _setup_logging(self):
        """Setup logging configuration."""
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def _init_face_engine(self):
        """Initialize face engine (if available)."""
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
        """Initialize gait engine (if available)."""
        try:
            from gait_subsystem.gait.gait_engine import GaitEngine
            from gait_subsystem.gait.config import default_gait_config
            
            gait_cfg = default_gait_config()
            self.gait_engine = GaitEngine(config=gait_cfg)
            logger.info("[SIMPLE-RUNNER] ✓ Gait engine initialized")
            
        except ImportError as e:
            logger.warning(f"[SIMPLE-RUNNER] Gait engine not available: {e}")
        except Exception as e:
            logger.error(f"[SIMPLE-RUNNER] Gait engine init failed: {e}")
    
    def _init_perception(self):
        """Initialize perception engine (lazy init)."""
        if self.perception_engine is not None:
            return
        
        try:
            from gait_subsystem.run_gait_standalone import GaitPerceptionEngine
            from gait_subsystem.gait.config import default_gait_config
            
            gait_cfg = default_gait_config()
            self.perception_engine = GaitPerceptionEngine(gait_cfg)
            logger.info("[SIMPLE-RUNNER] ✓ Perception engine initialized")
            
        except ImportError as e:
            logger.warning(f"[SIMPLE-RUNNER] Perception engine not available: {e}")
        except Exception as e:
            logger.error(f"[SIMPLE-RUNNER] Perception init failed: {e}")
    
    # ========================================================================
    # MAIN LOOP
    # ========================================================================
    
    def run(self):
        """
        Main processing loop.
        
        This is the simple, clean architecture:
        1. Read frame
        2. Run perception → tracklets
        3. For each tracklet:
           - Call face engine → face decision
           - Call gait engine → gait signal
           - Fuse → combined result
        4. Visualize with unified colors
        """
        import cv2
        
        logger.info("[SIMPLE-RUNNER] Starting main loop...")
        
        # Initialize frame source
        cap = self._open_video_source()
        if cap is None:
            logger.error("[SIMPLE-RUNNER] Failed to open video source")
            return
        
        # Initialize perception
        self._init_perception()
        
        self._running = True
        self._frame_count = 0
        
        try:
            while self._running:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    if self.config.video_file_path:
                        logger.info("[SIMPLE-RUNNER] End of video")
                        break
                    continue
                
                self._frame_count += 1
                
                # Skip frames if configured
                if self.config.frame_skip > 0:
                    if self._frame_count % (self.config.frame_skip + 1) != 0:
                        continue
                
                # Check max frames
                if self.config.max_frames and self._frame_count >= self.config.max_frames:
                    logger.info(f"[SIMPLE-RUNNER] Max frames reached: {self.config.max_frames}")
                    break
                
                # Process frame
                results = self._process_frame(frame)
                
                # Display
                if self.config.display_results:
                    self._visualize(frame, results)
                    
                    # Handle keyboard
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        logger.info("[SIMPLE-RUNNER] ESC pressed, stopping")
                        break
                    elif key == ord('q'):
                        break
                
                # Console output
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
        """Open camera or video file."""
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
    
    # ========================================================================
    # FRAME PROCESSING
    # ========================================================================
    
    def _process_frame(self, frame) -> List[FusionResult]:
        """
        Process a single frame through face + gait + fusion pipeline.
        
        This is the KEY LOGIC:
        1. Run perception → tracklets with bbox + keypoints
        2. For each tracklet:
           - Call face engine (unchanged) → read its decision
           - Call gait engine (unchanged) → read its signal
           - Fuse with simple weights
        """
        results = []
        timestamp = time.time()
        
        # Create Frame object
        frame_obj = Frame(
            frame_id=self._frame_count,
            ts=timestamp,
            camera_id="cam0",
            size=(frame.shape[1], frame.shape[0]),
            image=frame
        )
        
        # Run perception to get tracklets
        tracklets = self._run_perception(frame_obj)
        
        # Store tracklets for visualization (CRITICAL!)
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
        """Run perception to get tracklets with bbox and keypoints."""
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
        """
        Process a single tracklet through face + gait + fusion.
        
        This is where the MAGIC happens - but it's simple magic:
        1. Call face engine → get face decision
        2. Call gait engine → get gait signal
        3. Fuse → combined result
        """
        track_id = tracklet.track_id
        
        # ====================================================================
        # STEP 1: FACE ENGINE (call it, don't modify it)
        # ====================================================================
        face_input = None
        if self.face_engine:
            try:
                # Update signals (face engine does its thing)
                face_signals = self.face_engine.update_signals(
                    frame=frame_obj,
                    tracks=[tracklet]
                )
                
                # Get decision (face engine does its thing)
                if face_signals:
                    face_decisions = self.face_engine.decide(face_signals)
                    if face_decisions:
                        face_decision = face_decisions[0]
                        face_input = create_face_input_from_decision(face_decision)
                        
                        # Add bbox from tracklet
                        if face_input and hasattr(tracklet, 'last_box') and tracklet.last_box:
                            face_input.bbox = tuple(int(v) for v in tracklet.last_box)
            
            except Exception as e:
                logger.debug(f"[SIMPLE-RUNNER] Face engine error: {e}")
        
        # ====================================================================
        # STEP 2: GAIT ENGINE (call it, don't modify it)
        # ====================================================================
        gait_input = None
        if self.gait_engine:
            try:
                # Update signals (gait engine does its thing)
                gait_signals = self.gait_engine.update_signals(
                    frame=frame_obj,
                    tracks=[tracklet]
                )
                
                # Get gait state for quality/streak info
                gait_state = None
                if track_id in self.gait_engine._track_states:
                    gait_state = self.gait_engine._track_states[track_id]
                
                # Extract gait input from signal
                if gait_signals:
                    gait_signal = gait_signals[0]
                    gait_input = create_gait_input_from_signal(gait_signal, gait_state)
            
            except Exception as e:
                logger.debug(f"[SIMPLE-RUNNER] Gait engine error: {e}")
        
        # ====================================================================
        # STEP 3: SIMPLE FUSION (just combine with weights!)
        # ====================================================================
        result = self.fusion_engine.fuse(
            track_id=track_id,
            face_input=face_input,
            gait_input=gait_input,
            timestamp=timestamp
        )
        
        return result
    
    # ========================================================================
    # STICKY IDENTITY MANAGEMENT
    # ========================================================================
    
    def _update_sticky_identity(self, track_id: int, result: FusionResult, current_time: float):
        """
        Update sticky identity for a track.
        
        If recognized: Update the sticky entry
        If not recognized: Keep the old identity if within sticky_duration
        """
        # Check if this result has a valid recognition
        is_recognized = (
            result.display_name is not None and 
            result.combined_confidence >= 0.40  # Lower threshold for sticky (40%)
        )
        
        if is_recognized:
            # Update sticky identity
            self._sticky_identities[track_id] = {
                'name': result.display_name,
                'confidence': result.combined_confidence,
                'last_seen': current_time,
                'color_rgb': result.color_rgb,
                'state': result.state,
                'face_id': result.face_id,
                'gait_id': result.gait_id,
            }
        
        # Return sticky info if available and not expired
        if track_id in self._sticky_identities:
            sticky = self._sticky_identities[track_id]
            age = current_time - sticky['last_seen']
            if age <= self._sticky_duration:
                return sticky
            else:
                # Expired - remove
                del self._sticky_identities[track_id]
        
        return None
    
    def _cleanup_stale_sticky(self, current_time: float, active_track_ids: set):
        """Remove sticky entries for tracks that no longer exist."""
        stale_tracks = []
        for track_id, sticky in self._sticky_identities.items():
            # Remove if track gone AND expired
            if track_id not in active_track_ids:
                age = current_time - sticky['last_seen']
                if age > self._sticky_duration:
                    stale_tracks.append(track_id)
        
        for track_id in stale_tracks:
            del self._sticky_identities[track_id]
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def _visualize(self, frame, results: List[FusionResult]):
        """
        Visualize fusion results on frame.
        
        DRAWS FOR EVERY DETECTED PERSON:
        - Bounding box with color based on recognition state
        - Full skeleton (17 COCO keypoints) with SAME color
        - Identity label with confidence
        
        COLOR SCHEME:
        - YELLOW (255, 200, 0): Unknown / Scanning
        - GREEN (0, 255, 0): Recognized (both face+gait, or single modality)
        - RED (255, 0, 0): Conflict (face says A, gait says B)
        
        STICKY IDENTITY:
        - If recognized, keeps showing identity for 3 seconds even if face misses frames
        - Prevents flickering from intermittent detection
        """
        import cv2
        
        current_time = time.time()
        
        # Skeleton connections (COCO 17-keypoint)
        SKELETON = [
            (0, 1), (0, 2), (1, 3), (2, 4),           # Face
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),               # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
        ]
        
        # Build result map by track_id for quick lookup
        result_map = {r.track_id: r for r in results}
        
        # Use stored tracklets (from _process_frame)
        tracklets = self._current_tracklets
        
        # Cleanup stale sticky identities
        active_track_ids = {t.track_id for t in tracklets}
        self._cleanup_stale_sticky(current_time, active_track_ids)
        
        # Draw EVERY tracklet (not just those with fusion results)
        for tracklet in tracklets:
            track_id = tracklet.track_id
            
            # ============================================================
            # STICKY IDENTITY LOGIC
            # ============================================================
            # Get current frame's result
            result = result_map.get(track_id) or self._track_results.get(track_id)
            
            # Update sticky identity and get display info
            sticky_info = None
            if result:
                sticky_info = self._update_sticky_identity(track_id, result, current_time)
            else:
                # No result this frame - check if we have a sticky identity
                if track_id in self._sticky_identities:
                    sticky = self._sticky_identities[track_id]
                    age = current_time - sticky['last_seen']
                    if age <= self._sticky_duration:
                        sticky_info = sticky
            
            # Determine color and label based on sticky or current result
            if sticky_info:
                # We have a sticky identity - use GREEN
                color = sticky_info['color_rgb'][::-1]  # BGR for OpenCV
                display_name = sticky_info['name']
                display_conf = sticky_info['confidence']
            elif result and result.display_name:
                # Current frame has recognition
                color = result.color_rgb[::-1]
                display_name = result.display_name
                display_conf = result.combined_confidence
            else:
                # No recognition - YELLOW for scanning
                color = (0, 200, 255)  # Yellow in BGR
                display_name = None
                display_conf = 0.0
            
            # ============================================================
            # DRAW BOUNDING BOX
            # ============================================================
            if hasattr(tracklet, 'last_box') and tracklet.last_box:
                x1, y1, x2, y2 = [int(v) for v in tracklet.last_box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # ============================================================
            # DRAW SKELETON (all 17 keypoints + connections)
            # ============================================================
            if hasattr(tracklet, 'gait_sequence_data') and tracklet.gait_sequence_data:
                kp = tracklet.gait_sequence_data[-1]  # Latest keypoints (17, 3)
                
                # Draw skeleton connections first (lines)
                for (start, end) in SKELETON:
                    if start >= len(kp) or end >= len(kp):
                        continue
                    if kp[start, 2] < 0.3 or kp[end, 2] < 0.3:
                        continue
                    
                    pt1 = (int(kp[start, 0]), int(kp[start, 1]))
                    pt2 = (int(kp[end, 0]), int(kp[end, 1]))
                    cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
                
                # Draw keypoints (circles)
                for i, point in enumerate(kp):
                    if point[2] < 0.3:
                        continue
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)  # Border
            
            # ============================================================
            # DRAW LABEL (with sticky identity support)
            # ============================================================
            if sticky_info:
                # Use sticky identity for label
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
                # Current frame has recognition
                if len(display_name) > 15:
                    display_name = display_name[:13] + ".."
                conf_pct = int(display_conf * 100)
                label = f"{display_name} ({conf_pct}%) [F]"
            else:
                # No recognition
                label = f"Track-{track_id}: Scanning..."
            
            label_pos = self._get_label_position(tracklet, result)
            
            if label_pos:
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.6, 2)
                lx, ly = label_pos
                lx = max(0, min(lx, frame.shape[1] - tw - 10))
                ly = max(th + 10, min(ly, frame.shape[0] - 10))
                
                # Background with color
                cv2.rectangle(frame, (lx - 5, ly - th - 5), (lx + tw + 5, ly + 5), color, -1)
                # Text in black for contrast
                cv2.putText(frame, label, (lx, ly), font, 0.6, (0, 0, 0), 2)
        
        # ============================================================
        # DRAW FPS
        # ============================================================
        if self.config.display_fps:
            self._fps_counter.tick()
            fps_text = f"FPS: {self._fps_counter.fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow(self.config.window_name, frame)
    
    def _format_label_for_track(self, track_id: int, result: Optional[FusionResult]) -> str:
        """Format label for display based on track and fusion result."""
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
        """Get position for label (above head or bbox)."""
        if tracklet is None:
            return None
        
        # Try keypoints first (above head)
        if hasattr(tracklet, 'gait_sequence_data') and tracklet.gait_sequence_data:
            kp = tracklet.gait_sequence_data[-1]
            for idx in [0, 1, 2, 5, 6]:  # nose, eyes, shoulders
                if idx < len(kp) and kp[idx, 2] > 0.5:
                    return (int(kp[idx, 0]) - 50, int(kp[idx, 1]) - 40)
        
        # Fallback to bbox top
        if hasattr(tracklet, 'last_box') and tracklet.last_box:
            x1, y1, _, _ = tracklet.last_box
            return (int(x1), int(y1) - 10)
        
        return None
    
    def stop(self):
        """Stop the runner."""
        self._running = False


# ============================================================================
# FPS COUNTER
# ============================================================================

class FPSCounter:
    """Simple FPS counter."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = []
        self.fps = 0.0
    
    def tick(self):
        """Record a frame timestamp."""
        now = time.time()
        self.timestamps.append(now)
        
        # Keep only recent timestamps
        while len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)
        
        # Calculate FPS
        if len(self.timestamps) >= 2:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed > 0:
                self.fps = (len(self.timestamps) - 1) / elapsed


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_chimeric(
    camera_id: int = 0,
    video_path: Optional[str] = None,
    face_weight: float = 0.75,
    gait_weight: float = 0.25,
    max_frames: Optional[int] = None,
    display: bool = True
):
    """
    Run chimeric identity fusion.
    
    This is the simple entry point to run the system.
    
    Args:
        camera_id: Camera device ID (default 0)
        video_path: Video file path (if provided, uses video instead of camera)
        face_weight: Weight for face recognition (default 0.75)
        gait_weight: Weight for gait recognition (default 0.25)
        max_frames: Maximum frames to process (None = unlimited)
        display: Show visualization window
    """
    config = SimpleRunnerConfig(
        camera_device_id=camera_id,
        video_file_path=video_path,
        face_weight=face_weight,
        gait_weight=gait_weight,
        max_frames=max_frames,
        display_results=display
    )
    
    runner = SimpleChimericRunner(config)
    runner.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Chimeric Identity Runner")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument("--video", type=str, default=None, help="Video file path")
    parser.add_argument("--face-weight", type=float, default=0.75, help="Face weight (0-1)")
    parser.add_argument("--gait-weight", type=float, default=0.25, help="Gait weight (0-1)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--no-display", action="store_true", help="Disable display")
    
    args = parser.parse_args()
    
    run_chimeric(
        camera_id=args.camera,
        video_path=args.video,
        face_weight=args.face_weight,
        gait_weight=args.gait_weight,
        max_frames=args.max_frames,
        display=not args.no_display
    )
