
import pytest
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from collections import deque
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logging_setup import setup_logging
from core.config import load_config
from core.governance_metrics import get_metrics_collector

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_config():
    try:
        import os
        from pathlib import Path as PathlibPath
        
        tests_dir = PathlibPath(__file__).parent
        project_root = tests_dir.parent
        config_path = project_root / "config" / "default.yaml"
        
        cfg = load_config(str(config_path))
        log.info(f"✅ Test config loaded from: {config_path}")
        return cfg
    except Exception as e:
        log.error(f"❌ Failed to load test config: {e}")
        raise


@pytest.fixture(scope="session")
def metrics_collector():
    return get_metrics_collector()


@pytest.fixture
def reset_metrics(metrics_collector):
    metrics_collector.reset()
    yield


@pytest.fixture
def logger():
    return logging.getLogger("test")


@pytest.fixture
def numpy_seed():
    np.random.seed(42)
    yield


@pytest.fixture
def test_face_evidence(numpy_seed):
    def _create_evidence(
        track_id: int = 1,
        ts: float = 0.0,
        frame_id: int = 0,
        frame_idx: int = None,
        blur_score: float = 0.1,
        brightness: float = 0.5,
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
        scale: float = 0.5,
        quality_score: float = 0.9,
        quality: float = None
    ):
        try:
            from face.route import FaceEvidence
            
            q_val = quality if quality is not None else quality_score
            
            fid = frame_idx if frame_idx is not None else frame_id
            
            evidence = FaceEvidence(
                track_id=track_id,
                ts=ts,
                frame_id=fid,
                quality=q_val,
                embedding=np.random.rand(512),
                bbox_in_frame=(100, 100, 200, 200),
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                det_score=0.95,
                landmarks_2d=np.random.rand(5, 2) * 100 + 100,
            )
            return evidence
        except Exception as e:
            log.error(f"Failed to create test evidence: {e}")
            raise
    
    return _create_evidence


@pytest.fixture
def test_tracklet():
    def _create_tracklet(
        track_id: int,
        confidence: float = 0.95,
        last_ts: float = 10.0,
        identity_name: str = "John Doe",
        binding_state: str = "CONFIRMED",
    ):
        try:
            from schemas import Tracklet
            
            tracklet = Tracklet(
                track_id=track_id,
                camera_id="camera_0",
                last_frame_id=100,
                last_box=(100.0, 100.0, 200.0, 200.0),
                confidence=confidence,
                age_frames=10,
                lost_frames=0,
                history_boxes=[],
                last_ts=last_ts,
            )
            return tracklet
        except Exception as e:
            log.error(f"Failed to create test tracklet: {e}")
            raise
    
    return _create_tracklet


@pytest.fixture
def memory_tracker():
    import psutil
    
    class MemoryTracker:
        def __init__(self):
            self.process = psutil.Process()
            self.baseline = None
        
        def start(self):
            self.baseline = self.process.memory_info().rss / 1024 / 1024
            log.info(f"Memory baseline: {self.baseline:.1f} MB")
        
        def delta(self):
            if self.baseline is None:
                return 0
            current = self.process.memory_info().rss / 1024 / 1024
            delta = current - self.baseline
            log.info(f"Memory delta: {delta:+.1f} MB (now {current:.1f} MB)")
            return delta
    
    return MemoryTracker()


@pytest.fixture
def timer():
    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            if self.start_time is None:
                raise RuntimeError("Timer not started")
            self.elapsed = time.time() - self.start_time
            return self.elapsed
        
        def ms(self):
            if self.elapsed is None:
                raise RuntimeError("Timer not stopped")
            return self.elapsed * 1000
    
    return Timer()


@pytest.fixture
def fps_monitor():
    class FPSMonitor:
        def __init__(self, window_size: int = 100):
            self.frame_times = deque(maxlen=window_size)
            self.last_time = None
        
        def start(self):
            self.last_time = time.time()
        
        def tick(self):
            if self.last_time is None:
                self.start()
                return 0
            
            now = time.time()
            frame_time = now - self.last_time
            self.frame_times.append(frame_time)
            self.last_time = now
            return frame_time
        
        def get_fps(self) -> float:
            if not self.frame_times:
                return 0
            avg_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0
        
        def get_stats(self) -> Dict:
            if not self.frame_times:
                return {}
            
            times = list(self.frame_times)
            avg_fps = self.get_fps()
            min_frame_time = min(times)
            max_frame_time = max(times)
            p95_frame_time = np.percentile(times, 95)
            
            return {
                'avg_fps': avg_fps,
                'min_frame_time_ms': min_frame_time * 1000,
                'max_frame_time_ms': max_frame_time * 1000,
                'p95_frame_time_ms': p95_frame_time * 1000,
                'frame_count': len(times),
            }
    
    return FPSMonitor()


def pytest_configure(config):
    log.info("="*70)
    log.info("GAITGUARD VERIFICATION TEST SUITE")
    log.info("="*70)
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    log.info(f"Results directory: {results_dir}")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "phase_a" in item.nodeid:
            item.add_marker(pytest.mark.phase_a)
        elif "phase_b" in item.nodeid:
            item.add_marker(pytest.mark.phase_b)
        elif "phase_c" in item.nodeid:
            item.add_marker(pytest.mark.phase_c)
        elif "phase_d" in item.nodeid:
            item.add_marker(pytest.mark.phase_d)
        elif "phase_e" in item.nodeid:
            item.add_marker(pytest.mark.phase_e)
        elif "e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
        elif "perf" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        elif "stress" in item.nodeid:
            item.add_marker(pytest.mark.stress)


def pytest_addoption(parser):
    parser.addoption(
        "--verbose-output",
        action="store_true",
        default=False,
        help="Enable verbose test output"
    )


class TestResults:
    
    def __init__(self):
        self.results: List[Dict] = []
    
    def add(self, test_name: str, passed: bool, message: str = "", metrics: Dict = None):
        self.results.append({
            'name': test_name,
            'passed': passed,
            'message': message,
            'metrics': metrics or {},
            'timestamp': time.time(),
        })
    
    def summary(self) -> Dict:
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': (passed / total * 100) if total > 0 else 0,
        }
    
    def print_summary(self):
        summary = self.summary()
        print("\n" + "="*70)
        print("TEST RESULTS SUMMARY")
        print("="*70)
        print(f"Total tests: {summary['total']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Pass rate: {summary['pass_rate']:.1f}%")
        print("="*70)


@pytest.fixture
def test_results():
    return TestResults()


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_test_header(title: str):
    print("\n" + Colors.BOLD + Colors.BLUE + "="*70 + Colors.ENDC)
    print(Colors.BOLD + Colors.BLUE + title + Colors.ENDC)
    print(Colors.BOLD + Colors.BLUE + "="*70 + Colors.ENDC)


def print_pass(message: str):
    print(Colors.GREEN + f"✅ {message}" + Colors.ENDC)


def print_fail(message: str):
    print(Colors.RED + f"❌ {message}" + Colors.ENDC)


def print_warn(message: str):
    print(Colors.YELLOW + f"⚠️  {message}" + Colors.ENDC)


def print_info(message: str):
    print(Colors.BLUE + f"ℹ️  {message}" + Colors.ENDC)
