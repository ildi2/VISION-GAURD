
import logging
from typing import Dict, List, Optional, Callable
import numpy as np
from collections import namedtuple

log = logging.getLogger(__name__)


def generate_random_embedding(dim: int = 512) -> np.ndarray:
    embedding = np.random.randn(dim)
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def generate_similar_embedding(reference: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    embedding = reference + np.random.randn(len(reference)) * noise_level
    embedding = embedding / np.linalg.norm(embedding)
    return embedding


def generate_different_embedding(reference: np.ndarray, min_distance: float = 0.3) -> np.ndarray:
    while True:
        embedding = np.random.randn(len(reference))
        embedding = embedding / np.linalg.norm(embedding)
        
        distance = 1 - np.dot(reference, embedding)
        
        if distance >= min_distance:
            return embedding


TestEvidenceData = namedtuple('TestEvidenceData', [
    'blur_score',
    'brightness',
    'yaw',
    'pitch',
    'roll',
    'scale',
    'quality_score'
])


TestTrackletData = namedtuple('TestTrackletData', [
    'track_id',
    'confidence',
    'last_seen_ts',
    'identity_name',
    'binding_state'
])


TestMergeScenario = namedtuple('TestMergeScenario', [
    'tracklet_a',
    'tracklet_b',
    'expected_merge',
    'description'
])


HIGH_QUALITY_EVIDENCE = TestEvidenceData(
    blur_score=0.05,
    brightness=0.6,
    yaw=5.0,
    pitch=2.0,
    roll=2.0,
    scale=0.6,
    quality_score=0.95
)

LOW_QUALITY_EVIDENCE = TestEvidenceData(
    blur_score=0.9,
    brightness=0.1,
    yaw=45.0,
    pitch=30.0,
    roll=30.0,
    scale=0.1,
    quality_score=0.1
)

MARGINAL_QUALITY_EVIDENCE = TestEvidenceData(
    blur_score=0.3,
    brightness=0.45,
    yaw=15.0,
    pitch=10.0,
    roll=10.0,
    scale=0.4,
    quality_score=0.65
)


def validate_evidence_decision(decision, expected_decisions: List[str]) -> bool:
    decision_str = str(decision).upper()
    
    for expected in expected_decisions:
        if expected.upper() in decision_str:
            return True
    
    return False


def validate_binding_state(state, expected_states: List[str]) -> bool:
    state_str = str(state).upper()
    
    for expected in expected_states:
        if expected.upper() in state_str:
            return True
    
    return False


def validate_merge_decision(decision, should_merge: bool) -> bool:
    if hasattr(decision, 'should_merge'):
        return decision.should_merge == should_merge
    
    return False


class StatisticalValidator:
    
    @staticmethod
    def check_acceptance_rate(decisions: List, decision_type: str, 
                            min_rate: float, max_rate: float) -> bool:
        if not decisions:
            return False
        
        count = sum(1 for d in decisions if decision_type in str(d))
        rate = count / len(decisions)
        
        return min_rate <= rate <= max_rate
    
    @staticmethod
    def check_latency_percentile(latencies: List[float], percentile: int, 
                                max_latency_ms: float) -> bool:
        if not latencies:
            return False
        
        p_latency = np.percentile(latencies, percentile)
        return p_latency <= max_latency_ms
    
    @staticmethod
    def check_throughput(count: int, duration_sec: float, 
                        min_throughput: float) -> bool:
        if duration_sec <= 0:
            return False
        
        throughput = count / duration_sec
        return throughput >= min_throughput


class TestLogger:
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def header(self, title: str):
        self.logger.info("\n" + "="*70)
        self.logger.info(title)
        self.logger.info("="*70)
    
    def section(self, title: str):
        self.logger.info("\n" + "-"*70)
        self.logger.info(title)
        self.logger.info("-"*70)
    
    def test_case(self, description: str):
        self.logger.info(f"\nTest: {description}")
    
    def success(self, message: str):
        self.logger.info(f"✅ {message}")
    
    def failure(self, message: str):
        self.logger.error(f"❌ {message}")
    
    def warning(self, message: str):
        self.logger.warning(f"⚠️  {message}")
    
    def info(self, message: str):
        self.logger.info(f"ℹ️  {message}")
    
    def metric(self, name: str, value, unit: str = ""):
        self.logger.info(f"   {name}: {value} {unit}")
    
    def results(self, results: Dict[str, bool]):
        self.section("RESULTS")
        
        passed = sum(1 for v in results.values() if v)
        failed = len(results) - passed
        
        for name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.logger.info(f"{name}: {status}")
        
        self.logger.info(f"\nSummary: {passed} passed, {failed} failed")


def compare_embeddings(emb1: np.ndarray, emb2: np.ndarray) -> Dict:
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    
    cosine_sim = np.dot(emb1_norm, emb2_norm)
    
    euclidean_dist = np.linalg.norm(emb1 - emb2)
    
    return {
        'cosine_similarity': cosine_sim,
        'euclidean_distance': euclidean_dist,
        'is_identical': np.allclose(emb1, emb2),
        'is_very_similar': cosine_sim > 0.98,
        'is_similar': cosine_sim > 0.90,
        'is_different': cosine_sim < 0.5
    }


class TestAssertions:
    
    @staticmethod
    def assert_decision_in(decision, valid_decisions: List[str]):
        decision_str = str(decision).upper()
        
        for valid in valid_decisions:
            if valid.upper() in decision_str:
                return
        
        raise AssertionError(
            f"Decision '{decision}' not in {valid_decisions}"
        )
    
    @staticmethod
    def assert_acceptance_rate(decisions: List, expected_min: float, 
                              expected_max: float, decision_type: str = "ACCEPT"):
        if not decisions:
            raise AssertionError("No decisions to validate")
        
        count = sum(1 for d in decisions if decision_type in str(d))
        rate = count / len(decisions)
        
        assert expected_min <= rate <= expected_max, \
            f"Acceptance rate {rate:.2%} not in [{expected_min:.2%}, {expected_max:.2%}]"
    
    @staticmethod
    def assert_throughput(count: int, duration: float, 
                         min_throughput: float, operation: str = "ops"):
        if duration <= 0:
            raise AssertionError("Duration must be > 0")
        
        throughput = count / duration
        
        assert throughput >= min_throughput, \
            f"Throughput {throughput:.0f} {operation}/sec below minimum {min_throughput:.0f}"
    
    @staticmethod
    def assert_latency_p95(latencies: List[float], max_latency_ms: float):
        if not latencies:
            raise AssertionError("No latencies to validate")
        
        p95 = np.percentile(latencies, 95)
        
        assert p95 <= max_latency_ms, \
            f"P95 latency {p95:.2f}ms exceeds {max_latency_ms}ms"


class MockTracklet:
    
    def __init__(self, track_id: int, confidence: float = 0.9, 
                 last_seen_ts: float = 0.0, identity_name: str = "Test",
                 binding_state: str = "CONFIRMED"):
        self.track_id = track_id
        self.confidence = confidence
        self.last_seen_ts = last_seen_ts
        self.identity_name = identity_name
        self.binding_state = binding_state
        self.embeddings = [generate_random_embedding()]
        self.box_history = []
        self.appearance_history = []


class MockEvidence:
    
    def __init__(self, bbox=None, embedding=None, quality_metrics=None, pose=None):
        self.bbox = bbox or np.array([100, 100, 200, 200])
        self.landmark_2d = np.random.rand(5, 2) * 100 + 100
        self.embedding = embedding or generate_random_embedding()
        self.quality_metrics = quality_metrics or {
            'blur': 0.1,
            'brightness': 0.6,
            'scale': 0.5,
            'quality': 0.9
        }
        self.pose = pose or np.array([0.0, 0.0, 0.0])
        self.detector_confidence = 0.95


def generate_test_summary(results: Dict[str, Dict]) -> str:
    lines = []
    
    total_passed = 0
    total_failed = 0
    
    for test_name, result in results.items():
        status = "✅ PASS" if result.get('passed') else "❌ FAIL"
        lines.append(f"{test_name}: {status}")
        
        if result.get('passed'):
            total_passed += 1
        else:
            total_failed += 1
    
    lines.append("\n" + "="*70)
    lines.append(f"TOTAL: {total_passed} passed, {total_failed} failed")
    
    return "\n".join(lines)
