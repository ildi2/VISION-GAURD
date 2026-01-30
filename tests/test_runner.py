# tests/test_runner.py
"""
Comprehensive Test Runner
Executes all test suites and generates report
"""

import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime
import json

log = logging.getLogger(__name__)


class TestRunner:
    """Orchestrate test execution"""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_test_suite(self, test_file: str, suite_name: str) -> dict:
        """Run a single test suite"""
        print(f"\n{'='*70}")
        print(f"Running: {suite_name}")
        print(f"{'='*70}")
        
        test_path = self.test_dir / test_file
        
        if not test_path.exists():
            print(f"❌ Test file not found: {test_path}")
            return {'passed': 0, 'failed': 1, 'status': 'NOT_FOUND'}
        
        # Run pytest
        cmd = [
            sys.executable,
            "-m", "pytest",
            str(test_path),
            "-v",
            "--tb=short",
            "--color=yes"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Parse output
            passed = result.stdout.count(" PASSED")
            failed = result.stdout.count(" FAILED")
            skipped = result.stdout.count(" SKIPPED")
            
            status = 'PASS' if failed == 0 else 'FAIL'
            
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            
            return {
                'passed': passed,
                'failed': failed,
                'skipped': skipped,
                'status': status,
                'return_code': result.returncode
            }
        
        except subprocess.TimeoutExpired:
            print(f"❌ Test suite timed out")
            return {'status': 'TIMEOUT', 'failed': 1}
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return {'status': 'ERROR', 'failed': 1}
    
    def run_all_tests(self):
        """Run all test suites"""
        self.start_time = datetime.now()
        
        test_suites = [
            ("test_phase_a_config.py", "PHASE A: Config Governance"),
            ("test_phase_b_evidence_gate.py", "PHASE B: Evidence Gating"),
            ("test_phase_c_binding.py", "PHASE C: Binding State Machine"),
            ("test_phase_d_scheduler.py", "PHASE D: Scheduler"),
            ("test_phase_e_merge_manager.py", "PHASE E: Merge Manager"),
            ("test_e2e_integration.py", "END-TO-END: Integration Tests"),
            ("test_perf_load.py", "PERFORMANCE: Load Tests"),
            ("test_stress.py", "STRESS: Edge Cases"),
        ]
        
        print("\n" + "="*70)
        print("GAITGUARD VERIFICATION TEST SUITE")
        print("="*70)
        print(f"Start time: {self.start_time}")
        print(f"Test directory: {self.test_dir}")
        
        for test_file, suite_name in test_suites:
            result = self.run_test_suite(test_file, suite_name)
            self.results[suite_name] = result
        
        self.end_time = datetime.now()
        self.generate_report()
    
    def generate_report(self):
        """Generate test report"""
        print("\n" + "="*70)
        print("VERIFICATION TEST REPORT")
        print("="*70)
        print(f"Start: {self.start_time}")
        print(f"End: {self.end_time}")
        print(f"Duration: {self.end_time - self.start_time}")
        
        # Summary table
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        
        print("\n" + "-"*70)
        print(f"{'Test Suite':<40} {'Status':<10} {'Results':<15}")
        print("-"*70)
        
        for suite_name, result in self.results.items():
            status = result.get('status', 'UNKNOWN')
            
            if status == 'PASS':
                passed = result.get('passed', 0)
                failed = result.get('failed', 0)
                skipped = result.get('skipped', 0)
                total_passed += passed
                total_failed += failed
                total_skipped += skipped
                results_str = f"✅ {passed}P {failed}F {skipped}S"
            elif status == 'FAIL':
                failed = result.get('failed', 0)
                total_failed += failed
                results_str = f"❌ {failed}F"
            elif status == 'NOT_FOUND':
                results_str = f"⚠️  NOT FOUND"
            elif status == 'TIMEOUT':
                total_failed += 1
                results_str = f"⏱️  TIMEOUT"
            else:
                total_failed += 1
                results_str = f"❌ ERROR"
            
            status_emoji = "✅" if status == 'PASS' else "❌"
            print(f"{suite_name:<40} {status_emoji} {status:<8} {results_str:<15}")
        
        print("-"*70)
        print(f"TOTAL: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")
        
        if total_failed == 0 and total_passed > 0:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\n⚠️  {total_failed} test(s) failed or not found")
        
        print("="*70)
        
        # Save results to JSON
        report_file = self.test_dir / f"test_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                'start_time': str(self.start_time),
                'end_time': str(self.end_time),
                'total_passed': total_passed,
                'total_failed': total_failed,
                'total_skipped': total_skipped,
                'suites': self.results
            }, f, indent=2)
        
        print(f"\nReport saved: {report_file}")


def main():
    """Main entry point"""
    runner = TestRunner()
    runner.run_all_tests()
    
    # Exit with error code if any tests failed
    if any(r.get('status') != 'PASS' for r in runner.results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
