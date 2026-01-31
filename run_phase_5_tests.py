# run_phase_5_tests.py
"""
Phase 5 Test Runner

Executes all Phase 5 validation tests with proper environment setup.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Now import and run tests
import unittest

if __name__ == '__main__':
    # Discover and run all tests in test_phase_5_real_mode
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('tests.test_phase_5_real_mode')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with proper code
    sys.exit(0 if result.wasSuccessful() else 1)
