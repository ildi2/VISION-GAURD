
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import unittest

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('tests.test_phase_5_real_mode')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    sys.exit(0 if result.wasSuccessful() else 1)
