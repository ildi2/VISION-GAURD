#!/usr/bin/env python
"""Fix remaining second_best_score issues"""

import re

files = [
    'tests/test_e2e_integration.py',
    'tests/test_perf_load.py',
    'tests/test_stress.py',
]

for fname in files:
    with open(fname, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Pattern 1: score=X, quality=Y (no second_best_score)
    content = re.sub(
        r'score=([\d.]+),\s+quality=',
        lambda m: f'score={m.group(1)}, second_best_score={float(m.group(1))-0.15:.2f}, quality=',
        content
    )
    
    # Pattern 2: score=X, \n timestamp (missing second_best_score and quality)
    content = re.sub(
        r'score=([\d.]+),\s*timestamp=',
        lambda m: f'score={m.group(1)}, second_best_score={float(m.group(1))-0.15:.2f}, quality=0.8, timestamp=',
        content
    )
    
    # Pattern 3: Explicitly add missing params in multiline calls
    # score=X, \n                second_best_score=... quality=...
    # Already handled by pattern 1
    
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)

print("✅ Fixed all second_best_score issues")
