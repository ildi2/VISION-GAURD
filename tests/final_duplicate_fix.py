#!/usr/bin/env python3
"""
Final comprehensive fix for all duplicate second_best_score issues
"""

import re
from pathlib import Path

def remove_all_duplicate_second_best_score(content):
    """Remove ALL duplicate second_best_score keyword arguments"""
    
    # Match pattern: second_best_score=X, second_best_score=Y,
    # Keep first, remove second and beyond
    pattern = r'(second_best_score=[^,]+,)\s*second_best_score=[^,]+,'
    
    while re.search(pattern, content):
        content = re.sub(pattern, r'\1', content)
    
    return content

def main():
    tests_dir = Path(__file__).parent
    files = [
        tests_dir / "test_e2e_integration.py",
        tests_dir / "test_perf_load.py",
        tests_dir / "test_stress.py"
    ]
    
    print("Removing all remaining duplicate second_best_score parameters...")
    for fpath in files:
        if fpath.exists():
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            original = content
            content = remove_all_duplicate_second_best_score(content)
            
            if content != original:
                with open(fpath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ {fpath.name} - duplicates removed")
            else:
                print(f"⏭️  {fpath.name} - no duplicates")

if __name__ == "__main__":
    main()
