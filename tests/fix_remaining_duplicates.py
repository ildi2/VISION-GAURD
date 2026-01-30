#!/usr/bin/env python3
"""
Fix all remaining duplicate second_best_score keyword arguments in extended tests
"""

import re
import sys
from pathlib import Path

def fix_duplicate_kwargs(file_path):
    """Fix duplicate keyword arguments in a file"""
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    original = content
    
    # Pattern: score=X, second_best_score=Y, second_best_score=Z, quality=W
    # Replace with: score=X, second_best_score=Y, quality=W
    # This regex handles multiline scenarios
    pattern = r'(score=[^,]+,\s*second_best_score=[^,]+,)\s*second_best_score=[^,]+,\s*(quality=)'
    replacement = r'\1 \2'
    
    content = re.sub(pattern, replacement, content)
    
    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    tests_dir = Path(__file__).parent
    files_to_fix = [
        tests_dir / "test_e2e_integration.py",
        tests_dir / "test_perf_load.py",
        tests_dir / "test_stress.py"
    ]
    
    print("=" * 70)
    print("Fixing duplicate second_best_score keyword arguments")
    print("=" * 70)
    
    fixed_count = 0
    for file_path in files_to_fix:
        if file_path.exists():
            if fix_duplicate_kwargs(file_path):
                print(f"✅ Fixed: {file_path.name}")
                fixed_count += 1
            else:
                print(f"⏭️  No duplicates found in: {file_path.name}")
        else:
            print(f"⚠️  File not found: {file_path.name}")
    
    print("=" * 70)
    if fixed_count > 0:
        print(f"✅ Fixed {fixed_count} file(s)")
    else:
        print("✅ All files already clean")
    print("=" * 70)

if __name__ == "__main__":
    main()
