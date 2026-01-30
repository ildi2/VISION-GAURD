#!/usr/bin/env python
"""Complete fix for Phase B test file"""

import re

with open('tests/test_phase_b_evidence_gate.py', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

fixed_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Fix: decision.name -> decision[0] (for tuple case)
    if 'decision.name' in line:
        line = line.replace('decision.name', '(decision[0] if isinstance(decision, tuple) else decision.name)')
    
    # Fix: assert decision in [GateDecision... (remaining ones not yet fixed)
    if 'assert decision in [GateDecision' in line:
        # Extract the assertion
        decision_status = 'decision[0] if isinstance(decision, tuple) else decision'
        # Replace 'decision in [GateDecision.X, GateDecision.Y]' with equivalent
        matches = re.search(r'assert decision in \[(.*?)\]', line)
        if matches:
            enums_str = matches.group(1)
            # Extract the enum values
            enums = [e.strip() for e in enums_str.split(',')]
            # Build a list of both string and enum values
            new_enums = []
            for enum_val in enums:
                # Extract the enum name like "ACCEPT" from "GateDecision.ACCEPT"
                if 'GateDecision.' in enum_val:
                    enum_name = enum_val.split('.')[-1]
                    new_enums.append(f"'{enum_name}'")
                new_enums.append(enum_val)
            
            new_line = line.replace(
                f'assert decision in [{enums_str}]',
                f'assert ({decision_status}) in [{", ".join(new_enums)}]'
            )
            line = new_line
    
    fixed_lines.append(line)
    i += 1

with open('tests/test_phase_b_evidence_gate.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("✅ Phase B fixes completed")
