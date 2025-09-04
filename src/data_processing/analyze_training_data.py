#!/usr/bin/env python3
"""
Analyze the training data to understand what types of bugs the model learned
"""

import json
import re
from collections import defaultdict

def analyze_dataset(file_path, max_samples=None):
    """Analyze the patterns in the dataset"""
    
    print(f"Analyzing dataset: {file_path}")
    
    working_samples = []
    buggy_samples = []
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            data = json.loads(line)
            code = data['code']
            label = data['label']
            
            if label == 0:
                working_samples.append(code)
            else:
                buggy_samples.append(code)
    
    print(f"Total samples: {len(working_samples) + len(buggy_samples)}")
    print(f"Working samples: {len(working_samples)}")
    print(f"Buggy samples: {len(buggy_samples)}")
    print()
    
    # Analyze bug patterns
    print("ANALYSIS OF BUGGY SAMPLES:")
    print("=" * 50)
    
    bug_patterns = defaultdict(int)
    
    for code in buggy_samples[:50]:  # Analyze first 50 buggy samples
        # Check for common bug patterns
        if not code.strip().endswith(';') and not code.strip().endswith('}'):
            bug_patterns['missing_semicolon'] += 1
        
        if '/ b' in code or '/b' in code or '/ denominator' in code or '/ total' in code:
            bug_patterns['division_operation'] += 1
            
        if 'return a / b' in code:
            bug_patterns['simple_division'] += 1
            
        if '[index]' in code or '[0]' in code:
            bug_patterns['array_access'] += 1
            
        if '.Length' in code and 'input.' in code:
            bug_patterns['null_reference_risk'] += 1
            
        if code.count('{') != code.count('}'):
            bug_patterns['brace_mismatch'] += 1
            
        if code.count('(') != code.count(')'):
            bug_patterns['paren_mismatch'] += 1
            
        if 'foreach' in code and ('.Remove(' in code or '.Add(' in code):
            bug_patterns['collection_modification'] += 1
            
    print("Bug pattern frequencies (in first 50 buggy samples):")
    for pattern, count in sorted(bug_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count}")
    
    print("\nSample buggy codes with division:")
    division_samples = [code for code in buggy_samples if ('/' in code and 'b' in code)][:5]
    for i, code in enumerate(division_samples, 1):
        print(f"  {i}. {code}")
    
    print("\nSample working codes:")
    for i, code in enumerate(working_samples[:5], 1):
        print(f"  {i}. {code}")
    
    return working_samples, buggy_samples, bug_patterns

def find_division_patterns():
    """Find all division-related patterns in the training data"""
    print("\nSEARCHING FOR DIVISION PATTERNS:")
    print("=" * 50)
    
    # Check both training files
    for file_name in ['train_csharp_bugs_enhanced.jsonl', 'valid_csharp_bugs_enhanced.jsonl']:
        try:
            with open(file_name, 'r') as f:
                division_working = []
                division_buggy = []
                
                for line_num, line in enumerate(f):
                    data = json.loads(line)
                    code = data['code']
                    label = data['label']
                    
                    # Look for division operations
                    if '/' in code and any(var in code.lower() for var in ['a', 'b', 'numerator', 'denominator', 'total', 'value']):
                        if label == 0:
                            division_working.append(code)
                        else:
                            division_buggy.append(code)
                
                print(f"\n{file_name}:")
                print(f"  Division in working samples: {len(division_working)}")
                print(f"  Division in buggy samples: {len(division_buggy)}")
                
                if division_working:
                    print("  Sample working division code:")
                    print(f"    {division_working[0]}")
                    
                if division_buggy:
                    print("  Sample buggy division code:")
                    print(f"    {division_buggy[0]}")
                    
        except FileNotFoundError:
            print(f"  {file_name}: File not found")

if __name__ == "__main__":
    # Analyze training data
    try:
        working, buggy, patterns = analyze_dataset('train_csharp_bugs_enhanced.jsonl', max_samples=1000)
        find_division_patterns()
    except FileNotFoundError:
        try:
            working, buggy, patterns = analyze_dataset('train_csharp_bugs.jsonl', max_samples=1000) 
            find_division_patterns()
        except FileNotFoundError:
            print("Training data files not found. Please run generate_csharp_dataset.py first.")