#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from itertools import combinations


# ------------------------------------------------------------
# Basic I/O
# ------------------------------------------------------------

def load_rules(jsonl_path: str) -> List[Dict]:
    """Load rules from a JSONL file."""
    rules = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rules.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return rules


# ------------------------------------------------------------
# Global statistics
# ------------------------------------------------------------

def global_statistics(rules: List[Dict]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Global statistics:
    - EDR anomaly APIs (accumulated by total_occurrences)
    - Malware APIs (counted by number of appearances)
    """
    edr_api_counter = {}
    malware_api_counter = Counter()

    for rule in rules:
        edr_api = rule.get('edr_anomaly_api', '')
        total_occurrences = rule.get('total_occurrences', 0)
        if edr_api:
            edr_api_counter[edr_api] = edr_api_counter.get(edr_api, 0) + total_occurrences

        for pattern in rule.get('malware_api_patterns', []):
            for api in pattern:
                malware_api_counter[api] += 1

    edr_api_stats = sorted(edr_api_counter.items(), key=lambda x: x[1], reverse=True)[:50]
    malware_api_stats = malware_api_counter.most_common(50)

    return edr_api_stats, malware_api_stats


# ------------------------------------------------------------
# Stable co-occurring subsequence (Core of Plan A)
# ------------------------------------------------------------

def generate_subsequences(seq: List[str], min_len: int = 2, max_len: int = 3):
    """Generate all ordered (non-contiguous) subsequences of a sequence."""
    n = len(seq)
    for l in range(min_len, min(max_len, n) + 1):
        for idx in combinations(range(n), l):
            yield tuple(seq[i] for i in idx)


def frequent_subsequence(
    patterns: List[List[str]],
    min_support_ratio: float = 0.6,
    max_len: int = 3
) -> List[str]:
    """
    Find a stable co-occurring API subsequence from multiple patterns.
    """
    if not patterns:
        return []

    pattern_count = len(patterns)
    subseq_support = defaultdict(set)  # subseq -> set(pattern_idx)

    for idx, pattern in enumerate(patterns):
        seen = set()
        for subseq in generate_subsequences(pattern, 2, max_len):
            if subseq not in seen:
                subseq_support[subseq].add(idx)
                seen.add(subseq)

    min_support = max(1, int(pattern_count * min_support_ratio))

    candidates = [
        subseq for subseq, idxs in subseq_support.items()
        if len(idxs) >= min_support
    ]

    if not candidates:
        return []

    # Sorting strategy:
    # 1. Prefer longer subsequences
    # 2. Higher pattern coverage
    # 3. Stable lexicographic order
    candidates.sort(
        key=lambda s: (len(s), len(subseq_support[s]), s),
        reverse=True
    )

    return list(candidates[0])


# ------------------------------------------------------------
# Single rule analysis
# ------------------------------------------------------------

def analyze_single_rule(rule: Dict) -> Dict:
    patterns = rule.get('malware_api_patterns', [])

    common_pattern = frequent_subsequence(
        patterns,
        min_support_ratio=0.6,
        max_len=3
    )

    # Fallback to ensure non-empty output
    if not common_pattern and patterns:
        common_pattern = patterns[0][:2]

    return {
        'edr_anomaly_api': rule.get('edr_anomaly_api', ''),
        'malware_api_lcs': common_pattern,  # Field name kept unchanged for compatibility
        'total_occurrences': rule.get('total_occurrences', 0)
    }


# ------------------------------------------------------------
# Main workflow
# ------------------------------------------------------------

def main():
    print("=" * 70)
    print("Rule Analysis Tool (Stable Co-occurring Subsequence Version)")
    print("=" * 70)

    input_file = "rule_faclon.jsonl"
    output_file = "rule_faclon_analysis.jsonl"
    # input_file = "rule_wd.jsonl"
    # output_file = "rule_wd_analysis.jsonl"

    print(f"\nInput file:  {input_file}")
    print(f"Output file: {output_file}")

    # 1. Load rules
    print("\n[1/4] Loading rules...")
    rules = load_rules(input_file)
    print(f"  Loaded {len(rules)} rules")

    # 2. Global statistics
    print("\n[2/4] Computing global statistics...")
    edr_api_stats, malware_api_stats = global_statistics(rules)

    print("\n  ✓ Top 10 EDR anomaly APIs:")
    for i, (api, count) in enumerate(edr_api_stats[:10], 1):
        print(f"    {i:2d}. {api:50s} {count:6d}")

    print("\n  ✓ Top 10 malware APIs:")
    for i, (api, count) in enumerate(malware_api_stats[:10], 1):
        print(f"    {i:2d}. {api:50s} {count:6d}")

    # 3. Sort rules
    print("\n[3/4] Sorting rules by total occurrences...")
    sorted_rules = sorted(
        rules,
        key=lambda x: x.get('total_occurrences', 0),
        reverse=True
    )
    print("  Sorting completed")

    # 4. Analyze rules
    print("\n[4/4] Analyzing rules and saving results...")
    analyzed_rules = []

    for i, rule in enumerate(sorted_rules, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(sorted_rules)}")

        analyzed_rules.append(analyze_single_rule(rule))

    # Write output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps({
            'type': 'global_statistics',
            'total_rules': len(rules),
            'top_50_edr_apis': [
                {'api': api, 'occurrences': cnt}
                for api, cnt in edr_api_stats
            ],
            'top_50_malware_apis': [
                {'api': api, 'count': cnt}
                for api, cnt in malware_api_stats
            ]
        }, ensure_ascii=False) + '\n')

        for rule in analyzed_rules:
            f.write(json.dumps(rule, ensure_ascii=False) + '\n')

    print("\n" + "=" * 70)
    print("Analysis completed successfully!")
    print(f"Output file: {Path(output_file).absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
