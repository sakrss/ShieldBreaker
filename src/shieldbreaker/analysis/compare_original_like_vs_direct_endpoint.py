#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from detect_endpoint_lstm_anomalies import (
    DEFAULT_ATTACK_PROCESSES,
    ELASTIC_PROCESSES,
    detect_endpoint_anomalies,
    estimate_threshold_from_valid,
    extract_process_name,
    load_lstm_model,
    load_vocab,
    normalize_method,
    parse_stimulated_log,
)


def build_original_like_endpoint_events(
    input_path: Path,
    attack_processes: set[str],
) -> Dict[str, object]:
    target_processes = set(ELASTIC_PROCESSES) | attack_processes
    filtered_records: List[Dict[str, object]] = []
    total_lines = 0
    json_errors = 0
    skipped_sysret = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            total_lines = line_no
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                continue

            if obj.get("Plugin") == "sysret":
                skipped_sysret += 1
                continue

            proc = extract_process_name(obj.get("ProcessName", ""))
            if proc not in target_processes:
                continue

            filtered_records.append(
                {
                    "raw_line_no": line_no,
                    "timestamp": float(obj.get("TimeStamp", 0.0)),
                    "process": proc,
                    "method": normalize_method(obj.get("Method", "")),
                    "event_uid": obj.get("EventUID"),
                    "pid": obj.get("PID"),
                    "tid": obj.get("TID"),
                }
            )

    # Mimic the original preprocessing more closely: sort the joint extracted view
    # by timestamp first, then derive endpoint events from that sorted subset.
    filtered_records.sort(key=lambda x: float(x["timestamp"]))

    endpoint_events = [
        rec
        for rec in filtered_records
        if rec["process"] == "elastic-endpoint.exe" and rec["method"]
    ]

    return {
        "total_lines": total_lines,
        "json_errors": json_errors,
        "skipped_sysret": skipped_sysret,
        "filtered_record_count": len(filtered_records),
        "endpoint_events": endpoint_events,
    }


def event_key(event: Dict[str, object]) -> Tuple[object, ...]:
    return (
        event.get("raw_line_no"),
        event.get("timestamp"),
        event.get("method"),
        event.get("pid"),
        event.get("tid"),
        event.get("event_uid"),
    )


def anomaly_key(anomaly: Dict[str, object]) -> Tuple[object, ...]:
    return (
        anomaly.get("raw_line_no"),
        anomaly.get("timestamp"),
        anomaly.get("method"),
        anomaly.get("pid"),
        anomaly.get("tid"),
        anomaly.get("event_uid"),
    )


def compare_sequences(
    direct_events: List[Dict[str, object]],
    original_like_events: List[Dict[str, object]],
) -> Dict[str, object]:
    direct_keys = [event_key(ev) for ev in direct_events]
    original_keys = [event_key(ev) for ev in original_like_events]

    direct_set = set(direct_keys)
    original_set = set(original_keys)

    same_order = direct_keys == original_keys
    first_diff = None
    if not same_order:
        max_len = min(len(direct_keys), len(original_keys))
        for idx in range(max_len):
            if direct_keys[idx] != original_keys[idx]:
                first_diff = {
                    "index": idx,
                    "direct": direct_events[idx],
                    "original_like": original_like_events[idx],
                }
                break
        if first_diff is None and len(direct_keys) != len(original_keys):
            first_diff = {
                "index": max_len,
                "direct": direct_events[max_len] if len(direct_keys) > max_len else None,
                "original_like": original_like_events[max_len] if len(original_keys) > max_len else None,
            }

    direct_only = [ev for ev in direct_events if event_key(ev) not in original_set]
    original_only = [ev for ev in original_like_events if event_key(ev) not in direct_set]

    return {
        "direct_count": len(direct_events),
        "original_like_count": len(original_like_events),
        "shared_count": len(direct_set & original_set),
        "direct_only_count": len(direct_only),
        "original_like_only_count": len(original_only),
        "same_order": same_order,
        "first_order_difference": first_diff,
        "direct_only_examples": direct_only[:10],
        "original_like_only_examples": original_only[:10],
    }


def compare_anomalies(
    direct_anomalies: List[Dict[str, object]],
    original_like_anomalies: List[Dict[str, object]],
) -> Dict[str, object]:
    direct_keys = [anomaly_key(ev) for ev in direct_anomalies]
    original_keys = [anomaly_key(ev) for ev in original_like_anomalies]
    direct_set = set(direct_keys)
    original_set = set(original_keys)

    direct_only = [ev for ev in direct_anomalies if anomaly_key(ev) not in original_set]
    original_only = [ev for ev in original_like_anomalies if anomaly_key(ev) not in direct_set]

    shared = direct_set & original_set
    jaccard = (len(shared) / len(direct_set | original_set)) if (direct_set or original_set) else 1.0

    return {
        "direct_count": len(direct_anomalies),
        "original_like_count": len(original_like_anomalies),
        "shared_count": len(shared),
        "direct_only_count": len(direct_only),
        "original_like_only_count": len(original_only),
        "jaccard": jaccard,
        "direct_only_examples": direct_only[:10],
        "original_like_only_examples": original_only[:10],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare original-like extracted-view detection vs direct raw-log detection"
    )
    parser.add_argument("--input_path", required=True, help="Raw stimulated DRAKVUF JSON file")
    parser.add_argument("--model_dir", default="models/out_elastic_endpoint_lstm_strictpath", help="LSTM model directory")
    parser.add_argument("--vocab_path", default="datasets/main/baseline/elastic_endpoint_vocab.json", help="Vocabulary JSON path")
    parser.add_argument("--valid_txt_path", default="datasets/lm_data_elastic_endpoint/valid.txt", help="Baseline validation text path")
    parser.add_argument("--output_path", required=True, help="Comparison JSON output path")
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--threshold_percentile", type=float, default=99.5)
    parser.add_argument("--context_size", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attack_processes", nargs="*", default=sorted(DEFAULT_ATTACK_PROCESSES))
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    model_dir = Path(args.model_dir).resolve()
    vocab_path = Path(args.vocab_path).resolve()
    valid_txt_path = Path(args.valid_txt_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    attack_processes = {proc.lower() for proc in args.attack_processes}

    vocab = load_vocab(vocab_path)
    model = load_lstm_model(model_dir, device)
    threshold_stats = estimate_threshold_from_valid(
        model=model,
        vocab=vocab,
        valid_txt_path=valid_txt_path,
        block_size=args.block_size,
        percentile=args.threshold_percentile,
        device=device,
    )

    direct_parsed = parse_stimulated_log(input_path, attack_processes)
    direct_events = direct_parsed["endpoint_events"]

    original_like_parsed = build_original_like_endpoint_events(input_path, attack_processes)
    original_like_events = original_like_parsed["endpoint_events"]

    sequence_comparison = compare_sequences(direct_events, original_like_events)

    direct_detection = detect_endpoint_anomalies(
        model=model,
        vocab=vocab,
        endpoint_events=direct_events,
        block_size=args.block_size,
        threshold=threshold_stats["threshold"],
        device=device,
        context_size=args.context_size,
    )
    original_like_detection = detect_endpoint_anomalies(
        model=model,
        vocab=vocab,
        endpoint_events=original_like_events,
        block_size=args.block_size,
        threshold=threshold_stats["threshold"],
        device=device,
        context_size=args.context_size,
    )

    anomaly_comparison = compare_anomalies(
        direct_detection["anomalies"],
        original_like_detection["anomalies"],
    )

    report = {
        "input_path": str(input_path),
        "model_dir": str(model_dir),
        "vocab_path": str(vocab_path),
        "valid_txt_path": str(valid_txt_path),
        "threshold_stats": threshold_stats,
        "direct_parse_summary": {
            "total_lines": direct_parsed["total_lines"],
            "json_errors": direct_parsed["json_errors"],
            "elastic_counts": direct_parsed["elastic_counts"],
            "attack_counts": direct_parsed["attack_counts"],
            "endpoint_event_count": len(direct_events),
        },
        "original_like_summary": {
            "total_lines": original_like_parsed["total_lines"],
            "json_errors": original_like_parsed["json_errors"],
            "skipped_sysret": original_like_parsed["skipped_sysret"],
            "filtered_record_count": original_like_parsed["filtered_record_count"],
            "endpoint_event_count": len(original_like_events),
        },
        "sequence_comparison": sequence_comparison,
        "direct_detection_summary": {
            "anomaly_count": direct_detection["anomaly_count"],
            "max_anomaly_nll": direct_detection["max_anomaly_nll"],
            "unknown_token_ratio": direct_detection["unknown_token_ratio"],
        },
        "original_like_detection_summary": {
            "anomaly_count": original_like_detection["anomaly_count"],
            "max_anomaly_nll": original_like_detection["max_anomaly_nll"],
            "unknown_token_ratio": original_like_detection["unknown_token_ratio"],
        },
        "anomaly_comparison": anomaly_comparison,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Original-like vs Direct Endpoint Detection Comparison")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Threshold: {threshold_stats['threshold']:.6f}")
    print("-" * 80)
    print(
        f"[SEQUENCE] direct={sequence_comparison['direct_count']} "
        f"original_like={sequence_comparison['original_like_count']} "
        f"shared={sequence_comparison['shared_count']} "
        f"same_order={sequence_comparison['same_order']}"
    )
    print(
        f"[ANOMALY] direct={anomaly_comparison['direct_count']} "
        f"original_like={anomaly_comparison['original_like_count']} "
        f"shared={anomaly_comparison['shared_count']} "
        f"jaccard={anomaly_comparison['jaccard']:.6f}"
    )
    print(f"[REPORT] {output_path}")


if __name__ == "__main__":
    main()
