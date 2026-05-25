#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_windows(payload: Dict) -> List[Dict[str, object]]:
    threshold = float(payload["threshold_stats"]["threshold"])
    rows: List[Dict[str, object]] = []
    for window in sorted(payload["windows"], key=lambda item: int(item["window_index"])):
        start_event_index = int(window["start_event_index"])
        window_index = int(window["window_index"])
        for offset, nll in enumerate(window["nlls"]):
            event_index = start_event_index + offset
            value = float(nll)
            rows.append(
                {
                    "event_index": event_index,
                    "nll": value,
                    "window_index": window_index,
                    "window_offset": offset,
                    "above_threshold": value > threshold,
                }
            )
    return rows


def write_sample_outputs(full_path: Path, prefix: str) -> List[Dict[str, object]]:
    payload = load_json(full_path)
    rows = flatten_windows(payload)
    sample = full_path.parent.name
    threshold = float(payload["threshold_stats"]["threshold"])

    csv_path = full_path.parent / f"{prefix}_event_nll_sequence.csv"
    json_path = full_path.parent / f"{prefix}_event_nll_sequence.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "event_index", "nll", "window_index", "window_offset", "above_threshold"])
        for row in rows:
            writer.writerow(
                [
                    sample,
                    row["event_index"],
                    f"{float(row['nll']):.8f}",
                    row["window_index"],
                    row["window_offset"],
                    int(bool(row["above_threshold"])),
                ]
            )

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sample": sample,
                "source": str(full_path),
                "threshold": threshold,
                "endpoint_event_count": int(payload["endpoint_event_count"]),
                "sequence_length": len(rows),
                "event_indices": [int(row["event_index"]) for row in rows],
                "nlls": [float(row["nll"]) for row in rows],
                "above_threshold": [bool(row["above_threshold"]) for row in rows],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] sample={sample} points={len(rows)} csv={csv_path} json={json_path}")
    for row in rows:
        row["sample"] = sample
    return rows


def write_aggregate(root: Path, prefix: str, rows: List[Dict[str, object]]) -> None:
    csv_path = root / f"{prefix}_event_nll_sequence_all_samples.csv"
    json_path = root / f"{prefix}_event_nll_sequence_all_samples.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "event_index", "nll", "window_index", "window_offset", "above_threshold"])
        for row in rows:
            writer.writerow(
                [
                    row["sample"],
                    row["event_index"],
                    f"{float(row['nll']):.8f}",
                    row["window_index"],
                    row["window_offset"],
                    int(bool(row["above_threshold"])),
                ]
            )

    by_sample: Dict[str, Dict[str, object]] = {}
    for row in rows:
        sample = str(row["sample"])
        bucket = by_sample.setdefault(
            sample,
            {
                "sample": sample,
                "event_indices": [],
                "nlls": [],
                "above_threshold": [],
            },
        )
        bucket["event_indices"].append(int(row["event_index"]))
        bucket["nlls"].append(float(row["nll"]))
        bucket["above_threshold"].append(bool(row["above_threshold"]))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "root": str(root),
                "prefix": prefix,
                "sample_count": len(by_sample),
                "point_count": len(rows),
                "samples": list(by_sample.values()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[DONE] aggregate_csv={csv_path}")
    print(f"[DONE] aggregate_json={json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten full window NLL matrices into continuous event-level NLL sequences")
    parser.add_argument("--root", required=True)
    parser.add_argument("--prefix", default="gpt2_medium")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    full_files = sorted(root.rglob(f"{args.prefix}_window_nll_full.json"))
    if not full_files:
        raise RuntimeError(f"No {args.prefix}_window_nll_full.json files found under {root}")

    all_rows: List[Dict[str, object]] = []
    for full_path in full_files:
        all_rows.extend(write_sample_outputs(full_path, args.prefix))

    write_aggregate(root, args.prefix, all_rows)
    print(f"[DONE] samples={len(full_files)} points={len(all_rows)}")


if __name__ == "__main__":
    main()
