#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from detect_endpoint_gpt_anomalies import (
    DEFAULT_ATTACK_PROCESSES,
    ELASTIC_PROCESSES,
    estimate_threshold_from_valid,
    load_model_and_tokenizer,
    load_vocab,
    parse_stimulated_log,
)


def score_windows_full(
    model,
    token_ids: List[int],
    max_seq_len: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    window_body = max_seq_len - 2
    if window_body < 1:
        raise ValueError("max_seq_len is too small")

    total_event_count = max(0, len(token_ids) - 3)
    windows: List[Tuple[int, List[int]]] = []
    if len(token_ids) <= max_seq_len:
        windows = [(0, token_ids)]
    else:
        start_event_idx = 2
        while start_event_idx < len(token_ids):
            prefix = token_ids[:2]
            body = token_ids[start_event_idx:start_event_idx + window_body]
            chunk = prefix + body
            if not chunk:
                break
            windows.append((start_event_idx, chunk))
            start_event_idx += window_body

    full_windows: List[Dict[str, object]] = []
    with torch.no_grad():
        for start_pos, chunk_ids in windows:
            if len(chunk_ids) < 2:
                continue

            input_ids = torch.tensor(chunk_ids, dtype=torch.long, device=device).unsqueeze(0)
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]
            pred_logits = logits[:-1]
            targets = input_ids[0, 1:]
            nlls = loss_fct(pred_logits, targets).detach().cpu().tolist()

            event_scores: List[Tuple[int, float]] = []
            if start_pos == 0:
                for offset, nll in enumerate(nlls):
                    seq_pos = 1 + offset
                    if 2 <= seq_pos <= total_event_count + 1:
                        event_scores.append((seq_pos - 2, float(nll)))
            else:
                for local_target_idx in range(1, len(chunk_ids) - 1):
                    seq_pos = start_pos + local_target_idx - 1
                    if 2 <= seq_pos <= total_event_count + 1:
                        event_scores.append((seq_pos - 2, float(nlls[local_target_idx])))

            if not event_scores:
                continue

            nll_values = [nll for _, nll in event_scores]
            max_offset, (max_event_index, max_nll) = max(enumerate(event_scores), key=lambda item: item[1][1])
            top_items = sorted(event_scores, key=lambda item: item[1], reverse=True)[:10]
            top_values = [nll for _, nll in top_items]
            full_windows.append(
                {
                    "window_index": len(full_windows),
                    "start_event_index": int(event_scores[0][0]),
                    "end_event_index": int(event_scores[-1][0]),
                    "event_count": len(event_scores),
                    "avg_nll": float(sum(nll_values) / len(nll_values)),
                    "max_nll": float(max_nll),
                    "max_event_index": int(max_event_index),
                    "max_offset": int(max_offset),
                    "top10_mean_nll": float(sum(top_values) / len(top_values)),
                    "top10_event_indices": [int(event_index) for event_index, _ in top_items],
                    "nlls": nll_values,
                }
            )

    return full_windows


def build_tokens(endpoint_events: List[Dict[str, object]], endpoint_process: str) -> List[str]:
    return ["<BOS>", f"<PROC:{endpoint_process}>"] + [str(ev["method"]) for ev in endpoint_events] + ["<EOS>"]


def write_sample_outputs(
    input_path: Path,
    output_json: Path,
    output_csv: Path,
    model_dir: Path,
    vocab_path: Path,
    valid_txt_path: Path,
    max_seq_len: int,
    endpoint_process: str,
    threshold_stats: Dict[str, float],
    endpoint_events: List[Dict[str, object]],
    windows: List[Dict[str, object]],
    event_rows: List[Dict[str, object]],
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    for window in windows:
        start_event_index = int(window["start_event_index"])
        end_event_index = int(window["end_event_index"])
        if 0 <= start_event_index < len(endpoint_events):
            start_event = endpoint_events[start_event_index]
            window["start_timestamp"] = float(start_event["timestamp"])
            window["start_raw_line_no"] = int(start_event["raw_line_no"])
        if 0 <= end_event_index < len(endpoint_events):
            end_event = endpoint_events[end_event_index]
            window["end_timestamp"] = float(end_event["timestamp"])
            window["end_raw_line_no"] = int(end_event["raw_line_no"])

    window_payload = {
        "input_path": str(input_path),
        "model_dir": str(model_dir),
        "vocab_path": str(vocab_path),
        "valid_txt_path": str(valid_txt_path),
        "endpoint_process": endpoint_process,
        "max_seq_len": max_seq_len,
        "window_body": max_seq_len - 2,
        "threshold_stats": threshold_stats,
        "endpoint_event_count": len(endpoint_events),
        "window_count": len(windows),
        "windows": windows,
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(window_payload, f, ensure_ascii=False, indent=2)

    event_json = output_json.with_name(output_json.name.replace("_window_nll_full.json", "_event_nll_sequence.json"))
    event_payload = {
        "input_path": str(input_path),
        "model_dir": str(model_dir),
        "vocab_path": str(vocab_path),
        "valid_txt_path": str(valid_txt_path),
        "endpoint_process": endpoint_process,
        "threshold_stats": threshold_stats,
        "max_seq_len": max_seq_len,
        "endpoint_event_count": len(endpoint_events),
        "event_count": len(event_rows),
        "events": event_rows,
    }
    with event_json.open("w", encoding="utf-8") as f:
        json.dump(event_payload, f, ensure_ascii=False, indent=2)

    event_csv = output_json.with_name(output_json.name.replace("_window_nll_full.json", "_event_nll_sequence.csv"))
    with event_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_index", "raw_line_no", "timestamp", "pid", "tid", "event_uid", "method", "nll", "unknown_token"])
        for event in event_rows:
            writer.writerow(
                [
                    event["event_index"],
                    event["raw_line_no"],
                    event["timestamp"],
                    event["pid"],
                    event["tid"],
                    event["event_uid"],
                    event["method"],
                    f"{float(event['nll']):.8f}",
                    str(bool(event["unknown_token"])).lower(),
                ]
            )

    max_len = max((int(item["event_count"]) for item in windows), default=0)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample",
                "window_index",
                "start_event_index",
                "end_event_index",
                "event_count",
                "avg_nll",
                "max_nll",
                "top10_mean_nll",
                "max_event_index",
                "max_offset",
            ]
            + [f"pos_{idx:04d}" for idx in range(max_len)]
        )
        sample = input_path.parent.name
        for item in windows:
            nlls = item["nlls"]
            writer.writerow(
                [
                    sample,
                    item["window_index"],
                    item["start_event_index"],
                    item["end_event_index"],
                    item["event_count"],
                    f"{float(item['avg_nll']):.8f}",
                    f"{float(item['max_nll']):.8f}",
                    f"{float(item['top10_mean_nll']):.8f}",
                    item["max_event_index"],
                    item["max_offset"],
                ]
                + [f"{float(value):.8f}" for value in nlls]
                + [""] * (max_len - len(nlls))
            )


def process_one(
    input_path: Path,
    model,
    tokenizer,
    model_dir: Path,
    vocab_path: Path,
    valid_txt_path: Path,
    threshold_stats: Dict[str, float],
    max_seq_len: int,
    device: torch.device,
    endpoint_process: str,
    edr_processes: set[str],
    prefix: str,
) -> Tuple[Path, Path, List[Dict[str, object]]]:
    parsed = parse_stimulated_log(input_path, set(DEFAULT_ATTACK_PROCESSES), endpoint_process, edr_processes)
    endpoint_events = parsed["endpoint_events"]
    if not endpoint_events:
        raise RuntimeError(f"No {endpoint_process} events found in {input_path}")

    tokens = build_tokens(endpoint_events, endpoint_process)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    unknown_token_id = tokenizer.convert_tokens_to_ids("<UNK>")
    windows = score_windows_full(model, token_ids, max_seq_len, device)
    nll_by_event_index: Dict[int, Tuple[float, int, int]] = {}
    for window in windows:
        start_event_index = int(window["start_event_index"])
        window_index = int(window["window_index"])
        for offset, nll in enumerate(window["nlls"]):
            nll_by_event_index[start_event_index + offset] = (float(nll), window_index, offset)

    event_rows: List[Dict[str, object]] = []
    for event_index, event in enumerate(endpoint_events):
        nll_info = nll_by_event_index.get(event_index)
        if nll_info is None:
            continue
        nll, window_index, window_offset = nll_info
        seq_pos = event_index + 2
        event_rows.append(
            {
                "event_index": event_index,
                "raw_line_no": event["raw_line_no"],
                "timestamp": event["timestamp"],
                "pid": event.get("pid"),
                "tid": event.get("tid"),
                "event_uid": event.get("event_uid"),
                "method": event["method"],
                "nll": nll,
                "unknown_token": token_ids[seq_pos] == unknown_token_id,
                "window_index": window_index,
                "window_offset": window_offset,
            }
        )

    output_json = input_path.parent / f"{prefix}_window_nll_full.json"
    output_csv = input_path.parent / f"{prefix}_window_nll_full_matrix.csv"
    write_sample_outputs(
        input_path=input_path,
        output_json=output_json,
        output_csv=output_csv,
        model_dir=model_dir,
        vocab_path=vocab_path,
        valid_txt_path=valid_txt_path,
        max_seq_len=max_seq_len,
        endpoint_process=endpoint_process,
        threshold_stats=threshold_stats,
        endpoint_events=endpoint_events,
        windows=windows,
        event_rows=event_rows,
    )
    return output_json, output_csv, windows


def write_aggregate(root: Path, prefix: str, sample_rows: List[Dict[str, object]]) -> Tuple[Path, Path]:
    max_len = max((len(row["nlls"]) for row in sample_rows), default=0)
    matrix_path = root / f"{prefix}_window_nll_full_matrix_all_samples.csv"
    meta_path = root / f"{prefix}_window_nll_full_metadata_all_samples.json"

    with matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row_id",
                "sample",
                "window_index",
                "start_event_index",
                "end_event_index",
                "event_count",
                "avg_nll",
                "max_nll",
                "top10_mean_nll",
                "max_event_index",
                "max_offset",
            ]
            + [f"pos_{idx:04d}" for idx in range(max_len)]
        )
        for row_id, row in enumerate(sample_rows):
            nlls = row["nlls"]
            writer.writerow(
                [
                    row_id,
                    row["sample"],
                    row["window_index"],
                    row["start_event_index"],
                    row["end_event_index"],
                    row["event_count"],
                    f"{float(row['avg_nll']):.8f}",
                    f"{float(row['max_nll']):.8f}",
                    f"{float(row['top10_mean_nll']):.8f}",
                    row["max_event_index"],
                    row["max_offset"],
                ]
                + [f"{float(value):.8f}" for value in nlls]
                + [""] * (max_len - len(nlls))
            )

    metadata = {
        "root": str(root),
        "prefix": prefix,
        "row_count": len(sample_rows),
        "max_window_width": max_len,
        "rows": [
            {key: row[key] for key in row if key != "nlls"}
            for row in sample_rows
        ],
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return matrix_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export complete per-window token/event NLL matrices")
    parser.add_argument("--root", required=True, help="Root directory containing sample drakvuf.json files")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--valid_txt_path", required=True)
    parser.add_argument("--prefix", default="gpt2_medium")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--threshold_percentile", type=float, default=99.5)
    parser.add_argument("--endpoint_process", default="elastic-endpoint.exe")
    parser.add_argument("--edr_processes", nargs="*", default=sorted(ELASTIC_PROCESSES))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    model_dir = Path(args.model_dir).resolve()
    vocab_path = Path(args.vocab_path).resolve()
    valid_txt_path = Path(args.valid_txt_path).resolve()
    endpoint_process = args.endpoint_process.lower()
    edr_processes = {proc.lower() for proc in args.edr_processes}
    edr_processes.add(endpoint_process)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    input_files = sorted(root.rglob("drakvuf.json"))
    if not input_files:
        raise RuntimeError(f"No drakvuf.json files found under {root}")

    vocab = load_vocab(vocab_path)
    _ = vocab
    model, tokenizer = load_model_and_tokenizer(model_dir, device)
    threshold_stats = estimate_threshold_from_valid(
        model=model,
        tokenizer=tokenizer,
        valid_txt_path=valid_txt_path,
        max_seq_len=args.max_seq_len,
        percentile=args.threshold_percentile,
        device=device,
    )

    all_rows: List[Dict[str, object]] = []
    for idx, input_path in enumerate(input_files, 1):
        print(f"[{idx}/{len(input_files)}] {input_path}")
        output_json, output_csv, windows = process_one(
            input_path=input_path,
            model=model,
            tokenizer=tokenizer,
            model_dir=model_dir,
            vocab_path=vocab_path,
            valid_txt_path=valid_txt_path,
            threshold_stats=threshold_stats,
            max_seq_len=args.max_seq_len,
            device=device,
            endpoint_process=endpoint_process,
            edr_processes=edr_processes,
            prefix=args.prefix,
        )
        sample = input_path.parent.name
        for window in windows:
            all_rows.append({"sample": sample, **window})
        print(f"[OK] windows={len(windows)} json={output_json} csv={output_csv}")

    matrix_path, meta_path = write_aggregate(root, args.prefix, all_rows)
    print(f"[DONE] samples={len(input_files)} windows={len(all_rows)}")
    print(f"[DONE] aggregate_matrix={matrix_path}")
    print(f"[DONE] aggregate_metadata={meta_path}")


if __name__ == "__main__":
    main()
