#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Reuse the trained LSTM definition.
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from train_lstm import LSTMLanguageModel  # type: ignore


ELASTIC_PROCESSES = {
    "elastic-endpoint.exe",
    "elastic-agent.exe",
    "filebeat.exe",
    "metricbeat.exe",
}

DEFAULT_ATTACK_PROCESSES = {
    "powershell.exe",
    "pwsh.exe",
    "cmd.exe",
    "wscript.exe",
    "cscript.exe",
    "mshta.exe",
    "rundll32.exe",
    "regsvr32.exe",
    "certutil.exe",
}

SPECIAL_TOKENS = {"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
PROC_TOKEN_RE = re.compile(r"^<PROC:.*>$")


def extract_process_name(process_path: str) -> str:
    if not process_path:
        return ""
    process_path = process_path.replace("/", "\\")
    return process_path.split("\\")[-1].strip().lower()


def normalize_method(method: str) -> str:
    if not method:
        return "UnknownMethod"
    return re.sub(r"\s+", "", method.strip())


def is_special_or_proc_token(token: str) -> bool:
    return token in SPECIAL_TOKENS or bool(PROC_TOKEN_RE.match(token))


def load_vocab(vocab_path: Path) -> Dict[str, int]:
    with vocab_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_lstm_model(model_dir: Path, device: torch.device) -> LSTMLanguageModel:
    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = LSTMLanguageModel(
        vocab_size=cfg["vocab_size"],
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg.get("dropout", 0.2),
        tie_weights=True,
    )
    try:
        state_dict = torch.load(model_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def score_token_sequence(
    model: LSTMLanguageModel,
    token_ids: List[int],
    block_size: int,
    device: torch.device,
) -> List[Tuple[int, float]]:
    """
    Return [(sequence_position, nll)] for every predicted token.
    The position refers to the target token index inside token_ids.
    """
    criterion = nn.CrossEntropyLoss(reduction="none")
    results: List[Tuple[int, float]] = []

    with torch.no_grad():
        for start in range(0, len(token_ids) - 1, block_size):
            chunk = token_ids[start:start + block_size + 1]
            if len(chunk) < 2:
                continue

            inputs = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
            targets = torch.tensor(chunk[1:], dtype=torch.long, device=device)

            logits = model(inputs)[0]
            nlls = criterion(logits, targets).detach().cpu().tolist()

            for offset, nll in enumerate(nlls):
                seq_pos = start + 1 + offset
                results.append((seq_pos, float(nll)))

    return results


def estimate_threshold_from_valid(
    model: LSTMLanguageModel,
    vocab: Dict[str, int],
    valid_txt_path: Path,
    block_size: int,
    percentile: float,
    device: torch.device,
) -> Dict[str, float]:
    unk_id = vocab.get("<UNK>", 1)
    normal_nlls: List[float] = []

    with valid_txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            token_ids = [vocab.get(tok, unk_id) for tok in tokens]
            scores = score_token_sequence(model, token_ids, block_size, device)
            for seq_pos, nll in scores:
                token = tokens[seq_pos]
                if not is_special_or_proc_token(token):
                    normal_nlls.append(nll)

    if not normal_nlls:
        raise RuntimeError("No normal token NLLs were collected from validation data.")

    arr = np.array(normal_nlls, dtype=np.float64)
    threshold = float(np.percentile(arr, percentile))
    return {
        "token_count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "p995": float(np.percentile(arr, 99.5)),
        "percentile": float(percentile),
        "threshold": threshold,
    }


def parse_stimulated_log(
    input_path: Path,
    attack_processes: set[str],
) -> Dict[str, object]:
    endpoint_events: List[Dict[str, object]] = []
    elastic_counts: Counter[str] = Counter()
    attack_counts: Counter[str] = Counter()
    attack_samples: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    plugin_counts: Counter[str] = Counter()

    total_lines = 0
    json_errors = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            total_lines = line_no

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                continue

            plugin = obj.get("Plugin", "")
            plugin_counts[plugin] += 1
            if plugin == "sysret":
                continue

            proc = extract_process_name(obj.get("ProcessName", ""))
            if not proc:
                continue

            ts = float(obj.get("TimeStamp", 0.0))
            method = normalize_method(obj.get("Method", ""))

            if proc in ELASTIC_PROCESSES:
                elastic_counts[proc] += 1
                if proc == "elastic-endpoint.exe" and method:
                    endpoint_events.append(
                        {
                            "raw_line_no": line_no,
                            "timestamp": ts,
                            "process": proc,
                            "method": method,
                            "event_uid": obj.get("EventUID"),
                            "pid": obj.get("PID"),
                            "tid": obj.get("TID"),
                        }
                    )

            if proc in attack_processes:
                attack_counts[proc] += 1
                samples = attack_samples[proc]
                if len(samples) < 30:
                    samples.append(
                        {
                            "raw_line_no": line_no,
                            "timestamp": ts,
                            "method": method,
                            "pid": obj.get("PID"),
                            "tid": obj.get("TID"),
                        }
                    )

    endpoint_events.sort(key=lambda x: (float(x["timestamp"]), int(x["raw_line_no"])))

    return {
        "total_lines": total_lines,
        "json_errors": json_errors,
        "plugin_counts": dict(plugin_counts),
        "elastic_counts": dict(elastic_counts),
        "attack_counts": dict(attack_counts),
        "attack_samples": dict(attack_samples),
        "endpoint_events": endpoint_events,
    }


def build_endpoint_token_sequence(endpoint_events: List[Dict[str, object]]) -> List[str]:
    tokens = ["<BOS>", "<PROC:elastic-endpoint.exe>"]
    tokens.extend(str(ev["method"]) for ev in endpoint_events)
    tokens.append("<EOS>")
    return tokens


def detect_endpoint_anomalies(
    model: LSTMLanguageModel,
    vocab: Dict[str, int],
    endpoint_events: List[Dict[str, object]],
    block_size: int,
    threshold: float,
    device: torch.device,
    context_size: int,
) -> Dict[str, object]:
    unk_id = vocab.get("<UNK>", 1)
    tokens = build_endpoint_token_sequence(endpoint_events)
    token_ids = [vocab.get(tok, unk_id) for tok in tokens]
    unk_token_count = sum(1 for tok in tokens if vocab.get(tok, unk_id) == unk_id and tok != "<UNK>")

    scores = score_token_sequence(model, token_ids, block_size, device)
    anomalies: List[Dict[str, object]] = []

    for seq_pos, nll in scores:
        token = tokens[seq_pos]
        if is_special_or_proc_token(token):
            continue
        if nll <= threshold:
            continue

        event_idx = seq_pos - 2
        if event_idx < 0 or event_idx >= len(endpoint_events):
            continue

        event = endpoint_events[event_idx]
        ctx_start = max(0, event_idx - context_size)
        ctx_end = min(len(endpoint_events), event_idx + context_size + 1)

        anomalies.append(
            {
                "event_index": event_idx,
                "raw_line_no": event["raw_line_no"],
                "timestamp": event["timestamp"],
                "pid": event["pid"],
                "tid": event["tid"],
                "event_uid": event["event_uid"],
                "method": event["method"],
                "nll": nll,
                "unknown_token": vocab.get(token, unk_id) == unk_id,
                "context_before": [endpoint_events[i]["method"] for i in range(ctx_start, event_idx)],
                "context_after": [endpoint_events[i]["method"] for i in range(event_idx + 1, ctx_end)],
            }
        )

    anomalies.sort(key=lambda x: (-float(x["nll"]), float(x["timestamp"])))

    nll_values = [float(item["nll"]) for item in anomalies]
    return {
        "tokens": len(tokens),
        "endpoint_event_count": len(endpoint_events),
        "unknown_token_count": unk_token_count,
        "unknown_token_ratio": (unk_token_count / len(tokens)) if tokens else 0.0,
        "anomaly_count": len(anomalies),
        "max_anomaly_nll": max(nll_values) if nll_values else None,
        "anomalies": anomalies,
    }


def main():
    parser = argparse.ArgumentParser(description="Detect Elastic Endpoint anomalies with the trained LSTM model")
    parser.add_argument("--input_path", required=True, help="Raw stimulated DRAKVUF JSONL file")
    parser.add_argument("--model_dir", default="models/out_elastic_endpoint_lstm_strictpath", help="Trained LSTM model directory")
    parser.add_argument("--vocab_path", default="datasets/main/baseline/elastic_endpoint_vocab.json", help="Vocabulary JSON path")
    parser.add_argument("--valid_txt_path", default="datasets/lm_data_elastic_endpoint/valid.txt", help="Baseline validation text path")
    parser.add_argument("--output_path", default="runs/generated/lstm_endpoint_anomaly_report.json", help="Output JSON report")
    parser.add_argument("--block_size", type=int, default=256, help="Inference chunk size")
    parser.add_argument("--threshold_percentile", type=float, default=99.5, help="Normal token NLL percentile used as anomaly threshold")
    parser.add_argument("--context_size", type=int, default=5, help="How many neighboring endpoint APIs to save around each anomaly")
    parser.add_argument("--top_k", type=int, default=100, help="Keep the top-k anomalies in the compact output")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--attack_processes", nargs="*", default=sorted(DEFAULT_ATTACK_PROCESSES), help="Attack-side process names to summarize")
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    model_dir = Path(args.model_dir).resolve()
    vocab_path = Path(args.vocab_path).resolve()
    valid_txt_path = Path(args.valid_txt_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    attack_processes = {proc.lower() for proc in args.attack_processes}

    print("=" * 80)
    print("Elastic Endpoint LSTM Anomaly Detection")
    print("=" * 80)
    print(f"Input            : {input_path}")
    print(f"Model            : {model_dir}")
    print(f"Vocabulary       : {vocab_path}")
    print(f"Baseline valid   : {valid_txt_path}")
    print(f"Output           : {output_path}")
    print(f"Device           : {device}")
    print(f"Threshold pct    : {args.threshold_percentile}")
    print("-" * 80)

    start_time = time.perf_counter()
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
    print(
        f"[THRESHOLD] percentile={threshold_stats['percentile']:.2f} "
        f"threshold={threshold_stats['threshold']:.6f} "
        f"p99={threshold_stats['p99']:.6f} p99.5={threshold_stats['p995']:.6f}"
    )

    parsed = parse_stimulated_log(input_path, attack_processes)
    endpoint_events = parsed["endpoint_events"]
    if not endpoint_events:
        raise RuntimeError("No elastic-endpoint.exe syscall events were found in the stimulated log.")

    print(
        f"[PARSE] total_lines={parsed['total_lines']} json_errors={parsed['json_errors']} "
        f"endpoint_events={len(endpoint_events)}"
    )
    print(f"[PARSE] elastic_counts={parsed['elastic_counts']}")
    print(f"[PARSE] attack_counts={parsed['attack_counts']}")

    detection = detect_endpoint_anomalies(
        model=model,
        vocab=vocab,
        endpoint_events=endpoint_events,
        block_size=args.block_size,
        threshold=threshold_stats["threshold"],
        device=device,
        context_size=args.context_size,
    )

    top_anomalies = detection["anomalies"][:args.top_k]
    elapsed_sec = time.perf_counter() - start_time

    report = {
        "input_path": str(input_path),
        "model_dir": str(model_dir),
        "vocab_path": str(vocab_path),
        "valid_txt_path": str(valid_txt_path),
        "device": str(device),
        "block_size": args.block_size,
        "threshold_stats": threshold_stats,
        "parse_summary": {
            "total_lines": parsed["total_lines"],
            "json_errors": parsed["json_errors"],
            "plugin_counts": parsed["plugin_counts"],
            "elastic_counts": parsed["elastic_counts"],
            "attack_counts": parsed["attack_counts"],
            "attack_samples": parsed["attack_samples"],
        },
        "detection_summary": {
            "endpoint_event_count": detection["endpoint_event_count"],
            "token_count": detection["tokens"],
            "unknown_token_count": detection["unknown_token_count"],
            "unknown_token_ratio": detection["unknown_token_ratio"],
            "anomaly_count": detection["anomaly_count"],
            "max_anomaly_nll": detection["max_anomaly_nll"],
            "top_k": args.top_k,
        },
        "top_anomalies": top_anomalies,
        "elapsed_sec": elapsed_sec,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("-" * 80)
    print(
        f"[DONE] anomaly_count={detection['anomaly_count']} "
        f"top_saved={len(top_anomalies)} elapsed={elapsed_sec:.2f}s"
    )
    if top_anomalies:
        for idx, anomaly in enumerate(top_anomalies[:10], 1):
            print(
                f"  {idx:2d}. line={anomaly['raw_line_no']} "
                f"ts={anomaly['timestamp']:.6f} "
                f"method={anomaly['method']} nll={anomaly['nll']:.4f}"
            )
    print(f"[REPORT] {output_path}")


if __name__ == "__main__":
    main()
