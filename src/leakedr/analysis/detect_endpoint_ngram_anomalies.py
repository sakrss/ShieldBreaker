#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import pickle
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


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


class BackoffNGramScorer:
    def __init__(self, state: Dict):
        self.order = int(state["best_order"])
        self.alpha = float(state["alpha"])
        self.unk_token = state["unk_token"]
        self.vocab_tokens = list(state["vocab_tokens"])
        self.vocab_set = set(self.vocab_tokens)
        self.vocab_size = int(state["vocab_size"])
        self.total_tokens = int(state["total_tokens"])
        self.ngram_counts = state["ngram_counts"]
        self.context_counts = state["context_counts"]
        self._cache: Dict[tuple, float] = {}

    def normalize_token(self, token: str) -> str:
        return token if token in self.vocab_set else self.unk_token

    def probability(self, token: str, history: List[str], order: int | None = None) -> float:
        current_order = self.order if order is None else order
        history = tuple(history[-(current_order - 1):]) if current_order > 1 else ()
        token = self.normalize_token(token)
        key = (current_order, history, token)
        if key in self._cache:
            return self._cache[key]

        if current_order == 1:
            numerator = self.ngram_counts[0].get((token,), 0) + self.alpha
            denominator = self.total_tokens + self.alpha * self.vocab_size
            value = numerator / denominator
        else:
            lower_prob = self.probability(token, list(history[1:]), current_order - 1)
            context_count = self.context_counts[current_order - 1].get(history, 0)

            if context_count == 0:
                value = lower_prob
            else:
                full_count = self.ngram_counts[current_order - 1].get(history + (token,), 0)
                numerator = full_count + self.alpha * self.vocab_size * lower_prob
                denominator = context_count + self.alpha * self.vocab_size
                value = numerator / denominator

        self._cache[key] = value
        return value

    def score_tokens(self, tokens: List[str]) -> List[tuple[int, float]]:
        scores: List[tuple[int, float]] = []
        history: List[str] = []
        for idx, token in enumerate(tokens):
            prob = self.probability(token, history)
            scores.append((idx, -math.log(prob)))
            history.append(self.normalize_token(token))
        return scores


def load_vocab(vocab_path: Path) -> Dict[str, int]:
    with vocab_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_ngram_scorer(model_dir: Path) -> BackoffNGramScorer:
    with (model_dir / "ngram_model.pkl").open("rb") as f:
        state = pickle.load(f)
    return BackoffNGramScorer(state)


def estimate_threshold_from_valid(
    scorer: BackoffNGramScorer,
    valid_txt_path: Path,
    percentile: float,
) -> Dict[str, float]:
    normal_nlls: List[float] = []
    with valid_txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            for idx, nll in scorer.score_tokens(tokens):
                token = tokens[idx]
                if not is_special_or_proc_token(token):
                    normal_nlls.append(float(nll))

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
    scorer: BackoffNGramScorer,
    vocab: Dict[str, int],
    endpoint_events: List[Dict[str, object]],
    threshold: float,
    context_size: int,
) -> Dict[str, object]:
    tokens = build_endpoint_token_sequence(endpoint_events)
    unk_token_count = sum(1 for tok in tokens if tok not in vocab)
    scores = scorer.score_tokens(tokens)

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
                "nll": float(nll),
                "unknown_token": token not in vocab,
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
    parser = argparse.ArgumentParser(description="Detect Elastic Endpoint anomalies with the trained N-gram model")
    parser.add_argument("--input_path", required=True, help="Raw stimulated DRAKVUF JSONL file")
    parser.add_argument("--model_dir", default="models/out_elastic_endpoint_ngram_strictpath", help="Trained N-gram model directory")
    parser.add_argument("--vocab_path", default="datasets/main/baseline/elastic_endpoint_vocab.json", help="Vocabulary JSON path")
    parser.add_argument("--valid_txt_path", default="datasets/lm_data_elastic_endpoint/valid.txt", help="Baseline validation text path")
    parser.add_argument("--output_path", default="runs/generated/ngram_endpoint_anomaly_report.json", help="Output JSON report")
    parser.add_argument("--threshold_percentile", type=float, default=99.5, help="Normal token NLL percentile used as anomaly threshold")
    parser.add_argument("--context_size", type=int, default=5, help="How many neighboring endpoint APIs to save around each anomaly")
    parser.add_argument("--top_k", type=int, default=300, help="Keep the top-k anomalies in the compact output")
    parser.add_argument("--attack_processes", nargs="*", default=sorted(DEFAULT_ATTACK_PROCESSES), help="Attack-side process names to summarize")
    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    model_dir = Path(args.model_dir).resolve()
    vocab_path = Path(args.vocab_path).resolve()
    valid_txt_path = Path(args.valid_txt_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attack_processes = {proc.lower() for proc in args.attack_processes}

    print("=" * 80)
    print("Elastic Endpoint N-gram Anomaly Detection")
    print("=" * 80)
    print(f"Input            : {input_path}")
    print(f"Model            : {model_dir}")
    print(f"Vocabulary       : {vocab_path}")
    print(f"Baseline valid   : {valid_txt_path}")
    print(f"Output           : {output_path}")
    print(f"Threshold pct    : {args.threshold_percentile}")
    print("-" * 80)

    start_time = time.perf_counter()
    vocab = load_vocab(vocab_path)
    scorer = load_ngram_scorer(model_dir)

    threshold_stats = estimate_threshold_from_valid(
        scorer=scorer,
        valid_txt_path=valid_txt_path,
        percentile=args.threshold_percentile,
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
        scorer=scorer,
        vocab=vocab,
        endpoint_events=endpoint_events,
        threshold=threshold_stats["threshold"],
        context_size=args.context_size,
    )
    top_anomalies = detection["anomalies"][:args.top_k]
    elapsed_sec = time.perf_counter() - start_time

    report = {
        "input_path": str(input_path),
        "model_dir": str(model_dir),
        "vocab_path": str(vocab_path),
        "valid_txt_path": str(valid_txt_path),
        "order": scorer.order,
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
