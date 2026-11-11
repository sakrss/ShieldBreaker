"""Commercial-trace characterization: API baseline -> intervals -> co-occurrence."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from shieldbreaker.config import AnalysisConfig
from shieldbreaker.manifest import load_manifest
from shieldbreaker.traces import (
    partition_records,
    read_records,
    write_json,
    write_jsonl,
)
from shieldbreaker.training.models import NGramModel
from shieldbreaker.analysis.temporal import (
    multiscale_windows,
    baseline_threshold,
    localize,
)
from shieldbreaker.analysis.semantics import (
    validate_reconstruction,
    reconstruction_request,
    PROMPT_ID,
)
from shieldbreaker.analysis.recurrence import estimate_recurrence, evaluate_held_out
from shieldbreaker.analysis.evaluation import trace_distribution_summary


def read_execution(row, manifest):
    records = read_records(row["path"])
    if len({r["ExecutionID"] for r in records if "ExecutionID" in r}) > 1:
        raise ValueError(
            "Each analysis trace must contain exactly one independent execution"
        )
    edr, stimulus = partition_records(
        records,
        manifest["edr_processes"],
        row.get("stimulus_processes", ()),
        row.get("stimulus_pids", ()),
    )
    if not edr:
        raise ValueError(f"No observable EDR API calls in {row['execution_id']}")
    return edr, stimulus


def run_analysis(
    manifest_path,
    output_dir,
    model_kind="ngram",
    model_path=None,
    config=AnalysisConfig(),
    device=None,
    inventory_path=None,
):
    manifest = load_manifest(manifest_path)
    out = Path(output_dir)
    traces = manifest["traces"]
    discovery = [
        row
        for row in traces
        if row["condition"] == "baseline" and row["split"] == "discovery"
    ]
    validation = [
        row
        for row in traces
        if row["condition"] == "baseline" and row["split"] == "validation"
    ]
    if not discovery or not validation:
        raise ValueError(
            "Independent discovery and validation baseline traces are required"
        )
    if model_kind == "ngram":
        model = NGramModel().fit(
            [event.function for event in read_execution(row, manifest)[0]]
            for row in discovery
        )
        model.save(out / "baseline_ngram.json")
    elif model_kind == "gpt2":
        if model_path is None:
            raise ValueError("GPT-2 requires a trained API-only checkpoint")
        from shieldbreaker.training.anomaly_detector import EDRAnomalyDetector

        model = EDRAnomalyDetector(model_path, device=device)
        provenance = Path(model_path) / "split_manifest.json"
        if not provenance.is_file():
            raise ValueError(
                "GPT-2 checkpoint needs the corpus split_manifest.json for leakage checks"
            )
        splits = json.loads(provenance.read_text())
        trained = {row["execution_id"] for row in splits if row["split"] == "discovery"}
        if trained != {row["execution_id"] for row in discovery}:
            raise ValueError(
                "Checkpoint discovery IDs do not match this baseline manifest"
            )
    elif model_kind == "lstm":
        from shieldbreaker.training.lstm import LSTMModel

        model = LSTMModel.load(model_path, device=device)
        if set(model.discovery_ids) != {row["execution_id"] for row in discovery}:
            raise ValueError(
                "LSTM training provenance does not match baseline discovery IDs"
            )
    else:
        raise ValueError(f"Unknown model: {model_kind}")
    score_cache = {}

    def score(row):
        if row["execution_id"] in score_cache:
            return score_cache[row["execution_id"]]
        edr, stimulus = read_execution(row, manifest)
        if model_kind in {"gpt2", "lstm"} and model.device.type == "cuda":
            import torch

            torch.cuda.synchronize(model.device)
        started = time.perf_counter()
        nll = model.score([event.function for event in edr])
        if model_kind in {"gpt2", "lstm"} and model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        inference_seconds = time.perf_counter() - started
        score_cache[row["execution_id"]] = (
            multiscale_windows(nll, [event.timestamp for event in edr], config),
            stimulus,
            inference_seconds,
            len(edr),
        )
        return score_cache[row["execution_id"]]

    calibration = [score(row)[0] for row in validation]
    threshold = baseline_threshold(calibration, config.quantile)
    write_json(
        out / "calibration.json",
        {
            "configuration": manifest["configuration"],
            "method": config.to_dict(),
            "score": "max_multiscale_mean_nll",
            "threshold": threshold,
            "baseline_execution_ids": [row["execution_id"] for row in validation],
        },
    )
    observations, distribution = [], []
    for row in traces:
        started = time.perf_counter()
        windows, stimulus, inference_seconds, api_events = score(row)
        anomalous, clusters, tolerance = localize(windows, threshold)
        if row["condition"] == "stimulated":
            if row.get("annotation") is None:
                raise ValueError(
                    f"Missing explicit semantic annotation for {row['execution_id']}; export requests first"
                )
            annotation = json.loads(
                Path(row["annotation"]).read_text(encoding="utf-8-sig")
            )
            behaviors = validate_reconstruction(annotation, stimulus)
            observations.append(
                {
                    "execution_id": row["execution_id"],
                    "split": row["split"],
                    "configuration": manifest["configuration"],
                    "behaviors": behaviors,
                    "clusters": clusters,
                    "anomalous_windows": [w.to_dict() for w in anomalous],
                    "interval_tolerance": tolerance,
                    "scoreable": bool(windows),
                    "semantic_prompt_id": annotation.get("prompt_id"),
                    "unsupported_conclusions": annotation["unsupported_conclusions"],
                }
            )
        distribution.append(
            {
                "execution_id": row["execution_id"],
                "split": row["split"],
                "condition": row["condition"],
                "window_count": len(windows),
                "scoreable": bool(windows),
                "trace_p995": float(np.quantile([w.score for w in windows], 0.995))
                if windows
                else None,
                "model_inference_seconds": inference_seconds,
                "api_events": api_events,
                "processing_seconds_including_io": time.perf_counter() - started,
            }
        )
        score_cache.pop(row["execution_id"], None)
    inventory = json.loads(Path(inventory_path).read_text()) if inventory_path else None
    if inventory is not None:
        ids = [row["candidate_id"] for row in inventory]
        if len(set(ids)) != len(ids):
            raise ValueError(
                "External candidate IDs must be unique, even when behaviors repeat"
            )
    candidates = estimate_recurrence(
        [obs for obs in observations if obs["split"] == "discovery"], config, inventory
    )
    metrics = evaluate_held_out(
        candidates, [obs for obs in observations if obs["split"] == "test"]
    )
    write_jsonl(out / "observations.jsonl", observations)
    write_json(out / "candidates.json", candidates)
    write_json(out / "held_out_metrics.json", metrics)
    write_jsonl(out / "trace_scores.jsonl", distribution)
    write_json(
        out / "condition_summaries.json", trace_distribution_summary(distribution)
    )
    write_json(
        out / "run_metadata.json",
        {
            "configuration": manifest["configuration"],
            "model": model_kind,
            "manifest_sha256": hashlib.sha256(
                Path(manifest_path).read_bytes()
            ).hexdigest(),
            "config": config.to_dict(),
            "semantic_prompt_id": PROMPT_ID,
            "association": "nonempty closed-interval intersection; no causal or persistent response identity claim",
            "inventory_unit": metrics["candidate_unit"],
            "scope": "New measurements only; does not reconstruct missing historical commercial inventories/results",
        },
    )
    return metrics


def export_semantic_requests(manifest_path, output_dir):
    manifest = load_manifest(manifest_path)
    for index, row in enumerate(manifest["traces"]):
        if row["condition"] == "stimulated":
            _, stimulus = read_execution(row, manifest)
            write_json(
                Path(output_dir) / f"{index:05d}.json",
                {
                    "execution_id": row["execution_id"],
                    **reconstruction_request(stimulus),
                },
            )
