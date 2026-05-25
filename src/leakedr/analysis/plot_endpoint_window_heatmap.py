#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_samples(root: Path, prefix: str) -> List[Tuple[str, Path]]:
    files = sorted(root.rglob(f"{prefix}_window_max_nll_points.json"))
    samples: List[Tuple[str, Path]] = []
    for file_path in files:
        samples.append((file_path.parent.name, file_path))
    return samples


def build_matrix(
    samples: List[Tuple[str, Path]],
    metric: str,
    transform: str,
) -> Tuple[List[str], np.ndarray, List[Dict]]:
    rows: List[List[float]] = []
    labels: List[str] = []
    metadata: List[Dict] = []
    max_windows = 0

    for sample_name, max_path in samples:
        max_data = load_json(max_path)
        threshold = float(max_data.get("threshold", 0.0))
        max_points = max_data.get("window_max_nll_points", [])

        means_path = max_path.with_name(max_path.name.replace("_window_max_nll_points.json", "_window_nll_means.json"))
        mean_by_window: Dict[int, float] = {}
        if means_path.exists():
            mean_data = load_json(means_path)
            for item in mean_data.get("window_nll_means", []):
                mean_by_window[int(item["window_index"])] = float(item["avg_nll"])

        values: List[float] = []
        for point in sorted(max_points, key=lambda x: int(x["window_index"])):
            window_idx = int(point["window_index"])
            max_nll = float(point["nll"])
            avg_nll = mean_by_window.get(window_idx, np.nan)

            if metric == "max_nll":
                value = max_nll
            elif metric == "avg_nll":
                value = avg_nll
            elif metric == "max_minus_avg":
                value = max_nll - avg_nll if not np.isnan(avg_nll) else np.nan
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            if transform == "threshold_delta":
                value = value - threshold
            elif transform == "threshold_ratio":
                value = value / threshold if threshold else np.nan
            elif transform != "raw":
                raise ValueError(f"Unsupported transform: {transform}")

            values.append(float(value))

        labels.append(sample_name)
        rows.append(values)
        max_windows = max(max_windows, len(values))
        metadata.append(
            {
                "sample": sample_name,
                "source": str(max_path),
                "window_count": int(max_data.get("window_count", len(values))),
                "threshold": threshold,
                "endpoint_event_count": int(max_data.get("endpoint_event_count", 0)),
            }
        )

    matrix = np.full((len(rows), max_windows), np.nan, dtype=np.float64)
    for row_idx, values in enumerate(rows):
        matrix[row_idx, : len(values)] = values

    return labels, matrix, metadata


def write_csv(path: Path, labels: List[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample"] + [f"window_{idx}" for idx in range(matrix.shape[1])])
        for label, row in zip(labels, matrix):
            writer.writerow([label] + ["" if np.isnan(value) else f"{value:.8f}" for value in row])


def write_json(path: Path, labels: List[str], matrix: np.ndarray, metadata: List[Dict], metric: str, transform: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric": metric,
        "transform": transform,
        "sample_count": len(labels),
        "max_window_count": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "samples": metadata,
        "matrix": [
            {
                "sample": label,
                "values": [None if np.isnan(value) else float(value) for value in row],
            }
            for label, row in zip(labels, matrix)
        ],
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def plot_heatmap(
    path: Path,
    labels: List[str],
    matrix: np.ndarray,
    title: str,
    cmap_name: str,
    clip_percentile: Optional[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#eeeeee")

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise RuntimeError("No finite heatmap values were found.")

    if clip_percentile is not None:
        vmin = float(np.percentile(finite, 100.0 - clip_percentile))
        vmax = float(np.percentile(finite, clip_percentile))
    else:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0

    fig_width = max(10.0, min(26.0, matrix.shape[1] * 0.22))
    fig_height = max(5.5, len(labels) * 0.38 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=180)
    image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Window index")
    ax.set_ylabel("Sample")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)

    max_window = matrix.shape[1]
    step = max(1, int(np.ceil(max_window / 16)))
    xticks = np.arange(0, max_window, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks], fontsize=8)

    cbar = fig.colorbar(image, ax=ax, fraction=0.024, pad=0.015)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlim(-0.5, max_window - 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dense endpoint window heatmaps from per-window NLL outputs")
    parser.add_argument("--root", required=True, help="Root directory containing sample folders")
    parser.add_argument("--prefix", default="gpt2_medium", help="Result file prefix")
    parser.add_argument("--metric", choices=["max_nll", "avg_nll", "max_minus_avg"], default="max_nll")
    parser.add_argument("--transform", choices=["raw", "threshold_delta", "threshold_ratio"], default="raw")
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--cmap", default="magma")
    parser.add_argument("--clip_percentile", type=float, default=99.0, help="Clip color scale symmetrically by percentile; use 0 to disable")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    samples = find_samples(root, args.prefix)
    if not samples:
        raise RuntimeError(f"No {args.prefix}_window_max_nll_points.json files found under {root}")

    labels, matrix, metadata = build_matrix(samples, args.metric, args.transform)
    clip_percentile = None if args.clip_percentile <= 0 else args.clip_percentile
    title = args.title or f"{root.name}: {args.metric} ({args.transform})"

    write_csv(Path(args.output_csv), labels, matrix)
    write_json(Path(args.output_json), labels, matrix, metadata, args.metric, args.transform)
    plot_heatmap(Path(args.output_png), labels, matrix, title, args.cmap, clip_percentile)

    print(f"[OK] samples={len(labels)} max_windows={matrix.shape[1]}")
    print(f"[OK] png={Path(args.output_png).resolve()}")
    print(f"[OK] csv={Path(args.output_csv).resolve()}")
    print(f"[OK] json={Path(args.output_json).resolve()}")


if __name__ == "__main__":
    main()
