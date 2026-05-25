#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def load_report(report_path: Path) -> Dict:
    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cluster_anomalies(anomalies: List[Dict], max_gap_sec: float) -> List[List[Dict]]:
    anomalies = sorted(anomalies, key=lambda x: x["timestamp"])
    clusters: List[List[Dict]] = []
    current: List[Dict] = []
    prev_ts = None

    for anomaly in anomalies:
        ts = float(anomaly["timestamp"])
        if prev_ts is None or (ts - prev_ts) <= max_gap_sec:
            current.append(anomaly)
        else:
            clusters.append(current)
            current = [anomaly]
        prev_ts = ts

    if current:
        clusters.append(current)

    return clusters


def build_cluster_summary(clusters: List[List[Dict]], top_methods: int) -> List[Dict]:
    summaries: List[Dict] = []

    for idx, cluster in enumerate(clusters, 1):
        nlls = [float(item["nll"]) for item in cluster]
        methods = Counter(item["method"] for item in cluster).most_common(top_methods)
        summaries.append(
            {
                "cluster_id": idx,
                "start_ts": float(cluster[0]["timestamp"]),
                "end_ts": float(cluster[-1]["timestamp"]),
                "duration_sec": float(cluster[-1]["timestamp"]) - float(cluster[0]["timestamp"]),
                "anomaly_count": len(cluster),
                "max_nll": max(nlls),
                "avg_nll": sum(nlls) / len(nlls),
                "top_methods": methods,
            }
        )

    return summaries


def build_anomaly_points(clusters: List[List[Dict]]) -> List[Dict]:
    points: List[Dict] = []

    for cluster_id, cluster in enumerate(clusters, 1):
        for anomaly in cluster:
            points.append(
                {
                    "cluster_id": cluster_id,
                    "timestamp": anomaly["timestamp"],
                    "raw_line_no": anomaly["raw_line_no"],
                    "pid": anomaly.get("pid"),
                    "tid": anomaly.get("tid"),
                    "event_uid": anomaly.get("event_uid"),
                    "method": anomaly["method"],
                    "nll": anomaly["nll"],
                    "context_before": anomaly.get("context_before", []),
                    "context_after": anomaly.get("context_after", []),
                }
            )

    points.sort(key=lambda x: (-float(x["nll"]), float(x["timestamp"])))
    return points


def write_markdown(
    output_md_path: Path,
    endpoint_event_count: int,
    threshold: float,
    clusters: List[Dict],
    anomaly_points: List[Dict],
    top_points: int,
) -> None:
    lines: List[str] = []
    lines.append("# Endpoint Anomaly Points")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Endpoint events: {endpoint_event_count}")
    lines.append(f"- Total anomaly points: {len(anomaly_points)}")
    lines.append(f"- Total anomaly clusters: {len(clusters)}")
    lines.append(f"- Token-level threshold: {threshold:.6f}")
    lines.append("")
    lines.append("## Clusters")
    lines.append("")

    for cluster in clusters:
        lines.append(f"### Cluster {cluster['cluster_id']}")
        lines.append("")
        lines.append(f"- Time: {cluster['start_ts']:.6f} -> {cluster['end_ts']:.6f} ({cluster['duration_sec']:.3f}s)")
        lines.append(f"- Anomaly count: {cluster['anomaly_count']}")
        lines.append(f"- Max NLL: {cluster['max_nll']:.4f}")
        lines.append(f"- Avg NLL: {cluster['avg_nll']:.4f}")
        lines.append(f"- Top methods: {', '.join(f'{name} x{count}' for name, count in cluster['top_methods'])}")
        lines.append("")

    lines.append("## Top Points")
    lines.append("")

    for idx, point in enumerate(anomaly_points[:top_points], 1):
        before = " -> ".join(point["context_before"][-3:]) if point["context_before"] else ""
        after = " -> ".join(point["context_after"][:3]) if point["context_after"] else ""
        lines.append(f"### Point {idx}")
        lines.append("")
        lines.append(f"- Cluster: {point['cluster_id']}")
        lines.append(f"- Time: {float(point['timestamp']):.6f}")
        lines.append(f"- Raw line: {point['raw_line_no']}")
        lines.append(f"- Method: {point['method']}")
        lines.append(f"- NLL: {float(point['nll']):.4f}")
        if before:
            lines.append(f"- Context before: {before}")
        if after:
            lines.append(f"- Context after: {after}")
        lines.append("")

    with output_md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Export pure endpoint anomaly points without attack-process context")
    parser.add_argument("--report_path", required=True, help="Full endpoint anomaly report JSON path")
    parser.add_argument("--output_json", required=True, help="Output compact JSON path")
    parser.add_argument("--output_md", required=True, help="Output markdown path")
    parser.add_argument("--cluster_gap_sec", type=float, default=3.0, help="Max time gap inside one cluster")
    parser.add_argument("--top_methods", type=int, default=6, help="Top methods per cluster")
    parser.add_argument("--top_points", type=int, default=50, help="How many highest-scoring points to include in markdown")
    args = parser.parse_args()

    report_path = Path(args.report_path).resolve()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    report = load_report(report_path)
    anomalies = report["top_anomalies"]
    clusters = cluster_anomalies(anomalies, args.cluster_gap_sec)
    cluster_summaries = build_cluster_summary(clusters, args.top_methods)
    anomaly_points = build_anomaly_points(clusters)

    compact = {
        "source_report": str(report_path),
        "endpoint_event_count": report["detection_summary"]["endpoint_event_count"],
        "anomaly_count": len(anomaly_points),
        "cluster_count": len(cluster_summaries),
        "threshold": report["threshold_stats"]["threshold"],
        "cluster_gap_sec": args.cluster_gap_sec,
        "clusters": cluster_summaries,
        "anomaly_points": anomaly_points,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)

    write_markdown(
        output_md_path=output_md,
        endpoint_event_count=compact["endpoint_event_count"],
        threshold=compact["threshold"],
        clusters=cluster_summaries,
        anomaly_points=anomaly_points,
        top_points=args.top_points,
    )

    print(f"[OK] anomaly_count={compact['anomaly_count']}")
    print(f"[OK] cluster_count={compact['cluster_count']}")
    print(f"[OK] output_json={output_json}")
    print(f"[OK] output_md={output_md}")


if __name__ == "__main__":
    main()
