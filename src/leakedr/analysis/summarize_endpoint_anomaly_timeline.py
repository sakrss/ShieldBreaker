#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


def extract_process_name(process_path: str) -> str:
    if not process_path:
        return ""
    process_path = process_path.replace("/", "\\")
    return process_path.split("\\")[-1].strip().lower()


def load_report(report_path: Path) -> Dict:
    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_process_events(raw_log_path: Path, process_name: str) -> List[Dict]:
    events: List[Dict] = []
    with raw_log_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            obj = json.loads(line)
            if obj.get("Plugin") == "sysret":
                continue

            proc = extract_process_name(obj.get("ProcessName", ""))
            if proc != process_name:
                continue

            events.append(
                {
                    "raw_line_no": line_no,
                    "timestamp": float(obj.get("TimeStamp", 0.0)),
                    "method": obj.get("Method", ""),
                    "pid": obj.get("PID"),
                    "tid": obj.get("TID"),
                    "event_uid": obj.get("EventUID"),
                }
            )
    return events


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


def methods_summary(events: List[Dict], top_k: int) -> List[Tuple[str, int]]:
    return Counter(event["method"] for event in events).most_common(top_k)


def nearest_delta(process_events: List[Dict], ts: float) -> float | None:
    if not process_events:
        return None

    best = None
    for event in process_events:
        delta = abs(float(event["timestamp"]) - ts)
        if best is None or delta < best:
            best = delta
    return best


def select_window(events: List[Dict], start_ts: float, end_ts: float) -> List[Dict]:
    return [event for event in events if start_ts <= float(event["timestamp"]) <= end_ts]


def summarize_clusters(
    clusters: List[List[Dict]],
    powershell_events: List[Dict],
    pre_window_sec: float,
    post_window_sec: float,
    top_methods: int,
) -> List[Dict]:
    if powershell_events:
        ps_start = float(powershell_events[0]["timestamp"])
        ps_end = float(powershell_events[-1]["timestamp"])
    else:
        ps_start = None
        ps_end = None

    summaries: List[Dict] = []

    for idx, cluster in enumerate(clusters, 1):
        start_ts = float(cluster[0]["timestamp"])
        end_ts = float(cluster[-1]["timestamp"])
        window_start = start_ts - pre_window_sec
        window_end = end_ts + post_window_sec

        nearby_ps = select_window(powershell_events, window_start, window_end)
        overlap_ps = select_window(powershell_events, start_ts, end_ts)
        pre_ps = select_window(powershell_events, start_ts - pre_window_sec, start_ts)

        nlls = [float(item["nll"]) for item in cluster]
        top_items = sorted(cluster, key=lambda x: (-float(x["nll"]), float(x["timestamp"])))[:5]

        summary = {
            "cluster_id": idx,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_sec": end_ts - start_ts,
            "anomaly_count": len(cluster),
            "max_nll": max(nlls),
            "avg_nll": sum(nlls) / len(nlls),
            "top_endpoint_methods": methods_summary(cluster, top_methods),
            "within_powershell_window": bool(
                ps_start is not None and ps_end is not None and start_ts <= ps_end and end_ts >= ps_start
            ),
            "seconds_from_powershell_start": None if ps_start is None else start_ts - ps_start,
            "seconds_from_powershell_end": None if ps_end is None else start_ts - ps_end,
            "nearest_powershell_event_delta_sec": nearest_delta(powershell_events, start_ts),
            "powershell_events_in_cluster": len(overlap_ps),
            "powershell_events_in_pre_window": len(pre_ps),
            "powershell_events_in_window": len(nearby_ps),
            "top_powershell_methods_in_pre_window": methods_summary(pre_ps, top_methods),
            "top_powershell_methods_in_window": methods_summary(nearby_ps, top_methods),
            "top_anomalies": top_items,
        }
        summaries.append(summary)

    return summaries


def write_markdown(
    output_md_path: Path,
    report: Dict,
    cluster_summaries: List[Dict],
    powershell_events: List[Dict],
) -> None:
    ps_start = float(powershell_events[0]["timestamp"]) if powershell_events else None
    ps_end = float(powershell_events[-1]["timestamp"]) if powershell_events else None
    detection = report["detection_summary"]
    threshold_stats = report["threshold_stats"]

    lines: List[str] = []
    lines.append("# Elastic Endpoint LSTM Anomaly Timeline")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Endpoint events: {detection['endpoint_event_count']}")
    lines.append(f"- Detected anomalies: {detection['anomaly_count']}")
    lines.append(f"- Token-level threshold ({threshold_stats['percentile']}th percentile): {threshold_stats['threshold']:.6f}")
    if ps_start is not None and ps_end is not None:
        lines.append(f"- PowerShell active window: {ps_start:.6f} -> {ps_end:.6f} ({ps_end - ps_start:.3f}s)")
    lines.append("")
    lines.append("## Clusters")
    lines.append("")

    for cluster in cluster_summaries:
        lines.append(f"### Cluster {cluster['cluster_id']}")
        lines.append("")
        lines.append(f"- Time: {cluster['start_ts']:.6f} -> {cluster['end_ts']:.6f} ({cluster['duration_sec']:.3f}s)")
        lines.append(f"- Anomalies: {cluster['anomaly_count']}")
        lines.append(f"- Max NLL: {cluster['max_nll']:.4f}")
        lines.append(f"- Avg NLL: {cluster['avg_nll']:.4f}")
        lines.append(f"- Within PowerShell window: {cluster['within_powershell_window']}")
        if cluster["seconds_from_powershell_start"] is not None:
            lines.append(f"- Offset from PowerShell start: {cluster['seconds_from_powershell_start']:.3f}s")
        if cluster["seconds_from_powershell_end"] is not None:
            lines.append(f"- Offset from PowerShell end: {cluster['seconds_from_powershell_end']:.3f}s")
        if cluster["nearest_powershell_event_delta_sec"] is not None:
            lines.append(f"- Nearest PowerShell event delta: {cluster['nearest_powershell_event_delta_sec']:.3f}s")
        lines.append(f"- PowerShell events in cluster: {cluster['powershell_events_in_cluster']}")
        lines.append(f"- PowerShell events in pre-window: {cluster['powershell_events_in_pre_window']}")
        lines.append(f"- Top endpoint methods: {', '.join(f'{name} x{count}' for name, count in cluster['top_endpoint_methods'])}")
        if cluster["top_powershell_methods_in_pre_window"]:
            lines.append(
                f"- Top PowerShell methods in pre-window: {', '.join(f'{name} x{count}' for name, count in cluster['top_powershell_methods_in_pre_window'])}"
            )
        if cluster["top_powershell_methods_in_window"]:
            lines.append(
                f"- Top PowerShell methods in window: {', '.join(f'{name} x{count}' for name, count in cluster['top_powershell_methods_in_window'])}"
            )
        lines.append("- Top anomalies:")
        for anomaly in cluster["top_anomalies"]:
            lines.append(
                f"  - line {anomaly['raw_line_no']}, ts={anomaly['timestamp']:.6f}, method={anomaly['method']}, nll={anomaly['nll']:.4f}"
            )
        lines.append("")

    with output_md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Summarize Elastic Endpoint anomaly report into timeline clusters")
    parser.add_argument("--report_path", required=True, help="Full anomaly report JSON path")
    parser.add_argument("--raw_log_path", required=True, help="Raw stimulated DRAKVUF JSONL path")
    parser.add_argument("--output_json", required=True, help="Timeline summary JSON path")
    parser.add_argument("--output_md", required=True, help="Timeline summary Markdown path")
    parser.add_argument("--cluster_gap_sec", type=float, default=3.0, help="Max time gap within one anomaly cluster")
    parser.add_argument("--pre_window_sec", type=float, default=5.0, help="PowerShell context window before each cluster")
    parser.add_argument("--post_window_sec", type=float, default=2.0, help="PowerShell context window after each cluster")
    parser.add_argument("--top_methods", type=int, default=6, help="Top methods to keep per cluster")
    args = parser.parse_args()

    report_path = Path(args.report_path).resolve()
    raw_log_path = Path(args.raw_log_path).resolve()
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    report = load_report(report_path)
    anomalies = report["top_anomalies"]
    clusters = cluster_anomalies(anomalies, args.cluster_gap_sec)
    powershell_events = load_process_events(raw_log_path, "powershell.exe")
    cluster_summaries = summarize_clusters(
        clusters=clusters,
        powershell_events=powershell_events,
        pre_window_sec=args.pre_window_sec,
        post_window_sec=args.post_window_sec,
        top_methods=args.top_methods,
    )

    timeline = {
        "report_path": str(report_path),
        "raw_log_path": str(raw_log_path),
        "cluster_gap_sec": args.cluster_gap_sec,
        "pre_window_sec": args.pre_window_sec,
        "post_window_sec": args.post_window_sec,
        "cluster_count": len(cluster_summaries),
        "powershell_event_count": len(powershell_events),
        "clusters": cluster_summaries,
    }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)

    write_markdown(output_md, report, cluster_summaries, powershell_events)

    print(f"[OK] cluster_count={len(cluster_summaries)}")
    print(f"[OK] output_json={output_json}")
    print(f"[OK] output_md={output_md}")


if __name__ == "__main__":
    main()
