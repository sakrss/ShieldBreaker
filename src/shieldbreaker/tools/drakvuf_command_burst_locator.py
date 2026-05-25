
#!/usr/bin/env python3
"""
Locate likely command-execution bursts from DRAKVUF JSON/JSONL/ZIP traces
without knowing the exact command text in advance.

Design goals
------------
1) Work directly on standard DRAKVUF JSONL traces (or zipped JSONL).
2) Use process-local baseline rather than requiring known command labels.
3) Rank candidate "execution windows" inside interpreter-like processes.
4) Be robust to renamed PowerShell binaries by looking at behavior markers.

Example
-------
python drakvuf_command_burst_locator.py /path/to/drakvuf.json.zip \
    --top-k 5 \
    --candidate-limit 5 \
    --window-ms 500 \
    --idle-seed-sec 5 \
    --json-out result.full.json \
    --human-json-out result.human.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


# -----------------------------
# Domain heuristics
# -----------------------------

WAIT_METHODS = {
    "ntdelayexecution",
    "ntwaitforsingleobject",
    "ntwaitformultipleobjects",
    "ntwaitforworkviaworkerfactory",
    "ntwaitforalertbythreadid",
    "ntwaitforkeyedevent",
    "ntremoveiocompletionex",
}

SYNC_METHODS = {
    "ntsetevent",
    "ntclearevent",
    "ntreleasemutant",
    "ntreleasesemaphore",
    "ntpulseevent",
    "ntsignalandsingleobject",
    "ntsetinformationprocess",
    "ntyieldexecution",
}

FILE_METHODS = {
    "ntopenfile",
    "ntcreatefile",
    "ntreadfile",
    "ntwritefile",
    "ntqueryinformationfile",
    "ntsetinformationfile",
    "ntqueryattributesfile",
    "ntqueryfullattributesfile",
    "ntquerydirectoryfile",
    "ntquerydirectoryfileex",
    "ntdeviceiocontrolfile",
    "ntfscontrolfile",
    "ntlockfile",
}

THREAD_METHODS = {
    "ntcreatethreadex",
    "ntresumethread",
    "ntterminatethread",
    "ntqueryinformationthread",
    "ntopenthread",
    "ntopenthreadtoken",
    "ntopenthreadtokenex",
    "ntsetinformationthread",
}

PROCESS_METHODS = {
    "ntopenprocess",
    "ntqueryinformationprocess",
    "ntsetinformationprocess",
    "ntcreateuserprocess",
    "ntcreateprocessex",
    "ntterminateprocess",
}

MEMORY_METHODS = {
    "ntallocatevirtualmemory",
    "ntallocatevirtualmemoryex",
    "ntfreevirtualmemory",
    "ntprotectvirtualmemory",
    "ntqueryvirtualmemory",
    "ntmapviewofsection",
    "ntunmapviewofsection",
}

REGISTRY_METHODS = {
    "ntopenkey",
    "ntopenkeyex",
    "ntcreatekey",
    "ntqueryvaluekey",
    "ntenumeratekey",
    "ntenumeratevaluekey",
    "ntsetvaluekey",
    "ntdeletevaluekey",
    "ntdeletekey",
}

NETWORKISH_METHODS = {
    "ntdeviceiocontrolfile",
    "ntalpcsendwaitreceiveport",
}

POWERSHELL_MARKERS = (
    "microsoft.powershell.",
    "psreadline",
    "consolehost_history.txt",
    "system.management.automation",
    "powershell",
    "clr.dll",
)

INTERPRETER_NAME_HINTS = (
    "powershell",
    "pwsh",
    "cmd.exe",
    "wscript",
    "cscript",
    "mshta",
    "python",
    "bash",
    "sh",
)

PATH_FIELD_CANDIDATES = (
    "ObjectAttributes",
    "FileHandle",
    "KeyHandle",
    "FromModule",
    "FromParentModule",
    "ProcessName",
)

DEFAULT_REGEX = re.compile(r"(?:[a-zA-Z]:\\\\|\\\\\?\?\\\\|\\\\Device\\\\HarddiskVolume\d+\\\\)[^\"'\s,;]+")
MODULE_FILE_REGEX = re.compile(r"\.(?:psd1|psm1|dll|exe)\b", re.IGNORECASE)
BASENAME_REGEX = re.compile(r"[^\\/]+$")


# -----------------------------
# Data classes
# -----------------------------

@dataclass
class Event:
    idx: int
    ts: float
    plugin: str
    pid: int
    ppid: int
    tid: int
    process_name: str
    process_base: str
    method: str
    object_strings: Tuple[str, ...]
    raw: Dict[str, object]


@dataclass
class WindowFeatures:
    pid: int
    process_name: str
    process_base: str
    start_ts: float
    end_ts: float
    line_start: int
    line_end: int
    event_count: int
    unique_tids: int
    new_tids: int
    wait_ratio: float
    sync_ratio: float
    exec_ratio: float
    file_ratio: float
    dir_enum_count: int
    file_read_count: int
    registry_count: int
    process_thread_count: int
    memory_count: int
    module_probe_count: int
    powershell_marker_count: int
    new_object_count: int
    object_touch_count: int
    score: float = 0.0
    burst_label: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class Burst:
    pid: int
    process_name: str
    process_base: str
    start_ts: float
    end_ts: float
    line_start: int
    line_end: int
    window_count: int
    peak_score: float
    avg_score: float
    summary_label: str
    representative_methods: List[str]
    representative_objects: List[str]
    features: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# -----------------------------
# Parsing utilities
# -----------------------------

def normalize_process_name(name: object) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().replace("/", "\\")


def basename_from_path(path: str) -> str:
    if not path:
        return ""
    m = BASENAME_REGEX.search(path)
    return (m.group(0) if m else path).lower()


def normalize_method(method: object) -> str:
    return str(method or "").strip().lower()


def iter_json_lines(input_path: str) -> Iterator[Tuple[int, Dict[str, object]]]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            json_members = [
                name for name in zf.namelist()
                if not name.endswith("/")
                and "__MACOSX" not in name
                and (name.lower().endswith(".json") or name.lower().endswith(".jsonl"))
            ]
            if not json_members:
                raise ValueError("ZIP does not contain a .json or .jsonl member.")
            member = json_members[0]
            with zf.open(member, "r") as fh:
                for i, raw_line in enumerate(fh, start=1):
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        yield i, json.loads(line)
                    except json.JSONDecodeError:
                        continue
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield i, json.loads(line)
                except json.JSONDecodeError:
                    continue


def extract_object_strings(record: Dict[str, object]) -> Tuple[str, ...]:
    vals: List[str] = []
    for key in PATH_FIELD_CANDIDATES:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            vals.append(value.strip())

    # Also mine embedded paths/modules from all short-ish strings.
    for value in record.values():
        if isinstance(value, str) and 3 <= len(value) <= 600:
            for m in DEFAULT_REGEX.finditer(value):
                vals.append(m.group(0))
            if MODULE_FILE_REGEX.search(value):
                vals.append(value.strip())

    # Deduplicate while preserving order.
    seen = set()
    out = []
    for v in vals:
        nv = v.strip()
        if not nv:
            continue
        key = nv.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(nv)
    return tuple(out)


def build_events(input_path: str) -> List[Event]:
    events: List[Event] = []
    for idx, rec in iter_json_lines(input_path):
        plugin = str(rec.get("Plugin", "")).strip().lower()
        if plugin not in {"syscall", "sysret"}:
            continue
        try:
            ts = float(rec.get("TimeStamp"))
            pid = int(rec.get("PID", -1))
            ppid = int(rec.get("PPID", -1))
            tid = int(rec.get("TID", -1))
        except (TypeError, ValueError):
            continue

        proc = normalize_process_name(rec.get("ProcessName"))
        method = normalize_method(rec.get("Method"))
        events.append(
            Event(
                idx=idx,
                ts=ts,
                plugin=plugin,
                pid=pid,
                ppid=ppid,
                tid=tid,
                process_name=proc,
                process_base=basename_from_path(proc),
                method=method,
                object_strings=extract_object_strings(rec),
                raw=rec,
            )
        )
    events.sort(key=lambda e: (e.ts, e.idx))
    return events


# -----------------------------
# Candidate process discovery
# -----------------------------

def marker_count_for_event(e: Event) -> int:
    blob = " | ".join([e.process_name, e.method, *e.object_strings]).lower()
    return sum(1 for marker in POWERSHELL_MARKERS if marker in blob)


def discover_candidate_processes(
    events: Sequence[Event],
    candidate_limit: int = 5,
    explicit_regex: Optional[str] = None,
) -> List[int]:
    if explicit_regex:
        rx = re.compile(explicit_regex, re.IGNORECASE)
        pids = sorted({e.pid for e in events if rx.search(e.process_name)})
        return pids[:candidate_limit]

    pid_stats: Dict[int, Dict[str, object]] = defaultdict(lambda: {
        "count": 0,
        "name_hint": 0,
        "marker_hits": 0,
        "module_hits": 0,
        "first_ts": None,
        "process_name": "",
    })

    for e in events:
        s = pid_stats[e.pid]
        s["count"] += 1
        s["process_name"] = e.process_name
        blob = " | ".join([e.process_name, *e.object_strings]).lower()
        if any(hint in e.process_base for hint in INTERPRETER_NAME_HINTS):
            s["name_hint"] += 1
        mark = marker_count_for_event(e)
        s["marker_hits"] += mark
        if "microsoft.powershell." in blob or "psreadline" in blob:
            s["module_hits"] += 1
        if s["first_ts"] is None or e.ts < s["first_ts"]:
            s["first_ts"] = e.ts

    ranked: List[Tuple[float, int]] = []
    for pid, s in pid_stats.items():
        score = (
            4.0 * min(1.0, s["name_hint"])
            + 3.0 * min(10.0, s["marker_hits"]) / 10.0
            + 3.0 * min(10.0, s["module_hits"]) / 10.0
            + min(2.0, math.log1p(s["count"]) / 4.0)
        )
        ranked.append((score, pid))

    ranked.sort(reverse=True)
    return [pid for _, pid in ranked[:candidate_limit]]


# -----------------------------
# Feature extraction
# -----------------------------

def robust_median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return statistics.median(values)


def robust_mad(values: Sequence[float], center: Optional[float] = None) -> float:
    if not values:
        return 0.0
    c = robust_median(values) if center is None else center
    deviations = [abs(v - c) for v in values]
    return statistics.median(deviations)


def robust_z(x: float, baseline_values: Sequence[float], fallback_scale: float = 1.0) -> float:
    if not baseline_values:
        return 0.0
    med = robust_median(baseline_values)
    mad = robust_mad(baseline_values, med)
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = fallback_scale
    return (x - med) / scale


def classify_burst(method_counter: Counter, object_samples: Sequence[str]) -> str:
    methods = {m for m, c in method_counter.items() if c > 0}
    joined_objects = " | ".join(s.lower() for s in object_samples)

    if "ntquerydirectoryfileex" in methods or "ntquerydirectoryfile" in methods:
        return "directory_enumeration_burst"
    if "ntreadfile" in methods and ("system32" in joined_objects or "\\device\\" in joined_objects):
        return "file_reading_burst"
    if any(m in methods for m in REGISTRY_METHODS):
        return "registry_access_burst"
    if any(m in methods for m in PROCESS_METHODS | THREAD_METHODS):
        return "process_thread_burst"
    if "microsoft.powershell.utility" in joined_objects or "psreadline" in joined_objects:
        return "powershell_module_resolution_burst"
    return "generic_interpreter_activity_burst"


def build_windows_for_pid(
    events_for_pid: Sequence[Event],
    window_ms: int = 500,
    idle_seed_sec: float = 5.0,
) -> List[WindowFeatures]:
    if not events_for_pid:
        return []

    window_sec = max(window_ms / 1000.0, 0.05)
    t0 = events_for_pid[0].ts
    t1 = events_for_pid[-1].ts
    num_bins = max(1, int(math.ceil((t1 - t0) / window_sec)) + 1)

    bins: List[List[Event]] = [[] for _ in range(num_bins)]
    for e in events_for_pid:
        idx = int((e.ts - t0) / window_sec)
        idx = min(max(idx, 0), num_bins - 1)
        bins[idx].append(e)

    windows: List[WindowFeatures] = []
    seen_objects: set[str] = set()
    seen_tids: set[int] = set()
    process_name = events_for_pid[0].process_name
    process_base = events_for_pid[0].process_base
    pid = events_for_pid[0].pid

    for i, bucket in enumerate(bins):
        start_ts = t0 + i * window_sec
        end_ts = start_ts + window_sec
        if bucket:
            line_start = bucket[0].idx
            line_end = bucket[-1].idx
        else:
            line_start = line_end = -1

        method_counter = Counter(e.method for e in bucket)
        total = len(bucket)

        def count(method_set: set[str]) -> int:
            return sum(c for m, c in method_counter.items() if m in method_set)

        file_count = count(FILE_METHODS)
        dir_enum_count = method_counter.get("ntquerydirectoryfileex", 0) + method_counter.get("ntquerydirectoryfile", 0)
        file_read_count = method_counter.get("ntreadfile", 0)
        reg_count = count(REGISTRY_METHODS)
        proc_thread_count = count(PROCESS_METHODS | THREAD_METHODS)
        mem_count = count(MEMORY_METHODS)
        wait_count = count(WAIT_METHODS)
        sync_count = count(SYNC_METHODS)

        object_touches = []
        module_probe_count = 0
        powershell_marker_count = 0
        new_object_count = 0
        new_tids = 0

        bucket_tids = {e.tid for e in bucket}
        for tid in bucket_tids:
            if tid not in seen_tids:
                new_tids += 1

        for e in bucket:
            powershell_marker_count += marker_count_for_event(e)
            for obj in e.object_strings:
                object_touches.append(obj)
                lo = obj.lower()
                if lo not in seen_objects:
                    new_object_count += 1
                if any(marker in lo for marker in POWERSHELL_MARKERS):
                    module_probe_count += 1
                elif MODULE_FILE_REGEX.search(lo):
                    module_probe_count += 1

        # Update seen state after counting novelty for this window.
        seen_tids.update(bucket_tids)
        for obj in object_touches:
            seen_objects.add(obj.lower())

        exec_count = file_count + reg_count + proc_thread_count + mem_count

        windows.append(
            WindowFeatures(
                pid=pid,
                process_name=process_name,
                process_base=process_base,
                start_ts=start_ts,
                end_ts=end_ts,
                line_start=line_start,
                line_end=line_end,
                event_count=total,
                unique_tids=len(bucket_tids),
                new_tids=new_tids,
                wait_ratio=(wait_count / total) if total else 0.0,
                sync_ratio=(sync_count / total) if total else 0.0,
                exec_ratio=(exec_count / total) if total else 0.0,
                file_ratio=(file_count / total) if total else 0.0,
                dir_enum_count=dir_enum_count,
                file_read_count=file_read_count,
                registry_count=reg_count,
                process_thread_count=proc_thread_count,
                memory_count=mem_count,
                module_probe_count=module_probe_count,
                powershell_marker_count=powershell_marker_count,
                new_object_count=new_object_count,
                object_touch_count=len(object_touches),
                score=0.0,
                burst_label="",
            )
        )

    # Build a process-local baseline from the earliest windows, then back off to all windows if needed.
    seed_window_count = max(3, int(math.ceil(idle_seed_sec / window_sec)))
    seed = windows[:min(seed_window_count, len(windows))]
    if sum(w.event_count for w in seed) == 0:
        seed = windows

    def values(attr: str) -> List[float]:
        vals = [float(getattr(w, attr)) for w in seed]
        if all(v == 0 for v in vals):
            vals = [float(getattr(w, attr)) for w in windows]
        return vals

    event_vals = values("event_count")
    exec_vals = values("exec_ratio")
    file_vals = values("file_ratio")
    wait_vals = values("wait_ratio")
    obj_vals = values("new_object_count")
    module_vals = values("module_probe_count")
    thread_vals = values("new_tids")
    dir_vals = values("dir_enum_count")
    read_vals = values("file_read_count")
    mem_vals = values("memory_count")
    reg_vals = values("registry_count")
    proc_thread_vals = values("process_thread_count")
    ps_vals = values("powershell_marker_count")

    for w in windows:
        # Weighted process-local burst score.
        score = 0.0
        score += 2.0 * robust_z(w.event_count, event_vals, 1.0)
        score += 2.2 * robust_z(w.exec_ratio, exec_vals, 0.25)
        score += 1.2 * robust_z(w.file_ratio, file_vals, 0.25)
        score += 1.6 * robust_z(w.new_object_count, obj_vals, 1.0)
        score += 1.2 * robust_z(w.module_probe_count, module_vals, 1.0)
        score += 1.0 * robust_z(w.new_tids, thread_vals, 1.0)
        score += 1.3 * robust_z(w.dir_enum_count, dir_vals, 1.0)
        score += 1.0 * robust_z(w.file_read_count, read_vals, 1.0)
        score += 0.6 * robust_z(w.memory_count, mem_vals, 1.0)
        score += 0.6 * robust_z(w.registry_count, reg_vals, 1.0)
        score += 0.8 * robust_z(w.process_thread_count, proc_thread_vals, 1.0)
        score += 1.0 * robust_z(w.powershell_marker_count, ps_vals, 1.0)
        score -= 1.8 * max(0.0, robust_z(w.wait_ratio, wait_vals, 0.25))
        w.score = score
        w.burst_label = ""

    return windows


def merge_high_score_windows(
    windows: Sequence[WindowFeatures],
    events_for_pid: Sequence[Event],
    score_threshold: Optional[float] = None,
    merge_gap_windows: int = 1,
    min_windows: int = 1,
) -> List[Burst]:
    if not windows:
        return []

    scores = [w.score for w in windows]
    med = robust_median(scores)
    mad = robust_mad(scores, med)
    scale = 1.4826 * mad
    if scale < 1e-9:
        scale = 1.0

    if score_threshold is None:
        # Adaptive threshold: median + max(2.5*MAD, 1.5)
        score_threshold = med + max(2.5 * scale, 1.5)

    hot_indices = [i for i, w in enumerate(windows) if w.score >= score_threshold and w.event_count > 0]
    if not hot_indices:
        # Fallback: take the single hottest non-empty window.
        non_empty = [i for i, w in enumerate(windows) if w.event_count > 0]
        if not non_empty:
            return []
        hottest = max(non_empty, key=lambda i: windows[i].score)
        hot_indices = [hottest]

    groups: List[List[int]] = []
    current: List[int] = []
    last_i = None
    for i in hot_indices:
        if last_i is None or i - last_i <= merge_gap_windows + 1:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
        last_i = i
    if current:
        groups.append(current)

    bursts: List[Burst] = []
    for grp in groups:
        if len(grp) < min_windows:
            continue
        ws = [windows[i] for i in grp]
        start_ts = ws[0].start_ts
        end_ts = ws[-1].end_ts
        line_start = min(w.line_start for w in ws if w.line_start >= 0)
        line_end = max(w.line_end for w in ws if w.line_end >= 0)

        method_counter: Counter = Counter()
        object_counter: Counter = Counter()
        for e in events_for_pid:
            if start_ts <= e.ts <= end_ts:
                method_counter[e.method] += 1
                for obj in e.object_strings:
                    object_counter[obj] += 1

        rep_methods = [m for m, _ in method_counter.most_common(8)]
        rep_objects = [o for o, _ in object_counter.most_common(8)]
        label = classify_burst(method_counter, rep_objects)

        bursts.append(
            Burst(
                pid=ws[0].pid,
                process_name=ws[0].process_name,
                process_base=ws[0].process_base,
                start_ts=start_ts,
                end_ts=end_ts,
                line_start=line_start,
                line_end=line_end,
                window_count=len(ws),
                peak_score=max(w.score for w in ws),
                avg_score=sum(w.score for w in ws) / len(ws),
                summary_label=label,
                representative_methods=rep_methods,
                representative_objects=rep_objects,
                features={
                    "max_event_count": max(w.event_count for w in ws),
                    "max_exec_ratio": max(w.exec_ratio for w in ws),
                    "max_wait_ratio": max(w.wait_ratio for w in ws),
                    "sum_dir_enum_count": float(sum(w.dir_enum_count for w in ws)),
                    "sum_file_read_count": float(sum(w.file_read_count for w in ws)),
                    "sum_module_probe_count": float(sum(w.module_probe_count for w in ws)),
                    "sum_new_object_count": float(sum(w.new_object_count for w in ws)),
                    "sum_new_tids": float(sum(w.new_tids for w in ws)),
                },
            )
        )

    bursts.sort(key=lambda b: (b.peak_score, b.avg_score, -(b.start_ts)), reverse=True)
    return bursts


# -----------------------------
# Main analysis
# -----------------------------

def analyze(
    input_path: str,
    candidate_regex: Optional[str],
    candidate_limit: int,
    top_k: int,
    window_ms: int,
    idle_seed_sec: float,
) -> Dict[str, object]:
    events = build_events(input_path)
    if not events:
        raise ValueError("No valid syscall/sysret events found.")

    by_pid: Dict[int, List[Event]] = defaultdict(list)
    for e in events:
        by_pid[e.pid].append(e)

    candidate_pids = discover_candidate_processes(
        events,
        candidate_limit=candidate_limit,
        explicit_regex=candidate_regex,
    )

    analyses = []
    for pid in candidate_pids:
        pid_events = by_pid[pid]
        if len(pid_events) < 10:
            continue
        windows = build_windows_for_pid(
            pid_events,
            window_ms=window_ms,
            idle_seed_sec=idle_seed_sec,
        )
        bursts = merge_high_score_windows(windows, pid_events)
        analyses.append({
            "pid": pid,
            "process_name": pid_events[0].process_name,
            "process_base": pid_events[0].process_base,
            "event_count": len(pid_events),
            "windows": [w.to_dict() for w in windows],
            "bursts": [b.to_dict() for b in bursts[:top_k]],
        })

    analyses.sort(
        key=lambda x: (
            x["bursts"][0]["peak_score"] if x["bursts"] else float("-inf"),
            x["event_count"],
        ),
        reverse=True,
    )

    result = {
        "input_path": input_path,
        "total_events": len(events),
        "trace_start_ts": events[0].ts,
        "trace_end_ts": events[-1].ts,
        "candidate_pids": candidate_pids,
        "analyses": analyses,
    }
    return result


def shorten_object_for_humans(value: str, max_len: int = 120) -> str:
    if len(value) <= max_len:
        return value
    base = basename_from_path(value)
    if not base:
        return value[:max_len - 3] + "..."
    head_len = max_len - len(base) - 6
    if head_len < 12:
        return "..." + value[-(max_len - 3):]
    return value[:head_len] + "..." + base


def build_human_readable_result(
    result: Dict[str, object],
    top_k: int,
    min_peak_score: float,
) -> Dict[str, object]:
    trace_start_ts = float(result.get("trace_start_ts", 0.0))
    processes: List[Dict[str, object]] = []

    for proc in result.get("analyses", []):
        bursts = [
            b for b in proc.get("bursts", [])
            if float(b.get("peak_score", 0.0)) >= min_peak_score
        ][:top_k]

        if not bursts:
            continue

        human_bursts = []
        for rank, burst in enumerate(bursts, start=1):
            start_ts = float(burst["start_ts"])
            end_ts = float(burst["end_ts"])
            human_bursts.append({
                "rank": rank,
                "summary_label": burst["summary_label"],
                "score": {
                    "peak": round(float(burst["peak_score"]), 3),
                    "avg": round(float(burst["avg_score"]), 3),
                },
                "time": {
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "offset_from_trace_start_sec": round(start_ts - trace_start_ts, 3),
                    "duration_sec": round(end_ts - start_ts, 3),
                },
                "lines": {
                    "start": int(burst["line_start"]),
                    "end": int(burst["line_end"]),
                },
                "window_count": int(burst["window_count"]),
                "representative_methods": burst["representative_methods"][:6],
                "representative_objects": [
                    shorten_object_for_humans(obj)
                    for obj in burst["representative_objects"][:4]
                ],
                "key_features": burst["features"],
            })

        processes.append({
            "pid": proc["pid"],
            "process_name": proc["process_name"],
            "process_base": proc["process_base"],
            "event_count": proc["event_count"],
            "kept_bursts": len(human_bursts),
            "bursts": human_bursts,
        })

    processes.sort(
        key=lambda proc: proc["bursts"][0]["score"]["peak"] if proc["bursts"] else float("-inf"),
        reverse=True,
    )

    return {
        "input_path": result["input_path"],
        "total_events": result["total_events"],
        "trace_start_ts": result.get("trace_start_ts"),
        "trace_end_ts": result.get("trace_end_ts"),
        "candidate_pids": result["candidate_pids"],
        "focus": {
            "description": "Human-readable summary that keeps only high-score bursts.",
            "score_interpretation": "Higher score means the window looks more burst-like relative to that process's own baseline.",
            "min_peak_score": min_peak_score,
            "top_k_per_process": top_k,
            "low_score_windows_omitted": True,
        },
        "processes": processes,
    }


def print_human_summary(result: Dict[str, object], top_k: int) -> None:
    print(f"Input: {result['input_path']}")
    print(f"Total syscall/sysret events: {result['total_events']}")
    print(f"Candidate PIDs: {result['candidate_pids']}")
    print()

    analyses = result.get("analyses", [])
    if not analyses:
        print("No candidate process analysis available.")
        return

    for proc in analyses:
        print("=" * 100)
        print(f"PID {proc['pid']} | {proc['process_name']}")
        print(f"Events: {proc['event_count']}")
        bursts = proc.get("bursts", [])[:top_k]
        if not bursts:
            print("  No burst detected.")
            continue
        for i, b in enumerate(bursts, start=1):
            print(f"  [{i}] {b['summary_label']}")
            print(
                f"      ts={b['start_ts']:.6f} -> {b['end_ts']:.6f} | "
                f"lines={b['line_start']}-{b['line_end']} | "
                f"peak={b['peak_score']:.2f} avg={b['avg_score']:.2f} "
                f"windows={b['window_count']}"
            )
            print(f"      methods: {', '.join(b['representative_methods'][:6])}")
            objects = b["representative_objects"][:4]
            if objects:
                print(f"      objects: {', '.join(objects)}")
        print()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_path", help="Path to drakvuf.json, drakvuf.jsonl, or zip containing one of them")
    p.add_argument("--candidate-regex", default=None, help="Regex to force candidate process selection")
    p.add_argument("--candidate-limit", type=int, default=5, help="How many candidate PIDs to analyze")
    p.add_argument("--top-k", type=int, default=5, help="How many bursts per PID to keep")
    p.add_argument("--window-ms", type=int, default=500, help="Sliding bin/window size in milliseconds")
    p.add_argument("--idle-seed-sec", type=float, default=5.0, help="Early per-process time used as local baseline seed")
    p.add_argument("--json-out", default=None, help="Optional path to save full JSON output including all windows")
    p.add_argument("--human-json-out", default=None, help="Optional path to save a compact, human-readable JSON summary")
    p.add_argument("--min-peak-score", type=float, default=0.0, help="Only keep bursts at or above this peak score in the human-readable JSON")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = analyze(
        input_path=args.input_path,
        candidate_regex=args.candidate_regex,
        candidate_limit=args.candidate_limit,
        top_k=args.top_k,
        window_ms=args.window_ms,
        idle_seed_sec=args.idle_seed_sec,
    )
    print_human_summary(result, top_k=args.top_k)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
    if args.human_json_out:
        human_result = build_human_readable_result(
            result,
            top_k=args.top_k,
            min_peak_score=args.min_peak_score,
        )
        with open(args.human_json_out, "w", encoding="utf-8") as fh:
            json.dump(human_result, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
