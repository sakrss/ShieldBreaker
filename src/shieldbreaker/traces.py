"""Execution-preserving API traces (Eq. 2). Original call metadata is retained."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
from typing import Iterable


def process_name(path):
    return str(path or "").replace("\\", "/").rsplit("/", 1)[-1].strip().casefold()


def identifier(record, *names):
    for name in names:
        if name in record and record[name] is not None:
            return str(record[name])
    return None


@dataclass(frozen=True)
class Event:
    timestamp: float
    process: str
    process_id: str | None
    thread_id: str | None
    function: str
    event_id: str
    metadata: dict

    @classmethod
    def from_record(cls, record, index):
        timestamp = float(record["TimeStamp"])
        function = str(record.get("Method", "")).strip()
        if (
            not math.isfinite(timestamp)
            or not function
            or any(c.isspace() for c in function)
        ):
            raise ValueError(
                "Expected finite timestamp and one API function-name token"
            )
        return cls(
            timestamp,
            process_name(record.get("ProcessName", record.get("ExtractedProcessName"))),
            identifier(record, "PID", "ProcessId", "ProcessID"),
            identifier(record, "TID", "ThreadId", "ThreadID"),
            function,
            str(record.get("GlobalID", index)),
            dict(record),
        )


def read_records(path):
    records = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Expected JSON object")
                records.append(record)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return records


def events_from_records(records):
    events = []
    for index, record in enumerate(records, 1):
        if record.get("Plugin") == "sysret":
            continue  # Return records are not additional API-call tokens.
        try:
            events.append(Event.from_record(record, index))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Invalid API record {index}: {exc}") from exc
    # Stable sorting preserves recording order; ties are not causal order.
    return sorted(events, key=lambda event: event.timestamp)


def partition_records(records, edr_processes, stimulus_processes=(), stimulus_pids=()):
    """Separate X^E and X^M, including descendants with recorded PID/PPID links.

    Without PID/PPID evidence only explicitly named stimulus processes are selected.
    PID reuse cannot be resolved without process-lifetime metadata; callers should
    use one short controlled execution per input and explicit process identities.
    """
    edr_names = {process_name(p) for p in edr_processes}
    stimulus_names = {process_name(p) for p in stimulus_processes}
    if edr_names & stimulus_names:
        raise ValueError("EDR and stimulus process-name filters overlap")
    root_ids = {str(pid) for pid in stimulus_pids}
    for record in records:
        if (
            process_name(record.get("ProcessName", record.get("ExtractedProcessName")))
            in stimulus_names
        ):
            pid = identifier(record, "PID", "ProcessId", "ProcessID")
            if pid is not None:
                root_ids.add(pid)
    descendants = set(root_ids)
    changed = True
    while changed:
        changed = False
        for record in records:
            parent = identifier(record, "PPID", "ParentProcessId", "ParentProcessID")
            pid = identifier(record, "PID", "ProcessId", "ProcessID")
            if parent in descendants and pid is not None and pid not in descendants:
                descendants.add(pid)
                changed = True
    edr, stimulus = [], []
    for event in events_from_records(records):
        is_stimulus = event.process in stimulus_names or event.process_id in descendants
        is_edr = event.process in edr_names
        if is_stimulus and is_edr:
            raise ValueError(
                f"Ambiguous EDR/stimulus process identity at event {event.event_id}"
            )
        if is_edr:
            edr.append(event)
        elif is_stimulus:
            stimulus.append(event)
    return edr, stimulus


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, records: Iterable[dict]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
