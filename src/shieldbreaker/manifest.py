"""Validate independent executions, product configuration, and train/test boundaries."""

import json
from pathlib import Path


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if (
        not isinstance(manifest.get("configuration"), str)
        or not manifest["configuration"].strip()
    ):
        raise ValueError(
            "Declare a fixed configuration identity (product/version/policy)"
        )
    if not manifest.get("edr_processes") or not manifest.get("traces"):
        raise ValueError("Manifest requires EDR process filters and independent traces")
    ids, paths, groups = set(), set(), {}
    traces = []
    for row in manifest["traces"]:
        row = dict(row)
        execution_id = row["execution_id"]
        if not isinstance(execution_id, str) or not execution_id or execution_id in ids:
            raise ValueError("Nonempty unique execution_id required")
        ids.add(execution_id)
        if row["split"] not in {"discovery", "validation", "test"}:
            raise ValueError("Unknown execution split")
        if row["condition"] not in {"baseline", "stimulated", "benign"}:
            raise ValueError("Unknown trace condition")
        if (
            row.get("configuration", manifest["configuration"])
            != manifest["configuration"]
        ):
            raise ValueError("Different product configurations cannot share a baseline")
        group = row.get("group_id", execution_id)
        if group in groups and groups[group] != row["split"]:
            raise ValueError(
                f"Group {group} crosses discovery/validation/test boundaries"
            )
        groups[group] = row["split"]
        trace_path = (path.parent / row["path"]).resolve()
        if trace_path in paths or not trace_path.is_file():
            raise ValueError(f"Duplicate or missing trace file: {trace_path}")
        paths.add(trace_path)
        row["path"] = str(trace_path)
        if row.get("annotation") is not None:
            row["annotation"] = str((path.parent / row["annotation"]).resolve())
        traces.append(row)
    return {**manifest, "traces": traces}
