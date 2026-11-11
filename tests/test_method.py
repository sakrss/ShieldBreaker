import json
import pytest
from shieldbreaker.analysis.temporal import (
    multiscale_windows,
    baseline_threshold,
    merge_intervals,
    estimate_tolerance,
    temporal_overlap,
)
from shieldbreaker.analysis.recurrence import (
    wilson_interval,
    estimate_recurrence,
    evaluate_held_out,
)
from shieldbreaker.analysis.semantics import (
    validate_reconstruction,
    reconstruction_request,
)
from shieldbreaker.traces import partition_records, events_from_records
from shieldbreaker.preprocess.build_vocab import split_executions, build_vocab
from shieldbreaker.manifest import load_manifest
from shieldbreaker.analysis.evaluation import (
    variant_summary,
    paired_alert_comparison,
    holm_adjust,
)


def observation(i, segments=None, clusters=None):
    return {
        "execution_id": str(i),
        "behaviors": segments
        if segments is not None
        else [
            {"behavior": "rename", "start": 1, "end": 2},
            {"behavior": "rename", "start": 1.5, "end": 2.5},
        ],
        "clusters": clusters if clusters is not None else [[2, 3], [2.2, 4]],
    }


def test_stride_leaves_small_scale_gaps_and_uses_complete_windows():
    losses = [1.0] * 256 + [100.0] * 256
    windows = multiscale_windows(losses, list(range(512)))
    assert len(windows) == 1 and windows[0].scale == 512
    assert windows[0].score == 50.5  # Not an arbitrary 256-event segment.
    assert multiscale_windows([1.0] * 255, list(range(255))) == []
    ties = multiscale_windows([1.0] * 1024, list(range(1024)))
    assert [(w.anchor, w.scale) for w in ties] == [(0, 256), (512, 256)]


def test_baseline_quantile_and_strict_threshold():
    windows = multiscale_windows(
        [1.0] * 256 + [0.0] * 256 + [3.0] * 256, list(range(768))
    )
    assert baseline_threshold([windows]) == pytest.approx(2.99)
    with pytest.raises(ValueError, match="No finite"):
        baseline_threshold([[]])
    with pytest.raises(ValueError):
        multiscale_windows([float("nan")], [1])


def test_nested_windows_keep_furthest_endpoint():
    assert merge_intervals([[0, 10], [2, 3], [9, 12], [15, 16]]) == [[0, 12], [15, 16]]
    assert merge_intervals([[0, 10], [2, 3], [12, 13]], 2) == [[0, 13]]
    assert estimate_tolerance([[0, 10], [2, 3], [9, 12]]) == 0
    assert temporal_overlap([1, 2], [2, 3])
    assert not temporal_overlap([1, 2], [2.001, 3])


def test_wilson_uses_unrounded_boundary_and_minimum_trials():
    assert wilson_interval(18, 20)[0] == pytest.approx(0.698966, abs=0.000001)
    for k in (5, 8):
        assert not estimate_recurrence([observation(i) for i in range(k)])[0][
            "retained"
        ]
    assert estimate_recurrence([observation(i) for i in range(9)])[0]["retained"]
    assert not estimate_recurrence(
        [observation(i, clusters=[] if i >= 18 else None) for i in range(20)]
    )[0]["retained"]
    assert wilson_interval(0, 0) == (None, None)


def test_one_vote_per_execution_and_undefined_empty_denominator():
    candidates = estimate_recurrence([observation(i) for i in range(9)])
    assert candidates[0]["trials"] == candidates[0]["successes"] == 9
    assert (
        evaluate_held_out(candidates, [observation(10), observation(11, [], [])])[
            "eligible_traces"
        ]
        == 1
    )
    assert (
        evaluate_held_out(candidates, [observation(11, [], [])])[
            "conditional_trace_cooccurrence_rate"
        ]
        is None
    )
    with pytest.raises(ValueError, match="Duplicate"):
        estimate_recurrence([observation(1), observation(1)])


def test_distinct_external_candidates_may_share_behavior():
    inventory = [
        {"candidate_id": "a", "behavior": "rename"},
        {"candidate_id": "b", "behavior": "rename"},
    ]
    candidates = estimate_recurrence(
        [observation(i) for i in range(9)], inventory=inventory
    )
    result = evaluate_held_out(candidates, [observation(20)])
    assert (
        result["activated"] == 2
        and result["eligible_traces"] == 1
        and result["overlapping_traces"] == 1
    )
    assert result["candidate_unit"] == "external_inventory"


def test_pid_descendants_and_metadata_survive_partition():
    records = [
        {
            "TimeStamp": 1,
            "ProcessName": "C:\\EDR.EXE",
            "PID": 1,
            "TID": 4,
            "Method": "NtReadFile",
            "Arguments": {"File": "x"},
            "ReturnStatus": 0,
        },
        {
            "TimeStamp": 1,
            "ProcessName": "sample.exe",
            "PID": 2,
            "Method": "NtCreateProcess",
        },
        {
            "TimeStamp": 2,
            "ProcessName": "child.exe",
            "PID": 3,
            "PPID": 2,
            "Method": "NtOpenFile",
        },
        {
            "TimeStamp": 3,
            "ProcessName": "child.exe",
            "PID": 3,
            "Method": "NtClose",
            "Plugin": "sysret",
        },
        {"TimeStamp": 2, "ProcessName": "unrelated.exe", "PID": 9, "Method": "NtClose"},
    ]
    edr, stimulus = partition_records(records, ["edr.exe"], ["SAMPLE.EXE"])
    assert [e.process_id for e in stimulus] == ["2", "3"]
    assert edr[0].metadata["Arguments"] == {"File": "x"} and edr[0].thread_id == "4"
    assert edr[0].function == "NtReadFile"
    with pytest.raises(ValueError, match="overlap"):
        partition_records(records, ["sample.exe"], ["sample.exe"])


def test_semantics_require_grounded_time_and_evidence():
    events = events_from_records(
        [
            {"TimeStamp": 1, "Method": "NtOpenFile", "GlobalID": 1},
            {"TimeStamp": 2, "Method": "NtClose", "GlobalID": 2},
        ]
    )
    annotation = {
        "behaviors": [
            {
                "behavior": "file access",
                "start": 1,
                "end": 2,
                "evidence_event_ids": ["1", "2"],
            }
        ],
        "unsupported_conclusions": ["execution is not observed"],
    }
    assert validate_reconstruction(annotation, events) == annotation["behaviors"]
    annotation["behaviors"][0]["evidence_event_ids"] = ["missing"]
    with pytest.raises(ValueError, match="evidence"):
        validate_reconstruction(annotation, events)
    assert reconstruction_request(events)["model"] == "gpt-5.4"


def test_execution_splits_are_disjoint_and_vocab_is_function_only():
    result = split_executions([str(i) for i in range(100)])
    assert {
        split: list(result.values()).count(split) for split in set(result.values())
    } == {"discovery": 70, "validation": 10, "test": 20}
    assert result == split_executions([str(i) for i in range(100)])
    with pytest.raises(ValueError):
        build_vocab([["<PROC:edr.exe>"]])


def test_manifest_blocks_group_and_file_leakage(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).write_text("")
    rows = [
        {
            "execution_id": "a",
            "group_id": "sample",
            "split": "discovery",
            "condition": "baseline",
            "path": "a",
        },
        {
            "execution_id": "b",
            "group_id": "sample",
            "split": "test",
            "condition": "stimulated",
            "path": "b",
        },
    ]
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "configuration": "p/version/policy",
                "edr_processes": ["edr.exe"],
                "traces": rows,
            }
        )
    )
    with pytest.raises(ValueError, match="crosses"):
        load_manifest(path)


def test_rq4_statistics_use_paired_cases_and_conditional_groups():
    rows = (
        [{"alert": False, "anomaly": False}] * 13
        + [{"alert": False, "anomaly": True}]
        + [{"alert": True, "anomaly": True}] * 2
    )
    summary = variant_summary(rows)
    assert summary["overall_agreement"] == 15 / 16
    assert summary["conditional_agreement_nonalerting"] == 13 / 14
    left = {i: i < 9 for i in range(10)}
    right = {i: i == 9 for i in range(10)}
    assert paired_alert_comparison(left, right)["p_value"] == pytest.approx(0.021484375)
    assert holm_adjust([0.021484375, 0.00634765625, 0.75390625]) == pytest.approx(
        [0.04296875, 0.01904296875, 0.75390625]
    )


def test_distribution_summaries_and_case_intersections():
    from shieldbreaker.analysis.evaluation import (
        trace_distribution_summary,
        cross_product_nonalerting,
    )

    summary = trace_distribution_summary(
        [
            {"condition": "baseline", "trace_p995": 1},
            {"condition": "baseline", "trace_p995": 3},
            {"condition": "baseline", "trace_p995": None},
        ]
    )
    assert summary["baseline"] == {"traces": 3, "scoreable": 2, "median_trace_p995": 2}
    result = cross_product_nonalerting(
        {"a": {"x": False, "y": True}, "b": {"x": False, "y": False}}
    )
    assert result["pairwise_intersections"] == {"a/b": 1}
    assert sum(r["count"] for r in result["exclusive_regions"]) == 2
