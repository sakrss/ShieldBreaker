"""Execution-level support (Eq. 7) and conditional trace-level rates (Eqs. 8--10)."""

from collections import defaultdict
import math
from shieldbreaker.config import AnalysisConfig
from shieldbreaker.analysis.temporal import temporal_overlap


def wilson_interval(successes, trials, z=1.959964):
    if (
        not isinstance(trials, int)
        or not isinstance(successes, int)
        or not 0 <= successes <= trials
    ):
        raise ValueError("Require integer 0 <= successes <= trials")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    if trials == 0:
        return None, None
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = (
        z
        * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return max(0.0, center - half), min(1.0, center + half)


def execution_sets(observation):
    present, overlapping = set(), set()
    for segment in observation["behaviors"]:
        behavior = segment["behavior"]
        present.add(behavior)
        if any(
            temporal_overlap((segment["start"], segment["end"]), cluster)
            for cluster in observation["clusters"]
        ):
            overlapping.add(behavior)
    return present, overlapping


def unique_observations(observations):
    observations = list(observations)
    ids = [obs["execution_id"] for obs in observations]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate execution IDs would inflate recurrence support")
    return observations


def estimate_recurrence(observations, config=AnalysisConfig(), inventory=None):
    observations = unique_observations(observations)
    counts = defaultdict(lambda: [0, 0])
    for observation in observations:
        present, overlapping = execution_sets(observation)
        for behavior in present:
            counts[behavior][0] += 1
            counts[behavior][1] += behavior in overlapping
    # No historical commercial candidate keys are reconstructed. Without an
    # externally supplied record-level inventory the unit is a behavior category.
    if inventory is None:
        inventory = [
            {"behavior": name, "unit": "behavior_category"} for name in sorted(counts)
        ]
    result = []
    for candidate in inventory:
        behavior = candidate["behavior"]
        n, k = counts.get(behavior, [0, 0])
        lower, upper = wilson_interval(k, n, config.wilson_z)
        result.append(
            {
                **candidate,
                "trials": n,
                "successes": k,
                "wilson_lower": lower,
                "wilson_upper": upper,
                "retained": k >= config.min_successes
                and lower is not None
                and lower >= config.min_wilson_lower,
            }
        )
    return result


def evaluate_held_out(candidates, observations):
    observations = unique_observations(observations)
    retained = [r for r in candidates if r["retained"]]
    categories = {r["behavior"] for r in retained}
    activated = set()
    eligible = overlapping = 0
    for observation in observations:
        present, matches = execution_sets(observation)
        activated |= present & categories
        eligible += bool(present & categories)
        overlapping += bool(matches & categories)
    return {
        "candidate_unit": "external_inventory"
        if any("candidate_id" in r for r in retained)
        else "behavior_category",
        "retained_total": len(retained),
        "activated": sum(r["behavior"] in activated for r in retained),
        "eligible_traces": eligible,
        "overlapping_traces": overlapping,
        "conditional_trace_cooccurrence_rate": overlapping / eligible
        if eligible
        else None,
        "test_traces": len(observations),
    }
