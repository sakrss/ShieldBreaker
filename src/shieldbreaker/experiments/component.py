"""Section 6.1--6.2 abstract fragment experiment (Python standard-library RNG)."""

import random
import numpy as np
from shieldbreaker.analysis.evaluation import classification_metrics, select_threshold
from shieldbreaker.analysis.recurrence import wilson_interval
from shieldbreaker.analysis.temporal import temporal_overlap
from shieldbreaker.experiments.signatures import trigrams, jaccard, medoid
from shieldbreaker.traces import write_json, write_jsonl


def make_fragment(rng, behavior, substitution):
    tokens = [f"noise{rng.randrange(12)}" for _ in range(rng.randint(48, 72))]
    motif = [
        "shared0",
        "shared1",
        "shared2",
        "shared3",
        *[f"B{behavior}_{i}" for i in range(4)],
    ]
    offset = rng.randrange(len(tokens) - len(motif) + 1)
    tokens[offset : offset + 8] = motif
    return [
        f"noise{rng.randrange(12)}" if rng.random() < substitution else token
        for token in tokens
    ]


def generate(seed=20260910, counts=(80, 40, 60)):
    rng = random.Random(seed)
    for profile in range(3):
        for behavior in range(8):
            for split, count in zip(("discovery", "validation", "test"), counts):
                for index in range(count):
                    start = rng.uniform(15, 35)
                    end = start + 2
                    fragments = []
                    if rng.random() < (0.97, 0.93, 0.89)[profile] - 0.006 * behavior:
                        response_start = end + rng.uniform(-0.5, 4.5)
                        fragments.append(
                            {
                                "tokens": make_fragment(
                                    rng, behavior, (0.01, 0.04, 0.08)[profile]
                                ),
                                "interval": [
                                    response_start,
                                    response_start + rng.uniform(0.2, 1.2),
                                ],
                                "ground_truth_type": behavior,
                                "planted_target": True,
                            }
                        )
                    for _ in range(rng.randint(1, 4)):
                        kind = rng.randrange(8)
                        background_start = rng.uniform(0, 98)
                        fragments.append(
                            {
                                "tokens": make_fragment(
                                    rng, kind, (0.01, 0.04, 0.08)[profile]
                                ),
                                "interval": [
                                    background_start,
                                    background_start + rng.uniform(0.2, 1.2),
                                ],
                                "ground_truth_type": kind,
                                "planted_target": False,
                            }
                        )
                    yield {
                        "execution_id": f"S{profile + 1}/B{behavior}/{split}/{index}",
                        "profile": f"S{profile + 1}",
                        "behavior": behavior,
                        "split": split,
                        "behavior_interval": [start, end],
                        "fragments": fragments,
                    }


def in_horizon(row, fragment, shift=0):
    start, end = row["behavior_interval"]
    return temporal_overlap((start + shift, end + shift + 5), fragment["interval"])


def run(output_dir, seed=20260910, counts=(80, 40, 60)):
    records = list(generate(seed, counts))
    write_jsonl(output_dir / "executions.jsonl", records)
    summaries, supports = [], []
    for profile in ("S1", "S2", "S3"):
        rows = [r for r in records if r["profile"] == profile]
        representatives = {}
        for behavior in range(8):
            # Only observed fragments and timing enter medoid selection.
            fragments = [
                f["tokens"]
                for row in rows
                if row["behavior"] == behavior and row["split"] == "discovery"
                for f in row["fragments"]
                if in_horizon(row, f)
            ]
            representatives[behavior] = trigrams(medoid(fragments))
        validation_scores, validation_truth = [], []
        for row in rows:
            if row["split"] == "validation":
                for fragment in row["fragments"]:
                    validation_scores.append(
                        jaccard(
                            trigrams(fragment["tokens"]),
                            representatives[row["behavior"]],
                        )
                    )
                    validation_truth.append(
                        fragment["ground_truth_type"] == row["behavior"]
                    )
        threshold = select_threshold(
            validation_scores, validation_truth, np.arange(0.10, 0.801, 0.05)
        )

        def matches(row, shift=0):
            return [
                f
                for f in row["fragments"]
                if in_horizon(row, f, shift)
                and jaccard(trigrams(f["tokens"]), representatives[row["behavior"]])
                > threshold
            ]

        test = [r for r in rows if r["split"] == "test"]
        for behavior in range(8):
            discovery = [
                r
                for r in rows
                if r["split"] == "discovery" and r["behavior"] == behavior
            ]
            k = sum(bool(matches(row)) for row in discovery)
            lower, upper = wilson_interval(k, len(discovery))
            supports.append(
                {
                    "profile": profile,
                    "behavior": behavior,
                    "successes": k,
                    "trials": len(discovery),
                    "wilson_lower": lower,
                    "wilson_upper": upper,
                    "retained": k >= 5 and lower is not None and lower >= 0.70,
                }
            )
        truth, predictions = [], []
        for row in test:
            for fragment in row["fragments"]:
                truth.append(fragment["ground_truth_type"] == row["behavior"])
                predictions.append(
                    jaccard(
                        trigrams(fragment["tokens"]), representatives[row["behavior"]]
                    )
                    > threshold
                )
        hit_rows = [matches(row) for row in test]
        summaries.append(
            {
                "profile": profile,
                "candidates": 8,
                "retained": sum(
                    s["retained"] for s in supports if s["profile"] == profile
                ),
                "threshold": threshold,
                "test_executions": len(test),
                "strict_hits": sum(bool(hit) for hit in hit_rows),
                "target_hits": sum(
                    any(f["planted_target"] for f in hit) for hit in hit_rows
                ),
                "background_only_hits": sum(
                    bool(hit) and not any(f["planted_target"] for f in hit)
                    for hit in hit_rows
                ),
                "shifted_hits": sum(bool(matches(row, 50)) for row in test),
                "fragment_metrics": classification_metrics(truth, predictions),
            }
        )
        write_json(
            output_dir / f"{profile}_frozen_matching.json",
            {
                "threshold": threshold,
                "representatives": {
                    str(b): [[list(key), value] for key, value in signature.items()]
                    for b, signature in representatives.items()
                },
            },
        )
    result = {
        "experiment": "synthetic_component",
        "seed": seed,
        "post_behavior_seconds": 5,
        "negative_control_shift_seconds": 50,
        "coverage_by_construction": 1.0,
        "profiles": summaries,
        "support": supports,
        "scope": "New abstract-fragment measurements, not commercial EDR results or recovered historical cells",
    }
    write_json(output_dir / "summary.json", result)
    return result
