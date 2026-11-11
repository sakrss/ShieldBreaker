"""Section 6.3--6.5: API-observation guidance in an abstract editable-feature world."""

from collections import defaultdict, Counter
import numpy as np
from shieldbreaker.analysis.evaluation import classification_metrics, select_threshold
from shieldbreaker.analysis.recurrence import wilson_interval
from shieldbreaker.experiments.signatures import trigrams, jaccard, window_grid
from shieldbreaker.training.models import NGramModel
from shieldbreaker.traces import write_json, write_jsonl

ARMS = (
    "full",
    "frequency",
    "bigram",
    "temporal",
    "no_semantics",
    "no_signature",
    "single_scale",
    "unguided",
    "shuffled",
)


def background(rng, length=256):
    tokens, state = [], int(rng.integers(8))
    for _ in range(length):
        state = (state + int(rng.choice(3, p=[0.15, 0.8, 0.05]))) % 8
        tokens.append(f"bg{state}")
    return tokens


def make_trace(rng, behavior, profile, response):
    tokens = background(rng)
    visible = response and rng.random() >= (0.03, 0.10, 0.20)[profile]
    if visible:
        motif = ["r0", "r1", "r2", "r3", *[f"B{behavior}_{i}" for i in range(4)]]
        width, start = int(rng.choice([16, 32, 64])), int(rng.integers(85, 145))
        fragment = [
            motif[i % 8]
            if rng.random() >= (0.02, 0.06, 0.12)[profile]
            else f"bg{rng.integers(8)}"
            for i in range(width)
        ]
        tokens[start : start + width] = fragment
    if rng.random() < 0.35:
        tokens[224:248] = [f"nuisance{rng.integers(8)}" for _ in range(24)]
    return tokens, bool(visible)


def score_observation(tokens, lm, prototype):
    nll = np.concatenate(([0.0], np.cumsum(lm.score(tokens))))
    grid = window_grid(256, region=(64, 216))
    signature = trigrams(prototype, False)
    prototype_frequency = Counter(prototype)
    values = []
    for anchor, width in grid:
        fragment = tokens[anchor : anchor + width]
        mean = float((nll[anchor + width] - nll[anchor]) / width)
        values.append(
            (width, mean, jaccard(trigrams(fragment, False), signature), fragment)
        )
    winner = max(values, key=lambda v: v[1])
    single = max((v for v in values if v[0] == 64), key=lambda v: v[1])
    full = winner[1] * winner[2]
    return {
        "full": full,
        "no_semantics": full,
        "no_signature": winner[1],
        "frequency": max(jaccard(Counter(v[3]), prototype_frequency) for v in values),
        "bigram": max(v[1] for v in values if v[0] == 32),
        "single_scale": single[1] * single[2],
        "temporal": 32.0,
    }


def generate(seed, profile):
    rng = np.random.default_rng(seed + profile * 100000)
    pairs = [
        set(rng.choice(np.arange(1, 13), size=2, replace=False).tolist())
        for _ in range(6)
    ]
    families = rng.permutation(24).tolist()
    rows = []
    for position, family in enumerate(families):
        split = (
            "discovery" if position < 16 else "validation" if position < 20 else "test"
        )
        permutation = rng.permutation(13).tolist()  # surface ID -> canonical ID
        for behavior in range(6 if split != "test" else 5):
            for feature in [-1, *range(13)]:
                valid = feature != 0
                alert = valid and feature not in pairs[behavior]
                tokens, visible = make_trace(rng, behavior, profile, alert)
                rows.append(
                    {
                        "seed": seed,
                        "profile": f"S{profile + 1}",
                        "family": family,
                        "behavior": behavior,
                        "split": split,
                        "canonical_feature": feature,
                        "surface_feature": permutation.index(feature)
                        if feature >= 0
                        else -1,
                        "surface_to_canonical": permutation,
                        "tokens": tokens,
                        "ground_truth": {
                            "valid": valid,
                            "alert": alert,
                            "response": visible,
                            "hidden_predicate_features": sorted(pairs[behavior]),
                        },
                    }
                )
    baseline = [background(rng) for _ in range(80)]
    return rows, baseline


def rank_guidance(rows, arm, behavior):
    discovery = [
        r for r in rows if r["split"] == "discovery" and r["behavior"] == behavior
    ]
    canonical = arm != "no_semantics"
    score_name = "full" if arm in {"no_semantics", "unguided", "shuffled"} else arm
    by_family = defaultdict(dict)
    for row in discovery:
        feature = row["canonical_feature"] if canonical else row["surface_feature"]
        by_family[row["family"]][feature] = row["scores"][score_name]
    reductions = {
        feature: np.mean(
            [values[-1] - values[feature] for values in by_family.values()]
        )
        for feature in range(13)
    }
    return sorted(reductions, key=lambda f: (-reductions[f], f))[:4]


def run(output_dir, seeds=range(20260910, 20260915)):
    queries, support, metrics, ledgers, difficult = [], [], [], [], []
    for seed in seeds:
        for profile in range(3):
            rows, baseline = generate(seed, profile)
            write_jsonl(
                output_dir / f"{seed}_S{profile + 1}_baseline.jsonl",
                ({"tokens": tokens} for tokens in baseline),
            )
            lm = NGramModel(order=2).fit(baseline)
            prototypes = {}
            for behavior in range(6):
                candidates = []
                for row in rows:
                    if (
                        row["split"] == "discovery"
                        and row["behavior"] == behavior
                        and row["canonical_feature"] == -1
                    ):
                        scores = np.concatenate(
                            ([0.0], np.cumsum(lm.score(row["tokens"])))
                        )
                        for anchor, width in window_grid(256, region=(64, 216)):
                            candidates.append(
                                (
                                    (scores[anchor + width] - scores[anchor]) / width,
                                    row["tokens"][anchor : anchor + width],
                                )
                            )
                prototypes[behavior] = max(candidates, key=lambda pair: pair[0])[1]
            for row in rows:
                row["scores"] = score_observation(
                    row["tokens"], lm, prototypes[row["behavior"]]
                )
            validation = [row for row in rows if row["split"] == "validation"]
            test = [row for row in rows if row["split"] == "test"]
            thresholds = {
                arm: select_threshold(
                    [r["scores"][arm] for r in validation],
                    [r["ground_truth"]["response"] for r in validation],
                )
                for arm in ARMS[:7]
            }
            for arm in ARMS[:7]:
                metrics.append(
                    {
                        "seed": seed,
                        "profile": f"S{profile + 1}",
                        "arm": arm,
                        "threshold": thresholds[arm],
                        **classification_metrics(
                            [r["ground_truth"]["response"] for r in test],
                            [r["scores"][arm] > thresholds[arm] for r in test],
                        ),
                    }
                )
            rankings = {
                (arm, behavior): rank_guidance(rows, arm, behavior)
                for arm in ARMS[:7]
                for behavior in range(6)
            }
            rng = np.random.default_rng(seed + profile * 100)
            for behavior in range(6):
                rankings["unguided", behavior] = rng.permutation(13)[:4].tolist()
                rankings["shuffled", behavior] = list(
                    rankings["full", (behavior + 1) % 6]
                )
            for behavior in range(6):
                disc = [
                    r
                    for r in rows
                    if r["split"] == "discovery"
                    and r["behavior"] == behavior
                    and r["canonical_feature"] == -1
                ]
                k = sum(r["scores"]["full"] > thresholds["full"] for r in disc)
                lower, upper = wilson_interval(k, len(disc))
                support.append(
                    {
                        "seed": seed,
                        "profile": f"S{profile + 1}",
                        "behavior": behavior,
                        "trials": len(disc),
                        "successes": k,
                        "wilson_lower": lower,
                        "wilson_upper": upper,
                        "retained": k >= 5 and lower >= 0.7,
                        "active_on_test": behavior < 5,
                    }
                )
            for family in sorted({row["family"] for row in test}):
                for behavior in range(5):
                    unchanged = next(
                        r
                        for r in test
                        if r["family"] == family
                        and r["behavior"] == behavior
                        and r["canonical_feature"] == -1
                    )
                    for arm in ARMS:
                        success = False
                        for query, feature in enumerate(rankings[arm, behavior], 1):
                            canonical = (
                                unchanged["surface_to_canonical"][feature]
                                if arm == "no_semantics"
                                else feature
                            )
                            edited = next(
                                r
                                for r in test
                                if r["family"] == family
                                and r["behavior"] == behavior
                                and r["canonical_feature"] == canonical
                            )
                            ground_truth = edited["ground_truth"]
                            success |= (
                                ground_truth["valid"] and not ground_truth["alert"]
                            )
                            queries.append(
                                {
                                    "seed": seed,
                                    "profile": f"S{profile + 1}",
                                    "family": family,
                                    "behavior": behavior,
                                    "arm": arm,
                                    "query": query,
                                    "canonical_feature": canonical,
                                    "equivalent_nonalerting": ground_truth["valid"]
                                    and not ground_truth["alert"],
                                    "cumulative_success": bool(success),
                                    "invalid": not ground_truth["valid"],
                                }
                            )
            for unchanged in [r for r in test if r["canonical_feature"] == -1]:
                behavior = unchanged["behavior"]
                visible_tokens, visible = make_trace(rng, behavior, profile, True)
                while not visible:
                    visible_tokens, visible = make_trace(rng, behavior, profile, True)
                absent_tokens, _ = make_trace(rng, behavior, profile, False)
                changed_tokens, changed_visible = make_trace(
                    rng, behavior + 6, profile, True
                )
                settings = [
                    ("changed_response", changed_tokens, changed_visible, True),
                    ("lookalike", visible_tokens, True, False),
                    ("response_without_alert", visible_tokens, True, False),
                    ("alert_without_response", absent_tokens, False, True),
                    ("benign_no_alert", absent_tokens, False, False),
                ]
                for regime, tokens, response, alert in settings:
                    score = score_observation(tokens, lm, prototypes[behavior])["full"]
                    difficult.append(
                        {
                            "seed": seed,
                            "profile": f"S{profile + 1}",
                            "family": unchanged["family"],
                            "behavior": behavior,
                            "regime": regime,
                            "tokens": tokens,
                            "response_visible": response,
                            "alert": alert,
                            "score": score,
                            "match": score > thresholds["full"],
                        }
                    )
            ledgers.extend(rows)
            write_json(
                output_dir / f"{seed}_S{profile + 1}_frozen.json",
                {
                    "thresholds": thresholds,
                    "rankings": [
                        {"arm": arm, "behavior": b, "features": features}
                        for (arm, b), features in rankings.items()
                    ],
                    "prototypes": {str(b): tokens for b, tokens in prototypes.items()},
                },
            )
    summaries = []
    for arm in ARMS:
        arm_rows = [q for q in queries if q["arm"] == arm]
        summaries.append(
            {
                "arm": arm,
                "cases": sum(q["query"] == 1 for q in arm_rows),
                "prefix_success": [
                    np.mean(
                        [
                            q["cumulative_success"]
                            for q in arm_rows
                            if q["query"] == budget
                        ]
                    ).item()
                    for budget in range(1, 5)
                ],
                "first_edit_invalid_fraction": np.mean(
                    [q["invalid"] for q in arm_rows if q["query"] == 1]
                ).item(),
            }
        )
    from scipy.stats import t

    differences = []
    for seed in seeds:
        full = [
            q["cumulative_success"]
            for q in queries
            if q["seed"] == seed and q["arm"] == "full" and q["query"] == 4
        ]
        unguided = [
            q["cumulative_success"]
            for q in queries
            if q["seed"] == seed and q["arm"] == "unguided" and q["query"] == 4
        ]
        differences.append(float(np.mean(full) - np.mean(unguided)))
    half = (
        float(
            t.ppf(0.975, len(differences) - 1)
            * np.std(differences, ddof=1)
            / np.sqrt(len(differences))
        )
        if len(differences) > 1
        else None
    )
    paired = {
        "n_seeds": len(differences),
        "mean_difference": float(np.mean(differences)),
        "approximate_95_t_interval": [
            float(np.mean(differences) - half),
            float(np.mean(differences) + half),
        ]
        if half is not None
        else None,
    }
    difficult_summary = [
        {
            "regime": regime,
            "cases": sum(r["regime"] == regime for r in difficult),
            "predicted_matches": sum(
                r["regime"] == regime and r["match"] for r in difficult
            ),
        }
        for regime in (
            "changed_response",
            "lookalike",
            "response_without_alert",
            "alert_without_response",
            "benign_no_alert",
        )
    ]
    result = {
        "experiment": "synthetic_connected",
        "seeds": list(seeds),
        "trace_counts": dict(Counter(r["split"] for r in ledgers)),
        "query_count": len(queries),
        "arms": summaries,
        "response_metrics": metrics,
        "paired_full_minus_unguided_at_q4": paired,
        "difficult_cases": difficult_summary,
        "candidate_count": len(support),
        "active_candidates": sum(r["active_on_test"] for r in support),
        "passing_active": sum(r["active_on_test"] and r["retained"] for r in support),
        "scope": "Generator labels assess synthetic equivalence; canonical metadata is supplied, not LLM-validated",
    }
    write_jsonl(output_dir / "traces.jsonl", ledgers)
    write_jsonl(output_dir / "queries.jsonl", queries)
    write_jsonl(output_dir / "difficult_cases.jsonl", difficult)
    write_json(output_dir / "support.json", support)
    write_json(output_dir / "summary.json", result)
    return result
