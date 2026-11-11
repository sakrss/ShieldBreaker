"""Section 7: matched low-order statistics, retrieval, and independent FPR gate."""

from collections import Counter
import numpy as np
from scipy.stats import beta
from shieldbreaker.analysis.evaluation import classification_metrics, select_threshold
from shieldbreaker.experiments.connected import background
from shieldbreaker.experiments.signatures import ResponseScorer
from shieldbreaker.traces import write_json, write_jsonl

REGIMES = ("easy", "matched_bigrams", "rare_distractor", "noisy_order", "collision")
METHODS = (
    "full",
    "signature",
    "frequency",
    "bigram",
    "trigram",
    "nll_winner",
    "single_scale",
)


def reference_motifs():
    # Reordered hub excursions preserve all unigram and directed bigram counts.
    leaves = [f"api{i}" for i in range(15)]

    def trail(order):
        return [token for leaf in order for token in ("hub", leaf)] + ["hub", "end"]

    target, negative = trail(leaves), trail(list(reversed(leaves)))
    assert Counter(target) == Counter(negative)
    assert Counter(zip(target, target[1:])) == Counter(zip(negative, negative[1:]))
    return target, negative


def generate_trace(rng, regime, positive):
    target, negative = reference_motifs()
    tokens = background(rng)
    if rng.random() < 0.4:
        tokens[8:40] = target  # Observable background collisions are deliberate.
    start = int(rng.integers(85, 145))
    if positive:
        motif = list(target)
        if regime == "noisy_order":
            motif = [
                f"bg{rng.integers(8)}" if rng.random() < 0.12 else token
                for token in motif
            ]
        tokens[start : start + 32] = motif
    elif regime == "collision":
        tokens[start : start + 32] = target
    elif regime != "easy":
        tokens[start : start + 32] = negative
    if regime == "rare_distractor":
        tokens[192:240] = [f"rare{rng.integers(12)}" for _ in range(48)]
    return tokens


def balanced_records(rng, regime, count, split):
    if count % 2:
        raise ValueError("Balanced regimes require even trace counts")
    labels = [False] * (count // 2) + [True] * (count // 2)
    rng.shuffle(labels)
    return [
        {
            "split": split,
            "regime": regime,
            "positive": positive,
            "tokens": generate_trace(rng, regime, positive),
        }
        for positive in labels
    ]


def run(
    output_dir,
    seeds=range(20261001, 20261006),
    validation_count=60,
    test_count=100,
    retrieval_sets=30,
):
    all_records, cells, retrieval_ledger = [], [], []
    reference, _ = reference_motifs()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        baseline = [background(rng) for _ in range(100)]
        write_jsonl(
            output_dir / f"{seed}_baseline.jsonl",
            ({"tokens": tokens} for tokens in baseline),
        )
        scorer = ResponseScorer(baseline, reference)
        validation, test = [], []
        for regime in REGIMES:
            validation.extend(
                balanced_records(rng, regime, validation_count, "validation")
            )
            test.extend(balanced_records(rng, regime, test_count, "test"))
        for row in validation + test:
            row.update(seed=seed, scores=scorer.scores(row["tokens"]))
        thresholds = {
            name: select_threshold(
                [r["scores"][name] for r in validation],
                [r["positive"] for r in validation],
            )
            for name in METHODS
        }
        write_json(
            output_dir / f"{seed}_frozen.json",
            {"thresholds": thresholds, "reference": reference, "window_count": 85},
        )
        for regime in REGIMES:
            rows = [r for r in test if r["regime"] == regime]
            sets = []
            positives = [i for i, r in enumerate(rows) if r["positive"]]
            negatives = [i for i, r in enumerate(rows) if not r["positive"]]
            for index in range(retrieval_sets):
                candidates = [
                    int(rng.choice(positives)),
                    *rng.choice(negatives, size=3, replace=False).tolist(),
                ]
                rng.shuffle(candidates)
                sets.append(candidates)
                retrieval_ledger.append(
                    {
                        "seed": seed,
                        "regime": regime,
                        "set": index,
                        "test_pool_indices": candidates,
                        "reuses_traces": True,
                    }
                )
            for name in METHODS:
                result = classification_metrics(
                    [r["positive"] for r in rows],
                    [r["scores"][name] > thresholds[name] for r in rows],
                )
                hits = {budget: [] for budget in (1, 2, 4)}
                for candidates in sets:
                    ranking = sorted(
                        range(4),
                        key=lambda i: (-rows[candidates[i]]["scores"][name], i),
                    )
                    for budget in hits:
                        hits[budget].append(
                            any(
                                rows[candidates[i]]["positive"]
                                for i in ranking[:budget]
                            )
                        )
                cells.append(
                    {
                        "seed": seed,
                        "regime": regime,
                        "method": name,
                        **result,
                        "retrieval_prefix_success": {
                            str(b): float(np.mean(values)) for b, values in hits.items()
                        },
                    }
                )
        all_records.extend(validation + test)
    result = {
        "experiment": "synthetic_discrimination",
        "seeds": list(seeds),
        "window_grid_count": 85,
        "cells": cells,
        "scope": "Fresh deterministic synthetic measurements; retrieval sets can share traces",
    }
    write_jsonl(output_dir / "traces.jsonl", all_records)
    write_jsonl(output_dir / "retrieval_sets.jsonl", retrieval_ledger)
    write_json(output_dir / "summary.json", result)
    return result


def binomial_upper(false_positives, negatives, alpha=0.05 / 7):
    if not 0 <= false_positives <= negatives or negatives <= 0 or not 0 < alpha < 1:
        raise ValueError("Invalid exact-binomial confidence inputs")
    return (
        1.0
        if false_positives == negatives
        else float(
            beta.ppf(1 - alpha, false_positives + 1, negatives - false_positives)
        )
    )


def confirm_scorer(selection_scores, confirmation_scores, alpha=0.05 / 7, max_fpr=0.05):
    # A 2.5% selection upper tail; strict > handles ties conservatively.
    threshold = float(np.quantile(selection_scores, 0.975))
    fp = sum(score > threshold for score in confirmation_scores)
    upper = binomial_upper(fp, len(confirmation_scores), alpha)
    return {
        "threshold": threshold,
        "confirmation_fp": fp,
        "confirmation_n": len(confirmation_scores),
        "fpr_upper": upper,
        "enabled": upper <= max_fpr,
    }


def run_confirmation(
    output_dir, seeds=range(20261011, 20261016), negative_count=256, test_count=500
):
    ledgers, gates, pooled = [], [], {name: ([], []) for name in METHODS}
    reference, _ = reference_motifs()
    for seed in seeds:
        rng = np.random.default_rng(seed)
        baseline = [background(rng) for _ in range(100)]
        write_jsonl(
            output_dir / f"{seed}_baseline.jsonl",
            ({"tokens": tokens} for tokens in baseline),
        )
        scorer = ResponseScorer(baseline, reference)

        def sample(count, split, negative=False):
            rows = []
            for index in range(count):
                regime = str(rng.choice(REGIMES))
                positive = False if negative else bool(rng.integers(2))
                tokens = generate_trace(rng, regime, positive)
                rows.append(
                    {
                        "seed": seed,
                        "split": split,
                        "index": index,
                        "regime": regime,
                        "positive": positive,
                        "tokens": tokens,
                        "scores": scorer.scores(tokens),
                    }
                )
            return rows

        selection = sample(negative_count, "selection", True)
        confirmation = sample(negative_count, "confirmation", True)
        test = sample(test_count, "test")
        for name in METHODS:
            gate = confirm_scorer(
                [r["scores"][name] for r in selection],
                [r["scores"][name] for r in confirmation],
            )
            gates.append({"seed": seed, "method": name, **gate})
            truth, predictions = pooled[name]
            truth.extend(r["positive"] for r in test)
            # Disabled scorer/seed pairs stay in every metric denominator.
            predictions.extend(
                gate["enabled"] and r["scores"][name] > gate["threshold"] for r in test
            )
        ledgers.extend(selection + confirmation + test)
    result = {
        "experiment": "synthetic_independent_confirmation",
        "seeds": list(seeds),
        "seed_note": "Independent new seeds; original confirmation seeds are unspecified in the manuscript",
        "selection_tail": 0.025,
        "familywise_alpha": 0.05,
        "per_scorer_alpha": 0.05 / 7,
        "fpr_gate": 0.05,
        "gates": gates,
        "metrics": {name: classification_metrics(*pooled[name]) for name in METHODS},
    }
    write_jsonl(output_dir / "traces.jsonl", ledgers)
    write_json(output_dir / "summary.json", result)
    return result
