"""Section 7.4: finite AND/OR hypotheses, decoder controls, target validation gate."""

from itertools import combinations, product
import numpy as np
from shieldbreaker.analysis.evaluation import classification_metrics
from shieldbreaker.traces import write_json, write_jsonl

PAIRS = tuple(combinations(range(6), 2))
HYPOTHESES = tuple((pair, operator) for pair in PAIRS for operator in ("AND", "OR"))
INPUTS = np.array(list(product((0, 1), repeat=6)), dtype=bool)


def evaluate_rule(inputs, pair, operator):
    a, b = inputs[:, pair[0]], inputs[:, pair[1]]
    if operator == "AND":
        return a & b
    if operator == "OR":
        return a | b
    if operator == "XOR":
        return a ^ b
    raise ValueError("Unknown rule operator")


def fit_hypothesis(inputs, labels):
    errors = [
        float(np.mean(evaluate_rule(inputs, *hypothesis) != labels))
        for hypothesis in HYPOTHESES
    ]
    return HYPOTHESES[int(np.argmin(errors))]  # Fixed-order tie resolution.


def sample_observations(rng, count, pair, operator, inverted=False):
    x = INPUTS[rng.integers(len(INPUTS), size=count)]
    logical = evaluate_rule(x, pair, operator)
    encoded = logical ^ inverted
    noisy = encoded ^ (rng.random(count) < 0.03)
    return x, noisy


def run(output_dir, seeds=range(20261021, 20261026), budgets=(16, 64, 128)):
    records, observations = [], []
    conditions = (
        "same_logic",
        "new_predicate",
        "unseen_or",
        "unmodeled_xor",
        "response_drift",
    )
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pairs = rng.permutation(len(PAIRS))
        discovery_pairs = [PAIRS[i] for i in pairs[:6]]
        target_pairs = [PAIRS[i] for i in pairs[6:12]]
        for rule, discovery_pair in enumerate(discovery_pairs):
            x, labels = sample_observations(rng, 192, discovery_pair, "AND")
            frozen = fit_hypothesis(x, labels)
            observations.append(
                {
                    "seed": seed,
                    "rule": rule,
                    "split": "discovery",
                    "inputs": x.astype(int).tolist(),
                    "labels": labels.astype(int).tolist(),
                    "pair": list(discovery_pair),
                    "operator": "AND",
                }
            )
            for condition in conditions:
                pair = (
                    discovery_pair
                    if condition in {"same_logic", "response_drift"}
                    else target_pairs[rule]
                )
                operator = (
                    "OR"
                    if condition == "unseen_or"
                    else "XOR"
                    if condition == "unmodeled_xor"
                    else "AND"
                )
                inverted = condition == "response_drift"
                truth = evaluate_rule(INPUTS, pair, operator).tolist()
                records.append(
                    {
                        "seed": seed,
                        "rule": rule,
                        "condition": condition,
                        "method": "frozen",
                        "fitting_observations": 0,
                        "total_target_cost": 0,
                        "accepted": True,
                        "f1": classification_metrics(
                            truth, evaluate_rule(INPUTS, *frozen).tolist()
                        )["f1"],
                    }
                )
                for budget in budgets:
                    fit_x, fit_y = sample_observations(
                        rng, budget, pair, operator, inverted
                    )
                    validation_x, validation_y = sample_observations(
                        rng, 64, pair, operator, inverted
                    )
                    # Labeled signature controls estimate response encoding, not a
                    # new predicate. Both logical classes are represented.
                    control_truth = np.tile([False, True], 8)
                    control_response = (
                        control_truth ^ inverted ^ (rng.random(16) < 0.03)
                    )
                    observations.append(
                        {
                            "seed": seed,
                            "rule": rule,
                            "condition": condition,
                            "fitting_count": budget,
                            "fitting_inputs": fit_x.astype(int).tolist(),
                            "fitting_encoded_labels": fit_y.astype(int).tolist(),
                            "validation_inputs": validation_x.astype(int).tolist(),
                            "validation_encoded_labels": validation_y.astype(
                                int
                            ).tolist(),
                            "control_truth": control_truth.astype(int).tolist(),
                            "control_response": control_response.astype(int).tolist(),
                            "target_pair": list(pair),
                            "target_operator": operator,
                            "encoding_inverted": inverted,
                        }
                    )
                    decoder_inverted = bool(
                        np.mean(control_response != control_truth) > 0.5
                    )
                    for method, decoder, control_count in [
                        ("adapted", decoder_inverted, 16),
                        ("fixed_signature", False, 0),
                    ]:
                        fitted = fit_hypothesis(fit_x, fit_y ^ decoder)
                        error = float(
                            np.mean(
                                evaluate_rule(validation_x, *fitted)
                                != (validation_y ^ decoder)
                            )
                        )
                        accepted = error <= 0.12
                        records.append(
                            {
                                "seed": seed,
                                "rule": rule,
                                "condition": condition,
                                "method": method,
                                "fitting_observations": budget,
                                "validation_observations": 64,
                                "signature_controls": control_count,
                                "total_target_cost": budget + 64 + control_count,
                                "decoder_inverted": decoder,
                                "validation_error": error,
                                "accepted": accepted,
                                "fitted_pair": list(fitted[0]),
                                "fitted_operator": fitted[1],
                                "f1": classification_metrics(
                                    truth, evaluate_rule(INPUTS, *fitted).tolist()
                                )["f1"],
                                "predictions_deployable": accepted,
                            }
                        )
    cells = []
    for condition in conditions:
        for method, budget in [
            ("frozen", 0),
            *[(m, b) for b in budgets for m in ("adapted", "fixed_signature")],
        ]:
            rows = [
                r
                for r in records
                if r["condition"] == condition
                and r["method"] == method
                and r["fitting_observations"] == budget
            ]
            cells.append(
                {
                    "condition": condition,
                    "method": method,
                    "fitting_observations": budget,
                    "models": len(rows),
                    "accepted": sum(r["accepted"] for r in rows),
                    "offline_f1_including_abstentions": float(
                        np.mean([r["f1"] for r in rows])
                    ),
                }
            )
    result = {
        "experiment": "synthetic_predicate_adaptation",
        "seeds": list(seeds),
        "seed_note": "Independent deterministic seeds; manuscript does not specify original adaptation seeds",
        "hypotheses": len(HYPOTHESES),
        "inputs": len(INPUTS),
        "cells": cells,
        "scope": "Offline truth-table simulation; rejected-model predictions are never marked deployable",
    }
    write_jsonl(output_dir / "models.jsonl", records)
    write_jsonl(output_dir / "observations.jsonl", observations)
    write_json(output_dir / "summary.json", result)
    return result
