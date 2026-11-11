from collections import Counter
from shieldbreaker.experiments.signatures import window_grid, trigrams
from shieldbreaker.experiments.discrimination import (
    reference_motifs,
    confirm_scorer,
    binomial_upper,
)
from shieldbreaker.experiments.connected import generate, rank_guidance
from shieldbreaker.experiments.adaptation import (
    HYPOTHESES,
    INPUTS,
    evaluate_rule,
    fit_hypothesis,
    run,
)
from shieldbreaker.experiments.component import generate as component_records


def test_matched_statistics_and_corrected_trigram_grid():
    positive, negative = reference_motifs()
    assert len(positive) == len(negative) == 32
    assert Counter(positive) == Counter(negative)
    assert Counter(zip(positive, positive[1:])) == Counter(zip(negative, negative[1:]))
    assert trigrams(positive) != trigrams(negative)
    assert len(window_grid()) == 85


def test_confirmation_uses_independent_gate_and_strict_ties():
    assert confirm_scorer([0.0] * 256, [0.0] * 256)["enabled"]
    assert not confirm_scorer([0.0] * 256, [1.0] * 256)["enabled"]
    assert binomial_upper(0, 256) < 0.05
    assert binomial_upper(256, 256) == 1


def test_connected_design_and_guidance_do_not_use_alert_labels():
    rows, baseline = generate(20260910, 0)
    assert Counter(r["split"] for r in rows) == {
        "discovery": 1344,
        "validation": 336,
        "test": 280,
    }
    assert len(baseline) == 80 and all(len(r["tokens"]) == 256 for r in rows)
    assert not any(r["behavior"] == 5 and r["split"] == "test" for r in rows)
    for row in rows:
        row["scores"] = {
            "full": 100 if row["canonical_feature"] == -1 else row["canonical_feature"]
        }
    before = rank_guidance(rows, "full", 0)
    for row in rows:
        row["ground_truth"] = {"invalid": "not visible to the learner"}
    assert rank_guidance(rows, "full", 0) == before


def test_component_has_presplit_independent_executions():
    rows = list(component_records())
    assert len(rows) == 4320
    assert len({r["execution_id"] for r in rows}) == len(rows)
    assert all(
        len(f["tokens"]) in range(48, 73) for row in rows for f in row["fragments"]
    )


def test_logic_grammar_and_total_adaptation_budget(tmp_path):
    assert len(HYPOTHESES) == 30 and len(INPUTS) == 64
    labels = evaluate_rule(INPUTS, (2, 5), "OR")
    assert fit_hypothesis(INPUTS, labels) == ((2, 5), "OR")
    run(tmp_path, seeds=[20261021], budgets=(16,))
    import json

    rows = [
        json.loads(line)
        for line in (tmp_path / "models.jsonl").read_text().splitlines()
    ]
    assert {r["total_target_cost"] for r in rows if r["method"] == "adapted"} == {96}
    assert all(
        not r["predictions_deployable"]
        for r in rows
        if r["method"] != "frozen" and not r["accepted"]
    )
