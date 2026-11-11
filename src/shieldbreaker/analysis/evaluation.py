"""RQ1/2/4 descriptive measurements; no synthetic observations masquerade as real."""

import math
from collections import Counter
from statistics import median
import numpy as np
from shieldbreaker.analysis.recurrence import wilson_interval


def classification_metrics(truth, predictions):
    if len(truth) != len(predictions):
        raise ValueError("Labels and predictions have different lengths")
    tp = sum(bool(y) and bool(p) for y, p in zip(truth, predictions))
    fp = sum(not y and bool(p) for y, p in zip(truth, predictions))
    fn = sum(bool(y) and not p for y, p in zip(truth, predictions))
    tn = len(truth) - tp - fp - fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def select_threshold(scores, truth, thresholds=None):
    if (
        not len(scores)
        or len(scores) != len(truth)
        or not all(math.isfinite(x) for x in scores)
    ):
        raise ValueError("Need matching finite validation scores and labels")
    grid = (
        np.unique(np.quantile(scores, np.linspace(0, 1, 41)))
        if thresholds is None
        else thresholds
    )
    # Strict > matches anomaly localization. Include no threshold tuned on tests.
    return float(
        max(
            grid,
            key=lambda t: (
                classification_metrics(truth, [s > t for s in scores])["f1"],
                t,
            ),
        )
    )


def relative_time_bins(windows, behavior_end, bin_width=1.0, before=30, after=30):
    """Map event-anchor windows to wall-clock bins; missing bins remain null."""
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    result = []
    for left in np.arange(-before, after, bin_width):
        scores = [
            w.score
            for w in windows
            if left <= w.start - behavior_end < left + bin_width
        ]
        result.append(
            {
                "relative_start": float(left),
                "p995": float(np.quantile(scores, 0.995)) if scores else None,
            }
        )
    return result


def semantic_assessment_summary(ratings):
    counts = Counter(ratings)
    if set(counts) - {"high", "medium", "low"}:
        raise ValueError("Only human high/medium/low confidence ratings are accepted")
    total = len(ratings)
    return {
        name: {
            "count": counts[name],
            "fraction": counts[name] / total if total else None,
        }
        for name in ("high", "medium", "low")
    }


def variant_summary(cases):
    """Only supplied alert/anomaly outcomes and equivalence assessments are used."""
    counts = Counter((bool(row["alert"]), bool(row["anomaly"])) for row in cases)
    n = len(cases)
    absent = counts[False, False] + counts[False, True]
    present = n - absent
    agreement = counts[False, False] + counts[True, True]
    return {
        "cases": n,
        "contingency_absent_absent_to_present_present": [
            counts[False, False],
            counts[False, True],
            counts[True, False],
            counts[True, True],
        ],
        "alerts_absent": absent,
        "anomalies_absent_among_nonalerting": counts[False, False],
        "conditional_agreement_nonalerting": counts[False, False] / absent
        if absent
        else None,
        "alerts_present": present,
        "anomalies_present_among_alerting": counts[True, True],
        "conditional_agreement_alerting": counts[True, True] / present
        if present
        else None,
        "overall_agreement": agreement / n if n else None,
        "agreement_wilson": wilson_interval(agreement, n),
        "nonalerting_wilson": wilson_interval(absent, n),
        "equivalence_assessments": Counter(
            str(row.get("equivalent", "unverified")) for row in cases
        ),
    }


def paired_alert_comparison(left, right):
    from scipy.stats import binomtest

    if set(left) != set(right):
        raise ValueError("McNemar comparison requires the same case identities")
    b = sum(bool(left[key]) and not right[key] for key in left)
    c = sum(not left[key] and bool(right[key]) for key in left)
    return {
        "discordant": [b, c],
        "p_value": float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0,
    }


def holm_adjust(p_values):
    ordered = sorted(range(len(p_values)), key=lambda i: p_values[i])
    result, running = [0.0] * len(p_values), 0.0
    for rank, index in enumerate(ordered):
        running = max(running, (len(p_values) - rank) * p_values[index])
        result[index] = min(1.0, running)
    return result


def trace_distribution_summary(trace_scores):
    """Summarize supplied per-trace p99.5 values without replacing missing traces."""
    result = {}
    for condition in ("baseline", "benign", "stimulated"):
        rows = [row for row in trace_scores if row["condition"] == condition]
        values = [row["trace_p995"] for row in rows if row["trace_p995"] is not None]
        result[condition] = {
            "traces": len(rows),
            "scoreable": len(values),
            "median_trace_p995": median(values) if values else None,
        }
    return result


def cross_product_nonalerting(alert_outcomes):
    """Intersect identical case IDs, not unpaired counts or internal rule IDs."""
    names = list(alert_outcomes)
    if not names:
        return {"products": [], "exclusive_regions": {}}
    cases = set(alert_outcomes[names[0]])
    if any(set(alert_outcomes[name]) != cases for name in names):
        raise ValueError("Cross-product comparison requires identical selected cases")
    exclusive = Counter(
        tuple(name for name in names if not alert_outcomes[name][case])
        for case in cases
    )
    return {
        "products": names,
        "cases": len(cases),
        "exclusive_regions": [
            {"nonalerting_products": list(region), "count": count}
            for region, count in sorted(exclusive.items())
        ],
        "pairwise_intersections": {
            f"{a}/{b}": sum(
                not alert_outcomes[a][case] and not alert_outcomes[b][case]
                for case in cases
            )
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        },
        "all_product_intersection": sum(
            all(not alert_outcomes[name][case] for name in names) for case in cases
        ),
    }
