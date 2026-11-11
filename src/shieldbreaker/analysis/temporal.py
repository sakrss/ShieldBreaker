"""Multi-scale NLL localization and interval association, Eqs. 4--6."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import numpy as np

from shieldbreaker.config import AnalysisConfig


@dataclass(frozen=True)
class Window:
    anchor: int
    scale: int
    score: float
    start: float
    end: float

    def to_dict(self):
        return asdict(self)


def multiscale_windows(nlls, timestamps, config=AnalysisConfig()):
    if len(nlls) != len(timestamps):
        raise ValueError("One NLL and timestamp are required for every API event")
    if any(not math.isfinite(x) or x < 0 for x in nlls):
        raise ValueError("NLLs must be finite and nonnegative")
    if any(not math.isfinite(t) for t in timestamps) or any(
        a > b for a, b in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("Timestamps must be finite and nondecreasing")
    sums = np.concatenate(([0.0], np.cumsum(nlls, dtype=np.float64)))
    windows = []
    for anchor in range(0, len(nlls), config.stride):
        complete = [w for w in sorted(config.scales) if anchor + w <= len(nlls)]
        if not complete:
            continue
        # max() retains the first (smallest) scale on exact ties.
        scale = max(complete, key=lambda w: (sums[anchor + w] - sums[anchor]) / w)
        score = float((sums[anchor + scale] - sums[anchor]) / scale)
        windows.append(
            Window(
                anchor, scale, score, timestamps[anchor], timestamps[anchor + scale - 1]
            )
        )
    return windows


def baseline_threshold(window_sequences, quantile=0.995):
    if not 0 < quantile < 1:
        raise ValueError("Quantile must lie strictly between zero and one")
    scores = [window.score for windows in window_sequences for window in windows]
    if not scores or not all(math.isfinite(x) for x in scores):
        raise ValueError("No finite complete baseline windows for calibration")
    return float(np.quantile(scores, quantile, method="linear"))


def merge_intervals(intervals, tolerance=0.0):
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("Interval tolerance must be finite and nonnegative")
    ordered = sorted((float(start), float(end)) for start, end in intervals)
    if any(not math.isfinite(s) or not math.isfinite(e) or s > e for s, e in ordered):
        raise ValueError("Invalid interval")
    merged = []
    for start, end in ordered:
        if merged and max(0.0, start - merged[-1][1]) <= tolerance:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def estimate_tolerance(intervals):
    """Kneedle on sorted normalized gaps; no knee => zero tolerance.

    Running maximum endpoints, rather than previous-window endpoints, ensure
    nested intervals cannot create spurious gaps (Section 4.4 correction).
    """
    ordered = sorted(intervals)
    if len(ordered) < 2:
        return 0.0
    merge_intervals(ordered)  # Validate endpoints.
    end = ordered[0][1]
    gaps = []
    for start, stop in ordered[1:]:
        gaps.append(max(0.0, start - end))
        end = max(end, stop)
    if len(set(gaps)) < 2:
        return 0.0
    from kneed import KneeLocator

    values = np.sort(np.asarray(gaps, dtype=float))
    x = np.linspace(0.0, 1.0, len(values))
    y = (values - values.min()) / (values.max() - values.min())
    knee = KneeLocator(x, y, curve="convex", direction="increasing").knee
    return 0.0 if knee is None else float(np.interp(knee, x, values))


def temporal_overlap(left, right):
    """Closed-interval, nonempty intersection. No post-behavior response horizon."""
    if (
        not all(math.isfinite(t) for t in (*left, *right))
        or left[0] > left[1]
        or right[0] > right[1]
    ):
        raise ValueError("Invalid interval")
    return max(left[0], right[0]) <= min(left[1], right[1])


def localize(windows, threshold):
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    anomalous = [window for window in windows if window.score > threshold]
    intervals = [(window.start, window.end) for window in anomalous]
    tolerance = estimate_tolerance(intervals)
    return anomalous, merge_intervals(intervals, tolerance), tolerance


def associate_behaviors(configuration, behaviors, clusters):
    """Emit the within-execution observations o=(p,b,c) from Eqs. 1 and 6.

    Segment and cluster indices are local to this one execution. They do not
    define a commercial candidate key or a cross-execution response identity.
    All intersecting pairs are retained; recurrence still counts one success
    per candidate per execution rather than the number of pairs emitted here.
    """
    observations = []
    for segment_index, segment in enumerate(behaviors):
        interval = [segment["start"], segment["end"]]
        for cluster_index, cluster in enumerate(clusters):
            if temporal_overlap(interval, cluster):
                observations.append(
                    {
                        "configuration": configuration,
                        "behavior": segment["behavior"],
                        "behavior_segment_index": segment_index,
                        "cluster_index": cluster_index,
                        "behavior_interval": interval,
                        "cluster_interval": list(cluster),
                    }
                )
    return observations
