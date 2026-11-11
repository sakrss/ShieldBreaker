"""Function-name sequence models. Arguments/process tokens never enter NLL."""

from __future__ import annotations
from collections import Counter, defaultdict
import json
import math
from pathlib import Path


class NGramModel:
    def __init__(self, order=3, alpha=1.0):
        if order < 1 or alpha <= 0:
            raise ValueError("Require positive order and smoothing")
        self.order, self.alpha = order, alpha
        self.vocabulary = {"<UNK>"}
        self.counts = defaultdict(Counter)

    def fit(self, sequences):
        sequences = list(sequences)
        if not sequences or not any(sequences):
            raise ValueError("Baseline training sequences are empty")
        self.vocabulary |= {token for sequence in sequences for token in sequence}
        for sequence in sequences:
            history = ["<BOS>"] * (self.order - 1)
            for token in sequence:
                self.counts[
                    tuple(history[-(self.order - 1) :]) if self.order > 1 else ()
                ][token] += 1
                history.append(token)
        return self

    def score(self, tokens):
        history = ["<BOS>"] * (self.order - 1)
        losses = []
        for token in tokens:
            token = token if token in self.vocabulary else "<UNK>"
            context = tuple(history[-(self.order - 1) :]) if self.order > 1 else ()
            counts = self.counts.get(context, {})
            probability = (counts.get(token, 0) + self.alpha) / (
                sum(counts.values()) + self.alpha * len(self.vocabulary)
            )
            losses.append(-math.log(probability))
            history.append(token)
        return losses

    def save(self, path):
        from shieldbreaker.traces import write_json

        write_json(
            path,
            {
                "order": self.order,
                "alpha": self.alpha,
                "vocabulary": sorted(self.vocabulary),
                "counts": [
                    {"context": list(key), "counts": dict(value)}
                    for key, value in sorted(self.counts.items())
                ],
            },
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        model = cls(data["order"], data["alpha"])
        model.vocabulary = set(data["vocabulary"])
        model.counts = {
            tuple(row["context"]): Counter(row["counts"]) for row in data["counts"]
        }
        return model
