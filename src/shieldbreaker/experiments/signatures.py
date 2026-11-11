"""Frozen response signatures, multiset Jaccard, and matched window grids."""

from collections import Counter
import numpy as np
from shieldbreaker.training.models import NGramModel


def trigrams(tokens, multiset=True):
    result = Counter(zip(tokens, tokens[1:], tokens[2:]))
    return result if multiset else Counter({key: 1 for key in result})


def jaccard(left, right):
    union = sum((left | right).values())
    return sum((left & right).values()) / union if union else 0.0


def medoid(fragments):
    if not fragments:
        raise ValueError("No discovery response fragments")
    signatures = [trigrams(fragment) for fragment in fragments]
    return list(
        fragments[
            max(
                range(len(fragments)),
                key=lambda i: sum(
                    jaccard(signatures[i], other) for other in signatures
                ),
            )
        ]
    )


def window_grid(length=256, scales=(16, 32, 64), stride=8, region=None):
    start, stop = region or (0, length)
    return [
        (anchor, scale)
        for scale in scales
        for anchor in range(start, stop - scale + 1, stride)
    ]


class ResponseScorer:
    def __init__(
        self,
        baseline,
        reference,
        scales=(16, 32, 64),
        stride=8,
        multiset=True,
        region=None,
    ):
        self.bigram = NGramModel(order=2).fit(baseline)
        self.trigram = NGramModel(order=3).fit(baseline)
        self.reference = list(reference)
        self.signature = trigrams(reference, multiset)
        self.multiset = multiset
        self.scales, self.stride, self.region = scales, stride, region
        self.frequency = Counter(reference)

    def scores(self, tokens):
        nll2, nll3 = self.bigram.score(tokens), self.trigram.score(tokens)
        nll2 = np.concatenate(([0.0], np.cumsum(nll2)))
        nll3 = np.concatenate(([0.0], np.cumsum(nll3)))
        values = []
        for anchor, width in window_grid(
            len(tokens), self.scales, self.stride, self.region
        ):
            fragment = tokens[anchor : anchor + width]
            similarity = jaccard(trigrams(fragment, self.multiset), self.signature)
            mean2 = float((nll2[anchor + width] - nll2[anchor]) / width)
            mean3 = float((nll3[anchor + width] - nll3[anchor]) / width)
            # Frequency similarity is deliberately order-insensitive.
            frequency = jaccard(Counter(fragment), self.frequency)
            values.append((anchor, width, mean2, mean3, similarity, frequency))
        if not values:
            raise ValueError("No complete response windows")
        winner = max(values, key=lambda row: row[2])
        return {
            "full": max(row[2] * row[4] for row in values),
            "signature": max(row[4] for row in values),
            "frequency": max(row[5] for row in values),
            "bigram": max(row[2] for row in values),
            "trigram": max(row[3] for row in values),
            "nll_winner": winner[2] * winner[4],
            "single_scale": max(
                (row[2] * row[4] for row in values if row[1] == 64), default=0.0
            ),
            "temporal": 32.0,
        }  # Same event count by construction, Section 6.4.
