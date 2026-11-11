"""GPT-2 per-API next-token negative log likelihood (Eq. 3).

Windows/thresholds use NLL, not exponentiated sequence perplexity. Long inputs
are scored in overlapping context blocks with exactly one loss per API event.
"""

from __future__ import annotations
import math
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel


class EDRAnomalyDetector:
    def __init__(self, model_path, tokenizer_path=None, device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device).eval()
        if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
            raise ValueError("Tokenizer IDs and model embedding vocabulary differ")
        if any(token.startswith("<PROC:") for token in self.tokenizer.get_vocab()):
            raise ValueError(
                "Legacy process-token checkpoints must be retrained on API-only tokens"
            )
        self.context_size = int(self.model.config.max_position_embeddings)
        if self.tokenizer.bos_token_id is None or self.tokenizer.unk_token_id is None:
            raise ValueError("A function-name tokenizer with BOS and UNK is required")

    def score(self, api_calls, block_size=512):
        if block_size < 1 or self.context_size < 2:
            raise ValueError("Invalid inference context/block size")
        vocabulary = self.tokenizer.get_vocab()
        if any(
            not isinstance(api, str)
            or not api
            or any(ch.isspace() for ch in api)
            or api.startswith("<")
            for api in api_calls
        ):
            raise ValueError(
                "Pass API function names only, without BOS/PROC/argument tokens"
            )
        ids = [self.tokenizer.bos_token_id] + [
            vocabulary.get(api, self.tokenizer.unk_token_id) for api in api_calls
        ]
        losses = []
        step = min(block_size, self.context_size - 1)
        with torch.inference_mode():
            for start in range(0, len(api_calls), step):
                stop = min(start + step, len(api_calls))
                # Global target positions start+1 .. stop. Retain the available
                # left context within the checkpoint's finite context budget.
                left = max(0, stop + 1 - self.context_size)
                inputs = torch.tensor(
                    [ids[left : stop + 1]], dtype=torch.long, device=self.device
                )
                logits = self.model(input_ids=inputs, use_cache=False).logits[0]
                target_start = start + 1 - left
                targets = inputs[0, target_start:]
                prediction = logits[target_start - 1 : -1]
                losses.extend(
                    torch.nn.functional.cross_entropy(
                        prediction.float(), targets, reduction="none"
                    )
                    .cpu()
                    .tolist()
                )
        if len(losses) != len(api_calls) or not all(
            math.isfinite(loss) for loss in losses
        ):
            raise ValueError("Model failed to produce one finite NLL per API event")
        return losses

    def calculate_perplexity(self, sequence):
        """Diagnostic only; no localization/calibration uses perplexity."""
        api_calls = [token for token in sequence.split() if not token.startswith("<")]
        scores = self.score(api_calls)
        if not scores:
            raise ValueError("No API events")
        return math.exp(sum(scores) / len(scores))

    def estimate_threshold(self, normal_sequences, percentile=99.5, verbose=False):
        from shieldbreaker.analysis.temporal import (
            baseline_threshold,
            multiscale_windows,
        )

        windows = []
        for sequence in normal_sequences:
            tokens = sequence.split() if isinstance(sequence, str) else list(sequence)
            if any(token.startswith("<") for token in tokens):
                raise ValueError(
                    "Threshold calibration requires API-only baseline sequences"
                )
            windows.append(
                multiscale_windows(self.score(tokens), list(range(len(tokens))))
            )
        return baseline_threshold(windows, percentile / 100)
