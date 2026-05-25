#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import pickle
import time
from collections import Counter
from pathlib import Path


def load_sequences(file_path: str):
    sequences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sequences.append(tokens)
    return sequences


def load_vocab_tokens(vocab_file: str | None, train_sequences, valid_sequences, unk_token: str):
    if vocab_file:
        with open(vocab_file, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        if not isinstance(raw_vocab, dict):
            raise ValueError("vocab_file must contain a token->id JSON object")

        vocab_tokens = [
            token for token, _ in sorted(raw_vocab.items(), key=lambda item: item[1])
        ]
    else:
        vocab_tokens = sorted({
            token
            for seq in train_sequences + valid_sequences
            for token in seq
        })

    if unk_token not in vocab_tokens:
        vocab_tokens.append(unk_token)

    return vocab_tokens


class BackoffNGramModel:
    def __init__(self, max_order: int, vocab_tokens, alpha: float = 0.1, unk_token: str = "<UNK>"):
        if max_order < 1:
            raise ValueError("max_order must be >= 1")

        self.max_order = max_order
        self.alpha = alpha
        self.unk_token = unk_token
        self.vocab_tokens = list(vocab_tokens)
        self.vocab_set = set(self.vocab_tokens)
        self.vocab_size = len(self.vocab_tokens)

        self.ngram_counts = [Counter() for _ in range(max_order)]
        self.context_counts = [Counter() for _ in range(max_order)]
        self.total_tokens = 0
        self.train_sequences = 0

    def normalize_sequence(self, sequence):
        return tuple(
            token if token in self.vocab_set else self.unk_token
            for token in sequence
        )

    def fit(self, sequences):
        for sequence in sequences:
            tokens = self.normalize_sequence(sequence)
            if not tokens:
                continue

            self.train_sequences += 1
            self.total_tokens += len(tokens)

            for order in range(1, self.max_order + 1):
                if len(tokens) < order:
                    continue

                for idx in range(len(tokens) - order + 1):
                    ngram = tuple(tokens[idx: idx + order])
                    self.ngram_counts[order - 1][ngram] += 1
                    if order > 1:
                        self.context_counts[order - 1][ngram[:-1]] += 1

    def perplexity(self, sequences, order: int):
        if not 1 <= order <= self.max_order:
            raise ValueError(f"order must be between 1 and {self.max_order}")

        cache = {}

        def probability(token, history, current_order):
            history = tuple(history[-(current_order - 1):]) if current_order > 1 else ()
            key = (current_order, history, token)
            if key in cache:
                return cache[key]

            if current_order == 1:
                numerator = self.ngram_counts[0].get((token,), 0) + self.alpha
                denominator = self.total_tokens + self.alpha * self.vocab_size
                value = numerator / denominator
            else:
                lower_prob = probability(token, history[1:], current_order - 1)
                context_count = self.context_counts[current_order - 1].get(history, 0)

                if context_count == 0:
                    value = lower_prob
                else:
                    full_count = self.ngram_counts[current_order - 1].get(history + (token,), 0)
                    numerator = full_count + self.alpha * self.vocab_size * lower_prob
                    denominator = context_count + self.alpha * self.vocab_size
                    value = numerator / denominator

            cache[key] = value
            return value

        total_neg_log_likelihood = 0.0
        total_tokens = 0

        for sequence in sequences:
            tokens = self.normalize_sequence(sequence)
            history = []
            for token in tokens:
                prob = probability(token, tuple(history), order)
                total_neg_log_likelihood -= math.log(prob)
                total_tokens += 1
                history.append(token)

        if total_tokens == 0:
            return float("inf")

        return math.exp(total_neg_log_likelihood / total_tokens)

    def export_state(self, best_order: int):
        return {
            "model_type": "ngram",
            "best_order": best_order,
            "max_order_trained": self.max_order,
            "alpha": self.alpha,
            "unk_token": self.unk_token,
            "vocab_tokens": self.vocab_tokens,
            "vocab_size": self.vocab_size,
            "total_tokens": self.total_tokens,
            "train_sequences": self.train_sequences,
            "ngram_counts": self.ngram_counts[:best_order],
            "context_counts": self.context_counts[:best_order],
        }


def main():
    parser = argparse.ArgumentParser(description="Train an N-gram language model")
    parser.add_argument("--train_file", required=True, help="Training data file")
    parser.add_argument("--validation_file", required=True, help="Validation data file")
    parser.add_argument("--vocab_file", default=None, help="Optional token->id vocabulary JSON")
    parser.add_argument("--output_dir", default="models/out_ngram", help="Output directory")
    parser.add_argument("--min_order", type=int, default=1, help="Minimum n-gram order to evaluate")
    parser.add_argument("--max_order", type=int, default=5, help="Maximum n-gram order to evaluate")
    parser.add_argument("--alpha", type=float, default=0.1, help="Backoff smoothing strength")
    parser.add_argument("--unk_token", default="<UNK>", help="Unknown token placeholder")
    args = parser.parse_args()

    if args.min_order < 1 or args.max_order < args.min_order:
        raise ValueError("Require 1 <= min_order <= max_order")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    train_sequences = load_sequences(args.train_file)
    valid_sequences = load_sequences(args.validation_file)
    vocab_tokens = load_vocab_tokens(
        args.vocab_file, train_sequences, valid_sequences, args.unk_token
    )

    model = BackoffNGramModel(
        max_order=args.max_order,
        vocab_tokens=vocab_tokens,
        alpha=args.alpha,
        unk_token=args.unk_token,
    )
    model.fit(train_sequences)

    history = {
        "orders": [],
        "eval_perplexity": [],
        "train_sequence_count": len(train_sequences),
        "valid_sequence_count": len(valid_sequences),
        "train_token_count": sum(len(seq) for seq in train_sequences),
        "valid_token_count": sum(len(seq) for seq in valid_sequences),
        "vocab_size": len(vocab_tokens),
        "best_order": None,
        "best_eval_perplexity": None,
        "elapsed_time_sec": None,
    }

    best_order = None
    best_perplexity = float("inf")

    for order in range(args.min_order, args.max_order + 1):
        ppl = model.perplexity(valid_sequences, order)
        history["orders"].append(order)
        history["eval_perplexity"].append(ppl)
        print(f"Order {order}: eval_perplexity={ppl:.6f}")

        if ppl < best_perplexity:
            best_perplexity = ppl
            best_order = order

    history["best_order"] = best_order
    history["best_eval_perplexity"] = best_perplexity
    history["elapsed_time_sec"] = time.time() - start_time

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    config = {
        "model_type": "ngram",
        "best_order": best_order,
        "min_order": args.min_order,
        "max_order": args.max_order,
        "alpha": args.alpha,
        "unk_token": args.unk_token,
        "vocab_size": len(vocab_tokens),
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(output_dir / "ngram_model.pkl", "wb") as f:
        pickle.dump(model.export_state(best_order), f)

    print(f"Best order: {best_order}")
    print(f"Best eval_perplexity: {best_perplexity:.6f}")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
