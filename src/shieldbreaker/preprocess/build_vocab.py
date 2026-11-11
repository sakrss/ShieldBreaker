"""Build API-only tokens and split independent executions before fitting vocab."""

import argparse
from collections import defaultdict
from pathlib import Path
import random

from shieldbreaker.traces import (
    events_from_records,
    process_name,
    read_records,
    write_json,
    write_jsonl,
)

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


def split_executions(execution_ids, seed=42):
    ids = sorted(execution_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Execution IDs must be unique")
    if len(ids) < 10:
        raise ValueError(
            "At least ten independent executions are needed for a 70:10:20 split"
        )
    random.Random(seed).shuffle(ids)
    train_end, valid_end = int(0.7 * len(ids)), int(0.8 * len(ids))
    return {
        execution_id: (
            "discovery" if i < train_end else "validation" if i < valid_end else "test"
        )
        for i, execution_id in enumerate(ids)
    }


def build_vocab(token_seqs):
    vocabulary = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for token in sorted({token for sequence in token_seqs for token in sequence}):
        if token.startswith("<") or any(ch.isspace() for ch in token):
            raise ValueError(
                "API vocabulary cannot contain process/argument/boundary tokens"
            )
        vocabulary.setdefault(token, len(vocabulary))
    return vocabulary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_path",
        required=True,
        help="Single trace file or directory of independent JSONL traces",
    )
    parser.add_argument("--out_tokens_path", required=True)
    parser.add_argument("--out_vocab_path", required=True)
    parser.add_argument(
        "--processes", nargs="+", help="Optional case-insensitive EDR process filters"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = Path(args.input_path)
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    grouped = defaultdict(list)
    for file in files:
        for record in read_records(file):
            execution_id = str(
                record.get(
                    "ExecutionID",
                    file.relative_to(path).as_posix() if path.is_dir() else file.stem,
                )
            )
            grouped[execution_id].append(record)
    split = split_executions(grouped, args.seed)
    names = {process_name(name) for name in args.processes} if args.processes else None
    rows = []
    for execution_id, records in sorted(grouped.items()):
        tokens = [
            event.function
            for event in events_from_records(records)
            if names is None or event.process in names
        ]
        if not tokens:
            raise ValueError(f"No EDR API events in execution {execution_id}")
        rows.append(
            {
                "execution_id": execution_id,
                "split": split[execution_id],
                "tokens": tokens,
            }
        )
    vocabulary = build_vocab(
        row["tokens"] for row in rows if row["split"] == "discovery"
    )
    write_jsonl(args.out_tokens_path, rows)
    write_json(args.out_vocab_path, vocabulary)


if __name__ == "__main__":
    main()
