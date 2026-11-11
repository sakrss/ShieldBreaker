"""Preserve the execution split; segment only after splitting (70:10:20)."""

import argparse
import json
import hashlib
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from shieldbreaker.preprocess.build_vocab import SPECIAL_TOKENS
from shieldbreaker.traces import write_json


def segment_long_sequences(seqs, segment_length):
    if segment_length < 3:
        raise ValueError("segment_length must leave space for BOS, an API, and EOS")
    chunks = []
    for sequence in seqs:
        apis = [t for t in sequence if not t.startswith("<")]
        chunks.extend(
            [
                ["<BOS>", *apis[start : start + segment_length - 2], "<EOS>"]
                for start in range(0, len(apis), segment_length - 2)
            ]
        )
    return chunks


def build_wordlevel_tokenizer(vocab, tokenizer_json_path):
    if sorted(vocab.values()) != list(range(len(vocab))) or not all(
        token in vocab for token in SPECIAL_TOKENS
    ):
        raise ValueError(
            "Vocabulary must have contiguous IDs and all four special tokens"
        )
    core = Tokenizer(WordLevel(vocab, unk_token="<UNK>"))
    core.pre_tokenizer = WhitespaceSplit()
    core.save(str(tokenizer_json_path))
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token="<PAD>",
        unk_token="<UNK>",
        bos_token="<BOS>",
        eos_token="<EOS>",
        model_max_length=1024,
    )


def make_corpus(rows, vocab, out_dir, segment_length=1024):
    ids = [row["execution_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate executions in token corpus")
    if any(row["split"] not in {"discovery", "validation", "test"} for row in rows):
        raise ValueError("Unknown split")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, filename in [
        ("discovery", "train.txt"),
        ("validation", "valid.txt"),
        ("test", "test.txt"),
    ]:
        sequences = [row["tokens"] for row in rows if row["split"] == split]
        if not sequences:
            raise ValueError(
                f"Empty {split} split; do not reuse training data as validation"
            )
        chunks = segment_long_sequences(sequences, segment_length)
        (out_dir / filename).write_text(
            "".join(" ".join(chunk) + "\n" for chunk in chunks), encoding="utf-8"
        )
    observed = {
        token for row in rows if row["split"] == "discovery" for token in row["tokens"]
    }
    if set(vocab) - set(SPECIAL_TOKENS) != observed:
        raise ValueError(
            "Vocabulary must be fitted exclusively on discovery executions"
        )
    tokenizer = build_wordlevel_tokenizer(vocab, out_dir / "tokenizer.json")
    tokenizer.save_pretrained(out_dir)
    write_json(
        out_dir / "corpus_hashes.json",
        {
            name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
            for name in ("train.txt", "valid.txt", "test.txt")
        },
    )
    write_json(
        out_dir / "split_manifest.json",
        [{k: row[k] for k in ("execution_id", "split")} for row in rows],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens_path", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--segment_length", type=int, default=1024)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.tokens_path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not rows or not all(
        isinstance(row, dict) and {"execution_id", "split", "tokens"} <= row.keys()
        for row in rows
    ):
        raise ValueError(
            "Use execution-preserving build_vocab output, not a merged/segmented legacy corpus"
        )
    vocab = json.loads(Path(args.vocab_path).read_text(encoding="utf-8-sig"))
    make_corpus(rows, vocab, args.out_dir, args.segment_length)


if __name__ == "__main__":
    main()
