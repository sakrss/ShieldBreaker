#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import random
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


def load_tokens_jsonl(path: str):
    """
    Load token sequences from a JSONL file.
    Each line is expected to be a JSON list of tokens.
    """
    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seq = json.loads(line)  # list[str]
            seqs.append(seq)
    return seqs

def load_vocab(path: str):
    """
    Load vocabulary from JSON file. Returns token -> id mapping.
    """
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab

def ensure_special_tokens_in_vocab(vocab: dict):
    """
    Ensure that all required special tokens are present in the vocabulary.
    """
    for t in SPECIAL_TOKENS:
        if t not in vocab:
            vocab[t] = len(vocab)
    return vocab


def segment_long_sequences(seqs, segment_length: int):
    """
    If the corpus contains one very long sequence, split it into fixed-size
    segments so train/validation can be separated meaningfully.
    """
    if segment_length <= 0:
        return seqs

    segmented = []
    for seq in seqs:
        proc_tokens = [token for token in seq if token.startswith("<PROC:")]
        is_single_process_seq = (
            len(seq) >= 2 and
            seq[0] == "<BOS>" and
            seq[1].startswith("<PROC:") and
            len(set(proc_tokens)) == 1
        )

        if len(seq) <= segment_length:
            segmented.append(seq)
            continue

        if is_single_process_seq:
            prefix = seq[:2]
            suffix = ["<EOS>"] if seq[-1] == "<EOS>" else []
            body = seq[2:-1] if suffix else seq[2:]
            chunk_body_len = segment_length - len(prefix) - len(suffix)

            if chunk_body_len < 1:
                raise ValueError("segment_length is too small to preserve BOS/PROC context")

            for start in range(0, len(body), chunk_body_len):
                body_chunk = body[start:start + chunk_body_len]
                chunk = prefix + body_chunk + suffix
                if len(chunk) >= len(prefix) + 1:
                    segmented.append(chunk)
            continue

        for start in range(0, len(seq), segment_length):
            chunk = seq[start:start + segment_length]
            if len(chunk) >= 2:
                segmented.append(chunk)

    return segmented

def save_corpus(seqs, out_train: str, out_valid: str, valid_ratio: float = 0.05, seed: int = 42):
    """
    Shuffle and split sequences into training and validation sets, then save to files.
    """
    random.seed(seed)
    random.shuffle(seqs)
    n_valid = max(1, int(len(seqs) * valid_ratio)) if len(seqs) > 1 else 0

    valid = seqs[:n_valid]
    train = seqs[n_valid:] if n_valid > 0 else seqs

    def write_txt(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for seq in data:
                f.write(" ".join(seq) + "\n")

    write_txt(out_train, train)
    if n_valid > 0:
        write_txt(out_valid, valid)
    else:
        write_txt(out_valid, train)

def build_wordlevel_tokenizer(vocab: dict, tokenizer_json_path: str):
    """
    Build and save a WordLevel tokenizer using provided vocabulary.
    """
    model = WordLevel(vocab=vocab, unk_token="<UNK>")
    tok = Tokenizer(model)
    # Preserve process tokens such as <PROC:msmpeng.exe> as single units.
    tok.pre_tokenizer = WhitespaceSplit()

    tok.post_processor = TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[("<BOS>", vocab["<BOS>"]), ("<EOS>", vocab["<EOS>"])]
    )

    tok.save(tokenizer_json_path)



def main():
    parser = argparse.ArgumentParser(description="Build tokenizer files and train/valid corpus from token sequences")
    parser.add_argument("--tokens_path", type=str, default="datasets/baseline/tokens.jsonl", help="Input tokens JSONL file")
    parser.add_argument("--vocab_path", type=str, default="datasets/baseline/vocab.json", help="Input vocabulary JSON file")
    parser.add_argument("--out_dir", type=str, default="datasets/lm_data", help="Output directory")
    parser.add_argument("--valid_ratio", type=float, default=0.05, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--segment_length", type=int, default=4096, help="Split long token sequences into fixed-size segments before train/valid split")
    args = parser.parse_args()

    tokens_path = args.tokens_path
    vocab_path = args.vocab_path

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    seqs = load_tokens_jsonl(tokens_path)
    seqs = segment_long_sequences(seqs, args.segment_length)
    vocab = ensure_special_tokens_in_vocab(load_vocab(vocab_path))


    train_txt = str(out_dir / "train.txt")
    valid_txt = str(out_dir / "valid.txt")
    save_corpus(seqs, train_txt, valid_txt, valid_ratio=args.valid_ratio, seed=args.seed)


    tokenizer_json = str(out_dir / "tokenizer.json")
    build_wordlevel_tokenizer(vocab, tokenizer_json)


    tokenizer_cfg = {
        "model_max_length": 1024,
        "unk_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>"
    }
    with open(out_dir / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_cfg, f, ensure_ascii=False, indent=2)

    special_tokens_map = {
        "unk_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>"
    }
    with open(out_dir / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)

    # Summary
    print("[OK] Written files:")
    print(" -", train_txt)
    print(" -", valid_txt)
    print(" -", tokenizer_json)
    print(" -", str(out_dir / "tokenizer_config.json"))
    print(" -", str(out_dir / "special_tokens_map.json"))
    print("[INFO] Number of sequences =", len(seqs), "Vocabulary size =", len(vocab))

if __name__ == "__main__":
    main()
