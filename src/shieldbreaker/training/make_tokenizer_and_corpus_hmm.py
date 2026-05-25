#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a tokenizer and corpus from HMM-processed sequences
"""

import json
import random
from pathlib import Path
from collections import Counter

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


def load_hmm_sequences(path: str):

    seqs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) > 0:
                seqs.append(tokens)
    return seqs


def build_vocab_from_sequences(seqs):

    token_counter = Counter()
    for seq in seqs:
        token_counter.update(seq)
    
    vocab = {}
    

    for i, token in enumerate(SPECIAL_TOKENS):
        vocab[token] = i
    

    sorted_tokens = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
    for token, _ in sorted_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    
    return vocab


def save_corpus(seqs, out_train: str, out_valid: str, valid_ratio: float = 0.05, seed: int = 42):

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
    
    return len(train), len(valid)


def build_wordlevel_tokenizer(vocab: dict, tokenizer_json_path: str):

    model = WordLevel(vocab=vocab, unk_token="<UNK>")
    tok = Tokenizer(model)
    tok.pre_tokenizer = WhitespaceSplit()
    

    tok.post_processor = TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[("<BOS>", vocab["<BOS>"]), ("<EOS>", vocab["<EOS>"])]
    )
    
    tok.save(tokenizer_json_path)


def analyze_vocab(vocab: dict):

    proc_tokens = [t for t in vocab.keys() if t.startswith("<PROC:")]
    state_tokens = [t for t in vocab.keys() if t.startswith("<E_STATE_")]
    api_tokens = [t for t in vocab.keys() if not t.startswith("<") or t in SPECIAL_TOKENS]
    
    print("\nVocabulary statistics:")
    print(f"  Special tokens:  {len([t for t in vocab.keys() if t in SPECIAL_TOKENS])}")
    print(f"  Process tokens:  {len(proc_tokens)}")
    print(f"  State tokens:    {len(state_tokens)}")
    print(f"  API tokens:      {len(api_tokens) - len(SPECIAL_TOKENS)}")
    print(f"  Total:           {len(vocab)}")
    
    if proc_tokens:
        print(f"\nProcess token examples (first 5): {proc_tokens[:5]}")
    if state_tokens:
        print(f"State token examples (first 5): {state_tokens[:5]}")
    if len(api_tokens) > len(SPECIAL_TOKENS):
        other_apis = [t for t in api_tokens if t not in SPECIAL_TOKENS]
        print(f"API token examples (first 5):      {other_apis[:5]}")


def main():

    input_file = "../preprocess/kaba_input_proc_effective_state.txt"
    # Output directory
    out_dir = Path("lm_data_hmm")
    out_dir.mkdir(exist_ok=True)
    

    valid_ratio = 0.05
    
    print("=" * 70)
    print("Building Tokenizer and Corpus from HMM sequences")
    print("=" * 70)
    

    print(f"\n[1/5] Loading HMM sequences: {input_file}")
    seqs = load_hmm_sequences(input_file)
    print(f"  Number of sequences: {len(seqs)}")
    print(f"  Total tokens: {sum(len(seq) for seq in seqs)}")
    
    if len(seqs) > 0:
        print(f"  Average sequence length: {sum(len(seq) for seq in seqs) / len(seqs):.1f}")
        print(f"  Shortest sequence: {min(len(seq) for seq in seqs)}")
        print(f"  Longest sequence:  {max(len(seq) for seq in seqs)}")
    

    print("\n[2/5] Building vocabulary...")
    vocab = build_vocab_from_sequences(seqs)
    analyze_vocab(vocab)
    

    vocab_path = out_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"\n  Vocabulary saved: {vocab_path}")
    

    print(f"\n[3/5] Splitting corpus (validation ratio: {valid_ratio:.1%})...")
    train_txt = str(out_dir / "train.txt")
    valid_txt = str(out_dir / "valid.txt")
    
    n_train, n_valid = save_corpus(seqs, train_txt, valid_txt, valid_ratio)
    print(f"  Training set:   {n_train} sequences -> {train_txt}")
    print(f"  Validation set: {n_valid} sequences -> {valid_txt}")
    

    print("\n[4/5] Building tokenizer...")
    tokenizer_json = str(out_dir / "tokenizer.json")
    build_wordlevel_tokenizer(vocab, tokenizer_json)
    print(f"  Tokenizer saved: {tokenizer_json}")
    

    print("\n[5/5] Saving configuration...")
    tokenizer_cfg = {
        "model_max_length": 1024,
        "unk_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>"
    }
    config_path = out_dir / "tokenizer_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_cfg, f, ensure_ascii=False, indent=2)
    print(f"  Tokenizer config saved: {config_path}")
    

    special_tokens_map = {
        "unk_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>"
    }
    special_path = out_dir / "special_tokens_map.json"
    with open(special_path, "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)
    print(f"  Special tokens map saved: {special_path}")
    

    print("\n" + "=" * 70)
    print("✓ Completed! Generated files:")
    print(f"  - {train_txt}")
    print(f"  - {valid_txt}")
    print(f"  - {vocab_path}")
    print(f"  - {tokenizer_json}")
    print(f"  - {config_path}")
    print(f"  - {special_path}")
    print("=" * 70)
    print("\nUsage examples:")
    print(f"  python train_gpt2.py --tokenizer_name {out_dir} --train_file {train_txt} --validation_file {valid_txt} ...")
    print(f"  python train_lstm.py --vocab_file {vocab_path} --train_file {train_txt} --validation_file {valid_txt} ...")
    print(f"  python train_ngram.py --train_file {train_txt} --validation_file {valid_txt} ...")
    print("=" * 70)


if __name__ == "__main__":
    main()
