"""Train a GPT-2 API baseline without joining independent executions."""

import argparse
import inspect
import math
import shutil
import json
import hashlib
from pathlib import Path
import time

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from shieldbreaker.traces import write_json


def chunk_corpus(path, tokenizer, block_size):
    if block_size < 3:
        raise ValueError("block_size must be at least three")
    examples = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        tokens = [token for token in line.split() if token not in {"<BOS>", "<EOS>"}]
        if any(token.startswith("<PROC:") for token in tokens):
            raise ValueError(
                "Legacy process tokens are not part of the API-only baseline"
            )
        ids = [
            tokenizer.get_vocab().get(token, tokenizer.unk_token_id) for token in tokens
        ]
        for start in range(0, len(ids), block_size - 2):
            chunk = [
                tokenizer.bos_token_id,
                *ids[start : start + block_size - 2],
                tokenizer.eos_token_id,
            ]
            examples.append({"input_ids": chunk, "attention_mask": [1] * len(chunk)})
    if not examples:
        raise ValueError(f"Empty corpus: {path}")
    return Dataset.from_list(examples)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name_or_path", default="gpt2")
    p.add_argument("--tokenizer_name", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--validation_file", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=float, default=20)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--use_cpu", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--resume_from_checkpoint")
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_eval", action="store_true")
    p.add_argument(
        "--from_scratch",
        action="store_true",
        help="Initialize a new GPT-2 rather than download a checkpoint",
    )
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=12)
    args = p.parse_args()
    if Path(args.train_file).resolve() == Path(args.validation_file).resolve():
        raise ValueError("Training and validation must be independent files/executions")
    provenance = Path(args.tokenizer_name) / "split_manifest.json"
    hashes_path = Path(args.tokenizer_name) / "corpus_hashes.json"
    if not provenance.is_file() or not hashes_path.is_file():
        raise ValueError(
            "Build an execution-preserving tokenizer/corpus before training"
        )
    hashes = json.loads(hashes_path.read_text())
    for name, corpus in [
        ("train.txt", args.train_file),
        ("valid.txt", args.validation_file),
    ]:
        if hashlib.sha256(Path(corpus).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f"{name} does not match the declared corpus split")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if args.from_scratch:
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tokenizer),
                n_positions=args.block_size,
                n_embd=args.hidden_size,
                n_layer=args.layers,
                n_head=args.heads,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        )
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
        model.resize_token_embeddings(len(tokenizer))
    if args.block_size > model.config.max_position_embeddings:
        raise ValueError("block_size exceeds GPT-2 context size")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = not args.gradient_checkpointing
    training = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps or args.eval_steps,
        fp16=args.fp16,
        seed=args.seed,
        save_total_limit=2,
        report_to=[],
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy="steps" if args.do_train else "no",
        use_cpu=args.use_cpu,
    )
    strategy_key = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    training[strategy_key] = "steps" if args.do_eval else "no"
    trainer_kwargs = dict(
        model=model,
        args=TrainingArguments(**training),
        train_dataset=chunk_corpus(args.train_file, tokenizer, args.block_size)
        if args.do_train
        else None,
        eval_dataset=chunk_corpus(args.validation_file, tokenizer, args.block_size)
        if args.do_eval
        else None,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer_kwargs[
        "processing_class"
        if "processing_class" in inspect.signature(Trainer).parameters
        else "tokenizer"
    ] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if args.do_train:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        shutil.copyfile(provenance, Path(args.output_dir) / "split_manifest.json")
        shutil.copyfile(hashes_path, Path(args.output_dir) / "corpus_hashes.json")
    metrics = trainer.evaluate() if args.do_eval else {}
    if "eval_loss" in metrics:
        metrics["eval_perplexity"] = math.exp(min(metrics["eval_loss"], 700))
    write_json(
        Path(args.output_dir) / "resource_usage.json",
        {
            "model": "gpt2",
            "elapsed_seconds": time.perf_counter() - started,
            "gpu_peak_memory_mb": torch.cuda.max_memory_allocated() / 1024**2
            if torch.cuda.is_available()
            else None,
            "scope": "model training/evaluation; excludes sandbox collection and semantic reconstruction",
            "seed": args.seed,
            "metrics": metrics,
        },
    )


if __name__ == "__main__":
    main()
