#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import json
import math
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class PerplexityLoggerCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()
        self.history = {
            "steps": [],
            "eval_loss": [],
            "eval_perplexity": [],
            "elapsed_time_sec": [],
            "best_eval_loss": None,
            "best_eval_perplexity": None,
            "best_step": None,
        }

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or "eval_loss" not in metrics:
            return

        step = state.global_step
        eval_loss = metrics["eval_loss"]
        ppl = math.exp(eval_loss) if eval_loss < 100 else float("inf")
        elapsed = time.time() - self.start_time

        self.history["steps"].append(step)
        self.history["eval_loss"].append(eval_loss)
        self.history["eval_perplexity"].append(ppl)
        self.history["elapsed_time_sec"].append(elapsed)

        best_loss = self.history["best_eval_loss"]
        if best_loss is None or eval_loss < best_loss:
            self.history["best_eval_loss"] = eval_loss
            self.history["best_eval_perplexity"] = ppl
            self.history["best_step"] = step


def build_model(tokenizer, args):
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.n_inner or (4 * args.n_embd),
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    return GPT2LMHeadModel(config)


def main():
    parser = argparse.ArgumentParser(description="Train a small decoder-only Transformer from scratch")

    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer path or name")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data file")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to validation data file")
    parser.add_argument("--output_dir", type=str, default="models/out_transformer", help="Output directory")

    parser.add_argument("--block_size", type=int, default=256, help="Sequence block size")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=256, help="Hidden size")
    parser.add_argument("--n_inner", type=int, default=0, help="MLP inner size; 0 means 4 * n_embd")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout for embeddings, residuals, and attention")

    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=40, help="Maximum training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=None, help="Checkpoint save interval; defaults to eval_steps")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision")
    parser.add_argument("--early_stopping_patience", type=int, default=4, help="Stop after this many evals without improvement")
    parser.add_argument("--early_stopping_threshold", type=float, default=5e-4, help="Minimum eval_loss improvement to reset patience")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max(tokenizer.model_max_length, 10**9)

    raw_datasets = load_dataset(
        "text",
        data_files={
            "train": args.train_file,
            "validation": args.validation_file,
        },
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=False,
            add_special_tokens=True,
        )

    tokenized = raw_datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples}
        total_len = len(concatenated["input_ids"])
        total_len = (total_len // args.block_size) * args.block_size
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_len, args.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized.map(group_texts, batched=True)

    model = build_model(tokenizer, args)
    model.config.pad_token_id = tokenizer.pad_token_id
    model_param_count = model.num_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    save_steps = args.save_steps or args.eval_steps

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps" if args.do_eval else "no",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps" if args.do_train else "no",
        save_steps=save_steps,
        fp16=args.fp16,
        save_total_limit=2,
        load_best_model_at_end=bool(args.do_train and args.do_eval),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": lm_datasets["train"] if args.do_train else None,
        "eval_dataset": lm_datasets["validation"] if args.do_eval else None,
        "data_collator": data_collator,
    }

    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    ppl_callback = PerplexityLoggerCallback()
    trainer.add_callback(ppl_callback)
    if args.do_train and args.do_eval:
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if args.do_train:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)

    with open(Path(args.output_dir) / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(ppl_callback.history, f, indent=2)

    resource = {
        "model_parameter_count": model_param_count,
    }
    if torch.cuda.is_available():
        resource["gpu_peak_memory_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)

    with open(Path(args.output_dir) / "resource_usage.json", "w", encoding="utf-8") as f:
        json.dump(resource, f, indent=2)

    model_info = {
        "model_type": "transformer_decoder_from_scratch",
        "vocab_size": len(tokenizer),
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "n_inner": args.n_inner or (4 * args.n_embd),
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "best_eval_loss": ppl_callback.history["best_eval_loss"],
        "best_eval_perplexity": ppl_callback.history["best_eval_perplexity"],
        "best_step": ppl_callback.history["best_step"],
        "model_parameter_count": model_param_count,
    }
    with open(Path(args.output_dir) / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2)

    print("Training finished.")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
