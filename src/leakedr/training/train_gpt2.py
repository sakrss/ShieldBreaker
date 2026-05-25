#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import argparse
import inspect
import json
import math
import time
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
)
from datasets import load_dataset


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return None


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


def main():
    parser = argparse.ArgumentParser(description="Train GPT2 / GPT2-XL LM")
    
    # Data & model
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", help="Pretrained model path or name")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer path or name")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data file")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to validation data file")

    # Training parameters
    parser.add_argument("--output_dir", type=str, default="models/out_gpt2", help="Output directory")
    parser.add_argument("--block_size", type=int, default=256, help="Sequence block size")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Maximum training epochs")
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=None, help="Checkpoint save interval; defaults to eval_steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from a checkpoint directory")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision")
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Optimizer name, e.g. adamw_torch or adafactor")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to reduce activation memory")
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Force model weights to a specific dtype at load time",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa"],
        help="Attention backend to request from Transformers",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Stop after this many evals without improvement")
    parser.add_argument("--early_stopping_threshold", type=float, default=5e-4, help="Minimum eval_loss improvement to reset patience")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")

    args = parser.parse_args()
    if args.fp16 and args.model_dtype == "float16":
        raise ValueError(
            "--fp16 cannot be combined with --model_dtype float16 in this environment. "
            "Use fp16 mixed precision with default model weights, or drop --fp16 for pure float16 weights."
        )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # We chunk long sequences after tokenization, so avoid noisy warnings here.
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
            truncation=False,      # Do not truncate here
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


    model_kwargs = {}
    model_dtype = resolve_torch_dtype(args.model_dtype)
    if model_dtype is not None:
        model_kwargs["torch_dtype"] = model_dtype
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    pretrained_signature = inspect.signature(GPT2LMHeadModel.from_pretrained)
    if "low_cpu_mem_usage" in pretrained_signature.parameters:
        model_kwargs["low_cpu_mem_usage"] = True

    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, **model_kwargs)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()


    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    save_steps = args.save_steps or args.eval_steps

    training_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "eval_strategy": "steps" if args.do_eval else "no",
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps" if args.do_train else "no",
        "save_steps": save_steps,
        "fp16": args.fp16,
        "save_total_limit": 2,
        "load_best_model_at_end": bool(args.do_train and args.do_eval),
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": [],
    }

    training_signature = inspect.signature(TrainingArguments)
    if "optim" in training_signature.parameters:
        training_kwargs["optim"] = args.optim
    if "gradient_checkpointing" in training_signature.parameters:
        training_kwargs["gradient_checkpointing"] = args.gradient_checkpointing
    if "gradient_checkpointing_kwargs" in training_signature.parameters and args.gradient_checkpointing:
        training_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    training_args = TrainingArguments(**training_kwargs)

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
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)

    with open(Path(args.output_dir) / "training_history.json", "w") as f:
        json.dump(ppl_callback.history, f, indent=2)

    resource = {}
    if torch.cuda.is_available():
        resource["gpu_peak_memory_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
    resource["optim"] = args.optim
    resource["gradient_checkpointing"] = args.gradient_checkpointing
    resource["model_dtype"] = args.model_dtype
    resource["attn_implementation"] = args.attn_implementation

    with open(Path(args.output_dir) / "resource_usage.json", "w") as f:
        json.dump(resource, f, indent=2)

    print("Training finished.")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
