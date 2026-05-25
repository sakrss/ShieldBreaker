#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download GPT-2 XL with safetensors format.
This script avoids torch.load (.bin) and is safe for torch < 2.6.
"""

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import os

MODEL_NAME = "gpt2-xl"
CACHE_DIR = "models/pretrained"

def main():
    print("=" * 60)
    print("Downloading GPT-2 XL (safetensors)")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Cache dir: {os.path.abspath(CACHE_DIR)}")
    print("use_safetensors = True")
    print("=" * 60)


    print("\n[1/2] Downloading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
    )
    print("Tokenizer downloaded.")
    print(f"Vocab size: {len(tokenizer)}")


    print("\n[2/2] Downloading model weights (safetensors)...")
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        use_safetensors=True,
    )

    print("\nDownload finished successfully!")
    print(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")


    model_dir = os.path.join(CACHE_DIR, MODEL_NAME)
    print(f"Model stored at: {os.path.abspath(model_dir)}")


    print("\nFiles in model directory:")
    for f in os.listdir(model_dir):
        print("  ", f)


if __name__ == "__main__":
    main()
