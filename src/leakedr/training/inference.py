#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference using a trained GPT2 model
"""

import argparse
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel


def main():
    parser = argparse.ArgumentParser(description="GPT2 Model Inference")
    parser.add_argument("--model_path", type=str, default="models/out_gpt2_xl", help="Path to trained model")
    parser.add_argument("--tokenizer_path", type=str, default="datasets/lm_data", help="Path to tokenizer")
    parser.add_argument("--prompt", type=str, default="<BOS> <PROC:msmpeng.exe>", help="Input prompt")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-P sampling")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="Number of generated sequences")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Device selection
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("GPT2 Model Inference")
    print("=" * 60)
    print(f"Model Path : {args.model_path}")
    print(f"Tokenizer  : {args.tokenizer_path}")
    print(f"Device     : {device}")
    print(f"Prompt     : {args.prompt}")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Load model
    print("\nLoading model...")
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")
    
    # Encode input
    print("\nGenerating sequences...")
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    
    # Generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode outputs
    print("\n" + "=" * 60)
    print("Generated Results")
    print("=" * 60)
    
    for i, output in enumerate(outputs):
        text = tokenizer.decode(output, skip_special_tokens=False)
        print(f"\nSequence {i + 1}:")
        print("-" * 60)
        print(text)
        print("-" * 60)
        
        # Count tokens
        tokens = tokenizer.tokenize(text)
        print(f"Number of tokens: {len(tokens)}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
