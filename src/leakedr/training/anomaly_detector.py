#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, GPT2LMHeadModel
from typing import List, Tuple


class EDRAnomalyDetector:
    """
    EDR API Sequence Anomaly Detector based on GPT Perplexity.
    Supports automatic threshold learning from normal data.
    """

    def __init__(self, model_path: str, tokenizer_path: str = None, device: str = None):
        print(f"[INIT] Loading model from: {model_path}")
        self.model = GPT2LMHeadModel.from_pretrained(model_path)

        if tokenizer_path is None:
            tokenizer_path = model_path

        print(f"[INIT] Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[INIT] Device: {self.device}")
        print(f"[INIT] Vocabulary size: {len(self.tokenizer)}")
        print("[INIT] Model and tokenizer loaded successfully.\n")

    # ---------------------------------------------------------
    # Core metrics
    # ---------------------------------------------------------

    def calculate_perplexity(self, sequence: str) -> float:
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)

        return perplexity.item()

    def predict_next_api(self, sequence: str, top_k: int = 5) -> List[Tuple[str, float]]:
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits[0, -1], dim=-1)

        top_probs, top_ids = torch.topk(probs, top_k)

        predictions = []
        for prob, token_id in zip(top_probs, top_ids):
            token = self.tokenizer.decode([token_id])
            predictions.append((token.strip(), prob.item()))

        return predictions


    def estimate_threshold(
        self,
        normal_sequences: List[str],
        percentile: float = 95.0,
        verbose: bool = True
    ) -> float:
        """
        Automatically learn anomaly detection threshold from normal sequences.

        Args:
            normal_sequences: List of normal GPT input sequences.
            percentile: Percentile to use as threshold (e.g., 95 or 99).
            verbose: Print statistics.

        Returns:
            Learned threshold value.
        """
        print("=" * 60)
        print("[THRESHOLD LEARNING] Starting automatic threshold estimation")
        print(f"[THRESHOLD LEARNING] Number of normal sequences: {len(normal_sequences)}")
        print(f"[THRESHOLD LEARNING] Percentile: {percentile}")
        print("=" * 60)

        perplexities = []

        for i, seq in enumerate(normal_sequences, 1):
            ppl = self.calculate_perplexity(seq)
            perplexities.append(ppl)

            if verbose and i % 10 == 0:
                print(f"[THRESHOLD LEARNING] Processed {i}/{len(normal_sequences)} sequences")

        perplexities = np.array(perplexities)

        mean_ppl = perplexities.mean()
        std_ppl = perplexities.std()
        min_ppl = perplexities.min()
        max_ppl = perplexities.max()
        threshold = np.percentile(perplexities, percentile)

        print("\n[THRESHOLD LEARNING] Statistics:")
        print(f"  Mean Perplexity : {mean_ppl:.2f}")
        print(f"  Std  Perplexity : {std_ppl:.2f}")
        print(f"  Min  Perplexity : {min_ppl:.2f}")
        print(f"  Max  Perplexity : {max_ppl:.2f}")
        print(f"  ==> Learned Threshold ({percentile}th percentile): {threshold:.2f}")
        print("=" * 60)

        return float(threshold)


    def detect_anomaly(self, sequence: str, threshold: float) -> Tuple[bool, float, str]:
        perplexity = self.calculate_perplexity(sequence)

        if perplexity > threshold:
            if perplexity > threshold * 5:
                level = "[HIGH RISK]"
            else:
                level = "[SUSPICIOUS]"
            message = f"{level} Anomalous sequence detected! Perplexity: {perplexity:.2f}"
            return True, perplexity, message
        else:
            message = f"[NORMAL] Sequence is normal. Perplexity: {perplexity:.2f}"
            return False, perplexity, message

    def analyze_sequence(
        self,
        api_calls: List[str],
        process: str,
        threshold: float,
        verbose: bool = True
    ):
        sequence = f"<BOS> <PROC:{process}> " + " ".join(api_calls)

        if verbose:
            print("=" * 60)
            print("[SEQUENCE ANALYSIS]")
            print("=" * 60)
            print(f"Process   : {process}")
            print(f"API Calls : {' -> '.join(api_calls)}")
            print("-" * 60)

        is_anomaly, perplexity, message = self.detect_anomaly(sequence, threshold)
        print(message)

        if verbose:
            print("\n[Next API Prediction]")
            predictions = self.predict_next_api(sequence, top_k=5)
            for i, (api, prob) in enumerate(predictions, 1):
                print(f"  {i}. {api:30s} (Probability: {prob:.2%})")

            print("\n[Step-wise Perplexity Analysis]")
            for i in range(1, len(api_calls) + 1):
                partial_seq = f"<BOS> <PROC:{process}> " + " ".join(api_calls[:i])
                ppl = self.calculate_perplexity(partial_seq)
                status = "ALERT" if ppl > threshold else "OK"
                print(f"  [{status}] Step {i}: {api_calls[i-1]:30s} (Perplexity: {ppl:.2f})")

        print("=" * 60)
        return is_anomaly, perplexity


# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------

def build_sequences_from_api_calls(api_sequences: List[List[str]], process: str) -> List[str]:
    """
    Convert API call lists into GPT input format.
    """
    sequences = []
    for apis in api_sequences:
        seq = f"<BOS> <PROC:{process}> " + " ".join(apis)
        sequences.append(seq)
    return sequences


# ---------------------------------------------------------
# Example main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EDR API Sequence Anomaly Detector with Automatic Threshold Learning")
    parser.add_argument("--model_path", type=str, default="models/out_gpt2_xl", help="Model path")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path (optional)")
    parser.add_argument("--process", type=str, default="msmpeng.exe", help="Process name")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    args = parser.parse_args()

    detector = EDRAnomalyDetector(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device
    )

    # Example: normal API sequences (from baseline dataset)
    normal_api_sequences = [
        ["NtOpenKey", "NtQueryValueKey", "NtClose"],
        ["NtCreateFile", "NtReadFile", "NtQueryInformationFile", "NtClose"],
        ["NtOpenProcess", "NtQueryInformationProcess", "NtClose"],
        ["NtCreateFile", "NtReadFile", "NtClose"],
        ["NtOpenKey", "NtSetValueKey", "NtClose"],
    ]

    # Build GPT input sequences
    normal_sequences = build_sequences_from_api_calls(
        normal_api_sequences,
        args.process
    )

    # Learn threshold automatically
    auto_threshold = detector.estimate_threshold(
        normal_sequences,
        percentile=95
    )

    print(f"\n[FINAL] Learned anomaly detection threshold: {auto_threshold:.2f}\n")

    # Test a suspicious sequence
    test_apis = [
        "NtOpenProcess",
        "NtAllocateVirtualMemory",
        "NtWriteVirtualMemory",
        "NtCreateRemoteThread"
    ]

    detector.analyze_sequence(
        api_calls=test_apis,
        process=args.process,
        threshold=auto_threshold,
        verbose=True
    )


if __name__ == "__main__":
    main()
