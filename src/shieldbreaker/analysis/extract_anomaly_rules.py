#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import torch
from collections import defaultdict
import time
import datetime

# Add parent directory to path in order to import anomaly_detector
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from anomaly_detector import EDRAnomalyDetector


# ----------------------------------------------------------------------
# EDR process list
# ----------------------------------------------------------------------

EDR_PROCESSES = [
    "NisSrv.exe",
    "MsMpEng.exe",
    "SecurityHealthService.exe",
    "SecurityHealthSystray.exe"
]

# Falcon EDR process list (optional)
# EDR_PROCESSES = [
#     "CSFalconService.exe",
#     "CsSystemTray_7.15.18514.0.exe",
#     "CSFalconContainer.exe"
# ]

# Kaspersky EDR process list (optional)
# EDR_PROCESSES = [
#     "avp.exe",
#     "avpui.exe"
# ]


# ----------------------------------------------------------------------
# Target EDR processes for anomaly detection
# Only these processes will be passed into detect_anomalies
# ----------------------------------------------------------------------

ANOMALY_TARGET_EDR_PROCESSES = {
    "NisSrv.exe",
    "MsMpEng.exe",
    "SecurityHealthService.exe",
    "SecurityHealthSystray.exe"
}


# ----------------------------------------------------------------------
# Tee stdout to both console and log file
# ----------------------------------------------------------------------

class TeeStdout:
    def __init__(self, logfile_path: str):
        self.terminal = sys.__stdout__
        self.logfile = open(logfile_path, "a", encoding="utf-8")

        # Optional: write timestamp header into the log
        self.logfile.write("\n" + "=" * 80 + "\n")
        self.logfile.write(f"Run at {datetime.datetime.now().isoformat()}\n")
        self.logfile.write("=" * 80 + "\n")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def is_edr_process(process_name: str) -> bool:
    """Check whether the process is an EDR process."""
    return process_name in EDR_PROCESSES


def is_malware_process(process_name: str) -> bool:
    """Check whether the process is a malware process (starts with virussign.com_)."""
    return process_name.startswith("virussign.com_") and process_name.endswith(".exe")


def load_jsonl_records(jsonl_path: str) -> List[Dict]:
    """
    Load all records from a jsonl file.

    Returns:
        A list of records sorted by GlobalID.
    """
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                records.append(record)
            except json.JSONDecodeError:
                continue

    # Sort by GlobalID (just in case)
    records.sort(key=lambda x: x.get('GlobalID', 0))
    return records


def extract_edr_sequences(
    records: List[Dict],
    window_size: int = None
) -> Dict[str, List[Tuple[int, str]]]:
    """
    Extract full API sequences of EDR processes.

    Returns:
        {process_name: [(global_id, api_name), ...]}
    """
    process_sequences = defaultdict(list)

    for record in records:
        proc_name = record.get('ExtractedProcessName', '')
        if is_edr_process(proc_name):
            global_id = record.get('GlobalID', 0)
            api_name = record.get('Method', '')
            if api_name:
                process_sequences[proc_name].append((global_id, api_name))

    return dict(process_sequences)


# ----------------------------------------------------------------------
# Anomaly detection
# ----------------------------------------------------------------------

def detect_anomalies(
    detector: EDRAnomalyDetector,
    process_sequences: Dict[str, List[Tuple[int, str]]],
    vocab: Dict[str, int],
    threshold: float = 11.50,
    max_seq_len: int = 1024,
    inference_stats: Dict = None
) -> List[Tuple[int, str, str, float]]:
    """
    Detect anomalies using token-level NLL from a GPT (causal LM).

    Core idea:
    - Directly map APIs to token IDs using the training vocabulary
    - Use sliding window to handle long sequences
    - Compute NLL for each token
    - If NLL > threshold → anomaly
    """
    anomalies = []

    if inference_stats is not None:
        inference_stats.setdefault('times', [])
        inference_stats.setdefault('memory_peaks', [])
        inference_stats.setdefault('seq_lengths', [])

    model = detector.model
    device = detector.device
    unk_id = vocab.get("<UNK>", 1)
    bos_id = vocab.get("<BOS>", 2)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    # Process name → PROC token mapping (example: Kaspersky)
    proc_token_map = {
        "avp.exe": "<PROC:avp.exe>",
        "avpui.exe": "<PROC:avpui.exe>",
    }

    # Sliding window parameters
    window_size = max_seq_len - 2  # Reserve space for BOS and PROC
    stride = window_size           # No overlap by default

    for proc_name, api_seq in process_sequences.items():
        if len(api_seq) < 3:
            continue
        if proc_name not in ANOMALY_TARGET_EDR_PROCESSES:
            continue

        try:
            if len(api_seq) <= window_size:
                num_windows = 1
            else:
                num_windows = (len(api_seq) - window_size) // stride + 1
                if (num_windows - 1) * stride + window_size < len(api_seq):
                    num_windows += 1

            detected_anomaly_ids = set()

            for win_idx in range(num_windows):
                start_idx = win_idx * stride
                end_idx = min(start_idx + window_size, len(api_seq))
                window_seq = api_seq[start_idx:end_idx]

                tokens = []
                token_api_index = []

                # <BOS>
                tokens.append(bos_id)
                token_api_index.append(-1)

                # <PROC>
                proc_token = proc_token_map.get(proc_name, f"<PROC:{proc_name.lower()}>")
                proc_id = vocab.get(proc_token, unk_id)
                tokens.append(proc_id)
                token_api_index.append(-1)

                # APIs
                for i, (global_id, api_name) in enumerate(window_seq):
                    api_id = vocab.get(api_name, unk_id)
                    tokens.append(api_id)
                    token_api_index.append(start_idx + i)

                if len(tokens) < 3 or len(tokens) > max_seq_len:
                    continue

                input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

                # Inference timing and memory
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)

                start_time = time.perf_counter()
                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                end_time = time.perf_counter()

                infer_time_ms = (end_time - start_time) * 1000

                logits = outputs.logits[0]          # [T, V]
                pred_logits = logits[:-1]           # [T-1, V]
                targets = input_ids[0, 1:]          # [T-1]
                nlls = loss_fct(pred_logits, targets)

                if inference_stats is not None:
                    inference_stats['times'].append(infer_time_ms)
                    inference_stats['seq_lengths'].append(len(tokens))

                    if torch.cuda.is_available():
                        mem_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                        inference_stats['memory_peaks'].append(mem_peak)

                for t in range(2, len(tokens)):
                    orig_api_idx = token_api_index[t]
                    if orig_api_idx >= 0:
                        nll_value = nlls[t - 1].item()
                        if nll_value > threshold:
                            global_id, api_name = api_seq[orig_api_idx]
                            if global_id not in detected_anomaly_ids:
                                detected_anomaly_ids.add(global_id)
                                anomalies.append(
                                    (global_id, proc_name, api_name, nll_value)
                                )

        except Exception as e:
            print(f"Warning: error processing process {proc_name}: {e}")
            continue

    return anomalies


# ----------------------------------------------------------------------
# Malware API extraction
# ----------------------------------------------------------------------

def find_preceding_malware_apis(
    records: List[Dict],
    edr_global_id: int,
    max_unique: int = 5,
    max_globalid_gap: int = 2000
) -> List[str]:
    """
    New logic (two malware segments):
    1. Scan backward and find the nearest two continuous malware segments.
    2. Use the second (earlier) segment as anchor.
    3. If (edr_global_id - segment_start_gid) > max_globalid_gap → discard.
    4. Extract unique malware APIs from the start of that segment.
    """
    malware_segments = []
    current_segment = []

    for record in reversed(records):
        gid = record.get("GlobalID", 0)
        if gid >= edr_global_id:
            continue

        proc_name = record.get("ExtractedProcessName", "")
        if is_malware_process(proc_name):
            current_segment.append(record)
        else:
            if current_segment:
                malware_segments.append(current_segment)
                current_segment = []
                if len(malware_segments) >= 2:
                    break

    if current_segment and len(malware_segments) < 2:
        malware_segments.append(current_segment)

    if len(malware_segments) < 2:
        return []

    second_segment = malware_segments[1]
    second_segment_start_gid = second_segment[-1].get("GlobalID", 0)

    if (edr_global_id - second_segment_start_gid) > max_globalid_gap:
        return []

    unique_apis = []
    seen_apis = set()

    for record in reversed(second_segment):
        api = record.get("Method", "")
        if not api:
            continue
        if api not in seen_apis:
            seen_apis.add(api)
            unique_apis.append(api)
            if len(unique_apis) >= max_unique:
                break

    return unique_apis


# ----------------------------------------------------------------------
# Rule building and persistence
# ----------------------------------------------------------------------

def build_rules(
    anomalies: List[Tuple[int, str, str, float]],
    records: List[Dict],
    existing_rules: Dict[str, Dict[tuple, int]]
) -> Dict[str, Dict[tuple, int]]:
    """
    Build anomaly rules with counters.

    Returns:
        {edr_anomaly_api: {pattern_tuple: count}}
    """
    new_rules = defaultdict(lambda: defaultdict(int))

    for global_id, proc_name, anomaly_api, perplexity in anomalies:
        malware_apis = find_preceding_malware_apis(
            records,
            global_id,
            max_unique=15,
            max_globalid_gap=1000
        )

        if not malware_apis:
            continue

        rule_tuple = tuple(malware_apis)

        if anomaly_api not in existing_rules or rule_tuple not in existing_rules[anomaly_api]:
            new_rules[anomaly_api][rule_tuple] = 1
            existing_rules[anomaly_api][rule_tuple] = 1
        else:
            existing_rules[anomaly_api][rule_tuple] += 1
            new_rules[anomaly_api][rule_tuple] = existing_rules[anomaly_api][rule_tuple]

    return dict(new_rules)


def load_existing_rules(rule_file: str) -> Dict[str, Dict[tuple, int]]:
    """
    Load existing rules.

    Returns:
        {edr_api: {pattern_tuple: count}}
    """
    rules = defaultdict(lambda: defaultdict(int))

    if not Path(rule_file).exists():
        return rules

    with open(rule_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                edr_api = data.get('edr_anomaly_api', '')
                patterns = data.get('malware_api_patterns', [])
                counts = data.get('pattern_counts', [])

                for pattern_list, count in zip(patterns, counts):
                    pattern = tuple(pattern_list)
                    rules[edr_api][pattern] = count
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    return rules


def save_rules(
    rules: Dict[str, Dict[tuple, int]],
    rule_file: str,
    all_rules: Dict[str, Dict[tuple, int]]
):
    """
    Save rules to file (full rewrite with updated counts).
    """
    with open(rule_file, 'w', encoding='utf-8') as f:
        for edr_api, pattern_counts in sorted(all_rules.items()):
            if pattern_counts:
                patterns_list = []
                counts_list = []

                sorted_patterns = sorted(
                    pattern_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                for pattern_tuple, count in sorted_patterns:
                    patterns_list.append(list(pattern_tuple))
                    counts_list.append(count)

                record = {
                    'edr_anomaly_api': edr_api,
                    'malware_api_patterns': patterns_list,
                    'pattern_counts': counts_list,
                    'total_patterns': len(patterns_list),
                    'total_occurrences': sum(counts_list)
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_vocab(vocab_path: str) -> Dict[str, int]:
    """Load training vocabulary."""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------

def main():
    global EDR_PROCESSES, ANOMALY_TARGET_EDR_PROCESSES

    parser = argparse.ArgumentParser(description="Extract malware behavior to EDR anomaly rules")
    parser.add_argument("--model_path", default="models/out_gpt2", help="Trained GPT model directory")
    parser.add_argument("--vocab_path", default="datasets/baseline/vocab.json", help="Training vocabulary JSON")
    parser.add_argument("--processed_dir", default="datasets/stimulated/processed", help="Processed stimulated JSONL directory")
    parser.add_argument("--output_file", default="runs/rule.jsonl", help="Output rule JSONL file")
    parser.add_argument("--threshold", type=float, default=10.50, help="Token NLL threshold")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum model input length")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--edr_processes", nargs="+", default=EDR_PROCESSES, help="EDR process names")
    parser.add_argument(
        "--target_edr_processes",
        nargs="+",
        default=sorted(ANOMALY_TARGET_EDR_PROCESSES),
        help="EDR process names to score for anomalies",
    )
    args = parser.parse_args()

    EDR_PROCESSES = args.edr_processes
    ANOMALY_TARGET_EDR_PROCESSES = set(args.target_edr_processes)

    print("=" * 60)
    print("EDR + Malware Anomaly Rule Extraction Tool (Sliding Window Version)")
    print("=" * 60)

    model_path = args.model_path
    vocab_path = args.vocab_path
    processed_dir = args.processed_dir
    rule_file = Path(args.output_file)
    rule_file.parent.mkdir(parents=True, exist_ok=True)

    print("\nConfiguration:")
    print(f"  Model path: {model_path}")
    print(f"  Vocabulary path: {vocab_path}")
    print(f"  Data directory: {processed_dir}")
    print(f"  Output file: {rule_file}")
    print(f"  Anomaly threshold: {args.threshold}")
    print(f"  Sliding window: {args.max_seq_len - 2} APIs (model max {args.max_seq_len} tokens)")
    print("-" * 60)

    print("\n[Load] Loading vocabulary...")
    try:
        vocab = load_vocab(vocab_path)
        print(f"  Vocabulary size: {len(vocab)}")
    except Exception as e:
        print(f"Error: failed to load vocabulary - {e}")
        return

    print("\n[Init] Loading anomaly detection model...")
    try:
        detector = EDRAnomalyDetector(model_path, device=args.device)
        print(f"  Device: {detector.device}")
    except Exception as e:
        print(f"Error: failed to load model - {e}")
        return

    model_vocab_size = detector.model.get_input_embeddings().num_embeddings
    print(f"  Model embedding size: {model_vocab_size}")
    if len(vocab) != model_vocab_size:
        print(f"Warning: vocabulary size ({len(vocab)}) does not match model embeddings ({model_vocab_size})")

    print("\n[Load] Loading existing rules...")
    existing_rules = load_existing_rules(str(rule_file))
    existing_pattern_count = sum(len(v) for v in existing_rules.values())
    existing_occurrence_count = sum(sum(counts.values()) for counts in existing_rules.values())
    print(f"  Existing patterns: {existing_pattern_count}")
    print(f"  Existing total occurrences: {existing_occurrence_count}")

    processed_path = Path(processed_dir)
    jsonl_files = sorted(processed_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"Error: no .jsonl files found in {processed_dir}")
        return

    print(f"\n[Found] {len(jsonl_files)} files to process:")
    for f in jsonl_files:
        print(f"  - {f.name}")

    print("\n" + "=" * 60)
    print("Start processing...")
    print("=" * 60)

    total_new_rules = 0
    inference_stats = {'times': [], 'memory_peaks': [], 'seq_lengths': []}

    for i, jsonl_file in enumerate(jsonl_files, 1):
        print(f"\nProgress: [{i}/{len(jsonl_files)}] {jsonl_file.name}")

        try:
            records = load_jsonl_records(str(jsonl_file))
            sequences = extract_edr_sequences(records)

            anomalies = detect_anomalies(
                detector,
                sequences,
                vocab,
                threshold=args.threshold,
                max_seq_len=args.max_seq_len,
                inference_stats=inference_stats
            )

            new_rules = build_rules(anomalies, records, existing_rules)

            if new_rules:
                save_rules(new_rules, str(rule_file), existing_rules)
                pattern_count = sum(len(v) for v in new_rules.values())
                occurrence_count = sum(sum(counts.values()) for counts in new_rules.values())
                total_new_rules += occurrence_count
                print(f"  ✓ Updated {pattern_count} patterns, total occurrences {occurrence_count}")
            else:
                print("  - No new rules found")

        except Exception as e:
            print(f"  ✗ Failed to process file: {e}")
            continue

    print("\n" + "=" * 60)
    print("Processing completed!")
    print("=" * 60)
    print(f"Processed files: {len(jsonl_files)}")
    print(f"New/updated rule occurrences in this run: {total_new_rules}")
    print(f"Rule file: {rule_file.absolute()}")


if __name__ == "__main__":
    main()
