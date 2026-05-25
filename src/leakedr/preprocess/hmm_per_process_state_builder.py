#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from hmmlearn.hmm import CategoricalHMM

from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# =========================
# Configuration
# =========================

TARGET_PROCS = {
    "csfalconservice.exe",
    "cssystemtray_7.15.18514.0.exe",
    "csfalconcontainer.exe",
}

# # Windows Defender EDR process name list
# TARGET_PROCS = [
#     "nissrv.exe",
#     "msmpeng.exe",
#     "securityhealthservice.exe",
#     "securityhealthsystray.exe"
# ]

# # Kaspersky EDR process name list
# TARGET_PROCS = [
#     "avp.exe",
#     "avpui.exe"
# ]

WINDOW_SIZE = 50
N_STATES = 100
N_ITER = 100
MERGE_THRESHOLD = 0.15

PROC_PREFIX = "<PROC:"
PROC_SUFFIX = ">"


# =========================
# Basic utilities
# =========================

def normalize_proc_name(name):
    if not name:
        return "unknown.exe"
    name = name.replace("\\", "/").split("/")[-1]
    return name.lower().strip()


def normalize_method(method):
    if not method:
        return "UnknownMethod"
    return re.sub(r"\s+", "", method.strip())


def make_proc_token(proc):
    return f"{PROC_PREFIX}{proc}{PROC_SUFFIX}"


# =========================
# Data loading
# =========================

def load_events(jsonl_path):
    """
    Returns:
        dict(proc -> [(timestamp, api), ...])
    """
    proc_events = defaultdict(list)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                ts = float(obj.get("TimeStamp", 0))
                proc = normalize_proc_name(obj.get("ProcessName"))
                if proc not in TARGET_PROCS:
                    continue
                api = normalize_method(obj.get("Method"))
                proc_events[proc].append((ts, api))
            except Exception:
                continue

    # Sort events by timestamp for each process
    for proc in proc_events:
        proc_events[proc].sort(key=lambda x: x[0])

    return proc_events


# =========================
# Window splitting
# =========================

def split_windows(events, window_size):
    """
    Split events into fixed-size windows.
    Only keep windows whose size is at least half of window_size.
    """
    windows = []
    for i in range(0, len(events), window_size):
        w = events[i:i + window_size]
        if len(w) >= window_size // 2:
            windows.append(w)
    return windows


# =========================
# Encoding
# =========================

def build_vocab(all_events):
    """
    Build API vocabulary: api -> id
    """
    apis = set()
    for events in all_events.values():
        for _, api in events:
            apis.add(api)
    return {api: idx for idx, api in enumerate(sorted(apis))}


def encode_windows(windows, vocab):
    """
    Encode windows into numeric observations for HMM.
    """
    obs = []
    lengths = []

    for w in windows:
        encoded = [vocab[api] for _, api in w]
        obs.extend(encoded)
        lengths.append(len(encoded))

    return np.array(obs, dtype=np.int32).reshape(-1, 1), lengths


# =========================
# HMM
# =========================

def train_hmm(observations, lengths, n_states, n_iter, n_features):
    model = CategoricalHMM(
        n_components=n_states,
        n_features=n_features,
        n_iter=n_iter,
        random_state=42,
        verbose=False
    )
    model.fit(observations, lengths)
    return model


# =========================
# State utilities
# =========================

def dominant_state_by_longest_run(states):
    """
    Determine the dominant state by the longest continuous run.
    """
    current = states[0]
    length = 1
    runs = []

    for s in states[1:]:
        if s == current:
            length += 1
        else:
            runs.append((current, length))
            current = s
            length = 1

    runs.append((current, length))
    return max(runs, key=lambda x: x[1])[0]


def predict_windows(model, windows, vocab):
    """
    Predict hidden states for each window and find the dominant state.
    """
    results = []

    for w in windows:
        encoded = np.array([vocab[api] for _, api in w], dtype=np.int32).reshape(-1, 1)
        states = model.predict(encoded)
        dom_state = dominant_state_by_longest_run(states.tolist())
        results.append((dom_state, states.tolist()))

    return results


# =========================
# Post-hoc state merging
# =========================

def merge_hmm_states(model, threshold, eps=1e-12):
    """
    Merge latent HMM states into effective states using Jensen-Shannon
    divergence and hierarchical clustering.
    """
    emissions = model.emissionprob_.copy()  # (N_STATES, N_FEATURES)
    n_states, n_features = emissions.shape

    # ============================================================
    # 1. Mark valid states
    # Conditions:
    #   - No NaN / Inf
    #   - Sum of probabilities > 0 (state was actually visited)
    # ============================================================
    valid_mask = (
        np.isfinite(emissions).all(axis=1) &
        (emissions.sum(axis=1) > 0)
    )

    valid_states = np.where(valid_mask)[0]
    invalid_states = np.where(~valid_mask)[0]

    # Extreme case: only 0 or 1 valid state
    if len(valid_states) <= 1:
        # Map all states to effective state 0
        return {i: 0 for i in range(n_states)}

    # ============================================================
    # 2. Extract and fix emissions for valid states
    # ============================================================
    emissions_valid = emissions[valid_states]

    # Numerical stability fix
    emissions_valid = emissions_valid + eps
    emissions_valid = emissions_valid / emissions_valid.sum(
        axis=1, keepdims=True
    )

    # ============================================================
    # 3. Compute Jensen-Shannon divergence distance matrix
    # ============================================================
    k = emissions_valid.shape[0]
    D = np.zeros((k, k), dtype=np.float64)

    for i in range(k):
        for j in range(k):
            D[i, j] = jensenshannon(emissions_valid[i], emissions_valid[j])

    # Safety: eliminate any NaN / Inf
    D = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)

    # ============================================================
    # 4. Hierarchical clustering → effective states
    # ============================================================
    condensed_D = squareform(D, checks=False)
    Z = linkage(condensed_D, method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")

    # labels are 1-based, remap to 0-based
    uniq = sorted(set(labels))
    remap = {lab: idx for idx, lab in enumerate(uniq)}

    # ============================================================
    # 5. Build full mapping: latent → effective
    # ============================================================
    state_map = {}

    # Valid states: use clustering result
    for latent_id, lab in zip(valid_states, labels):
        state_map[int(latent_id)] = remap[int(lab)]

    # Invalid states: map to fallback (0)
    fallback = 0
    for latent_id in invalid_states:
        state_map[int(latent_id)] = fallback

    return state_map


# =========================
# GPT export
# =========================

def export_for_gpt(proc_windows, proc_states, proc_state_maps, out_path):
    """
    Export token sequences for GPT-style models.
    Each line format:
        <PROC:xxx.exe> <E_STATE_k> api1 api2 api3 ...
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for proc in proc_windows:
            proc_token = make_proc_token(proc)
            state_map = proc_state_maps[proc]

            for (dom_state, _), w in zip(proc_states[proc], proc_windows[proc]):
                eff_state = state_map[dom_state]
                tokens = [proc_token, f"<E_STATE_{eff_state}>"]
                tokens.extend(api for _, api in w)
                f.write(" ".join(tokens) + "\n")


# =========================
# Main pipeline
# =========================

def main():
    input_file = "datasets/main/baseline/faclon_edr_only.jsonl"
    output_file = "faclon_input_proc_effective_state.txt"

    print("=" * 70)
    print("Per-Process Categorical-HMM with Post-hoc State Merging")
    print("=" * 70)

    # 1. Load data
    proc_events = load_events(input_file)
    for proc, evs in proc_events.items():
        print(f"[LOAD] {proc}: {len(evs)} events")

    # 2. Build vocabulary
    vocab = build_vocab(proc_events)
    n_features = len(vocab)
    print(f"[VOCAB] size={n_features}")

    proc_windows = {}
    proc_states = {}
    proc_state_maps = {}

    # 3. Train one HMM per process
    for proc, events in proc_events.items():
        print(f"\n[PROCESS] {proc}")

        windows = split_windows(events, WINDOW_SIZE)
        print(f"  windows={len(windows)}")

        if len(windows) < 2:
            print("  ⚠ Too few windows, skipping")
            continue

        obs, lengths = encode_windows(windows, vocab)

        model = train_hmm(
            observations=obs,
            lengths=lengths,
            n_states=N_STATES,
            n_iter=N_ITER,
            n_features=n_features
        )

        # HMM prediction
        states = predict_windows(model, windows, vocab)

        # Post-hoc state merging
        state_map = merge_hmm_states(model, MERGE_THRESHOLD)
        eff_state_count = len(set(state_map.values()))

        print(f"  latent states={N_STATES} → effective states={eff_state_count}")

        proc_windows[proc] = windows
        proc_states[proc] = states
        proc_state_maps[proc] = state_map

    # 4. Export
    export_for_gpt(proc_windows, proc_states, proc_state_maps, output_file)

    print(f"\n✓ GPT input file generated: {Path(output_file).absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
