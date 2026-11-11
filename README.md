# ShieldBreaker

Differential characterization of endpoint-security runtime behavior. The analysis
unit is a recurrent **behavior–anomaly temporal co-occurrence**, not an inferred
vendor detection rule or a persistent cross-execution response identity.

This implementation aligns the methods, parameters and workflows with the paper.
Reproducing its historical numerical result cells is outside the alignment scope.

## Install

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[lm,test]'
pytest -q
```

The statistical/synthetic tools need only `pip install -e .`. GPT-2 and LSTM
require the `lm` extra. CUDA-enabled PyTorch can be installed separately.

## Method implemented

| Paper section | Code | Contract |
| --- | --- | --- |
| 4.1 | `traces.py`, `manifest.py`, `preprocess/` | Independent executions; timestamps, PID/TID and recorded metadata retained; EDR and stimulus descendants separated |
| 4.2 | `training/` | API function names only; GPT-2 per-event NLL, n-gram and LSTM comparators; no process/argument tokens in likelihood |
| 4.3 | `analysis/temporal.py` | Complete windows 256/512/1024; stride 512; max mean NLL; configuration-specific baseline quantile 0.995; strict `score > threshold`; ties favor smaller scales |
| 4.4 | `analysis/semantics.py`, `analysis/temporal.py` | Fixed log-grounded GPT-5.4 request; separate unsupported conclusions; validated behavior intervals; Kneedle gap tolerance; running maximum interval endpoints; closed-interval intersection |
| 4.5 | `analysis/recurrence.py` | At most one vote per behavior per execution; two-sided 95% Wilson lower endpoint, z=1.959964; k>=5 and **unrounded** lower bound >=0.70 |
| 5.2–5.6 | `analysis/evaluation.py`, `pipeline.py` | Trace p99.5 summaries, wall-clock bins, human confidence summaries, alert/anomaly contingencies, paired McNemar and Holm; conditional held-out rates |
| 6.1–6.2 | `experiments/component.py` | 4,320 abstract fragment executions; presplit medoids and frozen thresholds; five-second horizon; 50-second timing control |
| 6.3–6.5 | `experiments/connected.py` | 29,400 abstract traces, 10,800 recorded queries, nine arms, supplied canonical metadata, paired-seed uncertainty, five difficult regimes |
| 7.1–7.3 | `experiments/discrimination.py` | Signature-conditioned NLL before window selection; matched unigram/bigram statistics; 85-window trigram control; reused retrieval sets; independent binomial FPR confirmation |
| 7.4 | `experiments/adaptation.py` | 30 AND/OR hypotheses; six binary attributes; held-out predicate pairs; XOR rejection; decoder calibration; fitting+64+16 observation costs |

## Execution manifest

Use one JSONL file per independently observed execution. A single analysis run
must use one fixed product/version/policy configuration. `group_id` keeps related
samples or repeated observations on the same side of the data split.

```json
{
  "configuration": "product / version / fixed policy / database date",
  "edr_processes": ["sensor.exe"],
  "traces": [
    {"execution_id": "base-001", "group_id": "sample-001", "condition": "baseline", "split": "discovery", "path": "baseline/001.jsonl"},
    {"execution_id": "base-002", "group_id": "sample-002", "condition": "baseline", "split": "validation", "path": "baseline/002.jsonl"},
    {"execution_id": "stim-001", "group_id": "sample-001", "condition": "stimulated", "split": "discovery", "path": "stimulated/001.jsonl", "stimulus_processes": ["sample.exe"], "stimulus_pids": [100], "annotation": "behaviors/001.json"},
    {"execution_id": "stim-003", "group_id": "sample-003", "condition": "stimulated", "split": "test", "path": "stimulated/003.jsonl", "stimulus_processes": ["sample.exe"], "annotation": "behaviors/003.json"}
  ]
}
```

Paths are relative to the manifest. The example illustrates the schema, not a
sufficient discovery sample size or the full 70:10:20 experimental population.
`preprocess.build_vocab` makes the 70:10:20 execution split before segmentation
and fits vocabulary exclusively on the discovery partition. Related baseline and
stimulated observations must use that same sample/group split in the manifest.

Input events retain the original DRAKVUF fields:

```json
{"TimeStamp": 1234.5, "ProcessName": "C:\\sensor.exe", "PID": 10, "TID": 11, "Method": "NtOpenFile", "Plugin": "syscall", "Arguments": {"FileName": "example.txt"}, "ReturnStatus": 0}
```

`sysret` records are not additional API calls. Records without `Method` may
establish process ancestry but never become API tokens. Malformed call records
with a present `Method` are rejected. Available return status/arguments
on call records are retained. Missing return metadata is not synthesized. PID/PPID
links permit descendant selection; without those links, only explicitly named
stimulus processes are observed. PID lifetime/reuse and instrumentation coverage
remain properties of the supplied collection.

## Prepare and train a baseline

The package must be installed before using the module entry points.

```bash
python -m shieldbreaker.preprocess.extract_edr_processes \
  --input_dir datasets/baseline/raw --pattern drakvuf.json --recursive \
  --output_file datasets/baseline/edr_only.jsonl --processes sensor.exe

python -m shieldbreaker.preprocess.build_vocab \
  --input_path datasets/baseline/edr_only.jsonl \
  --out_tokens_path datasets/baseline/tokens.jsonl \
  --out_vocab_path datasets/baseline/vocab.json

python -m shieldbreaker.training.make_tokenizer_and_corpus \
  --tokens_path datasets/baseline/tokens.jsonl \
  --vocab_path datasets/baseline/vocab.json --out_dir datasets/lm_data

python -m shieldbreaker.training.train_gpt2 \
  --model_name_or_path gpt2 --tokenizer_name datasets/lm_data \
  --train_file datasets/lm_data/train.txt --validation_file datasets/lm_data/valid.txt \
  --output_dir models/gpt2 --do_train --do_eval
```

The corpus contains separate `train.txt`, `valid.txt`, `test.txt`, execution
provenance and corpus hashes. Training never concatenates separate executions or
drops short traces/tails. Legacy checkpoints with `<PROC:...>` vocabulary must be
retrained. Model-level timing excludes sandbox collection and LLM reconstruction.

The LSTM comparator uses the same split-preserving token records:

```bash
python -m shieldbreaker.training.lstm \
  --tokens-path datasets/baseline/tokens.jsonl --output-dir models/lstm
```

## Reconstruct stimulus behaviors and analyze

Export requests without sending logs to an external service:

```bash
shieldbreaker semantic-requests --manifest datasets/manifest.json --output-dir runs/semantic_requests
```

Use GPT-5.4 with the exported fixed prompt, or supply independently checked
annotations using the same schema. Every segment must cite actual stimulus event
IDs (the original `GlobalID`, or the one-based input-record index if absent):

```json
{
  "behaviors": [
    {"behavior": "file rename", "start": 1234.5, "end": 1234.8, "evidence_event_ids": ["10", "11"]}
  ],
  "unsupported_conclusions": ["Subsequent execution is not established by this log."]
}
```

The source repository did not contain the historical prompt, so the fixed prompt
here implements the paper's evidence constraints and has its own SHA-256 identity.
It is not represented as the original historical prompt. Human confidence ratings
remain separate from model-generated annotations.

```bash
shieldbreaker analyze --manifest datasets/manifest.json --model gpt2 \
  --model-path models/gpt2 --output-dir runs/analysis

# Fully offline comparator, fitted only on discovery baseline executions:
shieldbreaker analyze --manifest datasets/manifest.json --model ngram --output-dir runs/ngram
```

Calibration uses independent validation baseline windows from the same declared
configuration. Discovery stimulated executions supply recurrence support; test
stimulated executions supply the conditional held-out statistic. No post-behavior
horizon is added to the commercial overlap criterion. Short traces without a
complete window are explicitly marked `scoreable: false`.

Outputs include `calibration.json`, `observations.jsonl`, `candidates.json`,
`held_out_metrics.json`, `trace_scores.jsonl`, `behavior_aligned_scores.jsonl`,
`condition_summaries.json`, and `run_metadata.json`. `observations.jsonl` preserves
every intersecting behavior-segment/cluster pair under `cooccurrences`; their
indices are local to each execution. Aligned scores use window-start timestamps
relative to each behavior end in one-second bins over [-30, 30) seconds, storing
the bin-level p99.5 and leaving missing bins as `null`. A zero
eligible-trace denominator produces JSON `null`, not zero. Re-running recomputes
support rather than adding the same executions again.

Without the unavailable record-level commercial inventory, summaries use
**behavior categories** as their unit. They must not be compared as if they were
the historical 25/28/33 candidate inventories. Supply an actual external inventory
with `--inventory` to preserve candidate IDs, including distinct candidates that
share a behavior category:

```json
[{"candidate_id": "original-record-id", "behavior": "file rename"}]
```

## Synthetic studies

```bash
shieldbreaker simulate all --output-dir runs/synthetic
# Or: component, connected, discrimination, confirmation, adaptation
```

All generators emit abstract symbols and logical labels. No study executes a
commercial sensor, performs a real behavior transformation, or independently
verifies real-world functional equivalence. Synthetic response matching is
separate from the commercial within-execution temporal association.

Every study saves fresh observations, frozen parameters, decisions and metrics.
The paper's reported result cells are **not embedded in code**. The public source
lacked the original synthetic programs, detailed RNG draw order, some motif/filler
conventions, and confirmation/adaptation seeds. These implementations follow the
stated designs with explicit deterministic choices; they are not byte-for-byte
recovered historical experiments and do not promise the published numerical cells.
Original commercial traces, trained checkpoints, record-level inventories and
original synthetic programs are needed to audit those historical measurements.

Keep `datasets/`, `models/`, and `runs/` local. The repository contains source,
tests and configuration only; manuscript files are not changed by any command.
