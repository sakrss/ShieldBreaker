# ShieldBreaker

ShieldBreaker detects anomalous EDR behavior from API traces. It learns a baseline
language model over normal EDR API-call sequences, scores EDR behavior under
malicious stimulation, and extracts behavior-to-reaction rules.

This repository is organized as a GitHub-ready source-code project. Large local
datasets, trained models, and experiment outputs are intentionally excluded from
Git.

## Repository Layout

```text
src/shieldbreaker/
  analysis/      anomaly detection, NLL export, plotting, rule extraction
  preprocess/    trace filtering, tokenization, vocabulary builders
  training/      corpus builders, model training, inference scripts
  tools/         standalone utilities
docs/            method notes and repository maintenance docs
examples/        command examples
datasets/        local data placeholder, ignored by Git
models/          local model placeholder, ignored by Git
runs/            local experiment-output placeholder, ignored by Git
```

See [docs/repository_layout.md](docs/repository_layout.md) and
[docs/github_upload.md](docs/github_upload.md) for more details.

## Installation

Python 3.10+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build if GPU training or inference is required.

## Typical Workflow

1. Place raw DRAKVUF traces under `datasets/main/` or experiment outputs under
   `runs/`.
2. Extract EDR-only events and build token/vocabulary files.
3. Build tokenizer and train/validation corpora under `datasets/lm_data*`.
4. Train GPT, LSTM, N-gram, TCN, or Transformer baselines into `models/`.
5. Score stimulated traces and write reports under `runs/generated/`.
6. Localize multi-scale anomalies and extract behavior rules.

Example commands are available in [examples/README.md](examples/README.md).

## Multi-Scale Anomaly Localization

`src/shieldbreaker/analysis/localize_multiscale_anomalies.py` implements adaptive
multi-scale localization:

```powershell
python src\shieldbreaker\analysis\localize_multiscale_anomalies.py `
  --product elastic `
  --baseline datasets\main\baseline_sample\gpt2_medium_event_nll_sequence.json `
  --stimulated datasets\main\stimulated_sample\gpt2_medium_event_nll_sequence.json `
  --output_json runs\generated\elastic_multiscale_anomalies.json
```

The script estimates a product-specific threshold from baseline event-level NLL
sequences and reports anomaly points/clusters for stimulated traces.

## Data And Artifact Policy

The following folders are local-only and ignored by Git:

- `datasets/`
- `models/`
- `runs/`

Only their README placeholders are committed. Store large traces, checkpoints,
and result matrices in external artifact storage instead of GitHub.

