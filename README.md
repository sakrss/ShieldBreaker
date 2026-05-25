# ShieldBreaker

ShieldBreaker detects abnormal EDR reactions from API traces. It learns a normal
EDR behavior baseline with a GPT-style language model, scores EDR API calls in
malware-stimulated traces, and extracts association rules of the form:

```text
malware API pattern -> anomalous EDR API
```

The repository is intentionally source-only. Raw traces, trained checkpoints,
generated corpora, and rule outputs are local artifacts and are ignored by Git.

## Repository Layout

```text
src/shieldbreaker/
  preprocess/
    extract_edr_processes.py       filter baseline EDR events
    extract_malware_processes.py   filter stimulated EDR + malware events
    build_vocab.py                 build token sequences and vocabulary
  training/
    make_tokenizer_and_corpus.py   build tokenizer, train.txt, valid.txt
    train_gpt2.py                  train the EDR baseline language model
    anomaly_detector.py            GPT NLL scoring helper
    inference.py                   simple generation/debug entry point
  analysis/
    extract_anomaly_rules.py       detect anomalous EDR calls and build rules
    analyze_rules.py               summarize extracted rules
  tools/
    unzip.py                       unzip .gz trace files
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build separately if GPU training or inference is
needed.

## Data Format

Input traces are newline-delimited JSON records. The core scripts expect these
fields:

```json
{
  "TimeStamp": 1234567890.123,
  "ProcessName": "\\Device\\HarddiskVolume2\\Windows\\System32\\MsMpEng.exe",
  "Method": "NtOpenKey",
  "Plugin": "syscall"
}
```

`Plugin == "sysret"` records are ignored. Baseline logs should contain normal
EDR execution. Stimulated logs should contain both EDR behavior and the malware
sample process. By default, the scripts target Windows Defender process names;
pass `--processes` or `--edr_processes` for other EDR products.

## Typical Workflow

Create local artifact directories:

```bash
mkdir -p datasets/baseline datasets/stimulated datasets/lm_data models runs
```

Extract baseline EDR events:

```bash
python src/shieldbreaker/preprocess/extract_edr_processes.py \
  --input_dir datasets/baseline/raw \
  --pattern drakvuf.json \
  --recursive \
  --output_file datasets/baseline/edr_only.jsonl \
  --processes NisSrv.exe MsMpEng.exe SecurityHealthService.exe SecurityHealthSystray.exe
```

Build API tokens and vocabulary:

```bash
python src/shieldbreaker/preprocess/build_vocab.py \
  --input_path datasets/baseline/edr_only.jsonl \
  --out_tokens_path datasets/baseline/tokens.jsonl \
  --out_vocab_path datasets/baseline/vocab.json
```

Build the tokenizer and language-model corpus:

```bash
python src/shieldbreaker/training/make_tokenizer_and_corpus.py \
  --tokens_path datasets/baseline/tokens.jsonl \
  --vocab_path datasets/baseline/vocab.json \
  --out_dir datasets/lm_data
```

Train the GPT baseline:

```bash
python src/shieldbreaker/training/train_gpt2.py \
  --model_name_or_path gpt2 \
  --tokenizer_name datasets/lm_data \
  --train_file datasets/lm_data/train.txt \
  --validation_file datasets/lm_data/valid.txt \
  --output_dir models/out_gpt2 \
  --do_train \
  --do_eval
```

Prepare stimulated traces. Each `.log` filename is interpreted as the malware
sample name and converted to a malware process name with `.exe` suffix.

```bash
python src/shieldbreaker/preprocess/extract_malware_processes.py \
  --input_dir datasets/stimulated/raw \
  --output_dir datasets/stimulated/processed \
  --edr_processes NisSrv.exe MsMpEng.exe SecurityHealthService.exe SecurityHealthSystray.exe
```

Extract anomaly rules:

```bash
python src/shieldbreaker/analysis/extract_anomaly_rules.py \
  --model_path models/out_gpt2 \
  --vocab_path datasets/baseline/vocab.json \
  --processed_dir datasets/stimulated/processed \
  --output_file runs/rule.jsonl \
  --threshold 10.5
```

Analyze extracted rules:

```bash
python src/shieldbreaker/analysis/analyze_rules.py \
  --input_file runs/rule.jsonl \
  --output_file runs/rule_analysis.jsonl
```

## Artifact Policy

Keep these paths local:

- `datasets/`
- `models/`
- `runs/`

Do not commit raw DRAKVUF traces, generated JSONL corpora, model checkpoints, or
rule output files.
