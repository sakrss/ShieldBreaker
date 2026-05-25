# Repository Layout

This repository was reorganized on 2026-05-25 to separate source code, datasets,
model artifacts, experiment outputs, and documentation.

## Top-Level Folders

- `src/leakedr/` - project source code
  - `analysis/` - anomaly detection, matrix export, plotting, rule extraction
  - `preprocess/` - raw trace filtering and vocabulary/token builders
  - `training/` - tokenizer/corpus builders, model training, inference scripts
  - `tools/` - standalone utilities
- `datasets/` - input data and derived corpora
  - `main/` - former `data/`
  - `test/` - former `data-test/`
  - `lm_data*` - tokenizer/corpus folders formerly under `train/`
  - root-level raw/output JSON files moved here
- `models/` - trained models, checkpoints, and pretrained cache
- `runs/` - experiment outputs and per-sample anomaly reports
  - `summaries/` - former root-level run summary JSON files
  - `generated/` - default destination for newly generated reports
- `docs/` - repository notes, snapshots, and migration records

## Running Scripts

Run commands from the repository root unless a script says otherwise. Examples:

```powershell
python src\leakedr\analysis\detect_endpoint_gpt_anomalies.py --input_path datasets\main\win10.elastic\elastic-87-2026-03-23_190419\drakvuf.json --model_dir models\out_elastic_endpoint_gpt2_medium_strictpath --output_path runs\generated\gpt2_medium_endpoint_anomaly_report.json
```

```powershell
python src\leakedr\preprocess\extract_edr_processes.py --input_dir runs\baseline --pattern drakvuf.json --recursive
```

Historical JSON files still contain absolute `input_path`, `source_report`, and
`model_dir` metadata from the old layout. Those fields document how the result
was originally produced and were not bulk-rewritten.

