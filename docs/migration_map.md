# Migration Map

The original repository state was recorded before moving files. Snapshot text
files remain local-only and are ignored by GitHub upload rules.

## Source Code

| Old path | New path |
| --- | --- |
| `analysis/` | `src/leakedr/analysis/` |
| `preprocess/` | `src/leakedr/preprocess/` |
| `train/*.py`, `train/*.bat`, `train/*.ps1` | `src/leakedr/training/` |
| `tools/` | `src/leakedr/tools/` |
| `drakvuf_command_burst_locator.py` | `src/leakedr/tools/drakvuf_command_burst_locator.py` |
| `extract_notepad_sysret.py` | `src/leakedr/tools/extract_notepad_sysret.py` |

## Data And Corpora

| Old path | New path |
| --- | --- |
| `data/` | `datasets/main/` |
| `data-test/` | `datasets/test/` |
| `train/lm_data*` | `datasets/lm_data*` |
| `123456789.json` | `datasets/raw_123456789.json` |
| `drakvuf.json` | `datasets/raw_drakvuf.json` |
| `output-*.json` | `datasets/output-*.json` |
| `samples_md5.txt` | `datasets/samples_md5.txt` |

## Models And Runs

| Old path | New path |
| --- | --- |
| `train/out_*` | `models/out_*` |
| `train/models/` | `models/training_helpers/` |
| `train/baseline_train_logs/` | `runs/baseline_train_logs/` |
| `baseline/` | `runs/baseline/` |
| `liangxing/` | `runs/liangxing/` |
| `win10*`, `win11*` experiment folders | `runs/<same-name>/` |
| `*_run_summary_*.json` | `runs/summaries/` |
| `11/` | `runs/misc/11/` |
