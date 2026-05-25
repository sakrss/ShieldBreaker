# Datasets

Large DRAKVUF traces, derived JSONL files, and tokenizer corpora are stored here
locally but are intentionally ignored by Git.

Suggested local layout:

- `datasets/main/` - full raw and processed experiment data
- `datasets/test/` - small test traces
- `datasets/lm_data*` - language-model tokenizer and train/valid corpora

Do not upload raw traces, generated event-level NLL sequences, or large JSON/CSV
outputs to GitHub. Use external artifact storage if those files need to be
shared.

