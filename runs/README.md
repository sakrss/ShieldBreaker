# Runs

Experiment outputs, anomaly reports, matrices, Markdown summaries, and temporary
run folders are stored here locally but are intentionally ignored by Git.

Suggested local layout:

- `runs/generated/` - new reports produced by scripts
- `runs/summaries/` - compact run-level summaries
- `runs/<experiment-name>/` - per-experiment outputs

Keep GitHub focused on code and reproducible instructions. Store large run
artifacts outside the repository.

