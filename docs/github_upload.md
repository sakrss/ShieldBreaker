# GitHub Upload Checklist

This repository is now organized as a code-first GitHub project.

## Before Uploading

1. Keep `datasets/`, `models/`, and `runs/` local. They are ignored except for
   their README placeholders.
2. Check that no large artifacts are staged:

   ```powershell
   git status --short
   git ls-files --others --exclude-standard
   ```

3. If a large file appears, move it under `datasets/`, `models/`, or `runs/`, or
   add a narrower ignore rule.
4. Commit source, docs, examples, `requirements.txt`, and `.gitignore`.

## Suggested GitHub Contents

- `src/leakedr/` - source code
- `docs/` - method notes and repository layout
- `examples/` - reproducible command examples
- `datasets/README.md` - data placement guide
- `models/README.md` - model artifact guide
- `runs/README.md` - run artifact guide
- `requirements.txt`
- `.gitignore`
- `README.md`

