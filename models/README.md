# Models

Trained models, checkpoints, tokenizer caches, and pretrained weights are stored
here locally but are intentionally ignored by Git.

Suggested local layout:

- `models/out_*` - trained LeakEDR model outputs
- `models/pretrained/` - downloaded pretrained model cache
- `models/training_helpers/` - auxiliary local model resources

Model binaries such as `.bin`, `.pt`, `.pth`, `.ckpt`, and `.safetensors` should
not be committed to GitHub.

