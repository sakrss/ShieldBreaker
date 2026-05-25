#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512,
                 num_layers=2, dropout=0.2, tie_weights=True):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        if tie_weights:
            if embedding_dim != hidden_dim:
                raise ValueError("embedding_dim must equal hidden_dim when tying weights")
            self.fc.weight = self.embedding.weight

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        if self.fc.weight is not self.embedding.weight:
            self.fc.weight.data.uniform_(-init_range, init_range)

    def forward(self, input_ids):
        x = self.dropout(self.embedding(input_ids))
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return self.fc(x)

class TextDataset(Dataset):
    def __init__(self, file_path, vocab, block_size=256):
        self.examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                ids = [vocab.get(t, vocab.get("<UNK>", 1)) for t in tokens]
                for i in range(0, len(ids) - 1, block_size):
                    chunk = ids[i:i + block_size + 1]
                    if len(chunk) > 1:
                        self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        return (
            torch.tensor(chunk[:-1], dtype=torch.long),
            torch.tensor(chunk[1:], dtype=torch.long)
        )


def collate_fn(batch):
    x, y = zip(*batch)
    return (
        pad_sequence(x, batch_first=True, padding_value=0),
        pad_sequence(y, batch_first=True, padding_value=-100),
    )


class LSTMTrainer:
    def __init__(self, model, device, lr=1e-3, patience=3, min_delta=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.patience = patience
        self.min_delta = min_delta

        self.global_step = 0
        self.train_start_time = None
        self.best_perplexity = float("inf")
        self.best_step = 0
        self.best_epoch = 0
        self.best_state_dict = None
        self.no_improve_epochs = 0

        self.training_history = {
            "steps": [],
            "epochs": [],
            "eval_perplexity": [],
            "eval_at_epoch_end": [],
            "best_eval_perplexity": None,
            "best_step": None,
            "best_epoch": None,
            "epochs_trained": 0,
            "total_train_time_sec": None,
            "avg_step_time_ms": None,
            "peak_gpu_mem_mb": None,
        }

    def _record_eval(self, perplexity, epoch_idx, at_epoch_end):
        self.training_history["steps"].append(self.global_step)
        self.training_history["epochs"].append(epoch_idx)
        self.training_history["eval_perplexity"].append(perplexity)
        self.training_history["eval_at_epoch_end"].append(at_epoch_end)

        improved = perplexity < (self.best_perplexity - self.min_delta)
        if improved:
            self.best_perplexity = perplexity
            self.best_step = self.global_step
            self.best_epoch = epoch_idx
            self.best_state_dict = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            }
            self.training_history["best_eval_perplexity"] = perplexity
            self.training_history["best_step"] = self.global_step
            self.training_history["best_epoch"] = epoch_idx

        return improved

    def train_epoch(self, train_loader, eval_loader, epoch_idx, eval_steps=100):
        self.model.train()

        if self.train_start_time is None:
            self.train_start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        for x, y in tqdm(train_loader, desc="Training"):
            step_start = time.time()

            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(x)
            loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            self.global_step += 1

            if self.global_step % eval_steps == 0:
                ppl = self.evaluate(eval_loader)
                self._record_eval(ppl, epoch_idx, at_epoch_end=False)

            step_time = time.time() - step_start

        epoch_end_ppl = self.evaluate(eval_loader)
        improved = self._record_eval(epoch_end_ppl, epoch_idx, at_epoch_end=True)
        self.training_history["epochs_trained"] = epoch_idx
        return epoch_end_ppl, improved

    def evaluate(self, loader):
        self.model.eval()
        total_loss, total_tokens = 0.0, 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                mask = (y != -100)
                total_loss += loss.item() * mask.sum().item()
                total_tokens += mask.sum().item()

        self.model.train()
        return math.exp(total_loss / total_tokens)

    def finalize(self, output_dir):
        total_time = time.time() - self.train_start_time
        self.training_history["total_train_time_sec"] = total_time
        self.training_history["avg_step_time_ms"] = total_time / self.global_step * 1000

        if torch.cuda.is_available():
            self.training_history["peak_gpu_mem_mb"] = (
                torch.cuda.max_memory_allocated() / 1024 / 1024
            )

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)

        best_state = self.best_state_dict
        if best_state is None:
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            }

        model_path = Path(output_dir) / "pytorch_model.bin"
        torch.save(best_state, model_path)
        print(f"Model saved to {model_path}")

        config = {
            "model_type": "lstm",
            "vocab_size": self.model.vocab_size,
            "embedding_dim": self.model.embedding_dim,
            "hidden_dim": self.model.hidden_dim,
            "num_layers": self.model.num_layers,
            "dropout": 0.2,
            "best_eval_perplexity": self.training_history["best_eval_perplexity"],
            "best_step": self.training_history["best_step"],
            "best_epoch": self.training_history["best_epoch"],
        }
        with open(Path(output_dir) / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {Path(output_dir) / 'config.json'}")



def main():
    parser = argparse.ArgumentParser(description="Train LSTM language model")
    parser.add_argument("--train_file", required=True, help="Training data file")
    parser.add_argument("--validation_file", required=True, help="Validation data file")
    parser.add_argument("--vocab_file", required=True, help="Vocabulary JSON file")
    parser.add_argument("--output_dir", default="models/out_lstm", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs to train")
    parser.add_argument("--min_epochs", type=int, default=3, help="Train at least this many epochs before early stopping")
    parser.add_argument("--patience", type=int, default=3, help="Stop after this many non-improving epoch-end validations")
    parser.add_argument("--min_delta", type=float, default=1e-4, help="Minimum perplexity improvement to reset patience")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = json.load(open(args.vocab_file))
    model = LSTMLanguageModel(len(vocab))
    trainer = LSTMTrainer(
        model,
        device,
        patience=args.patience,
        min_delta=args.min_delta,
    )

    train_ds = TextDataset(args.train_file, vocab)
    val_ds = TextDataset(args.validation_file, vocab)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate_fn)

    for epoch_idx in range(1, args.epochs + 1):
        epoch_end_ppl, improved = trainer.train_epoch(
            train_loader, val_loader, epoch_idx, args.eval_steps
        )
        status = "improved" if improved else f"no improvement ({trainer.no_improve_epochs + 1}/{args.patience})"
        print(
            f"Epoch {epoch_idx}: epoch_end_eval_perplexity={epoch_end_ppl:.6f} "
            f"best={trainer.best_perplexity:.6f} [{status}]"
        )

        if improved:
            trainer.no_improve_epochs = 0
        else:
            trainer.no_improve_epochs += 1

        if epoch_idx >= args.min_epochs and trainer.no_improve_epochs >= args.patience:
            print(
                f"Early stopping triggered after epoch {epoch_idx}. "
                f"Best epoch={trainer.best_epoch}, best perplexity={trainer.best_perplexity:.6f}"
            )
            break

    trainer.finalize(args.output_dir)


if __name__ == "__main__":
    main()
