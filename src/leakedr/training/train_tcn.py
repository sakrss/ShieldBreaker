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



class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stacked Temporal Blocks for TCN"""
    def __init__(self, num_inputs, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else channels[i - 1]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, channels=(512, 512, 512, 512),
                 kernel_size=3, dropout=0.2, tie_weights=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.tcn = TemporalConvNet(embedding_dim, channels, kernel_size, dropout)
        self.fc = nn.Linear(channels[-1], vocab_size)
        self.dropout = nn.Dropout(dropout)

        if tie_weights:
            if embedding_dim != channels[-1]:
                raise ValueError("embedding_dim must equal last channel for weight tying")
            self.fc.weight = self.embedding.weight

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        if self.fc.weight is not self.embedding.weight:
            self.fc.weight.data.uniform_(-init_range, init_range)

    def forward(self, input_ids):
        x = self.embedding(input_ids)      # [B, T, D]
        x = self.dropout(x)
        x = x.transpose(1, 2)              # [B, D, T]
        x = self.tcn(x)                    # [B, C, T]
        x = x.transpose(1, 2)              # [B, T, C]
        return self.fc(x)                  # [B, T, V]


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
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


def collate_fn(batch):
    x, y = zip(*batch)
    return pad_sequence(x, batch_first=True, padding_value=0), pad_sequence(y, batch_first=True, padding_value=-100)



class TCNTrainer:

    def __init__(self, model, device, lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.global_step = 0
        self.train_start_time = None
        self.training_history = {
            "steps": [],
            "eval_perplexity": [],
            "total_train_time_sec": None,
            "avg_step_time_ms": None,
            "peak_gpu_mem_mb": None,
        }

    def train_epoch(self, train_loader, eval_loader, eval_steps=100):
        self.model.train()
        if self.train_start_time is None:
            self.train_start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

        for x, y in tqdm(train_loader, desc="Training"):
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
                self.training_history["steps"].append(self.global_step)
                self.training_history["eval_perplexity"].append(ppl)

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
            self.training_history["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)

        torch.save(self.model.state_dict(), Path(output_dir) / "pytorch_model.bin")
        print(f"Model saved to {Path(output_dir) / 'pytorch_model.bin'}")

        config = {
            "model_type": "tcn",
            "vocab_size": self.model.embedding.num_embeddings,
            "embedding_dim": self.model.embedding.embedding_dim,
            "channels": [self.model.tcn.network[i].conv1.out_channels for i in range(len(self.model.tcn.network))],
            "kernel_size": 3,
            "dropout": 0.2,
        }
        with open(Path(output_dir) / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to {Path(output_dir) / 'config.json'}")



def main():
    parser = argparse.ArgumentParser(description="Train TCN language model")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", required=True)
    parser.add_argument("--vocab_file", required=True)
    parser.add_argument("--output_dir", default="models/out_tcn")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = json.load(open(args.vocab_file))
    model = TCNLanguageModel(len(vocab))
    trainer = TCNTrainer(model, device)

    train_ds = TextDataset(args.train_file, vocab)
    val_ds = TextDataset(args.validation_file, vocab)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, collate_fn=collate_fn)

    for _ in range(args.epochs):
        trainer.train_epoch(train_loader, val_loader, args.eval_steps)

    trainer.finalize(args.output_dir)


if __name__ == "__main__":
    main()
