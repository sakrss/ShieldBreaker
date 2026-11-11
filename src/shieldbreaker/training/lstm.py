"""Optional LSTM baseline comparator using the same API-only abstraction."""

import argparse
import json
import time
from pathlib import Path
import torch
from shieldbreaker.preprocess.build_vocab import build_vocab
from shieldbreaker.traces import write_json


class Network(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size=64, hidden_size=128):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.output = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, ids, hidden=None):
        sequence, hidden = self.lstm(self.embedding(ids), hidden)
        return self.output(sequence), hidden


class LSTMModel:
    def __init__(self, vocab, embedding_size=64, hidden_size=128, device=None):
        self.vocab = vocab
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.sizes = {"embedding_size": embedding_size, "hidden_size": hidden_size}
        self.model = Network(len(vocab), **self.sizes).to(self.device)
        self.discovery_ids = []

    def score(self, tokens):
        if not tokens:
            return []
        ids = [self.vocab["<BOS>"]] + [
            self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens
        ]
        losses, hidden = [], None
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(tokens), 512):
                stop = min(start + 512, len(tokens))
                x = torch.tensor([ids[start:stop]], device=self.device)
                targets = torch.tensor(ids[start + 1 : stop + 1], device=self.device)
                logits, hidden = self.model(x, hidden)
                losses.extend(
                    torch.nn.functional.cross_entropy(
                        logits[0], targets, reduction="none"
                    )
                    .cpu()
                    .tolist()
                )
        return losses

    def fit(self, sequences, epochs=10, learning_rate=0.001, seed=42):
        torch.manual_seed(seed)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        for _ in range(epochs):
            for sequence in sequences:
                ids = [self.vocab["<BOS>"]] + [
                    self.vocab.get(t, self.vocab["<UNK>"]) for t in sequence
                ]
                hidden = None
                for start in range(0, len(sequence), 512):
                    stop = min(start + 512, len(sequence))
                    optimizer.zero_grad()
                    logits, hidden = self.model(
                        torch.tensor([ids[start:stop]], device=self.device), hidden
                    )
                    target = torch.tensor(ids[start + 1 : stop + 1], device=self.device)
                    loss = torch.nn.functional.cross_entropy(logits[0], target)
                    loss.backward()
                    optimizer.step()
                    hidden = tuple(t.detach() for t in hidden)
        return self

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "lstm.pt")
        write_json(
            path / "config.json",
            {"vocab": self.vocab, **self.sizes, "discovery_ids": self.discovery_ids},
        )

    @classmethod
    def load(cls, path, device=None):
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        model = cls(
            config["vocab"], config["embedding_size"], config["hidden_size"], device
        )
        model.model.load_state_dict(
            torch.load(path / "lstm.pt", map_location=model.device, weights_only=True)
        )
        model.discovery_ids = config["discovery_ids"]
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    rows = [
        json.loads(line)
        for line in Path(args.tokens_path).read_text().splitlines()
        if line.strip()
    ]
    discovery = [r for r in rows if r["split"] == "discovery"]
    sequences = [r["tokens"] for r in discovery]
    model = LSTMModel(build_vocab(sequences))
    started = time.perf_counter()
    model.fit(sequences, args.epochs, seed=args.seed)
    model.discovery_ids = [r["execution_id"] for r in discovery]
    model.save(args.output_dir)
    write_json(
        Path(args.output_dir) / "resource_usage.json",
        {
            "model": "lstm",
            "training_seconds": time.perf_counter() - started,
            "scope": "model only; excludes sandbox collection and semantic reconstruction",
        },
    )


if __name__ == "__main__":
    main()
