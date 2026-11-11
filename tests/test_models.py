import math
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from shieldbreaker.training.make_tokenizer_and_corpus import (
    make_corpus,
    segment_long_sequences,
)
from shieldbreaker.training.anomaly_detector import EDRAnomalyDetector
from shieldbreaker.training.train_gpt2 import chunk_corpus
from shieldbreaker.preprocess.build_vocab import build_vocab
from shieldbreaker.training.lstm import LSTMModel


def tiny_checkpoint(tmp_path):
    vocab = build_vocab([["A", "B"]])
    rows = [
        {"execution_id": str(i), "split": split, "tokens": ["A", "B"] * 10}
        for i, split in enumerate(["discovery", "validation", "test"])
    ]
    make_corpus(rows, vocab, tmp_path, segment_length=16)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_positions=16,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=vocab["<BOS>"],
            eos_token_id=vocab["<EOS>"],
        )
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    model.save_pretrained(tmp_path)
    return vocab


def test_long_gpt_trace_has_exactly_one_correct_loss_per_api(tmp_path):
    vocab = tiny_checkpoint(tmp_path)
    detector = EDRAnomalyDetector(tmp_path, device="cpu")
    losses = detector.score(["A", "B", "unknown"] * 400, block_size=7)
    assert len(losses) == 1200
    assert losses == pytest.approx([math.log(len(vocab))] * 1200, abs=1e-6)
    with pytest.raises(ValueError, match="function names"):
        detector.score(["<PROC:edr.exe>"])


def test_training_chunks_keep_short_executions_and_tail_tokens(tmp_path):
    tiny_checkpoint(tmp_path)
    detector = EDRAnomalyDetector(tmp_path)
    path = tmp_path / "corpus.txt"
    path.write_text("<BOS> A B <EOS>\n<BOS> B B B B B <EOS>\n")
    chunks = chunk_corpus(path, detector.tokenizer, 5)
    assert len(chunks) == 3
    # Both trace BOS tokens and the last two APIs survive.
    assert [len(row["input_ids"]) for row in chunks] == [4, 5, 4]
    with pytest.raises(ValueError, match="Vocabulary"):
        make_corpus(
            [
                {
                    "execution_id": str(i),
                    "split": s,
                    "tokens": ["A"] if s == "discovery" else ["B"],
                }
                for i, s in enumerate(["discovery", "validation", "test"])
            ],
            build_vocab([["A", "B"]]),
            tmp_path / "bad",
        )


def test_corpus_segmentation_rejects_non_api_tokens_instead_of_dropping_them():
    assert segment_long_sequences([["<BOS>", "A", "B", "C", "<EOS>"]], 4) == [
        ["<BOS>", "A", "B", "<EOS>"],
        ["<BOS>", "C", "<EOS>"],
    ]
    for token in ("<PROC:edr.exe>", "<PAD>", "A B", "", None, [], {}):
        with pytest.raises(ValueError, match="function-name tokens only"):
            segment_long_sequences([["A", token, "B"]], 8)


def test_lstm_roundtrip_and_sequence_boundary(tmp_path):
    torch.manual_seed(1)
    model = LSTMModel(
        build_vocab([["A", "B"]]), embedding_size=4, hidden_size=8, device="cpu"
    )
    model.fit([["A", "B"] * 5], epochs=1)
    model.discovery_ids = ["independent-trace"]
    model.save(tmp_path / "lstm")
    restored = LSTMModel.load(tmp_path / "lstm", device="cpu")
    assert restored.score(["A", "unknown", "B"]) == pytest.approx(
        model.score(["A", "unknown", "B"])
    )
    assert restored.discovery_ids == ["independent-trace"]
