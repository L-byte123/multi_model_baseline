"""Offline integration check with a real, randomly initialized tiny RoBERTa.

This is a plumbing test, not a pretrained language or fault diagnosis benchmark.
"""
import argparse
import json
from pathlib import Path

import torch

from .demo import make_demo
from .engine import predict, train
from .preprocess import prepare


def make_tiny_text(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel
    vocabulary = {token: index for index, token in enumerate(
        ["<s>", "<pad>", "</s>", "<unk>", "Bearing", "model", ":", "6205", ".",
         "Shaft", "speed", "1500", "1800", "rpm", "Load", "0", "1", "hp"])}
    core = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    core.pre_tokenizer = Whitespace()
    core.post_processor = TemplateProcessing(single="<s> $A </s>",
        special_tokens=[("<s>", 0), ("</s>", 2)])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<s>",
        eos_token="</s>", unk_token="<unk>", pad_token="<pad>", model_max_length=64)
    tokenizer.save_pretrained(path)
    cfg = RobertaConfig(vocab_size=len(vocabulary), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=128,
        pad_token_id=1, bos_token_id=0, eos_token_id=2)
    RobertaModel(cfg).save_pretrained(path)
    return path


def smoke(output):
    torch.set_num_threads(1)
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Smoke output must be empty")
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    text_model = make_tiny_text(root / "tiny_roberta")
    recordings = make_demo(root / "raw")
    manifest = prepare(recordings, root / "prepared")
    config = {"signal_encoder": "tiny", "input_kind": "spectrogram", "text_model": str(text_model),
        "freeze_text": True, "pretrained_signal": False, "fusion": "text", "height": 32,
        "width": 32, "projection_dim": 16, "hidden_dim": 16, "dropout": 0,
        "epochs": 2, "batch_size": 8, "local_files_only": True}
    cfg_path = root / "config.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    checkpoint = train(manifest, cfg_path, root / "training", "cpu")
    predict(checkpoint, manifest, root / "predictions.csv", device_name="cpu")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/smoke")
    print(smoke(parser.parse_args().output))
