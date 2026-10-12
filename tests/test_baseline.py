import csv
import json

import numpy as np
import pytest
import torch

from mm_baseline.config import Config
from mm_baseline.data import load_manifest, load_visual
from mm_baseline.demo import make_demo
from mm_baseline.engine import load_checkpoint, predict
from mm_baseline.metadata import fit_numeric, numeric_vector, operating_text
from mm_baseline.metrics import classification_metrics
from mm_baseline.model import FaultClassifier
from mm_baseline.preprocess import log_spectrogram, prepare, read_csv, validate_recordings, write_csv
from mm_baseline.smoke import make_tiny_text, smoke


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def test_stft_has_correct_frequency_peak_and_rejects_nyquist():
    sample_rate = 12000
    samples = np.sin(2*np.pi*1500*np.arange(2048)/sample_rate)
    spec = log_spectrogram(samples, sample_rate, n_fft=256)
    assert abs(spec.mean(1).argmax()*sample_rate/256-1500) < sample_rate/256
    assert spec.shape == (129, 29)
    with pytest.raises(ValueError, match="Nyquist"):
        log_spectrogram(samples, sample_rate, fmax=7000)
    with pytest.raises(ValueError):
        log_spectrogram(samples, float("nan"))


def test_prompt_excludes_labels_and_preserves_numbers():
    row = {"bearing_model": "6205", "rpm": "1797.5", "load": "1.5", "load_unit": "hp",
           "label": "inner_fault", "signal_path": "inner_fault.npy"}
    text = operating_text(row)
    assert "1797.5 rpm" in text and "1.5 hp" in text
    assert "inner_fault" not in text


def test_group_and_duplicate_path_leakage_rejected(tmp_path):
    rows = read_csv(make_demo(tmp_path / "raw"))
    rows[-1]["group_id"] = rows[0]["group_id"]
    with pytest.raises(ValueError, match="crosses"):
        validate_recordings(rows, tmp_path)
    rows = read_csv(tmp_path / "raw" / "recordings.csv")
    rows[-1]["signal_path"] = rows[0]["signal_path"]
    with pytest.raises(ValueError, match="crosses"):
        validate_recordings(rows, tmp_path)


def test_duplicate_raw_content_rejected(tmp_path):
    path = make_demo(tmp_path / "raw")
    rows = read_csv(path)
    source = np.load(path.parent / rows[0]["signal_path"])
    np.save(path.parent / rows[-1]["signal_path"], source)
    with pytest.raises(ValueError, match="Duplicate raw signal"):
        prepare(path, tmp_path / "prepared")


def test_training_only_normalization_and_png_mapping(tmp_path):
    manifest = prepare(make_demo(tmp_path / "raw"), tmp_path / "prepared")
    rows, _, stats, root = load_manifest(manifest)
    train_specs = [np.load(root / r["spectrogram_path"]).ravel() for r in rows if r["split"] == "train"]
    values = np.concatenate(train_specs).astype(np.float64)
    assert stats["normalization"]["mean"] == pytest.approx(values.mean())
    assert stats["normalization"]["std"] == pytest.approx(values.std())
    row = rows[0]
    numeric = load_visual(row, root, Config(signal_encoder="tiny", height=129, width=29), stats["normalization"])
    png = load_visual(row, root, Config(signal_encoder="tiny", input_kind="png", height=129, width=29), stats["normalization"])
    # PNG is clipped to +/-3 standard deviations and quantized, with matching frequency order.
    assert torch.max(torch.abs(numeric.clamp(-3, 3)-png)) < 6/255+1e-5


def test_numeric_encoding_uses_training_statistics_and_unknown_category():
    rows = [{"bearing_model": "6205", "rpm": "1500", "load": "1", "load_unit": "hp"},
            {"bearing_model": "6205", "rpm": "1800", "load": "2", "load_unit": "hp"}]
    stats = fit_numeric(rows)
    unseen = {"bearing_model": "6206", "rpm": "2100", "load": "3", "load_unit": "hp"}
    vector = numeric_vector(unseen, stats)
    assert vector.tolist() == pytest.approx([3, 3, 0, 1])
    with pytest.raises(ValueError, match="unit"):
        numeric_vector({**unseen, "load_unit": "N"}, stats)


def test_inference_preprocessing_reuses_training_statistics(tmp_path):
    raw = make_demo(tmp_path / "raw")
    prepared = prepare(raw, tmp_path / "prepared")
    reference = prepared.parent / "preprocessing.json"
    inference_rows = [r for r in read_csv(raw) if r["split"] == "test"]
    path = raw.parent / "inference.csv"
    write_csv(path, inference_rows)
    out = prepare(path, tmp_path / "inference", reference_preprocessing=reference)
    assert json.loads((out.parent / "preprocessing.json").read_text()) == json.loads(reference.read_text())
    with pytest.raises(ValueError, match="settings"):
        prepare(path, tmp_path / "bad_inference", n_fft=128, reference_preprocessing=reference)



def test_frozen_text_eval_and_gradients(tmp_path):
    text_path = make_tiny_text(tmp_path / "text")
    cfg = Config(signal_encoder="tiny", text_model=str(text_path), freeze_text=True,
                 pretrained_signal=False, projection_dim=16, hidden_dim=16, local_files_only=True)
    model = FaultClassifier(cfg, 4).train()
    assert not model.text_encoder.training
    assert not any(p.requires_grad for p in model.text_encoder.parameters())
    tokens = {"input_ids": torch.tensor([[0, 4, 5, 2], [0, 4, 7, 2]]),
              "attention_mask": torch.ones(2, 4, dtype=torch.long)}
    logits = model(torch.randn(2, 1, 32, 32), tokens)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert model.text_projection.weight.grad is not None
    assert all(p.grad is None for p in model.text_encoder.parameters())
    assert any(p.grad is not None for p in model.signal_encoder.parameters())


def test_ablation_models_forward_backward():
    for fusion in ("image", "numeric"):
        cfg = Config(signal_encoder="tiny", fusion=fusion, pretrained_signal=False, projection_dim=16)
        model = FaultClassifier(cfg, 4, numeric_dim=5)
        logits = model(torch.randn(2, 1, 32, 32), numeric=torch.randn(2, 5))
        assert logits.shape == (2, 4)
        logits.sum().backward()


def test_resnet_forward_without_download():
    pytest.importorskip("torchvision")
    cfg = Config(signal_encoder="resnet18", input_kind="png", fusion="image", pretrained_signal=False)
    model = FaultClassifier(cfg, 4).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 4)


def test_passt_forward_backward_without_download():
    pytest.importorskip("hear21passt.models.passt")
    cfg = Config(signal_encoder="passt", fusion="image", pretrained_signal=False, width=32)
    model = FaultClassifier(cfg, 4)
    logits = model(torch.randn(2, 1, 128, 32))
    assert logits.shape == (2, 4)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert model.signal_encoder.backbone.patch_embed.proj.weight.grad is not None



def test_metrics_known_confusion_matrix():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["macro_f1"] == pytest.approx((2/3+0.8)/2)


def test_offline_end_to_end_and_checkpoint_parity(tmp_path):
    root = smoke(tmp_path / "smoke")
    checkpoint = root / "training" / "best.pt"
    assert (root / "prepared" / "images" / "r00000_s000000000.png").is_file()
    assert len(json.loads((root / "training" / "history.json").read_text())) == 2
    model, tokenizer, saved = load_checkpoint(checkpoint, torch.device("cpu"))
    assert saved["classes"] == ["ball", "inner", "normal", "outer"]
    assert not model.training and tokenizer is not None
    # The training-time best checkpoint and reloaded inference must give identical probabilities.
    expected = read_csv(root / "training" / "test_predictions.csv")
    actual = read_csv(root / "predictions.csv")
    assert len(actual) == len(expected) == 16
    for a, b in zip(expected, actual):
        assert a["predicted_label"] == b["predicted_label"]
        for label in saved["classes"]:
            assert float(a[f"p_{label}"]) == pytest.approx(float(b[f"p_{label}"]), abs=1e-7)
    # Production inference must accept manifests without ground-truth labels.
    source = read_csv(root / "prepared" / "manifest.csv")
    for row in source:
        row.pop("label")
    path = root / "prepared" / "unlabeled.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source[0]))
        writer.writeheader()
        writer.writerows(source)
    assert len(predict(checkpoint, path, root / "unlabeled_predictions.csv", device_name="cpu")) == 16
