import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

from .metadata import operating_text


REQUIRED = {"signal_path", "sample_rate", "label", "group_id", "split", "bearing_model", "rpm", "load", "load_unit"}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_signal(path, key="", channel=""):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".mat":
        from scipy.io import loadmat
        if not key:
            raise ValueError("MAT files require an explicit signal_key")
        array = loadmat(path)[key]
    elif path.suffix.lower() == ".csv":
        array = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError("Supported raw signals: .npy, numeric headerless .csv, .mat")
    array = np.asarray(array, dtype=np.float64).squeeze()
    if array.ndim == 2:
        if channel == "":
            raise ValueError("Multichannel signals require channel (column index)")
        array = array[:, int(channel)]
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Signal must be a finite 1-D vector after channel selection")
    return array


def log_spectrogram(samples, sample_rate, n_fft=256, hop=64, fmin=0.0, fmax=None):
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be finite and positive")
    if not 0 < hop <= n_fft <= len(samples):
        raise ValueError("Require 0 < hop <= n_fft <= window length")
    fmax = sample_rate / 2 if fmax is None else fmax
    if not 0 <= fmin < fmax <= sample_rate / 2:
        raise ValueError("Require 0 <= fmin < fmax <= Nyquist frequency")
    frequencies, _, spectrum = scipy_signal.stft(
        samples - samples.mean(), fs=sample_rate, window="hann", nperseg=n_fft,
        noverlap=n_fft-hop, boundary=None, padded=False)
    keep = (frequencies >= fmin) & (frequencies <= fmax)
    if keep.sum() < 2:
        raise ValueError("Selected frequency range contains fewer than two bins")
    return (10 * np.log10(np.maximum(np.abs(spectrum[keep]) ** 2, 1e-12))).astype(np.float32)


def validate_recordings(rows, base, require_all_splits=True):
    if not rows or not REQUIRED.issubset(rows[0]):
        raise ValueError(f"Recording CSV requires {sorted(REQUIRED)}")
    groups, paths, rates = {}, {}, set()
    for row in rows:
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if not row["group_id"].strip() or not row["label"].strip():
            raise ValueError("group_id and label cannot be empty")
        operating_text(row)
        rates.add(float(row["sample_rate"]))
        group = row["group_id"]
        path = str((base / row["signal_path"]).resolve())
        if groups.setdefault(group, row["split"]) != row["split"]:
            raise ValueError(f"Group {group} crosses dataset splits")
        if paths.setdefault(path, row["split"]) != row["split"]:
            raise ValueError("The same raw recording crosses dataset splits")
    if len(rates) != 1:
        raise ValueError("Use one sample rate (resample beforehand); otherwise frequency pixels differ")
    if require_all_splits and {r["split"] for r in rows} != {"train", "val", "test"}:
        raise ValueError("Provide nonempty train, val, and test recording groups")


def prepare(recordings, output, window=2048, step=2048, n_fft=256, hop=64, fmin=0.0, fmax=None,
            reference_preprocessing=None):
    recordings, output = Path(recordings).resolve(), Path(output).resolve()
    rows = read_csv(recordings)
    validate_recordings(rows, recordings.parent, require_all_splits=reference_preprocessing is None)
    settings = {"window": window, "step": step, "n_fft": n_fft, "hop": hop, "fmin": fmin,
                "fmax": fmax, "sample_rate": float(rows[0]["sample_rate"])}
    reference = None
    if reference_preprocessing:
        reference = json.loads(Path(reference_preprocessing).read_text(encoding="utf-8"))
        if reference["preprocessing"] != settings:
            raise ValueError("Inference STFT settings must match reference_preprocessing")
    if window < n_fft or step <= 0:
        raise ValueError("window >= n_fft and step > 0 are required")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; use a new path")
    (output / "spectrograms").mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir()
    prepared, total, square, count = [], 0.0, 0.0, 0
    digests = {}
    for recording_index, row in enumerate(rows):
        raw_path = recordings.parent / row["signal_path"]
        samples = load_signal(raw_path, row.get("signal_key", ""), row.get("channel", ""))
        digest = hashlib.sha256(samples.astype("<f8").tobytes()).hexdigest()
        if digests.setdefault(digest, row["split"]) != row["split"]:
            raise ValueError("Duplicate raw signal content detected across splits")
        if len(samples) < window:
            raise ValueError(f"{raw_path.name}: shorter than window={window}")
        for start in range(0, len(samples)-window+1, step):
            spec = log_spectrogram(samples[start:start+window], float(row["sample_rate"]), n_fft, hop, fmin, fmax)
            sample_id = f"r{recording_index:05d}_s{start:09d}"
            spec_path = f"spectrograms/{sample_id}.npy"
            np.save(output / spec_path, spec, allow_pickle=False)
            prepared.append({"sample_id": sample_id, "spectrogram_path": spec_path,
                             "image_path": f"images/{sample_id}.png", "label": row["label"],
                             "split": row["split"], "group_id": row["group_id"],
                             "bearing_model": row["bearing_model"], "rpm": row["rpm"],
                             "load": row["load"], "load_unit": row["load_unit"],
                             "sample_rate": row["sample_rate"], "start": start})
            if row["split"] == "train":
                values = spec.astype(np.float64)
                total += values.sum()
                square += (values * values).sum()
                count += values.size
    if reference is None:
        mean = total/count
        std = max(float(np.sqrt(max(square/count-mean*mean, 0))), 1e-6)
        stats = {"mean": float(mean), "std": std, "png_low": float(mean-3*std), "png_high": float(mean+3*std)}
    else:
        stats = reference["normalization"]
    from PIL import Image
    for row in prepared:
        spec = np.load(output / row["spectrogram_path"], allow_pickle=False)
        gray = np.clip((spec-stats["png_low"])/(stats["png_high"]-stats["png_low"]), 0, 1)
        # PNGs show low frequencies at the bottom; loading flips them back.
        Image.fromarray(np.rint(np.flipud(gray)*255).astype(np.uint8)).save(output / row["image_path"])
    metadata = {"normalization": stats, "preprocessing": settings}

    (output / "preprocessing.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_csv(output / "manifest.csv", prepared)
    return output / "manifest.csv"
