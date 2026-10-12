import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .metadata import numeric_vector, operating_text
from .preprocess import read_csv


def validate_manifest(rows, root):
    if not rows:
        raise ValueError("Manifest is empty")
    required = {"sample_id", "spectrogram_path", "image_path", "label", "split", "group_id",
                "bearing_model", "rpm", "load", "load_unit"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Manifest requires {sorted(required)}")
    groups, paths, ids = {}, {}, set()
    for row in rows:
        if row["sample_id"] in ids:
            raise ValueError("Duplicate sample_id")
        ids.add(row["sample_id"])
        if row["split"] not in {"train", "val", "test"} or not row["group_id"].strip():
            raise ValueError("Invalid split or empty group_id")
        if groups.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise ValueError("A group crosses splits; split recordings before windowing")
        for key in ("spectrogram_path", "image_path"):
            resolved = str((root / row[key]).resolve())
            if paths.setdefault(resolved, row["split"]) != row["split"]:
                raise ValueError("A sample path crosses splits")
        operating_text(row)
    for split in ("train", "val", "test"):
        if not any(r["split"] == split for r in rows):
            raise ValueError(f"Missing {split} split")
    labels = sorted({r["label"] for r in rows if r["split"] == "train"})
    if any(r["label"] not in labels for r in rows):
        raise ValueError("Validation/test labels must occur in training")
    return labels


def load_manifest(path):
    path = Path(path).resolve()
    rows = read_csv(path)
    labels = validate_manifest(rows, path.parent)
    stats = json.loads((path.parent / "preprocessing.json").read_text(encoding="utf-8"))
    normalization = stats["normalization"]
    if not np.isfinite([normalization["mean"], normalization["std"]]).all() or normalization["std"] <= 0:
        raise ValueError("Invalid training normalization")
    return rows, labels, stats, path.parent


def load_visual(row, root, cfg, normalization):
    if cfg.input_kind == "spectrogram":
        values = np.load(root / row["spectrogram_path"], allow_pickle=False)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Spectrogram must be a finite 2-D log-power matrix")
        values = (values.astype(np.float32)-normalization["mean"])/normalization["std"]
        tensor = torch.from_numpy(values.copy())[None]
    else:
        from PIL import Image
        with Image.open(root / row["image_path"]) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float32)/255
        values = gray*(normalization["png_high"]-normalization["png_low"])+normalization["png_low"]
        values = np.flipud((values-normalization["mean"])/normalization["std"]).copy()
        tensor = torch.from_numpy(values)[None]
    tensor = F.interpolate(tensor[None], size=(cfg.height, cfg.width), mode="bilinear",
                           align_corners=False)[0]
    if cfg.signal_encoder == "resnet18":
        # Both PNG and NPY use the same training-statistics intensity mapping.
        tensor = ((tensor.clamp(-3, 3)+3)/6).expand(3, -1, -1)
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = (tensor-mean)/std
    return tensor


class FaultDataset(Dataset):
    def __init__(self, rows, root, cfg, labels, normalization, numeric_stats=None, inference=False):
        self.rows, self.root, self.cfg = rows, Path(root), cfg
        self.labels, self.normalization, self.numeric_stats = labels, normalization, numeric_stats
        self.inference = inference

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        visual = (torch.empty(0) if self.cfg.fusion == "text_only" else
                  load_visual(row, self.root, self.cfg, self.normalization))
        numeric = (torch.from_numpy(numeric_vector(row, self.numeric_stats))
                   if self.cfg.fusion == "numeric" else torch.empty(0))
        return {"visual": visual, "text": operating_text(row), "numeric": numeric,
                "label": -1 if self.inference else self.labels.index(row["label"]), "sample_id": row["sample_id"]}
