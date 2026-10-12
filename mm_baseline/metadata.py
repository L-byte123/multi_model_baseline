import math
import numpy as np


def operating_text(row):
    """Only allowlisted equipment fields; labels/paths never enter the prompt."""
    model = str(row["bearing_model"]).strip()
    unit = str(row["load_unit"]).strip()
    rpm, load = float(row["rpm"]), float(row["load"])
    if not model or not unit or not math.isfinite(rpm) or not math.isfinite(load):
        raise ValueError("Equipment fields must be present and numeric values finite")
    if rpm < 0 or load < 0:
        raise ValueError("rpm and load must be nonnegative")
    return f"Bearing model: {model}. Shaft speed: {rpm:g} rpm. Load: {load:g} {unit}."


def fit_numeric(rows):
    units = {r["load_unit"] for r in rows}
    if len(units) != 1:
        raise ValueError("Numeric ablation requires a single, consistent load unit")
    values = np.asarray([[float(r["rpm"]), float(r["load"])] for r in rows], dtype=np.float64)
    return {"mean": values.mean(0).tolist(), "std": np.maximum(values.std(0), 1e-6).tolist(),
            "models": sorted({r["bearing_model"] for r in rows}), "load_unit": next(iter(units))}


def numeric_vector(row, stats):
    if row["load_unit"] != stats["load_unit"]:
        raise ValueError("Load unit differs from the training unit; convert units before training")
    values = (np.asarray([float(row["rpm"]), float(row["load"])]) - stats["mean"]) / stats["std"]
    models = stats["models"]
    category = np.zeros(len(models) + 1, dtype=np.float32)
    category[models.index(row["bearing_model"]) if row["bearing_model"] in models else len(models)] = 1
    return np.concatenate([values, category]).astype(np.float32)
