from pathlib import Path

import numpy as np

from .preprocess import write_csv


def make_demo(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Demo output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    random = np.random.default_rng(42)
    rows = []
    sample_rate = 12000
    time = np.arange(4096)/sample_rate
    for split in ("train", "val", "test"):
        for label in ("normal", "inner", "outer", "ball"):
            for repeat in range(2):
                sid = f"{split}_{label}_{repeat}"
                # Each fault sees the same conditions, so metadata is not a label shortcut.
                rpm, load = 1500+repeat*300, float(repeat)
                frequency = {"normal": 80, "inner": 400, "outer": 700, "ball": 1000}[label]
                samples = np.sin(2*np.pi*frequency*time+random.uniform(0, 2*np.pi))
                samples += 0.1*random.normal(size=time.size)
                np.save(output / f"{sid}.npy", samples.astype(np.float32))
                rows.append({"signal_path": f"{sid}.npy", "sample_rate": sample_rate,
                    "label": label, "group_id": sid, "split": split, "bearing_model": "6205",
                    "rpm": rpm, "load": load, "load_unit": "hp"})
    write_csv(output / "recordings.csv", rows)
    return output / "recordings.csv"
