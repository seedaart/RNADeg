"""Export full-sequence predictions in the source JSONL row order."""
import argparse
import csv
import json
import subprocess
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from inference import DATA_DIR, HERE, MODEL_PATH, encode_sequence, load_model

SPLITS = {
    "train": "stanford-covid-vaccine/train.json",
    "public_test": "public_test/public_test.json",
    "private_test": "private_test/private_test.json",
}


def export_split(model, source, destination):
    predictions = []
    with source.open() as data:
        rows = [json.loads(line) for line in data]
    for row in rows:
        sequence = row["sequence"]
        predicted = model.inplace_predict(encode_sequence(sequence))
        if predicted.shape != (len(sequence),) or not np.isfinite(predicted).all():
            raise ValueError(f"Invalid checkpoint predictions for {row['id']}")
        predictions.append(predicted)

    values = np.stack(predictions).astype(np.float32, copy=False)[..., None]
    with destination.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("id", "position", "bt_xgb"))
        for row, prediction in zip(rows, values):
            writer.writerows(
                (row["id"], position, value)
                for position, value in enumerate(prediction[:, 0])
            )
    # The macOS XGBoost and PyTorch wheels can load different OpenMP runtimes.
    # Keep serialization in a process that imports only PyTorch and NumPy.
    with TemporaryDirectory() as staging:
        array_path = Path(staging) / "predictions.npy"
        np.save(array_path, values, allow_pickle=False)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, numpy, torch; "
                "torch.save(torch.from_numpy(numpy.load(sys.argv[1])), sys.argv[2])",
                str(array_path),
                str(destination.with_suffix(".pt")),
            ],
            check=True,
        )
    return values.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/predictions")
    args = parser.parse_args()
    model = load_model(MODEL_PATH)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, source in SPLITS.items():
        destination = args.output_dir / f"{split}_xgb"
        shape = export_split(model, args.data_dir / source, destination)
        print(f"{split}: {shape}", flush=True)


if __name__ == "__main__":
    main()
