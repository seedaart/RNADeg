"""Evaluate the released DegScore-XGBoost checkpoint on OpenVaccine JSONL."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from inference import DATA_DIR, HERE, MODEL_PATH, encode_sequence, load_model

SPLITS = ("public", "private")
TARGETS = ("reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C")


def evaluate(model, data_path, output_path, target, filtered):
    sum_squared_error = 0.0
    constructs = 0
    positions = 0
    with data_path.open() as data, output_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("id", "position", "target", "observed", "predicted"))
        for line in data:
            row = json.loads(line)
            if filtered and row["test_filter"] != 1:
                continue
            sequence = row["sequence"]
            scored = row["seq_scored"]
            observed = np.asarray(row[target][:scored], dtype=np.float64)
            if not 0 < scored <= len(sequence) or observed.shape != (scored,):
                raise ValueError(f"Invalid scored region for {row['id']}")
            if not np.isfinite(observed).all():
                raise ValueError(f"Non-finite {target} labels for {row['id']}")
            predicted = model.inplace_predict(encode_sequence(sequence)[:scored])
            if predicted.shape != observed.shape or not np.isfinite(predicted).all():
                raise ValueError(f"Invalid checkpoint predictions for {row['id']}")
            sum_squared_error += float(np.square(predicted - observed).sum())
            positions += scored
            constructs += 1
            writer.writerows(
                (row["id"], i, target, truth, prediction)
                for i, (truth, prediction) in enumerate(zip(observed, predicted))
            )
    if not positions:
        raise ValueError(f"No scored positions in {data_path}")
    return {
        "target": target,
        "constructs": constructs,
        "positions": positions,
        "rmse": float(np.sqrt(sum_squared_error / positions)),
        "test_filter": 1 if filtered else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--target", choices=TARGETS, required=True,
                        help="OpenVaccine assay to compare against the scalar checkpoint output")
    parser.add_argument("--private-test-filter", action="store_true",
                        help="Score only private constructs marked test_filter=1")
    args = parser.parse_args()

    try:
        model = load_model(args.model)
    except ValueError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for split in SPLITS:
        metrics[split] = evaluate(
            model,
            args.data_dir / f"{split}_test/{split}_test.json",
            args.output_dir / f"{split}_predictions.csv",
            args.target,
            split == "private" and args.private_test_filter,
        )
        print(f"{split}: {metrics[split]['rmse']:.5f} RMSE", flush=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
