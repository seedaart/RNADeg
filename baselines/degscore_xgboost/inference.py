"""Shared checkpoint loading and feature generation for DegScore-XGBoost."""
from pathlib import Path

import RNA
import xgboost as xgb

from vendor.encoder import encode_input, write_bprna_string

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[4] / "ProbML/Project/data"
MODEL_PATH = HERE / "checkpoint/bt_xgb.model"


def load_model(path):
    if not path.is_file() or path.stat().st_size < 1024:
        raise ValueError(f"Missing checkpoint at {path}. Run download_model.sh first.")
    model = xgb.Booster(params={"nthread": 1})
    model.load_model(path)
    if model.num_features() != 451:
        raise ValueError(f"Expected 451 DegScore features, got {model.num_features()}.")
    return model


def encode_sequence(sequence):
    structure, _ = RNA.fold(sequence)
    return encode_input(sequence, write_bprna_string(structure), window_size=20)
