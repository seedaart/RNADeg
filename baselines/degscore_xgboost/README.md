# OpenVaccine DegScore-XGBoost

Inference baseline for the ML4FG RNA degradation project. This uses the single
pretrained BT checkpoint and the loop annotation and feature encoder from the
[authors' KaggleOpenVaccine release](https://github.com/eternagame/KaggleOpenVaccine/tree/b95af835cc38bd7dc55ed6e7397d124787f86996).
It calls ViennaRNA directly through its Python bindings for the minimum-free-energy
fold. The encoder uses the full sequence, with a radius of 20 nucleotides around
each position and 451 features per position. Scoring uses the measured prefix
given by each record's `seq_scored`.

## Run

Use Python 3.13 and Git LFS. On macOS, install OpenMP for XGBoost with
`brew install libomp`.

From this directory:

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
./download_model.sh
python evaluate.py --target deg_Mg_pH10
```

The downloader verifies the published Git LFS SHA-256. The evaluator reads
`ProbML/Project/data/public_test/public_test.json` and
`ProbML/Project/data/private_test/private_test.json` by default.
Use `--data-dir` if the data live elsewhere. It writes one prediction CSV per
split with zero-based nucleotide positions and a `results/metrics.json` summary.
The default evaluates all 629 public and 3,005 private constructs in the local
files. To score the 1,172 private constructs flagged by the local release, add
`--private-test-filter` and use a separate `--output-dir`.

## Metric and scope

The released checkpoint gives one prediction per nucleotide. The upstream
notebook names its predictions for a one-day pH 10 assay and does not specify the
magnesium condition for the checkpoint. The example explicitly compares to
`deg_Mg_pH10` as a project choice. The script reports RMSE for the chosen assay.
The [paper's Table 1](https://doi.org/10.1038/s42256-022-00571-8) reports
three-target MCRMSE on leaderboard subsets of 400 public and 1,801 private
constructs. The local RMSE and those published MCRMSE scores measure different
quantities.

## Export predictions

```sh
python -m pip install torch
python export_predictions.py
```

This writes matching `train_xgb`, `public_test_xgb`, and `private_test_xgb`
CSV and PyTorch files under `results/predictions/`. Every input construct and
every nucleotide is included, including positions beyond `seq_scored`. Each
`.pt` file contains a float32 tensor with axes
`[construct, nucleotide, checkpoint output]`. Constructs follow the source
JSONL row order and nucleotide indices start at zero. The corresponding CSV
gives the construct ID and position for each prediction. The final axis has
one channel, `bt_xgb`, the single released DegScore-XGBoost output.
