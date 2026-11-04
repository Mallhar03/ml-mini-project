"""Shared experiment configuration and output paths."""

from pathlib import Path

SEED = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
CV_REPEATS = 10
NOISE_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]
NOISE_SEEDS = [42, 43, 44, 45, 46]
ENSEMBLE_SIZES = [1, 2, 5, 10, 20, 50, 100, 200, 300]

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

for directory in (FIGURES, TABLES):
    directory.mkdir(parents=True, exist_ok=True)
