"""Helpers for loading the prepared splits and building reference samples."""
import numpy as np
import pandas as pd

from src.config import ALL_FEATURES, PROCESSED_DIR, RANDOM_STATE, TARGET


def load_splits(processed_dir=PROCESSED_DIR):
    parts = {}
    for name in ["train", "valid", "test"]:
        path = processed_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.day2_prepare` first."
            )
        parts[name] = pd.read_parquet(path)
    return parts["train"], parts["valid"], parts["test"]


def make_reference(model, df, n=50000):
    """A sample of the training data plus the model's score on it.
    Later batches are compared with this sample to measure drift (PSI / CSI)."""
    sample = df.sample(min(n, len(df)), random_state=RANDOM_STATE)
    ref = sample[ALL_FEATURES].copy()
    ref["score"] = model.predict_proba(sample)
    ref[TARGET] = sample[TARGET].to_numpy()
    return ref.reset_index(drop=True)


def add_quarter(df):
    out = df.copy()
    out["quarter"] = out["issue_date"].dt.to_period("Q")
    return out
