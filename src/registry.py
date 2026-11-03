"""Saving and loading models on disk.

A model folder contains:
    model.joblib       the fitted model
    metadata.json      version, metrics, approval threshold, training window
    reference.parquet  a sample of the training data and its scores (used for drift checks)
The API loads the folder `models/champion/`.
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd


def save_model(model, directory, metadata, reference=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    meta = dict(metadata)
    meta.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    meta["model_type"] = getattr(model, "model_type", "unknown")
    with open(directory / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    if reference is not None:
        reference.to_parquet(directory / "reference.parquet", index=False)
    return directory


def load_model(directory):
    directory = Path(directory)
    model = joblib.load(directory / "model.joblib")
    with open(directory / "metadata.json") as f:
        metadata = json.load(f)
    return model, metadata


def load_reference(directory):
    return pd.read_parquet(Path(directory) / "reference.parquet")


def copy_model(src, dst):
    src, dst = Path(src), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst
