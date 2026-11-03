"""Synthetic stand-in for the prepared Lending Club splits, used by the tests and for smoke runs.
The relationship between inputs and default changes in later years, so drift is visible."""
import numpy as np
import pandas as pd

from src.config import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET,
                        TEST_YEARS, TRAIN_YEARS, VALID_YEARS)


def make_year(year, n, rng, decay=0.0, input_shift=0.8):
    shift = year - 2012                               # later years drift
    df = pd.DataFrame({
        "loan_amnt": rng.integers(2000, 35000, n) * (1 + 0.04 * shift),
        "annual_inc": rng.lognormal(11, 0.5, n),
        "dti": rng.uniform(0, 35, n) + input_shift * shift,
        "delinq_2yrs": rng.poisson(0.3, n),
        "fico_range_low": rng.normal(695 - 2 * shift, 30, n).round(),
        "inq_last_6mths": rng.poisson(0.8, n),
        "open_acc": rng.poisson(10, n),
        "pub_rec": rng.poisson(0.15, n),
        "revol_bal": rng.lognormal(9, 1, n),
        "revol_util": rng.uniform(0, 100, n),
        "total_acc": rng.poisson(24, n),
        "mort_acc": np.where(rng.random(n) < 0.25, np.nan, rng.poisson(1.2, n)),
        "pub_rec_bankruptcies": rng.poisson(0.1, n),
        "emp_length_years": np.where(rng.random(n) < 0.07, np.nan, rng.integers(0, 11, n)),
        "credit_history_years": rng.uniform(2, 35, n),
    })
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]
    df["home_ownership"] = rng.choice(["RENT", "OWN", "MORTGAGE"], n, p=[0.4, 0.1, 0.5])
    df["verification_status"] = rng.choice(["Verified", "Not Verified", "Source Verified"], n)
    df["purpose"] = rng.choice(["debt_consolidation", "credit_card", "car", "other"], n)
    df["addr_state"] = rng.choice(["CA", "NY", "TX", "FL", "IL"], n)
    dti_w = max(0.045 + 0.02 * shift - decay * shift, 0.0)   # dti matters more (or less, with decay) later
    logit = (-2.0 + dti_w * (df["dti"] - 18) - 0.012 * (df["fico_range_low"] - 690)
             + 0.5 * df["loan_to_income"].clip(upper=1) + 0.25 * (df["home_ownership"] == "RENT")
             + 0.15 * df["inq_last_6mths"] - 0.01 * (df["credit_history_years"] - 15))
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    months = rng.integers(1, 13, n)
    df["issue_date"] = pd.to_datetime({"year": year, "month": months, "day": 1})
    df["issue_year"] = year
    df["id"] = [f"{year}-{i}" for i in range(n)]
    return df


def make_splits(n_per_year=3000, seed=0, decay=0.0, input_shift=0.8):
    rng = np.random.default_rng(seed)
    def block(years):
        return pd.concat([make_year(y, n_per_year, rng, decay, input_shift) for y in years], ignore_index=True)
    return block(TRAIN_YEARS), block(VALID_YEARS), block(TEST_YEARS)


def write_splits(directory, n_per_year=3000, seed=0, decay=0.0, input_shift=0.8):
    directory.mkdir(parents=True, exist_ok=True)
    train, valid, test = make_splits(n_per_year, seed, decay, input_shift)
    for name, df in [("train", train), ("valid", valid), ("test", test)]:
        df.to_parquet(directory / f"{name}.parquet", index=False)
    return train, valid, test
