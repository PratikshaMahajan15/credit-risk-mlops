"""Step 2: define the target, keep only loans with finished outcomes, build features,
and split by issue year (train / validation / out-of-time test).
(The same step done in PySpark is in src/spark_prepare.py.)

Run from the project root:
    python -m src.day2_prepare
"""
import json

import pandas as pd

from src.config import (BAD_STATUS, CATEGORICAL_FEATURES, GOOD_STATUS,
                        NUMERIC_FEATURES, PROCESSED_DIR, RAW_SUBSET, TARGET,
                        TEST_YEARS, TERM, TRAIN_YEARS, VALID_YEARS)

EMP_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


def show_maturity_table(df):
    """Share of loans with a FINAL outcome, by issue year and term.
    This shows why we use 36-month loans only: later loans have not finished yet."""
    final_statuses = GOOD_STATUS + BAD_STATUS
    tmp = df.assign(is_final=df["loan_status"].isin(final_statuses))
    pivot = tmp.pivot_table(index="issue_year", columns="term", values="is_final", aggfunc="mean")
    print("\nShare of loans with a final outcome, by issue year and term:")
    print((pivot * 100).round(1).to_string())


def build_features(df):
    out = df.copy()
    out["emp_length_years"] = out["emp_length"].map(EMP_MAP)
    cr_line = pd.to_datetime(out["earliest_cr_line"], format="%b-%Y", errors="coerce")
    out["credit_history_years"] = (out["issue_date"] - cr_line).dt.days / 365.25
    out["loan_to_income"] = out["loan_amnt"] / out["annual_inc"].where(out["annual_inc"] > 0)
    return out


def prepare_frame(df, verbose=True):
    """Raw trimmed loans -> dict of train / valid / test DataFrames."""
    df = df.copy()
    df["term"] = df["term"].str.strip()   # the raw file has a leading space
    if verbose:
        print(f"Loaded {len(df):,} loans")
        show_maturity_table(df)

    # 1. Target: only loans whose outcome is final. Everything else (Current, Late, ...) is dropped.
    df = df[df["loan_status"].isin(GOOD_STATUS + BAD_STATUS)].copy()
    df[TARGET] = df["loan_status"].isin(BAD_STATUS).astype(int)
    # 2. One term only
    df = df[df["term"] == TERM].copy()
    # 3. Years used for the pipeline
    years = TRAIN_YEARS + VALID_YEARS + TEST_YEARS
    df = df[df["issue_year"].isin(years)].copy()
    if verbose:
        print(f"\nAfter final outcomes, {TERM} loans and issue years {min(years)}-{max(years)}: {len(df):,} loans")

    # 4. Features
    df = build_features(df)
    keep = ["id", "issue_date", "issue_year", TARGET] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    df = df[keep]

    # 5. Time-based split
    return {
        "train": df[df["issue_year"].isin(TRAIN_YEARS)],
        "valid": df[df["issue_year"].isin(VALID_YEARS)],
        "test": df[df["issue_year"].isin(TEST_YEARS)],
    }


def report_and_save(splits, processed_dir=PROCESSED_DIR):
    df = pd.concat(splits.values())
    print("\nSplit summary:")
    for name, part in splits.items():
        print(f"  {name:5s} years {[int(y) for y in sorted(part['issue_year'].unique())]}  "
              f"rows {len(part):>8,}  bad rate {part[TARGET].mean():.3f}")

    print("\nBad rate by issue quarter:")
    q = df.assign(quarter=df["issue_date"].dt.to_period("Q")).groupby("quarter")[TARGET].agg(["count", "mean"])
    print(q.round(3).to_string())

    print("\nMissing values (% of rows) in each split:")
    miss = pd.DataFrame({
        name: part[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().mean() * 100
        for name, part in splits.items()
    }).round(1)
    print(miss[(miss > 0).any(axis=1)].to_string())

    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(processed_dir / f"{name}.parquet", index=False)
    with open(processed_dir / "features.json", "w") as f:
        json.dump({"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES, "target": TARGET}, f, indent=2)
    print(f"\nSaved train/valid/test and features.json to {processed_dir}")


def main():
    df = pd.read_parquet(RAW_SUBSET)
    splits = prepare_frame(df)
    report_and_save(splits)


if __name__ == "__main__":
    main()
