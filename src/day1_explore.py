"""Step 1: load Lending Club accepted loans, look at the basics, save a trimmed Parquet copy.
(The same step done in PySpark is in src/spark_prepare.py.)

Run from the project root:
    python -m src.day1_explore
"""
import sys

import pandas as pd

from src.config import (BAD_STATUS, GOOD_STATUS, KEEP_COLUMNS, RAW_DIR,
                        RAW_SUBSET)


def find_raw_file(raw_dir=RAW_DIR):
    files = [
        p for p in raw_dir.glob("*")
        if (p.suffix == ".csv" or p.name.endswith(".csv.gz"))
        and "reject" not in p.name.lower()
    ]
    if not files:
        sys.exit(f"No accepted-loans file found in {raw_dir}. Put the downloaded file there.")
    return files[0]


def load_raw(path, verbose=True):
    """Read only the needed columns and add issue_date / issue_year."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    use = [c for c in KEEP_COLUMNS if c in header]
    missing = [c for c in KEEP_COLUMNS if c not in header]
    if verbose:
        print(f"Columns in file: {len(header)} | kept: {len(use)} | not found: {missing}")
    df = pd.read_csv(path, usecols=use, low_memory=False, dtype={"id": str})
    if verbose:
        print(f"\nRows read: {len(df):,}")
    # Some versions of the file end with summary rows that have no issue date.
    df = df.dropna(subset=["issue_d", "loan_status"])
    df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_date"])
    df["issue_year"] = df["issue_date"].dt.year
    if verbose:
        print(f"Rows after removing rows without a valid issue date: {len(df):,}")
    return df


def main():
    path = find_raw_file()
    print(f"Reading {path.name} ...")
    df = load_raw(path)

    print("\nLoan status counts:")
    print(df["loan_status"].value_counts())

    print("\nLoans per issue year (all statuses):")
    print(df["issue_year"].value_counts().sort_index())

    final = df[df["loan_status"].isin(GOOD_STATUS + BAD_STATUS)].copy()
    final["bad"] = final["loan_status"].isin(BAD_STATUS).astype(int)
    print("\nBad-loan rate by issue year (final outcomes only):")
    print(final.groupby("issue_year")["bad"].agg(["count", "mean"]).round(3))

    if "term" in df.columns:
        print("\nTerm values:")
        print(df["term"].value_counts())

    RAW_SUBSET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_SUBSET, index=False)
    print(f"\nSaved trimmed data to {RAW_SUBSET}")


if __name__ == "__main__":
    main()
