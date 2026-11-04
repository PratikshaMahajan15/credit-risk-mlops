"""PySpark data prep must give the same rows and values as the pandas version.
Skipped automatically if PySpark or Java is not available."""
import gzip
import shutil

import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Java not installed", allow_module_level=True)

from src import day1_explore, day2_prepare, spark_prepare
from src.config import KEEP_COLUMNS, NUMERIC_FEATURES


def raw_file(tmp_path, n=4000):
    rng = np.random.default_rng(5)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    years = rng.choice(range(2011, 2018), n)
    df = pd.DataFrame({
        "id": [str(i) for i in range(n)], "loan_amnt": rng.integers(1000, 35000, n).astype(float),
        "term": rng.choice([" 36 months", " 60 months"], n), "int_rate": rng.uniform(5, 25, n),
        "installment": rng.uniform(50, 900, n), "grade": "B", "sub_grade": "B2",
        "emp_length": rng.choice(["< 1 year", "1 year", "5 years", "10+ years", None], n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
        "annual_inc": np.where(rng.random(n) < 0.02, 0.0, rng.uniform(20000, 150000, n)),
        "verification_status": "Verified",
        "issue_d": [f"{months[rng.integers(0, 12)]}-{y}" for y in years],
        "loan_status": rng.choice(["Fully Paid", "Charged Off", "Current", "Default", "Late (31-120 days)"], n,
                                  p=[0.6, 0.2, 0.1, 0.02, 0.08]),
        "purpose": rng.choice(["debt_consolidation", "credit_card"], n), "addr_state": rng.choice(["CA", "NY"], n),
        "dti": rng.uniform(0, 40, n), "delinq_2yrs": 0.0, "earliest_cr_line": "Jan-2000",
        "fico_range_low": 680.0, "fico_range_high": 684.0, "inq_last_6mths": 1.0, "open_acc": 10.0,
        "pub_rec": 0.0, "revol_bal": 5000.0, "revol_util": rng.uniform(0, 100, n), "total_acc": 20.0,
        "application_type": "Individual", "mort_acc": np.where(rng.random(n) < 0.3, np.nan, 1.0),
        "pub_rec_bankruptcies": 0.0,
    })[KEEP_COLUMNS]
    path = tmp_path / "accepted_test.csv.gz"
    df.to_csv(path, index=False, compression="gzip")
    return path


def test_spark_matches_pandas(tmp_path):
    path = raw_file(tmp_path)
    pandas_splits = day2_prepare.prepare_frame(day1_explore.load_raw(path, verbose=False), verbose=False)
    spark_splits = spark_prepare.main(processed_dir=tmp_path / "spark_out", path=path)
    for name in ["train", "valid", "test"]:
        a = pandas_splits[name].assign(id=lambda d: d["id"].astype(str)).sort_values("id").reset_index(drop=True)
        b = spark_splits[name].assign(id=lambda d: d["id"].astype(str)).sort_values("id").reset_index(drop=True)
        assert len(a) == len(b) > 0
        assert (a["id"] == b["id"]).all()
        assert (a["bad"] == b["bad"]).all()
        for col in NUMERIC_FEATURES:
            np.testing.assert_allclose(a[col].astype(float), b[col].astype(float), rtol=1e-6, equal_nan=True)


def test_hdfs_etl_round_trip_matches_pandas(tmp_path):
    """The HDFS ETL (here pointed at a local folder instead of hdfs://) gives the same data as pandas,
    and really writes the three Parquet splits to the output location."""
    from src import spark_hdfs_etl

    path = raw_file(tmp_path)
    out_dir = tmp_path / "hdfs_sim"
    pandas_splits = day2_prepare.prepare_frame(day1_explore.load_raw(path, verbose=False), verbose=False)
    spark_splits = spark_hdfs_etl.run(str(path), out_dir.as_uri(), local_dir=tmp_path / "local_copy")
    for name in ["train", "valid", "test"]:
        assert (out_dir / name).exists() and any((out_dir / name).glob("*.parquet"))
        assert (tmp_path / "local_copy" / f"{name}.parquet").exists()
        a = pandas_splits[name].assign(id=lambda d: d["id"].astype(str)).sort_values("id").reset_index(drop=True)
        b = spark_splits[name].assign(id=lambda d: d["id"].astype(str)).sort_values("id").reset_index(drop=True)
        assert len(a) == len(b) > 0
        assert (a["bad"] == b["bad"]).all()
        for col in NUMERIC_FEATURES:
            np.testing.assert_allclose(a[col].astype(float), b[col].astype(float), rtol=1e-6, equal_nan=True)
