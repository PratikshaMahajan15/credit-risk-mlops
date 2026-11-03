"""PySpark version of the data steps: read the 2.2 million loan records, build the target and the
features, split by issue year, and write the same train / valid / test files as the pandas version.

Why Spark here: the raw file is a 2.2 million row, 150 column compressed CSV. Spark reads and
transforms it in parallel, and the same code would run on a cluster for much larger data.
The outputs are identical to `day1_explore` + `day2_prepare`, so everything downstream is unchanged.

Needs Java 17 or 21 installed (see README). Run from the project root:
    python -m src.spark_prepare
"""
import json
import sys

import pandas as pd

from src.config import (BAD_STATUS, CATEGORICAL_FEATURES, GOOD_STATUS,
                        KEEP_COLUMNS, NUMERIC_FEATURES, PROCESSED_DIR, RAW_DIR,
                        TARGET, TEST_YEARS, TERM, TRAIN_YEARS, VALID_YEARS)
from src.day1_explore import find_raw_file
from src.day2_prepare import EMP_MAP

RAW_NUMERIC = ["loan_amnt", "annual_inc", "dti", "delinq_2yrs", "fico_range_low", "inq_last_6mths",
               "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
               "pub_rec_bankruptcies"]


def get_spark(app_name="credit-risk-prep", memory="4g"):
    from src.spark_utils import get_spark as _get
    return _get(app_name, memory)


def read_raw(spark, path):
    """Read only the needed columns from the (gzipped) CSV."""
    from pyspark.sql import functions as F

    sdf = spark.read.option("header", True).option("multiLine", False).csv(str(path))
    use = [c for c in KEEP_COLUMNS if c in sdf.columns]
    missing = [c for c in KEEP_COLUMNS if c not in sdf.columns]
    print(f"Columns in file: {len(sdf.columns)} | kept: {len(use)} | not found: {missing}")
    sdf = sdf.select(*use)
    for c in RAW_NUMERIC:
        if c in sdf.columns:
            sdf = sdf.withColumn(c, F.col(c).cast("double"))
    return sdf


def prepare_spark(sdf):
    """Spark DataFrame of raw loans -> Spark DataFrame with target, features and issue year."""
    from pyspark.sql import functions as F

    sdf = sdf.filter(F.col("issue_d").isNotNull() & F.col("loan_status").isNotNull())
    sdf = sdf.withColumn("issue_date", F.to_date(F.col("issue_d"), "MMM-yyyy"))
    sdf = sdf.filter(F.col("issue_date").isNotNull())
    sdf = sdf.withColumn("issue_year", F.year("issue_date"))
    sdf = sdf.withColumn("term", F.trim(F.col("term")))
    return sdf


def build_dataset(sdf):
    """Target, filters, features. Mirrors day2_prepare.prepare_frame."""
    from pyspark.sql import functions as F

    final = GOOD_STATUS + BAD_STATUS
    years = TRAIN_YEARS + VALID_YEARS + TEST_YEARS

    d = sdf.filter(F.col("loan_status").isin(final))
    d = d.withColumn(TARGET, F.when(F.col("loan_status").isin(BAD_STATUS), 1).otherwise(0))
    d = d.filter(F.col("term") == TERM)
    d = d.filter(F.col("issue_year").isin(years))

    emp = F.create_map([F.lit(x) for kv in EMP_MAP.items() for x in kv])
    d = d.withColumn("emp_length_years", emp[F.col("emp_length")].cast("double"))
    cr = F.to_date(F.col("earliest_cr_line"), "MMM-yyyy")
    d = d.withColumn("credit_history_years", F.datediff(F.col("issue_date"), cr) / F.lit(365.25))
    d = d.withColumn("loan_to_income",
                     F.col("loan_amnt") / F.when(F.col("annual_inc") > 0, F.col("annual_inc")))
    keep = ["id", "issue_date", "issue_year", TARGET] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return d.select(*keep)


def show_spark_summaries(sdf):
    """The same overview tables as Step 1, computed by Spark."""
    from pyspark.sql import functions as F

    final = GOOD_STATUS + BAD_STATUS
    print("\nLoan status counts:")
    sdf.groupBy("loan_status").count().orderBy(F.desc("count")).show(12, truncate=False)

    print("Share of loans with a final outcome, by issue year and term (%):")
    (sdf.groupBy("issue_year").pivot("term", ["36 months", "60 months"])
        .agg(F.round(F.avg(F.col("loan_status").isin(final).cast("int")) * 100, 1))
        .orderBy("issue_year").show(20))

    print("Bad-loan rate by issue year (final outcomes only):")
    (sdf.filter(F.col("loan_status").isin(final))
        .groupBy("issue_year")
        .agg(F.count("*").alias("count"),
             F.round(F.avg(F.col("loan_status").isin(BAD_STATUS).cast("int")), 3).alias("bad_rate"))
        .orderBy("issue_year").show(20))


def to_pandas(sdf):
    """Collect a (small) Spark DataFrame into pandas. Uses Arrow, then pandas, so it works with
    both old and new pandas/pyspark versions."""
    try:
        table = sdf.toArrow()                 # PySpark 4+
        return table.to_pandas()
    except Exception:
        return sdf.toPandas()


def main(processed_dir=PROCESSED_DIR, path=None):
    path = path or find_raw_file(RAW_DIR)
    spark = get_spark()
    print(f"Spark {spark.version}. Reading {path.name} ...")
    raw = read_raw(spark, path)
    prepared = prepare_spark(raw)
    prepared.cache()
    print(f"Rows with a valid issue date: {prepared.count():,}")
    show_spark_summaries(prepared)

    data = build_dataset(prepared)
    pdf = to_pandas(data)
    pdf["issue_date"] = pd.to_datetime(pdf["issue_date"])
    pdf["issue_year"] = pdf["issue_year"].astype(int)
    print(f"Modelling dataset: {len(pdf):,} loans")

    processed_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": pdf[pdf["issue_year"].isin(TRAIN_YEARS)],
        "valid": pdf[pdf["issue_year"].isin(VALID_YEARS)],
        "test": pdf[pdf["issue_year"].isin(TEST_YEARS)],
    }
    for name, part in splits.items():
        part.to_parquet(processed_dir / f"{name}.parquet", index=False)
        print(f"  {name:5s} rows {len(part):>8,}  bad rate {part[TARGET].mean():.3f}")
    with open(processed_dir / "features.json", "w") as f:
        json.dump({"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES, "target": TARGET}, f, indent=2)
    print(f"\nSaved train/valid/test to {processed_dir}")
    spark.stop()
    return splits


if __name__ == "__main__":
    main()
