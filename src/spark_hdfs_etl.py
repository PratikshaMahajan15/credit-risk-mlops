"""PySpark ETL that reads the raw loan file from HDFS and writes the prepared splits back to HDFS as Parquet.

Flow
    HDFS  /lending/raw/accepted_2007_to_2018Q4.csv.gz        (2.26 million loans, 150 columns)
      -> PySpark: clean, build the target and features, split by issue year
      -> HDFS  /lending/processed/{train,valid,test}/         (Parquet, written by Spark)
      -> read back from HDFS and save a local copy in data/processed/ for the modelling steps

Runs inside the `spark-etl` Docker container on the same network as the Hadoop containers
(see docker-compose.hdfs.yml). The input and output locations are plain URIs, so the same code
also runs on a local folder: that is how it is tested.

    docker compose -f docker-compose.hdfs.yml --profile etl run --rm spark-etl
"""
import json
import os
from pathlib import Path

import pandas as pd

from src.config import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, PROCESSED_DIR,
                        TARGET, TEST_YEARS, TRAIN_YEARS, VALID_YEARS)
from src.spark_prepare import (build_dataset, get_spark, prepare_spark,
                               read_raw, show_spark_summaries, to_pandas)

HDFS_URI = os.getenv("HDFS_URI", "hdfs://namenode:9000")
RAW_PATH = os.getenv("HDFS_RAW", "/lending/raw/accepted_2007_to_2018Q4.csv.gz")
OUT_DIR = os.getenv("HDFS_OUT", "/lending/processed")
COLUMNS = ["id", "issue_date", "issue_year", TARGET] + NUMERIC_FEATURES + CATEGORICAL_FEATURES


def run(input_uri, output_uri, local_dir=PROCESSED_DIR, memory=None):
    from pyspark.sql import functions as F

    spark = get_spark("credit-risk-hdfs-etl", memory or os.getenv("SPARK_DRIVER_MEMORY", "4g"))
    print(f"Spark {spark.version}")
    print(f"Reading   {input_uri}")
    raw = read_raw(spark, input_uri)
    prepared = prepare_spark(raw).cache()
    print(f"Rows with a valid issue date: {prepared.count():,}")
    show_spark_summaries(prepared)

    data = build_dataset(prepared)
    conditions = {
        "train": F.col("issue_year").isin(TRAIN_YEARS),
        "valid": F.col("issue_year").isin(VALID_YEARS),
        "test": F.col("issue_year").isin(TEST_YEARS),
    }

    # 1. write the splits to HDFS as Parquet
    for name, cond in conditions.items():
        target = f"{output_uri}/{name}"
        data.filter(cond).write.mode("overwrite").parquet(target)
        print(f"Wrote     {target}")

    # 2. read them back from HDFS (proves the round trip) and keep a local copy for modelling
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for name in conditions:
        sdf = spark.read.parquet(f"{output_uri}/{name}")
        pdf = to_pandas(sdf)
        pdf["issue_date"] = pd.to_datetime(pdf["issue_date"])
        pdf["issue_year"] = pdf["issue_year"].astype(int)
        pdf = pdf[COLUMNS]
        pdf.to_parquet(local_dir / f"{name}.parquet", index=False)
        splits[name] = pdf
        print(f"  {name:5s} rows {len(pdf):>8,}  bad rate {pdf[TARGET].mean():.3f}")
    with open(local_dir / "features.json", "w") as f:
        json.dump({"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES, "target": TARGET}, f, indent=2)
    print(f"Saved local copies to {local_dir}")
    spark.stop()
    return splits


def main():
    run(f"{HDFS_URI}{RAW_PATH}", f"{HDFS_URI}{OUT_DIR}")


if __name__ == "__main__":
    main()
