"""Run the whole pipeline in order.

    python -m src.run_all                 # pandas data prep
    python -m src.run_all --spark         # PySpark data prep (needs Java)
    python -m src.run_all --mode scheduled
"""
import argparse

from src import day1_explore, day2_prepare, model_card, monitor, spark_prepare, train
from src.config import RAW_SUBSET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spark", action="store_true", help="use PySpark for the data preparation")
    ap.add_argument("--mode", choices=["drift", "scheduled"], default="drift")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    print("\n=== 1/4  Data preparation ===")
    if args.spark:
        spark_prepare.main()
    else:
        if not RAW_SUBSET.exists():
            day1_explore.main()
        day2_prepare.main()

    print("\n=== 2/4  Training ===")
    train.run(use_mlflow=not args.no_mlflow)

    print("\n=== 3/4  Monitoring replay and retraining ===")
    monitor.run(mode=args.mode, use_mlflow=not args.no_mlflow)

    print("\n=== 4/4  Model card ===")
    model_card.main()


if __name__ == "__main__":
    main()
