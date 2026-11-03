"""One place to create the local SparkSession (used by spark_prepare and record_matching)."""
import os
import sys


def get_spark(app_name, memory="4g"):
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        sys.exit("PySpark is not installed. Run: pip install pyspark")
    # On Windows Spark must use the same Python as this virtual environment.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    spark = (
        SparkSession.builder.master("local[*]").appName(app_name)
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
