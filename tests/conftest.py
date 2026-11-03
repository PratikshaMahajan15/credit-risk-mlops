import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLED", "1")          # tests never touch MLflow
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
