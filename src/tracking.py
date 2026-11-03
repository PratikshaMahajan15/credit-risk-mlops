"""MLflow helpers: experiment tracking and the model registry.

Everything here is best-effort. If MLflow is disabled (MLFLOW_DISABLED=1) or fails,
the pipeline keeps running and prints a warning, because tracking is not required
for the model files themselves (those are saved by src/registry.py).

Backends
    sqlite (default)  mlflow.db           -> browse with: mlflow ui --backend-store-uri sqlite:///mlflow.db
    file              mlruns/ folder      -> used automatically if the sqlite backend cannot start
                                             (for example when Windows blocks SQLAlchemy's compiled files),
                                             or force it with:  set MLFLOW_BACKEND=file
"""
import os
from pathlib import Path

from src.config import (MLFLOW_ARTIFACT_DIR, MLFLOW_DB, MLFLOW_EXPERIMENT,
                        REGISTERED_MODEL_NAME, ROOT)

_BACKEND = None


def enabled():
    return os.getenv("MLFLOW_DISABLED", "0") != "1"


def _configure(backend):
    import mlflow

    if backend == "file":
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"       # newer MLflow versions require this opt-in
        uri = (Path(ROOT) / "mlruns").resolve().as_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
    else:
        mlflow.set_tracking_uri(f"sqlite:///{Path(MLFLOW_DB).as_posix()}")
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        Path(MLFLOW_ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=Path(MLFLOW_ARTIFACT_DIR).resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return mlflow


def _setup():
    global _BACKEND
    if _BACKEND is not None:
        return _configure(_BACKEND)
    wanted = os.getenv("MLFLOW_BACKEND", "sqlite")
    try:
        mlflow = _configure(wanted)
        _BACKEND = wanted
    except Exception as exc:
        if wanted == "file":
            raise
        print(f"  [warning] MLflow sqlite backend failed ({type(exc).__name__}: {str(exc)[:90]}). "
              "Using the file backend (mlruns/) instead.")
        mlflow = _configure("file")
        _BACKEND = "file"
    return mlflow


def log_run(run_name, params=None, metrics=None, model_dir=None, files=(), tags=None,
            register=False, alias=None):
    """Log one run. If `register` is True the model folder is added to the registry
    (and `alias`, e.g. "champion", is pointed at the new version)."""
    if not enabled():
        return None
    try:
        mlflow = _setup()
        from mlflow.tracking import MlflowClient

        with mlflow.start_run(run_name=run_name) as run:
            if tags:
                mlflow.set_tags({k: str(v) for k, v in tags.items()})
            if params:
                mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            if metrics:
                clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v}
                mlflow.log_metrics(clean)
            if model_dir is not None:
                mlflow.log_artifacts(str(model_dir), artifact_path="model")
            for f in files:
                if Path(f).exists():
                    mlflow.log_artifact(str(f), artifact_path="reports")
            result = {"run_id": run.info.run_id, "version": None}

            if register and model_dir is not None:
                client = MlflowClient()
                try:
                    client.create_registered_model(REGISTERED_MODEL_NAME)
                except Exception:
                    pass                                     # already exists
                mv = client.create_model_version(
                    name=REGISTERED_MODEL_NAME,
                    source=f"{run.info.artifact_uri}/model",
                    run_id=run.info.run_id,
                )
                result["version"] = mv.version
                if alias:
                    client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, mv.version)
            return result
    except Exception as exc:                                 # pragma: no cover
        print(f"  [warning] MLflow logging skipped: {type(exc).__name__}: {str(exc)[:120]}")
        return None"""MLflow helpers: experiment tracking and the model registry.

Everything here is best-effort. If MLflow is disabled (MLFLOW_DISABLED=1) or fails,
the pipeline keeps running and prints a warning, because tracking is not required
for the model files themselves (those are saved by src/registry.py).

Backends
    sqlite (default)  mlflow.db           -> browse with: mlflow ui --backend-store-uri sqlite:///mlflow.db
    file              mlruns/ folder      -> used automatically if the sqlite backend cannot start
                                             (for example when Windows blocks SQLAlchemy's compiled files),
                                             or force it with:  set MLFLOW_BACKEND=file
"""
import os
from pathlib import Path

from src.config import (MLFLOW_ARTIFACT_DIR, MLFLOW_DB, MLFLOW_EXPERIMENT,
                        REGISTERED_MODEL_NAME, ROOT)

_BACKEND = None


def enabled():
    return os.getenv("MLFLOW_DISABLED", "0") != "1"


def _configure(backend):
    import mlflow

    if backend == "file":
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"       # newer MLflow versions require this opt-in
        uri = (Path(ROOT) / "mlruns").resolve().as_uri()
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
    else:
        mlflow.set_tracking_uri(f"sqlite:///{Path(MLFLOW_DB).as_posix()}")
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        Path(MLFLOW_ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=Path(MLFLOW_ARTIFACT_DIR).resolve().as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return mlflow


def _setup():
    global _BACKEND
    if _BACKEND is not None:
        return _configure(_BACKEND)
    wanted = os.getenv("MLFLOW_BACKEND", "sqlite")
    try:
        mlflow = _configure(wanted)
        _BACKEND = wanted
    except Exception as exc:
        if wanted == "file":
            raise
        print(f"  [warning] MLflow sqlite backend failed ({type(exc).__name__}: {str(exc)[:90]}). "
              "Using the file backend (mlruns/) instead.")
        mlflow = _configure("file")
        _BACKEND = "file"
    return mlflow


def log_run(run_name, params=None, metrics=None, model_dir=None, files=(), tags=None,
            register=False, alias=None):
    """Log one run. If `register` is True the model folder is added to the registry
    (and `alias`, e.g. "champion", is pointed at the new version)."""
    if not enabled():
        return None
    try:
        mlflow = _setup()
        from mlflow.tracking import MlflowClient

        with mlflow.start_run(run_name=run_name) as run:
            if tags:
                mlflow.set_tags({k: str(v) for k, v in tags.items()})
            if params:
                mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            if metrics:
                clean = {k: float(v) for k, v in metrics.items() if v is not None and v == v}
                mlflow.log_metrics(clean)
            if model_dir is not None:
                mlflow.log_artifacts(str(model_dir), artifact_path="model")
            for f in files:
                if Path(f).exists():
                    mlflow.log_artifact(str(f), artifact_path="reports")
            result = {"run_id": run.info.run_id, "version": None}

            if register and model_dir is not None:
                client = MlflowClient()
                try:
                    client.create_registered_model(REGISTERED_MODEL_NAME)
                except Exception:
                    pass                                     # already exists
                mv = client.create_model_version(
                    name=REGISTERED_MODEL_NAME,
                    source=f"{run.info.artifact_uri}/model",
                    run_id=run.info.run_id,
                )
                result["version"] = mv.version
                if alias:
                    client.set_registered_model_alias(REGISTERED_MODEL_NAME, alias, mv.version)
            return result
    except Exception as exc:                                 # pragma: no cover
        print(f"  [warning] MLflow logging skipped: {type(exc).__name__}: {str(exc)[:120]}")
        return None