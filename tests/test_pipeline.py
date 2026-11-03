"""End-to-end run on synthetic data: train -> monitor/retrain -> API -> model card."""
import json

import pytest
from fastapi.testclient import TestClient

from src import model_card, monitor, train
from tests.synth import write_splits


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("work")
    processed, models, reports = root / "processed", root / "models", root / "reports"
    # inputs shift and the signal weakens in later years, so the monitor should react
    write_splits(processed, n_per_year=2500, decay=0.03, input_shift=3.0)
    train.run(processed, models, reports, use_mlflow=False)
    out, summary = monitor.run(processed, models, reports, mode="drift", use_mlflow=False, verbose=False)
    return {"processed": processed, "models": models, "reports": reports, "log": out, "summary": summary}


def test_training_creates_models_and_reports(workspace):
    assert (workspace["models"] / "initial" / "model.joblib").exists()
    assert (workspace["models"] / "initial" / "reference.parquet").exists()
    metrics = json.load(open(workspace["reports"] / "metrics.json"))
    assert metrics["results"]["scorecard"]["test"]["auc"] > 0.55


def test_monitor_detects_drift_and_logs_every_quarter(workspace):
    log = workspace["log"]
    assert len(log) == 8
    assert (log["score_psi"] > 0.25).any()                 # drift was detected
    assert log["triggers"].fillna("").str.len().gt(0).any()
    assert (workspace["reports"] / "monitoring_timeline.png").exists()


def test_challenger_is_only_promoted_when_better(workspace):
    log = workspace["log"].dropna(subset=["holdout_challenger_auc"])
    for _, r in log.iterrows():
        gain = r["holdout_challenger_auc"] - r["holdout_champion_auc"]
        promoted = r["action"].startswith("PROMOTED")
        assert promoted == (gain >= 0.005)


def test_api_scores_with_the_champion(workspace, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(workspace["models"] / "champion"))
    from src.api import app
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        r = client.post("/score", json={"loan_amnt": 10000, "annual_inc": 60000, "dti": 30,
                                        "fico_range_low": 650, "home_ownership": "RENT"})
        body = r.json()
        assert r.status_code == 200
        assert 0 <= body["probability_of_default"] <= 1
        assert body["decision"] in {"approve", "decline"}
        assert len(body["reasons"]) <= 3
        assert "does not change the score" in body["explanation"]
        assert client.post("/score", json={"loan_amnt": -1, "annual_inc": 1000}).status_code == 422


def test_model_card_is_written(workspace):
    text = model_card.build(workspace["reports"])
    assert "Model Card" in text and "Out-of-time test" in text and "Limitations" in text
