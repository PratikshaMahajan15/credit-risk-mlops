"""Scoring API.

    uvicorn src.api:app --reload        then open http://127.0.0.1:8000/docs

The service loads the champion model from the folder in $MODEL_DIR (default models/champion),
so a newly promoted model is served after a restart.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import ALL_FEATURES, MODELS_DIR
from src.llm_explain import explain
from src.models import FEATURE_LABELS
from src.registry import load_model

STATE = {}


def model_dir():
    return Path(os.getenv("MODEL_DIR", str(MODELS_DIR / "champion")))


def load_champion():
    model, meta = load_model(model_dir())
    STATE["model"], STATE["meta"] = model, meta


@asynccontextmanager
async def lifespan(app):
    try:
        load_champion()
    except Exception as exc:                  # the API still starts; /score reports the problem
        STATE["error"] = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(title="Credit Risk Scoring API", version="1.0", lifespan=lifespan)


class Application(BaseModel):
    """One loan application. Only the first two fields are required; missing values are handled."""
    loan_amnt: float = Field(..., gt=0, description="Requested loan amount")
    annual_inc: float = Field(..., gt=0, description="Annual income")
    dti: Optional[float] = Field(None, ge=0, le=100, description="Debt-to-income ratio (%)")
    delinq_2yrs: Optional[float] = Field(None, ge=0)
    fico_range_low: Optional[float] = Field(None, ge=300, le=850)
    inq_last_6mths: Optional[float] = Field(None, ge=0)
    open_acc: Optional[float] = Field(None, ge=0)
    pub_rec: Optional[float] = Field(None, ge=0)
    revol_bal: Optional[float] = Field(None, ge=0)
    revol_util: Optional[float] = Field(None, ge=0, le=200)
    total_acc: Optional[float] = Field(None, ge=0)
    mort_acc: Optional[float] = Field(None, ge=0)
    pub_rec_bankruptcies: Optional[float] = Field(None, ge=0)
    emp_length_years: Optional[float] = Field(None, ge=0, le=10)
    credit_history_years: Optional[float] = Field(None, ge=0, le=80)
    home_ownership: Optional[str] = None
    verification_status: Optional[str] = None
    purpose: Optional[str] = None
    addr_state: Optional[str] = None

    model_config = {"json_schema_extra": {"example": {
        "loan_amnt": 12000, "annual_inc": 65000, "dti": 18.5, "fico_range_low": 690,
        "inq_last_6mths": 1, "open_acc": 9, "revol_util": 45, "total_acc": 22,
        "emp_length_years": 5, "credit_history_years": 12, "home_ownership": "RENT",
        "verification_status": "Verified", "purpose": "debt_consolidation", "addr_state": "CA"}}}


class Reason(BaseModel):
    feature: str
    label: str
    impact: float


class ScoreResponse(BaseModel):
    probability_of_default: float
    score: int
    decision: str
    risk_threshold: Optional[float]
    reasons: List[Reason]
    explanation: str
    explanation_source: str
    model_version: int


def _ready():
    if "model" not in STATE:
        try:
            load_champion()
        except Exception as exc:
            raise HTTPException(503, f"Model not loaded: {type(exc).__name__}: {exc}")
    return STATE["model"], STATE["meta"]


@app.get("/health")
def health():
    return {"status": "ok" if "model" in STATE else "model_not_loaded", "error": STATE.get("error")}


@app.get("/model-info")
def model_info():
    model, meta = _ready()
    return {"version": meta.get("version"), "type": meta.get("model_type"),
            "trained_on": meta.get("trained_on"), "features": meta.get("features"),
            "validation_auc": (meta.get("metrics", {}).get("valid") or {}).get("auc"),
            "risk_threshold": (meta.get("threshold") or {}).get("threshold")}


@app.post("/score", response_model=ScoreResponse)
def score(application: Application):
    model, meta = _ready()
    row = application.model_dump()
    row["loan_to_income"] = row["loan_amnt"] / row["annual_inc"]
    df = pd.DataFrame([row]).reindex(columns=ALL_FEATURES)
    df = df.apply(lambda c: c if c.name in model.categorical else pd.to_numeric(c, errors="coerce"))

    p = float(model.predict_proba(df)[0])
    points = float(model.points(df)[0])
    thr = (meta.get("threshold") or {}).get("threshold")
    decision = "approve" if (thr is None or p <= thr) else "decline"
    reasons = [{"feature": f, "label": FEATURE_LABELS.get(f, f), "impact": round(v, 4)}
               for f, v in model.top_reasons(df, n=3)[0]]
    text, source = explain(points, decision, reasons)
    return {"probability_of_default": round(p, 4), "score": int(round(points)), "decision": decision,
            "risk_threshold": thr, "reasons": reasons, "explanation": text,
            "explanation_source": source, "model_version": int(meta.get("version", 1))}
