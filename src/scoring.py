"""Score one application. Used by both the API and the demo app."""
import numpy as np
import pandas as pd

from src.config import ALL_FEATURES
from src.models import FEATURE_LABELS


def to_frame(application, model):
    """One application (dict of raw fields) -> a one-row DataFrame with every model column."""
    row = dict(application)
    row["loan_to_income"] = row["loan_amnt"] / row["annual_inc"]
    df = pd.DataFrame([row]).reindex(columns=ALL_FEATURES)
    return df.apply(lambda c: c if c.name in model.categorical else pd.to_numeric(c, errors="coerce"))


def score_one(model, meta, application, n_reasons=3):
    df = to_frame(application, model)
    p = float(model.predict_proba(df)[0])
    points = float(model.points(df)[0])
    thr = (meta.get("threshold") or {}).get("threshold")
    decision = "approve" if (thr is None or p <= thr) else "decline"

    contrib = model.contributions(df).iloc[0]
    risk = contrib[contrib > 0].sort_values(ascending=False).head(n_reasons)
    helping = contrib[contrib < 0].sort_values().head(n_reasons)

    def pack(series):
        return [{"feature": f, "label": FEATURE_LABELS.get(f, f), "impact": round(float(v), 4),
                 "value": None if pd.isna(df.iloc[0][f]) else df.iloc[0][f]} for f, v in series.items()]

    return {"probability": p, "points": points, "decision": decision, "threshold": thr,
            "risk_factors": pack(risk), "helping_factors": pack(helping), "frame": df}


def points_from_probability(p, scaling):
    """Convert default probabilities to score points with the model's scaling."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    factor = scaling["pdo"] / np.log(2)
    offset = scaling["base_score"] - factor * np.log(scaling["base_odds"])
    return offset - factor * np.log(p / (1 - p))
