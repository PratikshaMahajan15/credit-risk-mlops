"""Credit-risk metrics: AUC / Gini / KS, calibration, PSI / CSI and the approval threshold."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, roc_auc_score

EPS = 1e-6


def auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def gini(y, p):
    return 2 * auc(y, p) - 1


def ks(y, p):
    """Largest gap between the score distributions of bad and good loans."""
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def evaluate(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    return {
        "n": int(len(y)),
        "bad_rate": float(y.mean()),
        "auc": auc(y, p),
        "gini": gini(y, p),
        "ks": ks(y, p),
        "brier": float(brier_score_loss(y, p)),
    }


def calibration_table(y, p, bins=10):
    """Predicted vs observed default rate in score deciles (lowest risk first)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["decile"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False) + 1
    out = df.groupby("decile").agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return out.reset_index()


# ---------------------------------------------------------------- PSI / CSI --
def psi_numeric(expected, actual, bins=10):
    """Population Stability Index for a numeric variable. Missing values get their own bin."""
    expected = pd.Series(expected).astype(float)
    actual = pd.Series(actual).astype(float)
    e_valid, a_valid = expected.dropna(), actual.dropna()
    if len(e_valid) == 0 or len(a_valid) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e_valid, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # (almost) constant variable
        edges = np.array([-np.inf, edges[0], np.inf])
    else:
        edges = np.concatenate([[-np.inf], edges[1:-1], [np.inf]])
    e_counts = np.histogram(e_valid, bins=edges)[0].astype(float)
    a_counts = np.histogram(a_valid, bins=edges)[0].astype(float)
    e_counts = np.append(e_counts, expected.isna().sum())
    a_counts = np.append(a_counts, actual.isna().sum())
    return _psi_from_counts(e_counts, a_counts)


def psi_categorical(expected, actual):
    expected = pd.Series(expected).astype("object").fillna("__missing__")
    actual = pd.Series(actual).astype("object").fillna("__missing__")
    cats = sorted(set(expected.unique()) | set(actual.unique()))
    e_counts = expected.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    a_counts = actual.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    return _psi_from_counts(e_counts, a_counts)


def _psi_from_counts(e_counts, a_counts):
    e = np.clip(e_counts / max(e_counts.sum(), 1), EPS, None)
    a = np.clip(a_counts / max(a_counts.sum(), 1), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def csi_table(reference, current, numeric, categorical):
    """Characteristic Stability Index: PSI of every model input, largest first."""
    rows = []
    for f in numeric:
        rows.append({"feature": f, "csi": psi_numeric(reference[f], current[f])})
    for f in categorical:
        rows.append({"feature": f, "csi": psi_categorical(reference[f], current[f])})
    return pd.DataFrame(rows).sort_values("csi", ascending=False).reset_index(drop=True)


def psi_status(value, warn=0.10, alert=0.25):
    if value != value:       # NaN
        return "unknown"
    if value >= alert:
        return "alert"
    if value >= warn:
        return "warn"
    return "ok"


# ---------------------------------------------------------------- threshold --
def choose_threshold(y, p, cost_bad=5.0, cost_good=1.0, grid=200):
    """Approve a loan when predicted default risk <= threshold.
    Cost = cost_bad for every bad loan approved + cost_good for every good loan rejected."""
    y = np.asarray(y)
    p = np.asarray(p)
    candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, grid)))
    best = None
    for t in candidates:
        approved = p <= t
        cost = cost_bad * np.sum(approved & (y == 1)) + cost_good * np.sum(~approved & (y == 0))
        if best is None or cost < best["cost"]:
            best = {
                "threshold": float(t),
                "cost": float(cost),
                "approval_rate": float(approved.mean()),
                "bad_rate_approved": float(y[approved].mean()) if approved.any() else float("nan"),
            }
    return best
