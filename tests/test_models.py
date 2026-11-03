import numpy as np

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.metrics import auc
from src.models import FEATURE_LABELS, LogRegBaseline, ScorecardModel, XGBModel
from tests.synth import make_splits


def _data():
    train, valid, _ = make_splits(n_per_year=2500, seed=3)
    return train, valid


def test_scorecard_learns_and_handles_bad_inputs():
    train, valid = _data()
    sc = ScorecardModel(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(train, train[TARGET])
    p = sc.predict_proba(valid)
    assert ((p >= 0) & (p <= 1)).all()
    assert auc(valid[TARGET], p) > 0.6
    # every kept coefficient has the expected sign (WoE higher = safer, so negative)
    assert all(c < 0 for c in sc.coef_.values())
    # unseen category and a missing column must not crash
    row = valid.head(1).copy()
    row["addr_state"] = "ZZ"
    assert sc.predict_proba(row).shape == (1,)
    assert sc.predict_proba(valid.head(1).drop(columns=["mort_acc"])).shape == (1,)


def test_points_are_higher_for_safer_borrowers():
    train, valid = _data()
    sc = ScorecardModel(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(train, train[TARGET])
    p, pts = sc.predict_proba(valid), sc.points(valid)
    assert np.corrcoef(p, pts)[0, 1] < -0.95


def test_reason_codes_are_known_features():
    train, valid = _data()
    sc = ScorecardModel(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(train, train[TARGET])
    reasons = sc.top_reasons(valid.head(5), n=3)
    assert len(reasons) == 5
    for r in reasons:
        assert len(r) <= 3
        assert all(f in FEATURE_LABELS and v > 0 for f, v in r)


def test_challenger_and_baseline_run():
    train, valid = _data()
    xgb = XGBModel(NUMERIC_FEATURES, CATEGORICAL_FEATURES, n_estimators=80).fit(train, train[TARGET])
    base = LogRegBaseline(NUMERIC_FEATURES).fit(train, train[TARGET])
    assert auc(valid[TARGET], xgb.predict_proba(valid)) > 0.55
    assert auc(valid[TARGET], base.predict_proba(valid)) > 0.55
