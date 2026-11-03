import numpy as np
import pandas as pd

from src.metrics import (auc, choose_threshold, csi_table, evaluate, gini, ks,
                         psi_categorical, psi_numeric, psi_status)


def test_perfect_and_random_auc():
    y = np.array([0, 0, 1, 1])
    assert auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert gini(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert ks(y, [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_psi_is_zero_for_identical_data_and_grows_with_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20000)
    assert psi_numeric(a, a) < 1e-6
    small = psi_numeric(a, rng.normal(0.1, 1, 20000))
    large = psi_numeric(a, rng.normal(1.0, 1, 20000))
    assert small < 0.10 < 0.25 < large
    assert psi_status(large) == "alert" and psi_status(0.0) == "ok"


def test_psi_handles_missing_values_and_categories():
    a = pd.Series([1.0, 2.0, np.nan, 4.0] * 500)
    assert psi_numeric(a, a) < 1e-6
    c1 = pd.Series(["a", "b", "c"] * 300)
    c2 = pd.Series(["a", "a", "c"] * 300)
    assert psi_categorical(c1, c1) < 1e-6 and psi_categorical(c1, c2) > 0.05


def test_csi_table_ranks_the_shifted_feature_first():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 5000), "z": rng.normal(0, 1, 5000), "cat": rng.choice(["a", "b"], 5000)})
    cur = ref.copy()
    cur["x"] = cur["x"] + 2
    table = csi_table(ref, cur, ["x", "z"], ["cat"])
    assert table.loc[0, "feature"] == "x"


def test_evaluate_and_threshold():
    rng = np.random.default_rng(2)
    y = (rng.random(5000) < 0.2).astype(int)
    p = np.clip(0.2 + 0.3 * (y - 0.2) + rng.normal(0, 0.1, 5000), 0.01, 0.99)
    r = evaluate(y, p)
    assert 0.5 < r["auc"] < 1 and r["n"] == 5000
    t = choose_threshold(y, p, cost_bad=5, cost_good=1)
    assert 0 < t["approval_rate"] < 1
    assert t["bad_rate_approved"] < y.mean()
