"""Train the scorecard (champion), the XGBoost challenger and a simple baseline.

Outputs
    models/initial/     the first champion (never overwritten; monitoring starts from it)
    models/champion/    the model served by the API
    models/challenger_xgb/
    reports/            metrics.json, feature_report.csv, scorecard_bins.csv, charts
    MLflow              one run per model, scorecard registered with alias "champion"

Run from the project root:
    python -m src.train
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src import tracking
from src.config import (ALL_FEATURES, CATEGORICAL_FEATURES, COST_BAD, COST_GOOD,
                        MODELS_DIR, NUMERIC_FEATURES, PROCESSED_DIR, REPORTS_DIR,
                        TARGET, TEST_YEARS, TRAIN_YEARS, VALID_YEARS)
from src.data import load_splits, make_reference
from src.metrics import calibration_table, choose_threshold, evaluate
from src.models import LogRegBaseline, ScorecardModel, XGBModel
from src.registry import copy_model, save_model


def _window(df):
    years = sorted(int(y) for y in df["issue_year"].unique())
    return f"{years[0]}-{years[-1]}" if years else ""


def run(processed_dir=PROCESSED_DIR, models_dir=MODELS_DIR, reports_dir=REPORTS_DIR, use_mlflow=True):
    models_dir, reports_dir = type(MODELS_DIR)(models_dir), type(REPORTS_DIR)(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    train, valid, test = load_splits(processed_dir)
    splits = {"train": train, "valid": valid, "test": test}
    print(f"Train {len(train):,} | Valid {len(valid):,} | Test {len(test):,} loans")

    models = {}
    t0 = time.time()
    print("\nFitting scorecard (binning, WoE, logistic regression) ...")
    models["scorecard"] = ScorecardModel(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(train, train[TARGET])
    print(f"  done in {time.time() - t0:.0f}s")

    t0 = time.time()
    print("Fitting XGBoost challenger ...")
    models["xgboost"] = XGBModel(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(
        train, train[TARGET], valid, valid[TARGET])
    print(f"  done in {time.time() - t0:.0f}s")

    print("Fitting raw logistic-regression baseline ...")
    models["logreg_raw"] = LogRegBaseline(NUMERIC_FEATURES).fit(train, train[TARGET])

    # ------------------------------------------------------------- evaluate --
    results, preds = {}, {}
    for mname, model in models.items():
        results[mname] = {}
        for sname, part in splits.items():
            p = model.predict_proba(part)
            preds[(mname, sname)] = p
            results[mname][sname] = evaluate(part[TARGET], p)

    print("\nAUC / Gini / KS by model and period (test = out-of-time):")
    print(f"{'model':12s} {'split':6s} {'AUC':>7s} {'Gini':>7s} {'KS':>7s} {'bad rate':>9s}")
    for mname in models:
        for sname in splits:
            r = results[mname][sname]
            print(f"{mname:12s} {sname:6s} {r['auc']:7.3f} {r['gini']:7.3f} {r['ks']:7.3f} {r['bad_rate']:9.3f}")

    sc = models["scorecard"]
    print(f"\nScorecard uses {len(sc.selected_)} of {len(ALL_FEATURES)} features: {sc.selected_}")
    if sc.dropped_wrong_sign_:
        print(f"  dropped for a wrong-way sign: {sc.dropped_wrong_sign_}")

    # ------------------------------------------------------------ threshold --
    p_valid = preds[("scorecard", "valid")]
    thr = choose_threshold(valid[TARGET], p_valid, COST_BAD, COST_GOOD)
    p_test = preds[("scorecard", "test")]
    approved = p_test <= thr["threshold"]
    thr["test_approval_rate"] = float(approved.mean())
    thr["test_bad_rate_approved"] = float(test[TARGET].to_numpy()[approved].mean())
    print(f"\nApproval threshold (chosen on validation, cost ratio {COST_BAD:g}:{COST_GOOD:g}): "
          f"risk <= {thr['threshold']:.3f}")
    print(f"  on the test period: approves {thr['test_approval_rate']:.1%} of loans; "
          f"bad rate among approved {thr['test_bad_rate_approved']:.1%} "
          f"(vs {test[TARGET].mean():.1%} overall)")

    # ---------------------------------------------------------------- save --
    metadata = {
        "version": 1,
        "trained_on": _window(train),
        "n_train": int(len(train)),
        "metrics": results["scorecard"],
        "threshold": thr,
        "features": sc.selected_,
        "score_scaling": {"base_score": sc.base_score, "base_odds": sc.base_odds, "pdo": sc.pdo},
    }
    reference = make_reference(sc, train)
    initial = save_model(sc, models_dir / "initial", metadata, reference)
    copy_model(initial, models_dir / "champion")
    xgb_meta = {"version": 1, "trained_on": _window(train), "metrics": results["xgboost"]}
    save_model(models["xgboost"], models_dir / "challenger_xgb", xgb_meta)

    # -------------------------------------------------------------- reports --
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump({"results": results, "threshold": thr,
                   "split_years": {"train": TRAIN_YEARS, "valid": VALID_YEARS, "test": TEST_YEARS},
                   "selected_features": sc.selected_,
                   "dropped_wrong_sign": sc.dropped_wrong_sign_}, f, indent=2)
    sc.feature_report().to_csv(reports_dir / "feature_report.csv", index=False)
    bins = [t.assign(feature=name) for name, t in sc.binning_tables().items()]
    pd.concat(bins).to_csv(reports_dir / "scorecard_bins.csv", index=False)
    calibration_table(test[TARGET], p_test).to_csv(reports_dir / "calibration_test.csv", index=False)

    _plot_roc(test, preds, reports_dir / "roc_test.png")
    _plot_score_distribution(sc, test, reports_dir / "score_distribution_test.png")
    _plot_calibration(test[TARGET], p_test, reports_dir / "calibration_test.png")

    # --------------------------------------------------------------- mlflow --
    if use_mlflow and tracking.enabled():
        print("\nLogging to MLflow ...")
        chart_files = [reports_dir / n for n in ["roc_test.png", "score_distribution_test.png",
                                                 "calibration_test.png", "feature_report.csv"]]
        for mname in ["logreg_raw", "xgboost"]:
            tracking.log_run(
                run_name=f"{mname}_train", params={"model": mname, "train_years": TRAIN_YEARS},
                metrics={f"{s}_{k}": results[mname][s][k] for s in splits for k in ("auc", "gini", "ks")},
                tags={"stage": "initial"})
        info = tracking.log_run(
            run_name="scorecard_v1", params={"model": "scorecard", "features": ",".join(sc.selected_),
                                              "train_years": TRAIN_YEARS, "pdo": sc.pdo},
            metrics={f"{s}_{k}": results["scorecard"][s][k] for s in splits for k in ("auc", "gini", "ks")},
            model_dir=models_dir / "initial", files=chart_files, tags={"stage": "initial"},
            register=True, alias="champion")
        if info:
            print(f"  registered credit_risk_scorecard version {info['version']} (alias: champion)")
            print("  view it with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")

    print(f"\nSaved models to {models_dir} and charts/tables to {reports_dir}")
    return {"results": results, "threshold": thr, "models": models}


# ------------------------------------------------------------------ charts --
def _plot_roc(test, preds, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for mname in ["scorecard", "xgboost", "logreg_raw"]:
        fpr, tpr, _ = roc_curve(test[TARGET], preds[(mname, "test")])
        auc_value = evaluate(test[TARGET], preds[(mname, "test")])["auc"]
        ax.plot(fpr, tpr, label=f"{mname} (AUC {auc_value:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC on the out-of-time test period"); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_score_distribution(sc, test, path):
    pts = sc.points(test)
    y = test[TARGET].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(np.nanpercentile(pts, 0.5), np.nanpercentile(pts, 99.5), 40)
    ax.hist(pts[y == 0], bins=bins, alpha=0.6, density=True, label="repaid")
    ax.hist(pts[y == 1], bins=bins, alpha=0.6, density=True, label="defaulted")
    ax.set_xlabel("Score (points, higher = safer)"); ax.set_ylabel("Density")
    ax.set_title("Score distribution on the test period"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_calibration(y, p, path):
    table = calibration_table(y, p)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(table["predicted"], table["observed"], "o-", label="model")
    top = max(table["predicted"].max(), table["observed"].max()) * 1.05
    ax.plot([0, top], [0, top], "--", color="grey", label="perfect")
    ax.set_xlabel("Predicted default rate"); ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration by score decile (test)"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    run()
