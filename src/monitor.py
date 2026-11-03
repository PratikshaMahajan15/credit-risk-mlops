"""Replay the later periods quarter by quarter, watch for drift, and retrain when needed.

For every quarter the job
  1. scores the new loans with the current champion,
  2. measures input drift immediately (score PSI and per-feature CSI),
  3. measures performance once outcomes arrive (simulated delay of a few quarters),
  4. decides whether to retrain: a large drift, a drop in AUC, or (in "scheduled" mode) every quarter,
  5. trains a challenger, compares it with the champion on held-out recent quarters,
     and promotes it only if it is clearly better.
The original model (never retrained) is scored alongside, so the chart shows the benefit of retraining.

Run from the project root:
    python -m src.monitor                  # triggers: drift or AUC drop
    python -m src.monitor --mode scheduled # also review a challenger every quarter
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import tracking
from src.config import (ALL_FEATURES, AUC_DROP_TRIGGER, CATEGORICAL_FEATURES,
                        COOLDOWN_QUARTERS, LABEL_DELAY_QUARTERS, MODELS_DIR,
                        NUMERIC_FEATURES, PROCESSED_DIR, PSI_ALERT, PSI_WARN,
                        REPORTS_DIR, TARGET)
from src.data import add_quarter, load_splits, make_reference
from src.metrics import auc, csi_table, ks, psi_numeric, psi_status
from src.registry import copy_model, load_model, load_reference, save_model
from src.retrain import retrain_and_compare


def run(processed_dir=PROCESSED_DIR, models_dir=MODELS_DIR, reports_dir=REPORTS_DIR,
        mode="drift", use_mlflow=True, label_delay=LABEL_DELAY_QUARTERS, verbose=True):
    models_dir, reports_dir = type(MODELS_DIR)(models_dir), type(REPORTS_DIR)(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    train, valid, test = load_splits(processed_dir)
    train = add_quarter(train)
    stream = add_quarter(pd.concat([valid, test], ignore_index=True)).sort_values("issue_date").reset_index(drop=True)
    quarters = sorted(stream["quarter"].unique())

    # Always start from the first champion so the replay is repeatable.
    copy_model(models_dir / "initial", models_dir / "champion")
    champion, meta = load_model(models_dir / "champion")
    original, _ = load_model(models_dir / "initial")
    reference = load_reference(models_dir / "champion")
    version = int(meta.get("version", 1))
    baseline_auc = meta["metrics"]["valid"]["auc"]
    log(f"Champion v{version} trained on {meta.get('trained_on')}; baseline validation AUC {baseline_auc:.3f}")
    log(f"Replaying {len(quarters)} quarters ({quarters[0]} to {quarters[-1]}), mode = {mode}, "
        f"label delay = {label_delay} quarters\n")

    history = {}                  # quarter -> dict(y, p_current, p_original)
    versions = {version: original}
    promo_index = {}
    rows, promotions = [], []
    last_promotion_i = -10 ** 6

    for i, q in enumerate(quarters):
        batch = stream[stream["quarter"] == q]
        p_cur, p_orig = champion.predict_proba(batch), original.predict_proba(batch)
        history[q] = {"y": batch[TARGET].to_numpy(), "p_cur": p_cur, "p_orig": p_orig}

        score_psi = psi_numeric(reference["score"], p_cur)
        csi = csi_table(reference, batch, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        row = {
            "quarter": str(q), "n_loans": int(len(batch)), "champion_version": version,
            "score_psi": score_psi, "score_psi_status": psi_status(score_psi, PSI_WARN, PSI_ALERT),
            "max_csi": float(csi.loc[0, "csi"]), "max_csi_feature": csi.loc[0, "feature"],
            "labeled_quarter": None, "labeled_scored_by_version": None,
            "bad_rate_labeled": None, "auc_current": None,
            "auc_original": None, "ks_current": None, "triggers": "", "action": "none",
            "holdout_champion_auc": None, "holdout_challenger_auc": None,
        }

        # outcomes for an earlier quarter have now arrived
        j = i - label_delay
        if j >= 0:
            h = history[quarters[j]]
            row.update({
                "labeled_quarter": str(quarters[j]),
                "labeled_scored_by_version": rows[j]["champion_version"],
                "bad_rate_labeled": float(h["y"].mean()),
                "auc_current": auc(h["y"], h["p_cur"]),
                "auc_original": auc(h["y"], h["p_orig"]),
                "ks_current": ks(h["y"], h["p_cur"]),
            })

        # ------------------------------------------------------ triggers --
        triggers = []
        if score_psi >= PSI_ALERT:
            triggers.append(f"score PSI {score_psi:.2f} >= {PSI_ALERT}")
        if row["auc_current"] is not None and row["auc_current"] < baseline_auc - AUC_DROP_TRIGGER:
            triggers.append(f"AUC {row['auc_current']:.3f} is {baseline_auc - row['auc_current']:.3f} below baseline")
        if mode == "scheduled" and j >= 1:
            triggers.append("scheduled review")
        row["triggers"] = "; ".join(triggers)

        cooled_down = (i - last_promotion_i) > COOLDOWN_QUARTERS
        if triggers and j >= 0 and cooled_down:
            labeled_stream = stream[stream["quarter"].isin(quarters[: j + 1])]
            labeled = pd.concat([train, labeled_stream], ignore_index=True)
            challenger, cmp_, promote, gain, fit_df = retrain_and_compare(champion, labeled)
            row["holdout_champion_auc"] = cmp_["champion_auc"]
            row["holdout_challenger_auc"] = cmp_["challenger_auc"]
            if promote:
                version += 1
                champion = challenger
                versions[version] = challenger
                promo_index[version] = i
                reference = make_reference(champion, fit_df)
                baseline_auc = cmp_["challenger_auc"]
                last_promotion_i = i
                meta = {
                    "version": version, "trained_on": f"up to {quarters[j]}", "n_train": int(len(fit_df)),
                    "metrics": {"valid": {"auc": cmp_["challenger_auc"]}}, "features": champion.selected_,
                    "threshold": meta.get("threshold"), "promoted_at_quarter": str(q),
                    "reason": row["triggers"], "holdout_comparison": cmp_,
                }
                save_model(champion, models_dir / "champion", meta, reference)
                if use_mlflow and tracking.enabled():
                    tracking.log_run(
                        run_name=f"scorecard_v{version}", params={"trigger": row["triggers"], "quarter": str(q)},
                        metrics={"holdout_champion_auc": cmp_["champion_auc"],
                                 "holdout_challenger_auc": cmp_["challenger_auc"], "gain": gain},
                        model_dir=models_dir / "champion", tags={"stage": "retrained"},
                        register=True, alias="champion")
                row["action"] = f"PROMOTED v{version}"
                promotions.append({"quarter": str(q), "version": version, "gain": gain})
            else:
                row["action"] = "challenger rejected"
            log(f"{q}: {row['triggers']} -> {row['action']} "
                f"(holdout AUC champion {cmp_['champion_auc']:.3f} vs challenger {cmp_['challenger_auc']:.3f})")
        elif triggers:
            row["action"] = "trigger, waiting for labels" if j < 0 else "trigger, in cooldown"
            log(f"{q}: {row['triggers']} -> {row['action']}")

        row["champion_version_after"] = version
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(reports_dir / "monitoring_log.csv", index=False)
    backtest = _backtest(stream, quarters, versions, promo_index, original)
    backtest.to_csv(reports_dir / "post_promotion_backtest.csv", index=False)
    csi_table(reference, batch, NUMERIC_FEATURES, CATEGORICAL_FEATURES).to_csv(reports_dir / "csi_latest.csv", index=False)
    summary = _summary(out, promotions, backtest)
    with open(reports_dir / "monitoring_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _plot(out, backtest, reports_dir / "monitoring_timeline.png")

    if use_mlflow and tracking.enabled():
        tracking.log_run(
            run_name=f"monitoring_{mode}", params={"mode": mode, "label_delay": label_delay},
            metrics={"promotions": summary["promotions"],
                     "auc_original_last": summary.get("auc_original_last"),
                     "auc_current_last": summary.get("auc_current_last")},
            files=[reports_dir / "monitoring_log.csv", reports_dir / "monitoring_timeline.png"],
            tags={"stage": "monitoring"})

    log("\nSummary")
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log(f"\nSaved monitoring_log.csv, monitoring_timeline.png and monitoring_summary.json to {reports_dir}")
    return out, summary


def _backtest(stream, quarters, versions, promo_index, original):
    """For every promoted model, compare it with the original on the quarters AFTER its promotion.
    (In the replay their outcomes had not arrived yet; here all outcomes are known, so we can check.)"""
    rows = []
    for v, ip in promo_index.items():
        for q in quarters[ip + 1:]:
            batch = stream[stream["quarter"] == q]
            y = batch[TARGET]
            rows.append({"version": v, "quarter": str(q), "n_loans": int(len(batch)),
                         "auc_original": auc(y, original.predict_proba(batch)),
                         "auc_promoted": auc(y, versions[v].predict_proba(batch))})
    out = pd.DataFrame(rows, columns=["version", "quarter", "n_loans", "auc_original", "auc_promoted"])
    out["gain"] = out["auc_promoted"] - out["auc_original"]
    return out


def _summary(out, promotions, backtest):
    labeled = out.dropna(subset=["auc_original"])
    summary = {"quarters": int(len(out)), "promotions": len(promotions),
               "promotion_quarters": [p["quarter"] for p in promotions]}
    first_alert = out[out["score_psi_status"] == "alert"]
    summary["first_psi_alert_quarter"] = None if first_alert.empty else first_alert.iloc[0]["quarter"]
    if not labeled.empty:
        summary["auc_original_first"] = round(float(labeled["auc_original"].iloc[0]), 4)
        summary["auc_original_last"] = round(float(labeled["auc_original"].iloc[-1]), 4)
        summary["auc_current_last"] = round(float(labeled["auc_current"].iloc[-1]), 4)
    if len(backtest):
        last_v = backtest["version"].max()
        b = backtest[backtest["version"] == last_v]
        summary["backtest_quarters_after_last_promotion"] = int(len(b))
        summary["backtest_mean_auc_original"] = round(float(b["auc_original"].mean()), 4)
        summary["backtest_mean_auc_promoted"] = round(float(b["auc_promoted"].mean()), 4)
        summary["backtest_mean_auc_gain"] = round(float(b["gain"].mean()), 4)
    return summary


def _plot(out, backtest, path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    x = out["quarter"]
    ax = axes[0]
    ax.plot(x, out["score_psi"], "o-", color="tab:blue", label="score PSI")
    ax.axhline(PSI_WARN, ls="--", color="orange", label=f"warn {PSI_WARN}")
    ax.axhline(PSI_ALERT, ls="--", color="red", label=f"alert {PSI_ALERT}")
    ax.set_ylabel("Score PSI"); ax.legend(loc="upper left"); ax.set_title("Input drift")

    ax = axes[1]
    lab = out.dropna(subset=["auc_original"])
    ax.plot(lab["labeled_quarter"], lab["auc_original"], "o-", color="tab:red", label="original model (never retrained)")
    ax.plot(lab["labeled_quarter"], lab["auc_current"], "o-", color="tab:green", label="monitored model (retrained when needed)")
    if len(backtest):
        b = backtest[backtest["version"] == backtest["version"].max()]
        ax.plot(b["quarter"], b["auc_promoted"], "s--", color="tab:blue",
                label="promoted model, back-tested on later quarters")
    ax.set_ylabel("AUC on quarter whose outcomes arrived"); ax.legend(loc="lower left")
    ax.set_title("Model quality over time")

    ax = axes[2]
    ax.plot(lab["labeled_quarter"], lab["bad_rate_labeled"], "o-", color="tab:purple")
    ax.set_ylabel("Bad rate"); ax.set_title("Default rate of the labelled quarter")

    promoted = out[out["action"].str.startswith("PROMOTED")]
    for _, r in promoted.iterrows():
        for a in axes:
            a.axvline(r["quarter"], color="green", alpha=0.5, ls=":")
        axes[1].annotate(r["action"], (r["quarter"], axes[1].get_ylim()[1]), rotation=90,
                         va="top", ha="right", fontsize=8, color="green")
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["drift", "scheduled"], default="drift")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    run(mode=args.mode, use_mlflow=not args.no_mlflow)
