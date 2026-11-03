"""Write docs/MODEL_CARD.md from the saved reports, so the numbers always match the last run.

    python -m src.model_card
"""
import json
from datetime import date

import pandas as pd

from src.config import (COST_BAD, COST_GOOD, DOCS_DIR, EXCLUDED_FROM_MODEL,
                        LABEL_DELAY_QUARTERS, PSI_ALERT, REPORTS_DIR, SCORE_BASE,
                        SCORE_BASE_ODDS, SCORE_PDO)


def _load(path):
    return json.load(open(path)) if path.exists() else None


def _row(name, r):
    return f"| {name} | {r['n']:,} | {r['bad_rate']:.1%} | {r['auc']:.3f} | {r['gini']:.3f} | {r['ks']:.3f} |"


def build(reports_dir=REPORTS_DIR):
    m = _load(reports_dir / "metrics.json")
    if m is None:
        raise SystemExit("reports/metrics.json not found. Run `python -m src.train` first.")
    mon = _load(reports_dir / "monitoring_summary.json")
    rm = _load(reports_dir / "record_matching_metrics.json")
    res = m["results"]["scorecard"]
    xg = m["results"]["xgboost"]
    thr = m["threshold"]
    years = m["split_years"]
    feat = pd.read_csv(reports_dir / "feature_report.csv")
    used = feat[feat["selected"]].copy()

    lines = [
        "# Model Card: Credit-Risk Scorecard", "",
        f"_Generated on {date.today().isoformat()} from the last pipeline run._", "",
        "## Purpose",
        "Estimate the probability that a loan will default (be charged off) from information known at "
        "application time, and turn it into a credit-style score with reason codes. "
        "This is a learning and portfolio project on public data. **It must not be used for real lending decisions.**", "",
        "## Data",
        "- Source: public Lending Club accepted-loans data (Kaggle). US peer-to-peer loans, not Indian bank data.",
        f"- Only 36-month loans with a final outcome (Fully Paid vs Charged Off/Default) are used, "
        "because later loans have not finished yet.",
        f"- Time-based split by issue year: train {years['train']}, validation {years['valid']}, "
        f"out-of-time test {years['test']}. No random splitting.",
        f"- Excluded on purpose to avoid leakage: {', '.join(EXCLUDED_FROM_MODEL)}. "
        "Grade, sub-grade and interest rate come from the lender's own model, and the instalment is calculated from the rate.", "",
        "## Model",
        f"- Logistic regression on Weight-of-Evidence binned features (optbinning). "
        f"Features kept: {len(used)} of {len(feat)} after an information-value cut-off and a correlation filter.",
        f"- Points scaling: {SCORE_BASE} points at {SCORE_BASE_ODDS}:1 good:bad odds, {SCORE_PDO} points to double the odds. "
        "Higher score = lower risk.",
        "- Reason codes: the features that push a borrower's risk up the most (coefficient x WoE).",
        f"- Challenger: XGBoost (test AUC {xg['test']['auc']:.3f}); raw logistic-regression baseline also trained.", "",
        "### Features used",
        "| Feature | Information value | Coefficient |", "|---|---|---|",
    ]
    for _, r in used.iterrows():
        lines.append(f"| {r['feature']} | {r['iv']:.3f} | {r['coefficient']:.3f} |")
    lines += ["", "## Performance (scorecard)",
              "| Period | Loans | Bad rate | AUC | Gini | KS |", "|---|---|---|---|---|---|",
              _row("Train", res["train"]), _row("Validation", res["valid"]), _row("Out-of-time test", res["test"]), "",
              "## Decision threshold",
              f"Chosen on the validation period assuming approving a bad loan costs {COST_BAD:g}x as much as rejecting a good one "
              f"({COST_BAD:g}:{COST_GOOD:g}). Approve when predicted risk <= {thr['threshold']:.3f}. "
              f"On the test period this approves {thr['test_approval_rate']:.1%} of loans, with a bad rate of "
              f"{thr['test_bad_rate_approved']:.1%} among approved loans (overall bad rate {res['test']['bad_rate']:.1%}). "
              "The cost ratio is an assumption, not a measured business figure.", ""]
    if mon:
        lines += ["## Monitoring and retraining (simulated replay)",
                  f"- Later quarters are replayed in order. Drift is measured by PSI on the score and CSI on each input "
                  f"(PSI >= {PSI_ALERT} is an alert). Outcomes are assumed to arrive {LABEL_DELAY_QUARTERS} quarters after a loan.",
                  f"- Quarters replayed: {mon['quarters']}. Promotions of a retrained model: {mon['promotions']} "
                  f"{mon.get('promotion_quarters') or ''}. First PSI alert: {mon.get('first_psi_alert_quarter')}.",
                  "- A retrained challenger replaces the champion only if it beats it on held-out recent quarters."]
        if "backtest_mean_auc_gain" in mon:
            lines.append(f"- On the {mon['backtest_quarters_after_last_promotion']} quarters after the last promotion, mean AUC was "
                         f"{mon['backtest_mean_auc_promoted']:.3f} for the retrained model vs "
                         f"{mon['backtest_mean_auc_original']:.3f} for the original (gain {mon['backtest_mean_auc_gain']:+.3f}).")
        lines.append("")
    if rm:
        lines += ["## Record matching (separate PySpark exercise)",
                  f"On a fictional customer file ({rm['records']:,} records): precision {rm['precision']:.1%}, recall {rm['recall']:.1%}; "
                  f"blocking removed {rm['comparisons_saved_pct']}% of pair comparisons.", ""]
    lines += ["## Limitations",
              "- US data from one platform; Indian lending and bureau data differ.",
              "- Production is simulated. Label delay and drift are replayed from historical data, not observed live.",
              "- Later issue years are not used for training or evaluation because their outcomes are incomplete.",
              "- The cost ratio for the threshold is an assumption.",
              "- No fairness testing across demographic groups was done, and no protected attributes are used. "
              "A real lender would need that analysis.",
              "- The scorecard is a simple model. Higher AUC from XGBoost may not justify the loss of explainability.", ""]
    return "\n".join(lines)


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    text = build()
    (DOCS_DIR / "MODEL_CARD.md").write_text(text, encoding="utf-8")
    print(f"Wrote {DOCS_DIR / 'MODEL_CARD.md'}")


if __name__ == "__main__":
    main()
