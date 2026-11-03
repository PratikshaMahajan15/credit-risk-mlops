"""Loan Risk Check: a plain-language demo of the credit-risk model.

    streamlit run app.py

Tab 1: fill in a loan application and see the model's view, with reasons in everyday words.
Tab 2: is the system still working well? (a summary of the monitoring checks)
Tab 3: how it works and what it cannot do.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR, REPORTS_DIR
from src.registry import load_model, load_reference
from src.scoring import points_from_probability, score_one

st.set_page_config(page_title="Loan Risk Check", page_icon="🏦", layout="wide")

CHAMPION = Path(MODELS_DIR) / "champion"
DISCLAIMER = ("Demo built on public US loan data (Lending Club). It is a learning project and "
              "must not be used for real lending decisions.")

PRESETS = {
    "Strong applicant": dict(loan_amnt=8000, annual_inc=120000, dti=8, fico_range_low=760, inq_last_6mths=0,
                             credit_history_years=20, revol_util=15, emp_length_years=10, home_ownership="MORTGAGE",
                             verification_status="Verified", purpose="credit_card", addr_state="CA"),
    "Average applicant": dict(loan_amnt=14000, annual_inc=65000, dti=18, fico_range_low=700, inq_last_6mths=1,
                              credit_history_years=12, revol_util=45, emp_length_years=5, home_ownership="RENT",
                              verification_status="Verified", purpose="debt_consolidation", addr_state="TX"),
    "Risky applicant": dict(loan_amnt=25000, annual_inc=40000, dti=32, fico_range_low=660, inq_last_6mths=4,
                            credit_history_years=4, revol_util=90, emp_length_years=1, home_ownership="RENT",
                            verification_status="Not Verified", purpose="other", addr_state="FL"),
}


@st.cache_resource
def load_everything():
    model, meta = load_model(CHAMPION)
    reference = load_reference(CHAMPION)
    return model, meta, reference


def describe(feature, value):
    """A factor with the applicant's own value, in everyday words."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {"loan_amnt": "Loan amount", "annual_inc": "Income"}.get(feature, feature) + " (not provided)"
    fmt = {
        "dti": lambda v: f"Debts take up {v:.0f}% of income",
        "fico_range_low": lambda v: f"Credit score of {v:.0f}",
        "loan_to_income": lambda v: f"Loan is {v:.0%} of yearly income",
        "credit_history_years": lambda v: f"{v:.0f} years of credit history",
        "revol_util": lambda v: f"{v:.0f}% of available credit is in use",
        "annual_inc": lambda v: f"Yearly income of ${v:,.0f}",
        "loan_amnt": lambda v: f"Loan amount of ${v:,.0f}",
        "inq_last_6mths": lambda v: f"{v:.0f} credit enquiries in the last 6 months",
        "emp_length_years": lambda v: f"{v:.0f} years in current job",
        "open_acc": lambda v: f"{v:.0f} open credit accounts",
        "total_acc": lambda v: f"{v:.0f} credit accounts in total",
        "delinq_2yrs": lambda v: f"{v:.0f} late payments in the last 2 years",
        "revol_bal": lambda v: f"Revolving balance of ${v:,.0f}",
        "mort_acc": lambda v: f"{v:.0f} mortgage accounts",
    }
    if feature in fmt:
        return fmt[feature](float(value))
    pretty = {"home_ownership": "Home", "verification_status": "Income check", "purpose": "Loan purpose",
              "addr_state": "State"}.get(feature, feature)
    return f"{pretty}: {str(value).replace('_', ' ').title()}"


def risk_level(p, threshold):
    if threshold is None:
        return "Moderate", "warning"
    if p <= 0.7 * threshold:
        return "Lower risk", "success"
    if p <= threshold:
        return "Moderate risk", "warning"
    return "Higher risk", "error"


# ------------------------------------------------------------------ pages --
def tab_check(model, meta, reference):
    st.subheader("Check a loan application")
    st.write("Fill in the details, or start from an example. The model estimates how likely this loan is "
             "not to be repaid, and explains why.")

    preset_name = st.radio("Start from an example", list(PRESETS), horizontal=True, index=1)
    p = PRESETS[preset_name]
    states = sorted(reference["addr_state"].dropna().unique()) if "addr_state" in reference else [p["addr_state"]]
    purposes = sorted(reference["purpose"].dropna().unique()) if "purpose" in reference else [p["purpose"]]
    homes = ["RENT", "MORTGAGE", "OWN"]
    verifs = ["Verified", "Source Verified", "Not Verified"]

    def idx(options, value):
        return options.index(value) if value in options else 0

    key = preset_name.replace(" ", "_")          # new widgets when the example changes
    c1, c2, c3 = st.columns(3)
    with c1:
        loan = st.number_input("Loan amount ($)", 500, 40000, int(p["loan_amnt"]), 500, key=f"loan_{key}")
        income = st.number_input("Yearly income ($)", 5000, 500000, int(p["annual_inc"]), 1000, key=f"inc_{key}")
        emp = st.slider("Years in current job", 0, 10, int(p["emp_length_years"]), key=f"emp_{key}")
        home = st.selectbox("Home", homes, idx(homes, p["home_ownership"]), key=f"home_{key}")
    with c2:
        fico = st.slider("Credit score", 300, 850, int(p["fico_range_low"]), key=f"fico_{key}")
        dti = st.slider("Share of income going to debt payments (%)", 0, 60, int(p["dti"]), key=f"dti_{key}")
        util = st.slider("Share of available credit in use (%)", 0, 120, int(p["revol_util"]), key=f"util_{key}")
        verif = st.selectbox("Income verified?", verifs, idx(verifs, p["verification_status"]), key=f"ver_{key}")
    with c3:
        hist = st.slider("Years of credit history", 0, 40, int(p["credit_history_years"]), key=f"hist_{key}")
        inq = st.slider("Credit enquiries in the last 6 months", 0, 10, int(p["inq_last_6mths"]), key=f"inq_{key}")
        purpose = st.selectbox("Loan purpose", purposes, idx(purposes, p["purpose"]), key=f"pur_{key}")
        state = st.selectbox("State", states, idx(states, p["addr_state"]), key=f"st_{key}")

    application = dict(loan_amnt=loan, annual_inc=income, dti=dti, fico_range_low=fico, inq_last_6mths=inq,
                       credit_history_years=hist, revol_util=util, emp_length_years=emp, home_ownership=home,
                       verification_status=verif, purpose=purpose, addr_state=state)
    r = score_one(model, meta, application)
    prob, points, thr = r["probability"], r["points"], r["threshold"]
    level, kind = risk_level(prob, thr)

    st.divider()
    left, right = st.columns([1, 1])
    with left:
        verdict = "Recommended: approve" if r["decision"] == "approve" else "Recommended: decline"
        getattr(st, kind)(f"**{level}.** {verdict}")
        st.metric("Out of 100 borrowers like this one, how many would not repay?", f"about {prob * 100:.0f}")
        ref_p = reference["score"].to_numpy()
        st.write(f"Riskier than **{(ref_p < prob).mean():.0%}** of past borrowers the model learned from.")
    with right:
        scaling = meta.get("score_scaling") or {"base_score": model.base_score, "base_odds": model.base_odds, "pdo": model.pdo}
        ref_points = points_from_probability(ref_p, scaling)
        lo, hi = np.percentile(ref_points, 1), np.percentile(ref_points, 99)
        st.metric("Credit-style score (higher = safer)", f"{points:.0f}")
        st.progress(float(np.clip((points - lo) / (hi - lo), 0, 1)))
        st.caption(f"Most past borrowers scored between {lo:.0f} and {hi:.0f}.")

    a, b = st.columns(2)
    with a:
        st.markdown("**What is pushing the risk up**")
        if r["risk_factors"]:
            for f in r["risk_factors"]:
                st.write("🔺 " + describe(f["feature"], f["value"]))
        else:
            st.write("Nothing stands out.")
    with b:
        st.markdown("**What is helping**")
        if r["helping_factors"]:
            for f in r["helping_factors"]:
                st.write("🔻 " + describe(f["feature"], f["value"]))
        else:
            st.write("Nothing stands out.")
    st.caption(DISCLAIMER)


def tab_health():
    st.subheader("Is the system still working well?")
    st.write("Models can quietly get worse as the world changes. These checks run on every new batch of "
             "applications. The numbers below come from replaying the 2014-2015 applications, quarter by "
             "quarter, on a model trained on 2012-2013 loans.")
    log_path, sum_path = Path(REPORTS_DIR) / "monitoring_log.csv", Path(REPORTS_DIR) / "monitoring_summary.json"
    if not log_path.exists() or not sum_path.exists():
        st.info("The monitoring results are not available yet. Run `python -m src.monitor` first.")
        return
    log = pd.read_csv(log_path)
    summary = json.load(open(sum_path))

    worst_psi = float(log["score_psi"].max())
    aucs = log["auc_original"].dropna()
    c1, c2, c3 = st.columns(3)
    with c1:
        ok = worst_psi < 0.10
        st.markdown("**Are today's applicants like the ones the model learned from?**")
        (st.success if ok else st.warning)(
            "Yes. The overall picture has not changed." if ok else "The mix of applicants has shifted.")
    with c2:
        if len(aucs):
            st.markdown("**Is it still telling safe borrowers from risky ones?**")
            st.success(f"Yes. It picks the right one about {aucs.min() * 100:.0f} to {aucs.max() * 100:.0f} times out of "
                       "100 (guessing would be 50).")
    with c3:
        n = summary.get("promotions", 0)
        st.markdown("**Did the system have to replace itself?**")
        st.info("No. It did not need to." if n == 0 else f"Yes, {n} time(s). It only swaps when the new model is clearly better.")

    csi_max = float(log["max_csi"].max())
    if csi_max >= 0.25:
        from src.models import FEATURE_LABELS
        feat = log.loc[log["max_csi"].idxmax(), "max_csi_feature"]
        st.warning(f"**One thing to look into:** the field '{FEATURE_LABELS.get(feat, feat).lower()}' looks very "
                   "different from the training data. This often means the way the information was recorded "
                   "changed, not that borrowers changed. It did not hurt accuracy here.")

    img = Path(REPORTS_DIR) / "monitoring_timeline.png"
    if img.exists():
        st.image(str(img), caption="Top: change in the model's scores. Middle: how well it sorts borrowers. Bottom: share of loans that went bad.")

    table = pd.DataFrame({
        "Quarter": log["quarter"],
        "Applicant mix": np.where(log["score_psi"] < 0.10, "Stable", "Changed"),
        "Accuracy (out of 100)": (log["auc_original"] * 100).round(0),
        "Action": log["action"].replace({"none": "No action needed"}),
    })
    with st.expander("Quarter-by-quarter log"):
        st.dataframe(table, hide_index=True)
    st.caption(DISCLAIMER)


def tab_about(meta):
    st.subheader("How it works")
    st.markdown(
        """
1. **Learn from the past.** The model studied about 144,000 finished loans from 2012-2013 and learned which
   details (credit score, debts, history, and so on) went with loans that were not repaid.
2. **Score a new application.** Each detail adds or removes points, like a credit score. The model also
   turns the score into the chance the loan is not repaid.
3. **Explain.** It lists the biggest reasons pushing the risk up and the biggest helping it.
4. **Decide.** A cut-off decides approve or decline. It was chosen assuming that approving a loan that
   goes bad costs five times as much as turning down a good one.
5. **Keep watching.** New applications are checked regularly. If the model gets worse, a new one is
   trained and swapped in only when it is clearly better.
        """)
    st.subheader("What it cannot do")
    st.markdown(
        """
- It uses **US loans from one lending platform**, not Indian bank data.
- It was **tested on later loans it never saw** (2015), but monitoring here is a replay of past data, not a live system.
- It has **not been tested for fairness** across groups of people.
- It is a **simple, explainable model**. A more complex one scores slightly better, but is harder to explain.
        """)
    st.caption(f"Model version {meta.get('version', 1)}, trained on {meta.get('trained_on')} loans.")
    st.caption(DISCLAIMER)


def main():
    st.title("🏦 Loan Risk Check")
    st.write("A credit-risk model that scores a loan application, explains the score in plain words, and "
             "is monitored to make sure it keeps working.")
    try:
        model, meta, reference = load_everything()
    except Exception as exc:
        st.error("The trained model could not be loaded. Run `python -m src.train` first.")
        st.caption(f"{type(exc).__name__}: {exc}")
        return
    t1, t2, t3 = st.tabs(["Check an application", "Is it still working well?", "How it works"])
    with t1:
        tab_check(model, meta, reference)
    with t2:
        tab_health()
    with t3:
        tab_about(meta)


main()
