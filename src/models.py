"""Models: a WoE logistic-regression scorecard (champion) and an XGBoost challenger.

All models share the same small interface:
    fit(df, y)            train on a DataFrame that contains the feature columns
    predict_proba(df)     probability of default (bad loan) for each row
"""
import warnings

import numpy as np
import pandas as pd
from optbinning import BinningProcess
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import (MAX_CORR, MIN_IV, RANDOM_STATE, SCORE_BASE,
                        SCORE_BASE_ODDS, SCORE_PDO)

warnings.filterwarnings("ignore")

# Plain-language names used in reason codes and explanations.
FEATURE_LABELS = {
    "loan_amnt": "Loan amount",
    "annual_inc": "Annual income",
    "dti": "Debt-to-income ratio",
    "delinq_2yrs": "Delinquencies in the last 2 years",
    "fico_range_low": "Credit (FICO) score at application",
    "inq_last_6mths": "Credit enquiries in the last 6 months",
    "open_acc": "Number of open credit accounts",
    "pub_rec": "Public derogatory records",
    "revol_bal": "Revolving balance",
    "revol_util": "Revolving credit utilisation",
    "total_acc": "Total number of credit accounts",
    "mort_acc": "Number of mortgage accounts",
    "pub_rec_bankruptcies": "Public bankruptcies",
    "emp_length_years": "Length of employment",
    "credit_history_years": "Length of credit history",
    "loan_to_income": "Loan amount relative to income",
    "home_ownership": "Home ownership status",
    "verification_status": "Income verification status",
    "purpose": "Loan purpose",
    "addr_state": "State of residence",
}


def _prepare(df, features, categorical):
    """Select the model columns; categorical values become plain strings (NaN stays missing)."""
    X = df.reindex(columns=features).copy()
    for c in categorical:
        X[c] = X[c].astype("object").where(X[c].notna(), None)
    return X


class ScorecardModel:
    """Binning (WoE) + logistic regression, with points scaling and reason codes."""

    model_type = "scorecard"

    def __init__(self, numeric, categorical, pdo=SCORE_PDO, base_score=SCORE_BASE,
                 base_odds=SCORE_BASE_ODDS, min_iv=MIN_IV, max_corr=MAX_CORR, C=1.0):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.features = self.numeric + self.categorical
        self.pdo, self.base_score, self.base_odds = pdo, base_score, base_odds
        self.min_iv, self.max_corr, self.C = min_iv, max_corr, C

    # ------------------------------------------------------------------ fit --
    def fit(self, df, y):
        y = np.asarray(y)
        X = _prepare(df, self.features, self.categorical)
        self.bp_ = BinningProcess(
            variable_names=self.features,
            categorical_variables=self.categorical,
            min_prebin_size=0.02,
        )
        self.bp_.fit(X, y)
        summary = self.bp_.summary()
        self.iv_ = dict(zip(summary["name"], summary["iv"].astype(float)))
        woe = self.bp_.transform(X, metric="woe")

        # 1. keep features with enough information value
        candidates = sorted([f for f in self.features if self.iv_[f] >= self.min_iv],
                            key=lambda f: -self.iv_[f])
        if len(candidates) < 3:                                   # tiny data fallback
            candidates = sorted(self.features, key=lambda f: -self.iv_[f])[:5]

        # 2. drop features that are almost duplicates of a stronger one
        corr = woe[candidates].corr().abs()
        selected = []
        for f in candidates:
            if all(corr.loc[f, s] < self.max_corr for s in selected):
                selected.append(f)

        # 3. logistic regression; WoE is "higher = safer", so every coefficient should be negative.
        #    A positive coefficient is a wrong-way sign, so that feature is removed and the model refitted.
        self.dropped_wrong_sign_ = []
        while True:
            lr = LogisticRegression(C=self.C, max_iter=1000)
            lr.fit(woe[selected], y)
            coefs = pd.Series(lr.coef_[0], index=selected)
            wrong = coefs[coefs > 0]
            if wrong.empty or len(selected) <= 2:
                break
            worst = wrong.idxmax()
            selected.remove(worst)
            self.dropped_wrong_sign_.append(worst)
        self.selected_ = selected
        self.lr_ = lr
        self.coef_ = dict(zip(selected, lr.coef_[0]))
        self.intercept_ = float(lr.intercept_[0])
        return self

    # -------------------------------------------------------------- predict --
    def _woe(self, df):
        X = _prepare(df, self.features, self.categorical)
        return self.bp_.transform(X, metric="woe")[self.selected_]

    def predict_proba(self, df):
        return self.lr_.predict_proba(self._woe(df))[:, 1]

    def points(self, df):
        """Credit-score style points: higher = safer. base_score at base_odds, +pdo doubles the odds."""
        factor = self.pdo / np.log(2)
        offset = self.base_score - factor * np.log(self.base_odds)
        logit = self.lr_.decision_function(self._woe(df))
        return offset - factor * logit

    # ---------------------------------------------------------------- reasons --
    def contributions(self, df):
        """Contribution of each selected feature to the log-odds of default.
        Positive = pushes the risk up (a reason for a worse score)."""
        woe = self._woe(df)
        coefs = np.array([self.coef_[f] for f in self.selected_])
        return pd.DataFrame(woe.to_numpy() * coefs, columns=self.selected_, index=df.index)

    def top_reasons(self, df, n=3):
        contrib = self.contributions(df)
        reasons = []
        for _, row in contrib.iterrows():
            top = row[row > 0].sort_values(ascending=False).head(n)
            reasons.append([(f, float(v)) for f, v in top.items()])
        return reasons

    # ---------------------------------------------------------------- reports --
    def feature_report(self):
        rows = []
        for f in self.features:
            rows.append({
                "feature": f,
                "iv": self.iv_.get(f),
                "selected": f in self.selected_,
                "coefficient": self.coef_.get(f),
            })
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def binning_tables(self):
        tables = {}
        for f in self.selected_:
            t = self.bp_.get_binned_variable(f).binning_table.build()
            t = t[t["Bin"].astype(str) != ""].copy()
            tables[f] = t
        return tables


class XGBModel:
    """Gradient-boosted trees (challenger). Categorical columns use XGBoost's native support."""

    model_type = "xgboost"

    def __init__(self, numeric, categorical, **params):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        self.features = self.numeric + self.categorical
        self.params = dict(
            n_estimators=600, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=50, reg_lambda=5.0,
            tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
            eval_metric="auc", n_jobs=-1, random_state=RANDOM_STATE,
        )
        self.params.update(params)

    def _x(self, df):
        X = df.reindex(columns=self.features).copy()
        for c in self.numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        for c in self.categorical:
            X[c] = pd.Categorical(X[c].astype("object"), categories=self.cats_[c])
        return X

    def fit(self, df, y, eval_df=None, eval_y=None):
        y = np.asarray(y)
        self.cats_ = {c: sorted(df[c].dropna().astype(str).unique()) for c in self.categorical}
        params = dict(self.params)
        fit_kwargs = {}
        if eval_df is not None:
            params["early_stopping_rounds"] = 40
            fit_kwargs["eval_set"] = [(self._x(eval_df), np.asarray(eval_y))]
            fit_kwargs["verbose"] = False
        self.model_ = XGBClassifier(**params)
        self.model_.fit(self._x(df), y, **fit_kwargs)
        return self

    def predict_proba(self, df):
        return self.model_.predict_proba(self._x(df))[:, 1]


class LogRegBaseline:
    """Simple baseline: logistic regression on the raw numeric features (no binning)."""

    model_type = "logreg_raw"

    def __init__(self, numeric, categorical=None):
        self.numeric = list(numeric)
        self.categorical = []
        self.features = self.numeric

    def fit(self, df, y):
        X = df.reindex(columns=self.features).apply(pd.to_numeric, errors="coerce")
        self.pipe_ = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )
        self.pipe_.fit(X, np.asarray(y))
        return self

    def predict_proba(self, df):
        X = df.reindex(columns=self.features).apply(pd.to_numeric, errors="coerce")
        return self.pipe_.predict_proba(X)[:, 1]
