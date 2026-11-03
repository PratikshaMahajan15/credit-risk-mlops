"""Champion-challenger retraining.

A challenger is trained on the loans whose outcomes are already known, EXCLUDING the most
recent labelled quarters. Those quarters are held out and both models are scored on them.
The challenger replaces the champion only if it is clearly better.
"""
import pandas as pd

from src.config import (CATEGORICAL_FEATURES, HOLDOUT_QUARTERS, MIN_AUC_GAIN,
                        NUMERIC_FEATURES, TARGET)
from src.metrics import auc, ks
from src.models import ScorecardModel


def split_for_retraining(labeled, holdout_quarters=HOLDOUT_QUARTERS):
    """`labeled` needs a 'quarter' column. Returns (fit_df, holdout_df)."""
    quarters = sorted(labeled["quarter"].unique())
    hold_q = quarters[-holdout_quarters:] if len(quarters) > holdout_quarters else quarters[-1:]
    is_hold = labeled["quarter"].isin(hold_q)
    return labeled[~is_hold], labeled[is_hold]


def train_challenger(fit_df):
    return ScorecardModel(NUMERIC_FEATURES, CATEGORICAL_FEATURES).fit(fit_df, fit_df[TARGET])


def compare(champion, challenger, holdout):
    y = holdout[TARGET]
    p_champ, p_chall = champion.predict_proba(holdout), challenger.predict_proba(holdout)
    return {
        "holdout_n": int(len(holdout)),
        "champion_auc": auc(y, p_champ),
        "challenger_auc": auc(y, p_chall),
        "champion_ks": ks(y, p_champ),
        "challenger_ks": ks(y, p_chall),
    }


def should_promote(comparison, min_gain=MIN_AUC_GAIN):
    gain = comparison["challenger_auc"] - comparison["champion_auc"]
    return bool(gain >= min_gain), float(gain)


def retrain_and_compare(champion, labeled):
    """Convenience wrapper: returns (challenger, comparison, promote, gain, fit_df)."""
    fit_df, holdout = split_for_retraining(labeled)
    challenger = train_challenger(fit_df)
    comparison = compare(champion, challenger, holdout)
    promote, gain = should_promote(comparison)
    return challenger, comparison, promote, gain, fit_df
