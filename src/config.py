"""Shared paths and settings for the credit-risk pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_SUBSET = INTERIM_DIR / "loans_raw_subset.parquet"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"

# ---- MLflow ---------------------------------------------------------------
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_ARTIFACT_DIR = ROOT / "mlartifacts"
MLFLOW_EXPERIMENT = "credit-risk-mlops"
REGISTERED_MODEL_NAME = "credit_risk_scorecard"

# ---- Data definition ------------------------------------------------------
# Outcomes that are final (the loan is finished). Everything else is excluded from training.
GOOD_STATUS = ["Fully Paid"]
BAD_STATUS = ["Charged Off", "Default"]

# Columns to read from the raw file. Only fields known at application time,
# plus the outcome (loan_status) and the issue date used for the time-based split.
KEEP_COLUMNS = [
    "id", "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status", "issue_d",
    "loan_status", "purpose", "addr_state", "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "application_type", "mort_acc",
    "pub_rec_bankruptcies",
]

TERM = "36 months"             # only 36-month loans: their outcomes are complete earliest
TARGET = "bad"

# Time-based split by loan issue year (never a random split).
TRAIN_YEARS = [2012, 2013]
VALID_YEARS = [2014]
TEST_YEARS = [2015]

# Deliberately NOT used as model inputs:
#  grade, sub_grade, int_rate : assigned by Lending Club's own scoring model
#  installment                : calculated from loan amount, int_rate and term, so it leaks int_rate
#  application_type           : constant in the early years, so it only creates fake drift
#  fico_range_high            : almost identical to fico_range_low
EXCLUDED_FROM_MODEL = [
    "grade", "sub_grade", "int_rate", "installment",
    "application_type", "fico_range_high",
]

NUMERIC_FEATURES = [
    "loan_amnt", "annual_inc", "dti", "delinq_2yrs", "fico_range_low",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
    "total_acc", "mort_acc", "pub_rec_bankruptcies",
    "emp_length_years", "credit_history_years", "loan_to_income",
]
CATEGORICAL_FEATURES = ["home_ownership", "verification_status", "purpose", "addr_state"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---- Scorecard settings ---------------------------------------------------
SCORE_BASE = 600        # points at the base odds
SCORE_BASE_ODDS = 50    # good:bad odds at the base score
SCORE_PDO = 20          # points to double the odds
MIN_IV = 0.02           # drop features with weaker information value
MAX_CORR = 0.80         # drop a feature if its WoE is this correlated with a stronger one

# ---- Approval threshold (cost based) --------------------------------------
# Assumption: approving a loan that goes bad costs 5x what rejecting a good loan costs.
COST_BAD = 5.0
COST_GOOD = 1.0

# ---- Monitoring and retraining --------------------------------------------
PSI_WARN = 0.10                 # small shift
PSI_ALERT = 0.25                # large shift -> investigate / retrain
AUC_DROP_TRIGGER = 0.02         # retrain if AUC falls this far below the baseline
MIN_AUC_GAIN = 0.005            # a challenger must beat the champion by this much
LABEL_DELAY_QUARTERS = 2        # simulated: outcomes arrive this many quarters after the loan
HOLDOUT_QUARTERS = 2            # most recent labelled quarters used to compare models
COOLDOWN_QUARTERS = 2           # wait this long after a promotion before retraining again

RANDOM_STATE = 42
