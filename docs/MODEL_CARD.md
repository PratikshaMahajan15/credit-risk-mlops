# Model Card: Credit-Risk Scorecard

_Generated on 2026-11-23 from the last pipeline run._

## Purpose
Estimate the probability that a loan will default (be charged off) from information known at application time, and turn it into a credit-style score with reason codes. This is a learning and portfolio project on public data. **It must not be used for real lending decisions.**

## Data
- Source: public Lending Club accepted-loans data (Kaggle). US peer-to-peer loans, not Indian bank data.
- Only 36-month loans with a final outcome (Fully Paid vs Charged Off/Default) are used, because later loans have not finished yet.
- Time-based split by issue year: train [2012, 2013], validation [2014], out-of-time test [2015]. No random splitting.
- Excluded on purpose to avoid leakage: grade, sub_grade, int_rate, installment, application_type, fico_range_high. Grade, sub-grade and interest rate come from the lender's own model, and the instalment is calculated from the rate.

## Model
- Logistic regression on Weight-of-Evidence binned features (own implementation in src/binning.py). Features kept: 9 of 20 after an information-value cut-off and a correlation filter.
- Points scaling: 600 points at 50:1 good:bad odds, 20 points to double the odds. Higher score = lower risk.
- Reason codes: the features that push a borrower's risk up the most (coefficient x WoE).
- Challenger: XGBoost (test AUC 0.664); raw logistic-regression baseline also trained.

### Features used
| Feature | Information value | Coefficient |
|---|---|---|
| fico_range_low | 0.130 | -0.825 |
| annual_inc | 0.096 | -0.569 |
| loan_to_income | 0.046 | -0.654 |
| mort_acc | 0.042 | -0.268 |
| dti | 0.041 | -0.559 |
| home_ownership | 0.035 | -0.334 |
| inq_last_6mths | 0.028 | -1.176 |
| revol_util | 0.028 | -0.141 |
| credit_history_years | 0.021 | -0.185 |

## Performance (scorecard)
| Period | Loans | Bad rate | AUC | Gini | KS |
|---|---|---|---|---|---|
| Train | 143,892 | 12.7% | 0.645 | 0.290 | 0.208 |
| Validation | 162,570 | 13.7% | 0.644 | 0.288 | 0.208 |
| Out-of-time test | 283,026 | 14.9% | 0.646 | 0.293 | 0.209 |

## Decision threshold
Chosen on the validation period assuming approving a bad loan costs 5x as much as rejecting a good one (5:1). Approve when predicted risk <= 0.156. On the test period this approves 72.4% of loans, with a bad rate of 11.7% among approved loans (overall bad rate 14.9%). The cost ratio is an assumption, not a measured business figure.

## Monitoring and retraining (simulated replay)
- Later quarters are replayed in order. Drift is measured by PSI on the score and CSI on each input (PSI >= 0.25 is an alert). Outcomes are assumed to arrive 2 quarters after a loan.
- Quarters replayed: 8. Promotions of a retrained model: 0 . First PSI alert: None.
- A retrained challenger replaces the champion only if it beats it on held-out recent quarters.

## Limitations
- US data from one platform; Indian lending and bureau data differ.
- Production is simulated. Label delay and drift are replayed from historical data, not observed live.
- Later issue years are not used for training or evaluation because their outcomes are incomplete.
- The cost ratio for the threshold is an assumption.
- No fairness testing across demographic groups was done, and no protected attributes are used. A real lender would need that analysis.
- The scorecard is a simple model. Higher AUC from XGBoost may not justify the loss of explainability.
