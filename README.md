# data-hunt-fraud-m5
[![CI](https://github.com/yokea1/data-hunt-fraud-m5/actions/workflows/ci.yml/badge.svg)](https://github.com/yokea1/data-hunt-fraud-m5/actions)
[![Lint](https://github.com/yokea1/data-hunt-fraud-m5/actions/workflows/lint.yml/badge.svg)](https://github.com/yokea1/data-hunt-fraud-m5/actions)
![License](https://img.shields.io/badge/license-MIT-informational)

**Cost-Sensitive, Time-Split Benchmark for IEEE-CIS Fraud & M5 (with Calibration & Threshold Sweep)**

> Turn AUC/PR into **Expected Cost** under **temporal drift**, report **calibration** (ECE), and ship **reproducible** scripts & artifacts.

## Why this repo
- Business-aligned: decision threshold by **Expected Cost**, not only AUC/PR/F1
- Temporal robustness: **time-split** + rolling-origin backtests
- Calibration: **Reliability** diagram / ECE
- Reproducible: fixed seeds; figures & metrics exported to `reports/`

## Highlights
- **Best-F1 ≠ Min-Cost** under FP:FN=1:10 → fewer FN & lower business loss
- Artifacts-first: PR/ROC, threshold–F1/cost, confusion matrices, reliability; versioned

## Repo structure

data-hunt-fraud-m5/
├─ fraud_baseline.py
├─ m5_baseline.py
├─ scripts/
│ ├─ plot_reliability.py
│ └─ plot_expected_cost.py
├─ reports/
│ └─ figs/
├─ tests/
│ └─ test_runs.py
├─ .github/workflows/{ci.yml,lint.yml}
├─ requirements.txt · .gitignore · LICENSE
└─ README.md
## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python fraud_baseline.py
python scripts/plot_reliability.py --csv reports/ieee_baseline/preds.csv --out reports/figs/reliability.png
python scripts/plot_expected_cost.py --csv reports/ieee_baseline/preds.csv --out reports/figs/fraud_expected_cost.png

Results — Fraud (IEEE-CIS)

Test (HistGradientBoosting): PR-AUC 0.4616, ROC-AUC 0.8776, F1@0.50 0.3767

Validation: best-F1 ≈ 0.72, min-cost ≈ 0.56 (FP:FN = 1:10)












Calibration

Results — M5 (Forecasting)

Features: wide→long, lag1/7/28, rolling_mean_7, weekday/month/holiday

Backtesting: 3-fold expanding-window with fixed seeds

80/20: Naive(t-1) → Ridge improves to MAE 0.4961 (from 0.7097) and MAPE 47.30% (from 69.35%)

3-fold CV mean: MAE 0.7723 / MAPE 68.71%

Data placement

IEEE-CIS CSVs → data/ieee/

M5 CSVs → data/m5/
(Original datasets are not distributed; please respect licenses.)

Reproducibility & Ops

Fixed seeds; artifacts in reports/

Minimal tests via pytest -q

Lint/format: pre-commit (Black/Ruff/Isort)

Citation

If this work helps, star ⭐ and cite a release tag (e.g., v1.0.1).

License

MIT
