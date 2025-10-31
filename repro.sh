set -e
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python fraud_baseline.py
python scripts/plot_reliability.py --csv reports/ieee_baseline/preds.csv --out reports/figs/reliability.png
python scripts/plot_expected_cost.py --csv reports/ieee_baseline/preds.csv --out reports/figs/fraud_expected_cost.png
pytest -q
