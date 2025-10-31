## 📊 Results

**Fraud (IEEE-CIS)**
- **Best-F1 ≠ Min-Cost**: under FP:FN=1:10 the cost-optimal threshold is typically **lower** than best-F1 → fewer **FN** and lower business loss.
- Test (HGB): **PR-AUC 0.4616**, **ROC-AUC 0.8776**, **F1@0.50 0.3767**; **best-F1 ≈ 0.72**, **min-cost ≈ 0.56** (valid).

![PR curve (LogReg)](reports/figs/fraud_pr_curve.png)
![PR curve (HGB)](reports/figs/hgb_pr_curve.png)
![Expected-Cost](reports/figs/fraud_expected_cost.png)
![Threshold sweep](reports/figs/fraud_threshold_sweep.png)
![Confusion matrix @ best-F1](reports/figs/hgb_cm_bestF1.png)
![Confusion matrix @ 0.50](reports/figs/hgb_cm_default.png)

**Calibration**
![Reliability](reports/figs/reliability.png)
*Reliability diagram; closer to y=x indicates better calibration.*

**M5 (Forecasting)**
![Last-fold actual vs prediction](reports/figs/m5_pred_plot.png)
*Expanding-window backtest; last fold shown.*

## 📊 Results

**Fraud (IEEE-CIS)**
- **Best-F1 ≠ Min-Cost**: under FP:FN=1:10 the cost-optimal threshold is typically **lower** than best-F1 → fewer **FN** and lower business loss.
- Test (HGB): **PR-AUC 0.4616**, **ROC-AUC 0.8776**, **F1@0.50 0.3767**; **best-F1 ≈ 0.72**, **min-cost ≈ 0.56** (valid).

![PR curve (LogReg)](reports/figs/fraud_pr_curve.png)
![PR curve (HGB)](reports/figs/hgb_pr_curve.png)
![Expected-Cost](reports/figs/fraud_expected_cost.png)
![Threshold sweep](reports/figs/fraud_threshold_sweep.png)
![Confusion matrix @ best-F1](reports/figs/hgb_cm_bestF1.png)
![Confusion matrix @ 0.50](reports/figs/hgb_cm_default.png)

**Calibration**
![Reliability](reports/figs/reliability.png)
*Reliability diagram; closer to y=x indicates better calibration.*

**M5 (Forecasting)**
![Last-fold actual vs prediction](reports/figs/m5_pred_plot.png)
*Expanding-window backtest; last fold shown.*
