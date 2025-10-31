# fraud_baseline.py (with imputers)
# Usage:
#   python fraud_baseline.py
# Outputs:
#   reports/fraud_metrics.txt
#   reports/fraud_pr_curve.png
#   reports/fraud_threshold_sweep.png
#   reports/fraud_cm_default.png
#   reports/fraud_cm_bestF1.png

from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay)

# =========== CONFIG ===========
CSV_PATH = Path("ieee_fraud_full_train.csv")  # 如果文件名不同在此修改
TARGET   = "isFraud"
TEST_RATIO = 0.2
COST_FP = 1.0
COST_FN = 10.0
MISSING_COL_THRESHOLD = 0.5   # 缺失>50%的列会被丢弃
LOW_CARD_LIMIT = 50           # 仅保留基数≤50的少量类别列
MAX_CAT_COLS   = 10           # 最多取10个低基数类别列
OUTDIR = Path("reports")
# ==============================

OUTDIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(CSV_PATH, low_memory=False)

# 目标
y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET])

# 将 inf/-inf 先转成 NaN，方便填充
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# 丢弃缺失占比太高的列（>50%）
na_ratio = X.isna().mean()
keep_cols = na_ratio[na_ratio <= MISSING_COL_THRESHOLD].index.tolist()
X = X[keep_cols].copy()

# 时间切分优先（避免信息泄漏）
if "TransactionDT" in df.columns:
    # 重新从 X 对应索引切分
    order = df["TransactionDT"].rank(method="first").astype(int).values
    X["_order_"] = order
    X = X.sort_values("_order_").drop(columns=["_order_"])
    y = y.loc[X.index]
    split_idx = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=42, stratify=y
    )

# 特征选择：数值 + 少量低基数类别列
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cands = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

# 仅保留低基数类别列
low_card = []
for c in cat_cands:
    # 统计含 NaN 在内的唯一值个数（填充前粗估）
    uniq = X_train[c].nunique(dropna=True)
    if uniq <= LOW_CARD_LIMIT:
        low_card.append(c)
    if len(low_card) >= MAX_CAT_COLS:
        break

features = num_cols + low_card
X_train = X_train[features].copy()
X_test  = X_test[features].copy()

# 预处理：数值 -> 中位数填充 + 标准化；类别 -> 常量"missing"填充 + 顺序编码
num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False))
])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
    ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

pre = ColumnTransformer([
    ("num", num_pipe, [c for c in features if c in num_cols]),
    ("cat", cat_pipe, low_card)
], remainder="drop")

clf = Pipeline([
    ("pre", pre),
    ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", solver="saga"))
])

clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)[:, 1]

# 指标（默认阈值 0.5）
pred_05 = (proba >= 0.5).astype(int)
pr_auc = average_precision_score(y_test, proba)
roc_auc = roc_auc_score(y_test, proba)
f1_pos = f1_score(y_test, pred_05, pos_label=1)

# PR 曲线
prec, rec, thr = precision_recall_curve(y_test, proba)
plt.figure()
plt.plot(rec, prec)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve (AP={pr_auc:.4f})")
plt.tight_layout(); plt.savefig(OUTDIR/"fraud_pr_curve.png", dpi=140); plt.close()

# 阈值扫描（F1 & 期望成本）
ths = np.linspace(0.01, 0.99, 99)
f1s, costs = [], []
from sklearn.metrics import confusion_matrix
for t in ths:
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
    f1s.append(f1_score(y_test, pred, zero_division=0))
    cost = fp * COST_FP + fn * COST_FN
    costs.append(cost)
best_f1_idx   = int(np.argmax(f1s))
best_cost_idx = int(np.argmin(costs))
best_t_f1     = ths[best_f1_idx]
best_t_cost   = ths[best_cost_idx]

plt.figure()
ax1 = plt.gca()
ax1.plot(ths, f1s, label="F1 (pos)", linewidth=2)
ax1.set_xlabel("Threshold"); ax1.set_ylabel("F1")
ax2 = ax1.twinx()
ax2.plot(ths, costs, linestyle="--", label="Expected Cost", color="gray")
ax2.set_ylabel("Expected Cost")
plt.title(f"Threshold Sweep | bestF1@{best_t_f1:.2f}, minCost@{best_t_cost:.2f}")
plt.tight_layout(); plt.savefig(OUTDIR/"fraud_threshold_sweep.png", dpi=140); plt.close()

# 混淆矩阵（阈值0.5 与 最佳F1阈值）
from sklearn.metrics import ConfusionMatrixDisplay
for t, name in [(0.5, "default"), (best_t_f1, "bestF1")]:
    pred = (proba >= t).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix @ threshold={t:.2f}")
    plt.tight_layout(); plt.savefig(OUTDIR/f"fraud_cm_{name}.png", dpi=140); plt.close()

# 写指标
with open(OUTDIR/"fraud_metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"PR-AUC: {pr_auc:.6f}\n")
    f.write(f"ROC-AUC: {roc_auc:.6f}\n")
    f.write(f"F1(pos) @0.50: {f1_pos:.6f}\n")
    f.write(f"Best F1 threshold: {best_t_f1:.4f}, F1={f1s[best_f1_idx]:.6f}\n")
    f.write(f"Min Expected Cost threshold: {best_t_cost:.4f}, Cost={costs[best_cost_idx]:.2f} (FP={COST_FP}, FN={COST_FN})\n")

print("✅ Done. See reports/fraud_metrics.txt & PNGs.")
