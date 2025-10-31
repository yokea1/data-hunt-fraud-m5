# fraud_tree_compare.py
# Usage:
#   python fraud_tree_compare.py
# Outputs (in reports/):
#   fraud_tree_metrics.txt
#   hgb_pr_curve.png, hgb_threshold_sweep.png, hgb_cm_default.png, hgb_cm_bestF1.png
#   (optional XGB) xgb_pr_curve.png, xgb_threshold_sweep.png, xgb_cm_default.png, xgb_cm_bestF1.png

from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay)

# ---------------- CONFIG ----------------
CSV_PATH = Path("ieee_fraud_full_train.csv")
TARGET   = "isFraud"
TEST_RATIO = 0.2
COST_FP = 1.0
COST_FN = 10.0
FAST_ROWS = 120_000     # 抽样加速；全量可设 None 或更大
MAX_V_NUM = 80          # V1..V339 只取前80个
LOW_CARD_LIMIT = 50     # 仅纳入低基数类别列以做简单编码
MAX_CAT_COLS   = 10
OUTDIR = Path("reports")
# ----------------------------------------

OUTDIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(CSV_PATH, low_memory=False)

# 抽样加速（可注释掉）
if FAST_ROWS:
    df = df.sample(n=min(FAST_ROWS, len(df)), random_state=42)

y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET]).copy()
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# 时间切分
if "TransactionDT" in df.columns:
    order = df["TransactionDT"].rank(method="first").astype(int).values
    X["_order_"] = order
    X = X.sort_values("_order_").drop(columns=["_order_"])
    y = y.loc[X.index]
    split_idx = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
else:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=42, stratify=y
    )

# 数值列 + 低基数类别列
num_all = X_train.select_dtypes(include=[np.number]).columns.tolist()
v_cols = [c for c in num_all if c.startswith("V")]
other_nums = [c for c in num_all if not c.startswith("V")]
num_cols = other_nums + v_cols[:MAX_V_NUM]

cat_cands = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
low_card = []
for c in cat_cands:
    if X_train[c].nunique(dropna=True) <= LOW_CARD_LIMIT:
        low_card.append(c)
    if len(low_card) >= MAX_CAT_COLS:
        break

# —— 模型1：HistGradientBoosting（支持缺失、速度快）——
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
    ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

pre_hgb = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, low_card)
], remainder="drop")

hgb = Pipeline([
    ("pre", pre_hgb),
    ("clf", HistGradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=None,
        max_iter=250,
        class_weight={0:1.0, 1:15.0},   # 简单不平衡处理；可视数据调整
        random_state=42
    ))
])

models = [("hgb", hgb)]

# —— 模型2：XGBoost（可选）——
try:
    import xgboost as xgb
    from sklearn.preprocessing import OneHotEncoder
    # 用 OneHot 仅对低基数类别做展开；数值保持原状（缺失交给 XGB 处理也可）
    pre_xgb = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), low_card)
    ], remainder="drop")
    xgb_clf = Pipeline([
        ("pre", pre_xgb),
        ("clf", xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=15.0,  # 不平衡
            random_state=42,
            n_jobs= max(1, 0)
        ))
    ])
    models.append(("xgb", xgb_clf))
except Exception:
    pass  # 没装 xgboost 就跳过

def evaluate_and_save(name, proba, y_true):
    # 指标
    pr_auc = average_precision_score(y_true, proba)
    roc = roc_auc_score(y_true, proba)
    pred05 = (proba >= 0.5).astype(int)
    f1_05 = f1_score(y_true, pred05, pos_label=1)

    # PR 曲线
    prec, rec, thr = precision_recall_curve(y_true, proba)
    plt.figure(); plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{name.upper()} PR (AP={pr_auc:.4f})")
    plt.tight_layout(); plt.savefig(OUTDIR/f"{name}_pr_curve.png", dpi=140); plt.close()

    # 阈值扫描
    ths = np.linspace(0.01, 0.99, 99)
    f1s, costs = [], []
    for t in ths:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        f1s.append(f1_score(y_true, pred, zero_division=0))
        costs.append(fp*COST_FP + fn*COST_FN)
    best_f1_idx = int(np.argmax(f1s))
    best_cost_idx = int(np.argmin(costs))
    best_t_f1, best_t_cost = ths[best_f1_idx], ths[best_cost_idx]

    plt.figure()
    ax1 = plt.gca(); ax1.plot(ths, f1s, label="F1", linewidth=2)
    ax1.set_xlabel("Threshold"); ax1.set_ylabel("F1")
    ax2 = ax1.twinx(); ax2.plot(ths, costs, "--", color="gray", label="Cost")
    ax2.set_ylabel("Expected Cost")
    plt.title(f"{name.upper()} Threshold | bestF1@{best_t_f1:.2f}, minCost@{best_t_cost:.2f}")
    plt.tight_layout(); plt.savefig(OUTDIR/f"{name}_threshold_sweep.png", dpi=140); plt.close()

    # 混淆矩阵
    for t, tag in [(0.5, "default"), (best_t_f1, "bestF1")]:
        pred = (proba >= t).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
        disp.plot(values_format="d"); plt.title(f"{name.upper()} @ {t:.2f}")
        plt.tight_layout(); plt.savefig(OUTDIR/f"{name}_cm_{tag}.png", dpi=140); plt.close()

    return pr_auc, roc, f1_05, best_t_f1, best_t_cost

rows = []
for name, pipe in models:
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:,1]
    pr_auc, roc, f1_05, t_f1, t_cost = evaluate_and_save(name, proba, y_test)
    rows.append([name.upper(), pr_auc, roc, f1_05, t_f1, t_cost])

# 写汇总
with open(OUTDIR/"fraud_tree_metrics.txt", "w", encoding="utf-8") as f:
    f.write("Model\tPR-AUC\tROC-AUC\tF1@0.50\tbestF1_th\tminCost_th\n")
    for r in rows:
        f.write("\t".join([r[0]] + [f"{x:.6f}" if isinstance(x, float) else str(x) for x in r[1:]])+"\n")

print("✅ Done. See reports/fraud_tree_metrics.txt and PNGs.")
