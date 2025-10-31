# m5_baseline.py
# Usage:
#   python m5_baseline.py
# Outputs:
#   reports/m5_metrics.txt
#   reports/m5_pred_plot.png

from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

DATA = Path("m5_sample_long.csv")
CAL  = Path("calendar.csv")   # 可选
OUT  = Path("reports"); OUT.mkdir(parents=True, exist_ok=True)

def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.where(y_true==0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 可选：节假日/事件
if CAL.exists():
    cal = pd.read_csv(CAL)[["date","event_name_1","event_type_1"]]
    cal["date"] = pd.to_datetime(cal["date"])
    df = df.merge(cal, on="date", how="left")
    df["is_event"] = (~df["event_name_1"].isna()).astype(int)
else:
    df["is_event"] = 0

# 特征工程
df["lag1"] = df["value"].shift(1)
df["dow"]  = df["date"].dt.weekday   # 0=Mon
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["dow"]>=5).astype(int)
df = df.dropna().reset_index(drop=True)  # 去掉第一个缺lag1的样本

# 时间切分
n = len(df); split = int(n*0.8)
train, test = df.iloc[:split], df.iloc[split:]

# Baseline: Naive y_{t-1}
naive_pred = test["lag1"].values
mae_naive  = mean_absolute_error(test["value"], naive_pred)
mape_naive = mape(test["value"], naive_pred)

# 线性回归（含类别哑变量）
X_cols_num = ["lag1", "is_event", "is_weekend", "month"]  # month 当数值也可
X_cols_cat = ["dow"]                                       # 对星期做 one-hot
y_col = "value"

pre = ColumnTransformer([
    ("num", StandardScaler(with_mean=False), X_cols_num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), X_cols_cat)
], remainder="drop")

model = Pipeline([
    ("pre", pre),
    ("reg", Ridge(alpha=1.0))
])

model.fit(train[X_cols_num + X_cols_cat], train[y_col])
pred = model.predict(test[X_cols_num + X_cols_cat])

mae_reg  = mean_absolute_error(test[y_col], pred)
mape_reg = mape(test[y_col], pred)

# 指标写入
with open(OUT/"m5_metrics.txt", "w", encoding="utf-8") as f:
    f.write("=== Naive(t-1) ===\n")
    f.write(f"MAE:  {mae_naive:.4f}\nMAPE: {mape_naive:.2f}%\n\n")
    f.write("=== Ridge (lag1 + weekday/month/weekend + is_event) ===\n")
    f.write(f"MAE:  {mae_reg:.4f}\nMAPE: {mape_reg:.2f}%\n")

# 画 真实 vs 预测（只画测试段）
plt.figure(figsize=(10,4))
plt.plot(test["date"], test[y_col], label="Actual")
plt.plot(test["date"], naive_pred, label="Naive(t-1)", alpha=0.8)
plt.plot(test["date"], pred, label="Ridge", alpha=0.8)
plt.legend(); plt.title("M5 - Actual vs Predictions (test fold)")
plt.tight_layout(); plt.savefig(OUT/"m5_pred_plot.png", dpi=140); plt.close()

print("✅ Done. See reports/m5_metrics.txt & m5_pred_plot.png")
