from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

DATA = Path("m5_sample_long.csv")
CAL  = Path("calendar.csv")  # 可选
OUT  = Path("reports"); OUT.mkdir(parents=True, exist_ok=True)

def mape(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.where(y_true==0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / denom))*100

df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# 可选：节假日
if CAL.exists():
    cal = pd.read_csv(CAL)[["date","event_name_1","event_type_1"]]
    cal["date"] = pd.to_datetime(cal["date"])
    df = df.merge(cal, on="date", how="left")
    df["is_event"] = (~df["event_name_1"].isna()).astype(int)
else:
    df["is_event"] = 0

# 基础特征
df["lag1"]  = df["value"].shift(1)
df["lag7"]  = df["value"].shift(7)
df["lag28"] = df["value"].shift(28)
df["roll_mean_7"] = df["value"].rolling(7).mean()
df["dow"] = df["date"].dt.weekday
df["month"] = df["date"].dt.month
df["is_weekend"] = (df["dow"]>=5).astype(int)

df = df.dropna().reset_index(drop=True)

# 构造 3 折 expanding-window
n = len(df)
i1 = int(n*0.6); i2 = int(n*0.8)
folds = [((0, i1), (i1, i2)), ((0, i2), (i2, n))]
i0 = int(n*0.2)
folds.append(((i0, i2), (i2, n)))

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

X_num = ["lag1","lag7","lag28","roll_mean_7","is_event","is_weekend","month"]
X_cat = ["dow"]
ycol  = "value"

pre = ColumnTransformer([
    ("num", StandardScaler(with_mean=False), X_num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), X_cat)
], remainder="drop")

reg = Pipeline([
    ("pre", pre),
    ("ridge", Ridge(alpha=1.0))
])

rows = []
last_fold_plot = None

for k,(tr,te) in enumerate(folds, start=1):
    tr_s, tr_e = tr; te_s, te_e = te
    train = df.iloc[tr_s:tr_e]; test = df.iloc[te_s:te_e]
    reg.fit(train[X_num+X_cat], train[ycol])
    pred = reg.predict(test[X_num+X_cat])
    mae  = mean_absolute_error(test[ycol], pred)
    mpe  = mape(test[ycol], pred)
    rows.append([k, len(train), len(test), mae, mpe])
    last_fold_plot = (test["date"], test[ycol].values, pred)

arr = np.array([[r[3], r[4]] for r in rows])
mean_mae, mean_mpe = arr[:,0].mean(), arr[:,1].mean()

with open(OUT/"m5_cv_metrics.txt","w",encoding="utf-8") as f:
    f.write("Fold\tTrainN\tTestN\tMAE\tMAPE(%)\n")
    for r in rows:
        f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.4f}\t{r[4]:.2f}\n")
    f.write(f"\nMean\t-\t-\t{mean_mae:.4f}\t{mean_mpe:.2f}\n")

dates, y_true, y_pred = last_fold_plot
plt.figure(figsize=(10,4))
plt.plot(dates, y_true, label="Actual")
plt.plot(dates, y_pred, label="Ridge (lags+calendar)")
plt.legend(); plt.title("M5 - Expanding Window CV (last fold)")
plt.tight_layout(); plt.savefig(OUT/"m5_cv_lastfold_plot.png", dpi=140); plt.close()

print("✅ Done. See reports/m5_cv_metrics.txt & m5_cv_lastfold_plot.png")
