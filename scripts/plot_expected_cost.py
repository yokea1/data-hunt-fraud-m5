import argparse, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def expected_cost(y_true, y_prob, t, fp_cost, fn_cost):
    y_pred = (y_prob >= t).astype(int)
    fp = np.sum((y_pred==1) & (y_true==0))
    fn = np.sum((y_pred==0) & (y_true==1))
    return fp*fp_cost + fn*fn_cost

def best_f1(y_true, y_prob, grid):
    best=(0.5,-1.0)
    for t in grid:
        y_pred = (y_prob>=t).astype(int)
        tp = np.sum((y_pred==1)&(y_true==1))
        fp = np.sum((y_pred==1)&(y_true==0))
        fn = np.sum((y_pred==0)&(y_true==1))
        prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
        f1 = 2*prec*rec/(prec+rec+1e-12)
        if f1>best[1]: best=(float(t), float(f1))
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reports/ieee_baseline/preds.csv")
    ap.add_argument("--fp-cost", type=float, default=1.0)
    ap.add_argument("--fn-cost", type=float, default=10.0)
    ap.add_argument("--out", default="reports/figs/fraud_expected_cost.png")
    ap.add_argument("--points", type=int, default=199)
    args = ap.parse_args()

    if os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        y_true = df["y_true"].values.astype(int)
        y_prob = df["y_prob"].values.astype(float)
    else:
        rng = np.random.default_rng(7); n=5000
        y_true = (rng.random(n)<0.06).astype(int)
        y_prob = np.clip(0.06 + 0.15*rng.normal(size=n), 0, 1)

    grid = np.linspace(0.01, 0.99, args.points)
    costs = [expected_cost(y_true, y_prob, t, args.fp_cost, args.fn_cost) for t in grid]
    t_min = float(grid[int(np.argmin(costs))]); c_min = float(np.min(costs))
    f1s=[]
    for t in grid:
        y_pred = (y_prob>=t).astype(int)
        tp = np.sum((y_pred==1)&(y_true==1))
        fp = np.sum((y_pred==1)&(y_true==0))
        fn = np.sum((y_pred==0)&(y_true==1))
        prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
        f1s.append(2*prec*rec/(prec+rec+1e-12))
    t_f1 = float(grid[int(np.argmax(f1s))])

    plt.figure(figsize=(6,4))
    plt.plot(grid, costs)
    plt.axvline(t_min, linestyle="--")
    plt.axvline(t_f1, linestyle=":")
    plt.xlabel("Threshold"); plt.ylabel("Expected Cost")
    plt.title(f"Expected-Cost Curve  (minCost@{t_min:.2f}={c_min:.0f}, bestF1@{t_f1:.2f})")
    plt.tight_layout(); os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=180); print(f"[OK] saved -> {args.out}")

if __name__=="__main__": main()
