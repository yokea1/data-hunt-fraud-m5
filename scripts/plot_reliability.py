import argparse, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def ece(y_true, y_prob, bins=15):
    cuts = np.linspace(0,1,bins+1); n=len(y_true); e=0.0
    for i in range(bins):
        idx = (y_prob>=cuts[i]) & (y_prob<cuts[i+1])
        if idx.sum()==0: continue
        acc = (y_true[idx] == (y_prob[idx]>=0.5)).mean()
        conf = y_prob[idx].mean()
        e += (idx.sum()/n)*abs(acc-conf)
    return float(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="reports/ieee_baseline/preds.csv")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--out", default="reports/figs/reliability.png")
    args = ap.parse_args()

    if os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        y_true = df["y_true"].values.astype(int)
        y_prob = df["y_prob"].values.astype(float)
    else:
        rng = np.random.default_rng(42); n=5000
        y_true = (rng.random(n)<0.06).astype(int)
        y_prob = np.clip(0.06 + 0.15*rng.normal(size=n), 0, 1)

    cuts = np.linspace(0,1,args.bins+1); xs, ys = [], []
    for i in range(args.bins):
        idx = (y_prob>=cuts[i]) & (y_prob<cuts[i+1])
        if idx.sum()==0: continue
        xs.append((cuts[i]+cuts[i+1])/2)
        ys.append((y_true[idx]==(y_prob[idx]>=0.5)).mean())
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1],"--",linewidth=1); plt.plot(xs, ys, marker="o")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy")
    plt.title(f"Reliability")
    plt.tight_layout(); os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=180); print(f"[OK] saved -> {args.out}")

if __name__=="__main__": main()
