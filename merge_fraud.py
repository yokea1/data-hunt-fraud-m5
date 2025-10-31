# merge_fraud.py
# Usage:
#   python merge_fraud.py                # full merge
#   python merge_fraud.py --sample 0.3   # 30% sample (by rows) to save memory
#
# Inputs expected in the current folder:
#   - train_transaction.csv
#   - train_identity.csv
#
# Output:
#   - ieee_fraud_full_train.csv

import argparse
import pandas as pd
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trans", default="train_transaction.csv")
    p.add_argument("--iden",  default="train_identity.csv")
    p.add_argument("--out",   default="ieee_fraud_full_train.csv")
    p.add_argument("--sample", type=float, default=None, help="Row sample frac for transaction table (e.g., 0.3)")
    args = p.parse_args()

    trans_path = Path(args.trans)
    iden_path  = Path(args.iden)
    if not trans_path.exists():
        raise FileNotFoundError(f"Missing file: {trans_path.resolve()}")
    if not iden_path.exists():
        raise FileNotFoundError(f"Missing file: {iden_path.resolve()}")

    print(f"Reading {trans_path} ...")
    trans = pd.read_csv(trans_path, low_memory=False)
    if args.sample is not None:
        frac = max(0.0, min(1.0, args.sample))
        if frac > 0 and frac < 1:
            trans = trans.sample(frac=frac, random_state=42)
            print(f"Sampled transaction rows -> {len(trans):,}")

    print(f"Reading {iden_path} ...")
    iden  = pd.read_csv(iden_path, low_memory=False)

    print("Merging on TransactionID (left join) ...")
    df = trans.merge(iden, on="TransactionID", how="left")

    out_path = Path(args.out)
    df.to_csv(out_path, index=False)
    print(f"Saved merged dataset -> {out_path.resolve()}")
    print(f"Shape: {df.shape}")
    print("Head columns:", df.columns[:12].tolist())

if __name__ == "__main__":
    main()
