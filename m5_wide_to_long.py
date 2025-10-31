import argparse
import pandas as pd
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sales", default="sales_train_validation.csv")
    p.add_argument("--cal",   default="calendar.csv")
    p.add_argument("--item",  default=None)
    p.add_argument("--store", default=None)
    p.add_argument("--out",   default="m5_sample_long.csv")
    args = p.parse_args()

    sales = pd.read_csv(args.sales)
    cal   = pd.read_csv(args.cal)[["d","date"]]

    if args.item and args.store:
        row = sales[(sales["item_id"]==args.item) & (sales["store_id"]==args.store)].head(1).copy()
        if row.empty:
            raise ValueError("No matching item_id+store_id found. Try --item FOODS_3_090 --store CA_1")
    else:
        row = sales.iloc[0:1].copy()

    value_cols = [c for c in row.columns if c.startswith("d_")]
    long_df = row.melt(value_vars=value_cols, var_name="d", value_name="value")
    long_df = long_df.merge(cal, on="d", how="left").drop(columns=["d"])
    long_df = long_df.rename(columns={"date":"date"})
    long_df["date"] = pd.to_datetime(long_df["date"])

    out = Path(args.out)
    long_df.to_csv(out, index=False)
    print(f"Saved -> {out.resolve()}  shape={long_df.shape}")

if __name__ == "__main__":
    main()
