# sweetviz_analysis_advanced.py
# Examples:
#   python sweetviz_analysis_advanced.py --iris --open
#   python sweetviz_analysis_advanced.py --csv train.csv --target Survived --open
#   python sweetviz_analysis_advanced.py --csv train.csv --test test.csv --target Survived --sample 10000 --outdir reports --open

from __future__ import annotations
import argparse, sys, webbrowser, datetime as dt
from pathlib import Path

# ---- NumPy 2.x compatibility shim for Sweetviz ----
import numpy as _np
if not hasattr(_np, "VisibleDeprecationWarning"):
    class _VisibleDeprecationWarning(UserWarning): pass
    _np.VisibleDeprecationWarning = _VisibleDeprecationWarning
# ---------------------------------------------------

import pandas as pd
import sweetviz as sv


def load_data_from_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")
    try:
        # Low-memory off avoids dtype guess issues on wide CSVs
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV '{path}': {e}") from e


def load_iris() -> pd.DataFrame:
    try:
        from sklearn.datasets import load_iris
    except Exception as e:
        raise ImportError("scikit-learn is required for --iris. Install with: pip install scikit-learn") from e
    iris = load_iris(as_frame=True)
    return iris.frame


def maybe_sample(df: pd.DataFrame, n: int | None) -> pd.DataFrame:
    if n and len(df) > n:
        return df.sample(n=n, random_state=42)
    return df


def make_timestamped_name(base: str, suffix: str = "html") -> str:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{base}_{ts}.{suffix}"


def generate_report(
    df: pd.DataFrame,
    target: str | None,
    test_df: pd.DataFrame | None,
    outdir: Path,
    base_name: str
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    if test_df is not None:
        report = sv.compare([df, "Train"], [test_df, "Test"], target_feat=target)
        outfile = outdir / make_timestamped_name(base_name + "_compare")
    else:
        report = sv.analyze(source=df, target_feat=target)
        outfile = outdir / make_timestamped_name(base_name)

    report.show_html(str(outfile))
    return outfile


def main():
    ap = argparse.ArgumentParser(description="Advanced Sweetviz runner (single analysis or train/test comparison).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--iris", action="store_true", help="Use the Iris dataset (requires scikit-learn).")
    src.add_argument("--csv", type=str, help="Path to source CSV file.")

    ap.add_argument("--test", type=str, help="Optional path to TEST CSV for comparison (same schema recommended).")
    ap.add_argument("--target", type=str, help="Optional target column for supervised insights.")
    ap.add_argument("--sample", type=int, help="Optional row sample cap (e.g., 10000).")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory for HTML reports.")
    ap.add_argument("--name", type=str, default="sweetviz_report", help="Base filename (timestamp will be appended).")
    ap.add_argument("--open", dest="auto_open", action="store_true", help="Open the report in a browser when done.")
    args = ap.parse_args()

    try:
        # Load source data
        if args.iris:
            df = load_iris()
        else:
            df = load_data_from_csv(Path(args.csv))

        df = maybe_sample(df, args.sample)

        # Load optional test data
        test_df = None
        if args.test:
            test_df = load_data_from_csv(Path(args.test))
            test_df = maybe_sample(test_df, args.sample)

        # Basic sanity checks
        if args.target and args.target not in df.columns:
            raise ValueError(f"--target column '{args.target}' not found in source data.")
        if test_df is not None and args.target and args.target not in test_df.columns:
            # Not strictly required for compare(), but helpful if you expect the target in both
            print(f"⚠️  Note: target '{args.target}' not found in TEST data. Proceeding without test target split.", file=sys.stderr)

        # Generate report
        outfile = generate_report(
            df=df,
            target=args.target,
            test_df=test_df,
            outdir=Path(args.outdir),
            base_name=args.name
        )

        # Console summary
        print(f"✅ Sweetviz report generated: {outfile}")
        print(f"   Source rows: {len(df):,} | columns: {df.shape[1]}")
        if args.target:
            print(f"   Target: {args.target}")
        if test_df is not None:
            print(f"   Test rows: {len(test_df):,} | columns: {test_df.shape[1]}")

        if args.auto_open:
            webbrowser.open_new_tab(Path(outfile).resolve().as_uri())

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
