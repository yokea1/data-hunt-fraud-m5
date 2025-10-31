# eda_toolkit.py (robust + mode-aware, fixed)
# Run: python eda_toolkit.py
# Output: /reports/ (HTML + PNGs)

from __future__ import annotations
import io, warnings, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None
try:
    import missingno as msno
except Exception:
    msno = None

if not hasattr(np, "VisibleDeprecationWarning"):
    class _VisibleDeprecationWarning(UserWarning): ...
    np.VisibleDeprecationWarning = _VisibleDeprecationWarning

# ========= SWITCH HERE =========
DATA_MODE = "fraud"          # ← "fraud" 跑欺诈；"m5" 跑时间序列
CSV_PATH_OVERRIDE = None     # ← 想自定义路径就填 Path("xxx.csv")，否则留 None
# ===============================

FRAUD_DEFAULT_CSV   = Path("ieee_fraud_full_train.csv")  # merge_fraud.py 生成
FRAUD_FALLBACK_2COL = Path("ieee-fraud-train.csv")       # 仅两列（信息少）
M5_DEFAULT_CSV      = Path("m5_sample_long.csv")         # m5_wide_to_long.py 生成

MAX_ROWS_FOR_PLOTS = 5000
OUTDIR = Path("reports")

def resolve_dataset_and_target():
    if DATA_MODE.lower() == "fraud":
        target = "isFraud"
        csv_path = CSV_PATH_OVERRIDE or FRAUD_DEFAULT_CSV
        if not csv_path.exists():
            if FRAUD_FALLBACK_2COL.exists():
                print(f"⚠️ 未找到 {csv_path.name}，改用 {FRAUD_FALLBACK_2COL.name}（仅两列，图会较少）")
                csv_path = FRAUD_FALLBACK_2COL
            else:
                raise FileNotFoundError(
                    "未找到合并后的富特征文件。先运行：\n"
                    "  python merge_fraud.py    # 或 --sample 0.3"
                )
        return csv_path, target
    elif DATA_MODE.lower() == "m5":
        target = None
        csv_path = CSV_PATH_OVERRIDE or M5_DEFAULT_CSV
        if not csv_path.exists():
            msg = [f"未找到 {csv_path.name}。"]
            if Path("sales_train_validation.csv").exists() and Path("calendar.csv").exists():
                msg.append("检测到宽表与日历，请先运行：python m5_wide_to_long.py")
            else:
                msg.append("请先下载 M5 的 sales_train_validation.csv 与 calendar.csv；再运行：python m5_wide_to_long.py")
            raise FileNotFoundError("\n".join(msg))
        return csv_path, target
    else:
        raise ValueError("DATA_MODE 只能是 'fraud' 或 'm5'")

def read_csv_robust(p: Path) -> pd.DataFrame:
    if not p.exists():
        raise FileNotFoundError(p)
    try:
        df = pd.read_csv(p, low_memory=False)
        if df.shape[1] == 1 and "\r\n" in df.columns[0]:
            text = p.read_text(encoding="utf-8", errors="ignore").replace("\r\n","\n").replace("\r","\n")
            df = pd.read_csv(io.StringIO(text), low_memory=False)
        return df
    except Exception:
        text = p.read_text(encoding="utf-8", errors="ignore").replace("\r\n","\n").replace("\r","\n")
        return pd.read_csv(io.StringIO(text), low_memory=False)

def basic_checks(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    print("=== SHAPE ===", df.shape, file=buf)
    print("\n=== HEAD ===", file=buf); print(df.head(10), file=buf)
    print("\n=== INFO ===", file=buf); df.info(buf=buf)
    print("\n=== DESCRIBE (numeric) ===", file=buf)
    try:
        num_desc = df.describe(include=[np.number])
        if isinstance(num_desc, pd.DataFrame) and num_desc.shape[1]>0:
            print(num_desc, file=buf)
        else:
            print("(No numeric columns)", file=buf)
    except Exception as e:
        print(f"(Numeric describe skipped: {e})", file=buf)
    print("\n=== DESCRIBE (categorical) ===", file=buf)
    non_num = df.select_dtypes(include=['object','category','bool'])
    if non_num.shape[1]>0:
        try: print(non_num.describe(), file=buf)
        except Exception as e: print(f"(Non-numeric describe skipped: {e})", file=buf)
    else:
        print("(No object/category/bool columns to describe)", file=buf)
    print("\n=== NULL COUNTS ===", file=buf); print(df.isna().sum().sort_values(ascending=False), file=buf)
    return buf.getvalue()

def safe_sample(df, n):
    return df.sample(n=n, random_state=42) if len(df)>n else df

def plot_distributions(df, outdir: Path, target: str|None):
    outdir.mkdir(parents=True, exist_ok=True)
    dfp = safe_sample(df, MAX_ROWS_FOR_PLOTS)

    # Numeric distributions
    num_cols = dfp.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols[:20]:
        plt.figure()
        try:
            if sns is not None:
                sns.histplot(data=dfp, x=col, kde=True)
            else:
                plt.hist(dfp[col].dropna().values, bins=50)
        except Exception:
            plt.hist(dfp[col].dropna().values, bins=50)
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        plt.savefig(outdir/f"dist_{col}.png", dpi=120)
        plt.close()

    # Categorical counts
    cat_cols = dfp.select_dtypes(include=['object','category','bool']).columns.tolist()
    for col in cat_cols[:20]:
        vc = dfp[col].value_counts(dropna=False)
        if len(vc)>40:
            continue
        plt.figure(figsize=(6,4))
        if sns is not None:
            sns.countplot(data=dfp, x=col, order=vc.index.astype(str))
        else:
            plt.bar(vc.index.astype(str), vc.values)
        plt.title(f"Counts: {col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outdir/f"count_{col}.png", dpi=120)
        plt.close()

    # Correlation heatmap (numeric)
    if len(num_cols)>=2:
        plt.figure(figsize=(8,6))
        corr = dfp[num_cols].corr(numeric_only=True)
        try:
            if sns is not None:
                sns.heatmap(corr, cmap="viridis", center=0)
            else:
                plt.imshow(corr, aspect='auto')
                plt.colorbar()
        except Exception:
            plt.imshow(corr, aspect='auto')
            plt.colorbar()
        plt.title("Correlation heatmap (numeric)")
        plt.tight_layout()
        plt.savefig(outdir/"corr_heatmap.png", dpi=140)
        plt.close()

    # Pairplot (cap)
    few = [c for c in num_cols if c != (target or "")][:5]
    if len(few)>=2 and len(dfp)<=2000 and sns is not None:
        try:
            import seaborn as _sns
            _sns.pairplot(dfp[few], diag_kind="hist")
            plt.savefig(outdir/"pairplot.png", dpi=120)
            plt.close()
        except Exception:
            pass

    # Target-related
    if target and target in dfp.columns:
        vc = dfp[target].value_counts(dropna=False).sort_index()
        plt.figure()
        vc.plot(kind="bar")
        plt.title(f"Target distribution: {target}")
        plt.tight_layout()
        plt.savefig(outdir/f"target_distribution_{target}.png", dpi=120)
        plt.close()

        for col in num_cols[:6]:
            if col == target:
                continue
            plt.figure(figsize=(6,4))
            try:
                if sns is not None:
                    sns.boxplot(data=dfp, x=target, y=col)
                else:
                    uniq = sorted(dfp[target].dropna().unique())
                    data = [dfp[dfp[target]==v][col].dropna().values for v in uniq]
                    plt.hist(data, bins=40, label=[str(v) for v in uniq])
                    plt.legend()
                plt.title(f"{col} by {target}")
                plt.tight_layout()
                plt.savefig(outdir/f"{col}_by_{target}.png", dpi=120)
            except Exception:
                pass
            finally:
                plt.close()

def plot_missingness(df, outdir: Path):
    if msno is None:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    small = df.iloc[:, :min(200, df.shape[1])].copy()
    try:
        plt.figure(); msno.bar(small); plt.tight_layout(); plt.savefig(outdir/"missing_bar.png", dpi=140); plt.close()
    except Exception:
        pass
    try:
        plt.figure(); msno.matrix(small.sample(min(len(small), MAX_ROWS_FOR_PLOTS), random_state=42)); plt.tight_layout(); plt.savefig(outdir/"missing_matrix.png", dpi=140); plt.close()
    except Exception:
        pass
    try:
        plt.figure(); msno.heatmap(small); plt.tight_layout(); plt.savefig(outdir/"missing_heatmap.png", dpi=140); plt.close()
    except Exception:
        pass

def run_ydata_profile(df, outdir: Path, title: str):
    try:
        from ydata_profiling import ProfileReport
    except Exception as e:
        warnings.warn(f"YData-Profiling not available: {e}")
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    rep = ProfileReport(df, title=title, minimal=True)
    out = outdir/"ydata_profiling_report.html"
    rep.to_file(str(out))
    return out

def run_sweetviz(df, outdir: Path, target: str|None):
    try:
        import sweetviz as sv
    except Exception as e:
        warnings.warn(f"Sweetviz not available: {e}")
        return None
    outdir.mkdir(parents=True, exist_ok=True)
    html = outdir/f"sweetviz_report_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
    try:
        report = sv.analyze(
            df,
            target_feat=(target if target in df.columns else None),
            pairwise_analysis='off'   # 关闭成对变量分析，避免超慢与警告
        )
        report.show_html(str(html), open_browser=False)
        return html
    except Exception as e:
        warnings.warn(f"Sweetviz failed: {e}")
        return None

def maybe_time_series_quicklook(df, outdir: Path):
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    for col in date_cols[:1]:
        try:
            s = pd.to_datetime(df[col], errors="coerce")
            if s.notna().sum()>0:
                ts = pd.Series(1, index=s).resample("D").sum()
                plt.figure(figsize=(8,3))
                ts.plot()
                plt.title(f"Daily counts over time: {col}")
                plt.tight_layout()
                plt.savefig(outdir/f"time_series_daily_{col}.png", dpi=140)
                plt.close()
        except Exception:
            pass

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    csv_path, target = resolve_dataset_and_target()
    df = read_csv_robust(csv_path)
    print(f"✅ Loaded dataset: {csv_path.name}")
    print(f"   Rows: {len(df):,}, Columns: {list(df.columns[:12])}{' ...' if df.shape[1]>12 else ''}\n")
    (OUTDIR/"summary.txt").write_text(basic_checks(df), encoding="utf-8")
    print("📝 Wrote summary.txt")
    yp = run_ydata_profile(df, OUTDIR, f"Profile: {csv_path.name}")
    if yp: print(f"📘 YData-Profiling → {yp}")
    sv = run_sweetviz(df, OUTDIR, target)
    if sv: print(f"🍬 Sweetviz → {sv}")
    plot_dir = OUTDIR/"plots"; plot_distributions(df, plot_dir, target); print(f"📊 Basic plots → {plot_dir}")
    miss_dir = OUTDIR/"missing"; plot_missingness(df, miss_dir)
    if msno is not None: print(f"🕳️ missingno visuals → {miss_dir}")
    ts_dir = OUTDIR/"timeseries"; maybe_time_series_quicklook(df, ts_dir)
    print("\n✅ All done. Check 'reports' folder.")

if __name__ == "__main__":
    main()
