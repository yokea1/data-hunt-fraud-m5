# sweetviz_from_csv.py
# Reads sample_data.csv, generates an EDA report, and opens it automatically.

import numpy as np
import pandas as pd
import sweetviz as sv
import webbrowser
from pathlib import Path
import datetime as dt

# ---- Fix Sweetviz compatibility with NumPy 2.x ----
if not hasattr(np, "VisibleDeprecationWarning"):
    class _VisibleDeprecationWarning(UserWarning): pass
    np.VisibleDeprecationWarning = _VisibleDeprecationWarning
# ---------------------------------------------------

# 1️⃣ Load your dataset (CSV must be in the same folder)
csv_path = Path("sample_data.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"Dataset not found: {csv_path.resolve()}")

df = pd.read_csv(csv_path)
print(f"✅ Loaded dataset: {csv_path.name}")
print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")

# 2️⃣ (Optional) define target variable for supervised analysis
target = "Survived"   # comment this line if no target column

# 3️⃣ Generate Sweetviz report
report = sv.analyze(df, target_feat=target)

# 4️⃣ Save with timestamp and open automatically
timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
output_file = f"sweetviz_report_{timestamp}.html"
report.show_html(output_file)
print(f"✅ Report generated: {output_file}")

# 5️⃣ Auto-open in browser
webbrowser.open_new_tab(Path(output_file).resolve().as_uri())
