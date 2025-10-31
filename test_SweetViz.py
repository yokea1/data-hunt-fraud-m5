# sweetviz_analysis.py

import pandas as pd
import sweetviz as sv

# 1️⃣ Load your dataset
# Example: Iris dataset from seaborn or local CSV
# Option A: Local CSV file
# df = pd.read_csv("your_dataset.csv")

# Option B: Use built-in dataset (for testing)
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.frame

# 2️⃣ Generate the Sweetviz report
report = sv.analyze(df)

# 3️⃣ Display the report in a browser
report.show_html('sweetviz_report.html')

print("✅ Sweetviz report generated! Check 'sweetviz_report.html' in your project folder.")
