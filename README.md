# data-hunt-fraud-m5
[![CI](https://github.com/yokea1/data-hunt-fraud-m5/actions/workflows/ci.yml/badge.svg)](https://github.com/yokea1/data-hunt-fraud-m5/actions)

**Cost-Sensitive, Time-Split Benchmark for IEEE-CIS Fraud & M5 (with Calibration & Threshold Sweep)**  
**面向 IEEE-CIS 反欺诈 & M5 预测的「代价敏感 + 时间切分」评测基准（含校准与阈值扫描）**

> Turn AUC/PR into **Expected Cost** under **temporal drift**, with reproducible scripts & figures.  
> 在**时序漂移**下，把传统 AUC/PR **转化为业务“期望成本”**，并提供**可复现**脚本与图表。

## 🔍 Why this repo / 为什么要做
- 业务对齐（Expected Cost / 阈值-成本曲线） · 时序鲁棒（time-split & rolling） · 置信校准（ECE±CI） · 可复现（脚本+reports）

## ✨ Highlights / 亮点
- Threshold sweep（best-F1 vs min-Expected-Cost） · Cost-aware（自定义 FP:FN） · Artifacts（metrics.json/figs） · SLA hooks

## 🗂 Repo Structure / 目录
data-hunt-fraud-m5/
├─ fraud_baseline.py
├─ m5_baseline.py
├─ eda_toolkit.py
├─ data/ # ✅ 数据放这里（不进 Git）
│ ├─ ieee/
│ └─ m5/
├─ reports/
│ └─ figs/ # 结果图
├─ tests/
│ └─ test_runs.py # 最小CI测试
├─ requirements.txt
├─ .gitignore
└─ .github/workflows/ci.yml

bash
复制代码

## ⚙️ Setup & Quickstart / 安装与快速开始
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python fraud_baseline.py
python m5_baseline.py
📊 Results (sample) / 结果示例


📥 Data Placement / 数据放置
IEEE-CIS → data/ieee/；M5 → data/m5/（不入库，遵守许可）

🛡 SLA & Ops (optional) / SLA 与运维（可选）
P95/P99 · QPS · 错误率 · 成本/请求；canary→监控→自动回滚

📎 Citation / 引用
Star & cite tag（e.g., v1.0.0）

📜 License
MIT
