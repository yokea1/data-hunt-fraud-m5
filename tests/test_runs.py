import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def _run(script):
    p = subprocess.run([sys.executable, str(ROOT / script)], cwd=str(ROOT))
    assert p.returncode == 0, f"{script} exited with {p.returncode}"
def test_fraud_baseline_runs():
    _run("fraud_baseline.py")
def test_m5_baseline_runs():
    _run("m5_baseline.py")
