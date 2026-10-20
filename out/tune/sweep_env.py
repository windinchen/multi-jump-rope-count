"""能量包络回捞的参数扫描: 每个配置打印总数 / 完整率 / 双计特征 / 回捞数。"""
from __future__ import annotations

import os
import pickle
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [
    ("关掉回捞(基线)", ["envelope_detect=false"]),
    ("默认 frac=0.10 votes=2 raw=0.45", []),
    ("frac=0.05", ["envelope_frac=0.05"]),
    ("frac=0.07", ["envelope_frac=0.07"]),
    ("frac=0.15", ["envelope_frac=0.15"]),
    ("frac=0.25", ["envelope_frac=0.25"]),
    ("votes=1(不投票)", ["envelope_lead_votes=1"]),
    ("votes=3", ["envelope_lead_votes=3"]),
    ("raw=0.30", ["recover_raw_ratio=0.30"]),
    ("raw=0.60", ["recover_raw_ratio=0.60"]),
    ("空档门槛1.3", ["search_back_ratio=1.3"]),
    ("投票容差0.5", ["envelope_lead_tol=0.5"]),
    ("投票容差0.2", ["envelope_lead_tol=0.2"]),
]

for name, overrides in CONFIGS:
    cmd = [os.path.join("..", "..", ".venv", "bin", "python"), "quick.py"] + overrides
    print(f"\n### {name}: {' '.join(overrides)}", flush=True)
    out = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    body = out.stdout.strip().splitlines()
    keep = [line for line in body if line.startswith(("离线", "回捞"))]
    print("\n".join(keep) or out.stderr[-400:], flush=True)
