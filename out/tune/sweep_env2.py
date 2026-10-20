"""加上"人必须在画面里"这道闸后的参数组合扫描。"""
from __future__ import annotations

import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [
    ("关掉回捞(基线)", ["envelope_detect=false"]),
    ("frac.10 raw.45 votes2", []),
    ("frac.07 raw.35 votes2", ["envelope_frac=0.07", "recover_raw_ratio=0.35"]),
    ("frac.05 raw.35 votes2", ["envelope_frac=0.05", "recover_raw_ratio=0.35"]),
    ("frac.07 raw.30 votes2", ["envelope_frac=0.07", "recover_raw_ratio=0.30"]),
    ("frac.05 raw.30 votes1", ["envelope_frac=0.05", "recover_raw_ratio=0.30", "envelope_lead_votes=1"]),
    ("frac.07 raw.35 votes3", ["envelope_frac=0.07", "recover_raw_ratio=0.35", "envelope_lead_votes=3"]),
    ("frac.07 raw.35 观测闸0.5", ["envelope_frac=0.07", "recover_raw_ratio=0.35", "min_fill_coverage=0.5"]),
]

for name, overrides in CONFIGS:
    cmd = [os.path.join("..", "..", ".venv", "bin", "python"), "quick.py"] + overrides
    out = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    body = out.stdout.strip().splitlines()
    keep = [line for line in body if line.startswith(("离线", "回捞", "节奏离散"))]
    print(f"### {name}", flush=True)
    print("\n".join(keep) or out.stderr[-400:], flush=True)
