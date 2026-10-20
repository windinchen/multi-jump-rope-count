"""补漏护栏松紧对照: 逐条配置整段复算, 看能捞回多少、有没有数过头。

只看离线链路(实时链路结论一致且慢一倍)。虚高用两条证据衡量:
完整率 > 1.05 的轨迹条数, 以及"数到"与"按节奏应有"的比值。
"""
from __future__ import annotations

import dataclasses
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from harness import CACHE, VIDEOS, replay  # noqa: E402
from jumpcount.config import Config  # noqa: E402

CONFIGS = [
    ("现默认", {}),
    ("关掉补漏", {"interpolate_gaps": False}),
    ("单空档补到 4", {"max_fill_per_gap": 4}),
    ("单空档补到 8", {"max_fill_per_gap": 8}),
    ("空档观测门槛 0.5", {"min_fill_coverage": 0.5, "max_fill_per_gap": 4}),
    ("节律门槛 0.25", {"keep_rhythm_strength": 0.25, "max_fill_per_gap": 4}),
    ("节律门槛 0.15", {"keep_rhythm_strength": 0.15, "max_fill_per_gap": 4}),
    ("不补但放宽判定", {"interpolate_gaps": False, "keep_rhythm_strength": 0.2}),
]


def main() -> None:
    caches = {}
    for video in VIDEOS:
        path = os.path.join(CACHE, os.path.basename(video) + ".pkl")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                caches[video] = pickle.load(fh)
    header = ["配置", "计入", "低置信", "合计", "应有", "完整率", "数过头条数", "剔除仍数到"]
    print(f"{header[0]:<18}{header[1]:>8}{header[2]:>8}{header[3]:>8}{header[4]:>9}{header[5]:>8}{header[6]:>11}{header[7]:>11}")
    for name, overrides in CONFIGS:
        cfg = Config()
        for key, value in overrides.items():
            setattr(cfg, key, value)
        kept = prov = lost = expected = 0.0
        over = 0
        for video, data in caches.items():
            rows, _actions, k, p, exp = replay(cfg, video, data)
            kept += k
            prov += p
            expected += exp
            lost += sum(r[2] for r in rows if r[1] == "out")
            for row in rows:
                if row[1] != "out" and row[2] > 1.05 * row[3]:
                    over += 1
        total = kept + prov
        ratio = total / max(1.0, expected)
        print(f"{name:<20}{kept:>7.0f}{prov:>7.0f}{total:>7.0f}{expected:>9.0f}{ratio:>8.3f}{over:>11}{lost:>11.0f}", flush=True)


if __name__ == "__main__":
    main()
