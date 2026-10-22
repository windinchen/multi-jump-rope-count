"""变速假设: 一条轨迹内节奏是否在漂移? 若漂移明显, 单一全局周期会让补齐/回捞在快的那半失效。

用法: .venv/bin/python out/tune/probe_pace.py [key=value ...]
判据:
  lag-1 自相关 >0 = 相邻间隔成段变长/变短 = 真变速(不是随机抖动)
  pace = 后 1/3 间隔中位数 / 前 1/3 间隔中位数, <0.77 或 >1.3 = 前后节奏差 >=30%
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from harness import CACHE, VIDEOS, _detections  # noqa: E402
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import count_track, participation  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def ac1(x):
    x = np.asarray(x, float)
    if x.size < 8:
        return 0.0
    x = x - x.mean()
    d = float((x * x).sum())
    return float((x[:-1] * x[1:]).sum() / d) if d > 1e-9 else 0.0


def main() -> None:
    cfg = Config()
    for item in sys.argv[1:]:
        key, value = item.split("=", 1)
        low = value.lower()
        cast = low == "true" if low in ("true", "false") else None
        if cast is None:
            try:
                cast = int(value)
            except ValueError:
                cast = float(value)
        setattr(cfg, key, cast)
    rows = []
    for video in VIDEOS:
        path = os.path.join(CACHE, os.path.basename(video) + ".pkl")
        if not os.path.exists(path):
            continue
        data = pickle.load(open(path, "rb"))
        fps = data["fps"]
        tracker = Tracker(cfg)
        samples, centers = {}, {}
        for index in range(data["frames"]):
            for tid, det in tracker.update(index, _detections(data, index, cfg)):
                samples.setdefault(tid, []).append((index, index / fps, det.box, None, det.keypoints))
                centers.setdefault(tid, []).append(((det.box[0] + det.box[2]) / 2,
                                                    (det.box[1] + det.box[3]) / 2))
        slots = [build_slot(t, samples[t], centers[t]) for t in sorted(samples) if len(samples[t]) >= 2]
        slots, _ = consolidate(slots, cfg)
        for slot in slots:
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            verdict = participation(res.stats, cfg)[0]
            if verdict == "out" or not res.period_sec:
                continue
            beats = [e.frame for e in res.events if e.kind in ("jump", "bump")]
            if len(beats) < 14:
                continue
            iv = np.diff(beats).astype(float)
            third = max(3, len(iv) // 3)
            early, late = float(np.median(iv[:third])), float(np.median(iv[-third:]))
            rows.append((os.path.basename(video)[:12], slot.track_id, verdict,
                         len(beats), res.count + res.stats.get("filled", 0),
                         res.period_sec * fps, ac1(iv), late / max(1e-6, early),
                         res.stats.get("filled", 0), res.stats.get("recovered", 0)))
    a1 = np.array([r[6] for r in rows])
    pace = np.array([r[7] for r in rows])
    print(f"条数 {len(rows)}")
    print(f"lag-1 自相关: 均值 {a1.mean():+.3f}  中位 {np.median(a1):+.3f}  >0.25 的条数 {(a1>0.25).sum()}"
          f"  <-0.25 的 {(a1<-0.25).sum()}")
    print(f"前后节奏比: 中位 {np.median(pace):.2f}  差>=30% 的条数 {((pace<0.77)|(pace>1.3)).sum()}"
          f"  差>=50% 的 {((pace<0.67)|(pace>1.5)).sum()}")
    print("\n漂移最明显的 12 条(|log pace| 最大):")
    for r in sorted(rows, key=lambda r: -abs(np.log(max(1e-6, r[7]))))[:12]:
        print(f"  {r[0]} #{r[1]:<4} {r[2]:<12} 峰 {r[3]:<4} 计数 {r[4]:<4} 周期 {r[5]:.1f}f "
              f"lag1 {r[6]:+.2f} 前后比 {r[7]:.2f} 补 {r[8]} 捞 {r[9]}")


if __name__ == "__main__":
    main()
