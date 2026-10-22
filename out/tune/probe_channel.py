"""通道择优的机会成本: 同一条轨迹强制用踝/髋/头各数一遍, 看被选中那条少数了多少。

如果大量轨迹在另一条通道上数得出更多起跳, 说明"选清晰度最高的通道"在快跳人群里选错了人:
头颈在快跳时几乎不动但很平滑, 自相关反而最漂亮, 于是慢摆冒充了节律。
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
from jumpcount.counter import count_track, participation, pick_source  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402

SOURCES = ("ankle", "hip", "head", "box_bottom")


def main() -> None:
    cfg = Config()
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
            chosen = pick_source(slot.samples, cfg, fps)
            totals = {}
            for src in SOURCES:
                one = Config()
                one.signal_source = src
                res = count_track(slot.track_id, slot.samples, one, fps, source=src)
                if res.stats.get("reject"):
                    totals[src] = 0
                    continue
                ok = participation(res.stats, one)[0]
                totals[src] = (res.count + res.stats.get("filled", 0)) if ok else 0
            rows.append((os.path.basename(video)[:12], slot.track_id, chosen, totals))

    total_chosen = max_alt = 0
    gain = 0.0
    worse = 0
    print(f"{'视频':<14}{'轨迹':>4} {'选中':<11}" + "".join(f"{s:>11}" for s in SOURCES) + "   差值")
    for video, tid, chosen, totals in rows:
        best = max(SOURCES, key=lambda s: totals.get(s, 0))
        delta = totals.get(best, 0) - totals.get(chosen, 0)
        total_chosen += totals.get(chosen, 0)
        max_alt += totals.get(best, 0)
        gain += max(0, delta)
        worse += 1 if delta > 0 else 0
        if delta > 0 and totals.get(best, 0) >= 5:
            print(f"{video:<14}{tid:>4} {chosen:<11}" + "".join(f"{totals.get(s, 0):>11}" for s in SOURCES)
                  + f"   +{delta}")
    print()
    print(f"选中通道合计 {total_chosen} 次; 每条各取最优合计 {max_alt} 次; "
          f"机会成本 {gain} 次({gain / max(1, total_chosen):.1%}), 受影响的轨迹 {worse} 条")


if __name__ == "__main__":
    main()
