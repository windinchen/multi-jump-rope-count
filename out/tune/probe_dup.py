"""同一画面里的轨迹是否重叠: 被 height_cv 剔除的轨迹, 到底是"另一个没数到的人"还是"已经数到的人的重复分身"。

用法: .venv/bin/python out/tune/probe_dup.py [视频关键字] [key=value ...]
判法: 两条轨迹时间重叠多 + 横向中心接近 + 框高接近 => 同一个人被拆成两条(重复计数风险);
      时间重叠多但横向明显分开 => 两个不同的人(被误杀)。
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from harness import CACHE, VIDEOS, _detections  # noqa: E402
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import count_track, participation  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if "=" not in a]
    keyword = args[0] if args else "1790234843854"
    cfg = Config()
    for item in sys.argv[1:]:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        low = value.lower()
        cast = low == "true" if low in ("true", "false") else None
        if cast is None:
            try:
                cast = int(value)
            except ValueError:
                try:
                    cast = float(value)
                except ValueError:
                    cast = value
        setattr(cfg, key, cast)

    started = time.time()
    for video in VIDEOS:
        if keyword not in video:
            continue
        path = os.path.join(CACHE, os.path.basename(video) + ".pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            data = pickle.load(fh)
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

        infos = []
        for slot in slots:
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            verdict, reason = participation(res.stats, cfg)
            total = res.count + res.stats.get("filled", 0)
            if total < 3:
                continue
            fr = np.array([int(s[0]) for s in slot.samples], dtype=float)
            bx = np.array([s[2] for s in slot.samples], dtype=float)
            infos.append({
                "tid": slot.track_id, "verdict": verdict, "total": total,
                "f0": fr[0] / fps, "f1": fr[-1] / fps,
                "x": float(np.median(bx[:, 0] + bx[:, 2]) / 2), "xr": (float(np.percentile(bx[:, 0] + bx[:, 2], 5) / 2),
                                                                       float(np.percentile(bx[:, 0] + bx[:, 2], 95) / 2)),
                "h": float(np.median(bx[:, 3] - bx[:, 1])),
                "cv": res.stats.get("height_cv"), "period": res.period_sec or 0.0,
                "reason": reason,
            })
        infos.sort(key=lambda r: -r["total"])
        w = float(data.get("width") or 1280)
        print(f"\n=== {os.path.basename(video)}  画面宽 {w:.0f}  时长 {data['frames'] / fps:.1f}s  "
              f"数到 >=3 次的轨迹 {len(infos)} 条, 合计 {sum(r['total'] for r in infos)} 次, 用时 {time.time() - started:.1f}s")
        print(f"{'轨迹':>5}{'判定':>5}{'次数':>5}{'时长区间':>16}{'横向中位':>10}{'横向p5-p95':>18}{'框高中位':>10}{'cv':>6}{'周期':>7}  剔除理由")
        for r in infos:
            span = f"{r['f0']:.0f}-{r['f1']:.0f}s"
            xband = f"{r['xr'][0]:.0f}-{r['xr'][1]:.0f}"
            print(f"#{r['tid']:<4}{r['verdict'][:4]:>5}{r['total']:>5}{span:>16}{r['x']:>10.0f}{xband:>18}"
                  f"{r['h']:>10.0f}{(r['cv'] or 0):>6.2f}{r['period']:>7.2f}  {r['reason'][:40]}")

        print("\n两两重叠检查(时间重叠 >50% 且横向中位差 < 0.5 倍框高): 疑似同一人被拆成两条")
        flagged = 0
        for i in range(len(infos)):
            for j in range(i + 1, len(infos)):
                a, b = infos[i], infos[j]
                ov = min(a["f1"], b["f1"]) - max(a["f0"], b["f0"])
                base = min(a["f1"] - a["f0"], b["f1"] - b["f0"])
                if base <= 0 or ov / base < 0.5:
                    continue
                dx = abs(a["x"] - b["x"])
                if dx > 0.5 * min(a["h"], b["h"]):
                    continue
                flagged += 1
                print(f"  #{a['tid']}({a['verdict']},{a['total']}次) <-> #{b['tid']}({b['verdict']},{b['total']}次) "
                      f"时间重叠 {ov / base * 100:.0f}%, 横向差 {dx:.0f}px (框高 {min(a['h'], b['h']):.0f}px)")
        if not flagged:
            print("  无")


if __name__ == "__main__":
    main()
