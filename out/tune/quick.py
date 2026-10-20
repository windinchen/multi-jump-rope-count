"""改动前后的快速对照: 离线总数 + 跟踪/计数耗时 + 双计特征 + 虚高条数。

用法: .venv/bin/python out/tune/quick.py [key=value ...]
双计特征 = 相邻起跳间隔 < 0.7 倍周期的比例(一次起跳被数成两个的直接证据)。
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
from jumpcount.online import OnlineCounter  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


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
                try:
                    cast = float(value)
                except ValueError:
                    cast = value
        setattr(cfg, key, cast)

    caches = {}
    for video in VIDEOS:
        path = os.path.join(CACHE, os.path.basename(video) + ".pkl")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                caches[video] = pickle.load(fh)

    kept = prov = lost = expected = 0.0
    recovered = 0
    over = 0
    cvs: list = []
    tight = double_check = rec_tight = rec_check = 0
    track_ms = count_ms = 0.0
    online_total = online_people = 0
    online_frame_ms = []
    for video, data in caches.items():
        fps = data["fps"]
        t0 = time.perf_counter()
        tracker = Tracker(cfg)
        samples, centers = {}, {}
        per_frame = []
        for index in range(data["frames"]):
            a = time.perf_counter()
            pairs = tracker.update(index, _detections(data, index, cfg))
            per_frame.append((time.perf_counter() - a) * 1000.0)
            for tid, det in pairs:
                samples.setdefault(tid, []).append((index, index / fps, det.box, None, det.keypoints))
                centers.setdefault(tid, []).append(((det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2))
        track_ms += float(np.median(per_frame))
        b = time.perf_counter()
        slots = [build_slot(t, samples[t], centers[t]) for t in sorted(samples) if len(samples[t]) >= 2]
        slots, _actions = consolidate(slots, cfg)
        for slot in slots:
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            verdict, _reason = participation(res.stats, cfg)
            total = res.count + res.stats.get("filled", 0)
            recovered += int(res.stats.get("recovered", 0))
            exp = res.stats.get("expected_count", 0.0)
            if verdict == "in":
                kept += total
                expected += exp
            elif verdict == "provisional":
                prov += total
                expected += exp
            else:
                lost += total
            if verdict != "out" and exp and total > 1.05 * exp:
                over += 1
            intervals = np.diff([e.frame for e in res.events if e.kind in ("jump", "bump")])
            if intervals.size >= 4:
                cvs.append(float(np.std(intervals) / max(1e-6, np.mean(intervals))))
            jumps = [e.frame for e in res.events if e.kind in ("jump", "bump")]
            marks = [bool(getattr(e, "recovered", False)) for e in res.events if e.kind in ("jump", "bump")]
            if res.period_sec:
                period = res.period_sec * fps
                for x, y in zip(jumps, jumps[1:]):
                    double_check += 1
                    if y - x < 0.7 * period:
                        tight += 1
                for i, flag in enumerate(marks):
                    if not flag:
                        continue
                    rec_check += 1
                    near = [j for j in (i - 1, i + 1) if 0 <= j < len(marks)
                            and abs(jumps[j] - jumps[i]) < 0.7 * period]
                    if near:
                        rec_tight += 1
        count_ms += (time.perf_counter() - b) * 1000.0 / data["frames"]

        # 实时链路
        tracker = Tracker(cfg)
        counter = OnlineCounter(cfg, fps)
        online_ms = []
        for index in range(data["frames"]):
            c0 = time.perf_counter()
            dets = _detections(data, index, cfg)
            counter.update(index, index / fps, [(t, d.box, d.keypoints, d.conf) for t, d in tracker.update(index, dets)])
            online_ms.append((time.perf_counter() - c0) * 1000.0)
        snaps = counter.snapshot(data["frames"])
        online_total += sum(s.get("total_count", s["count"]) for s in snaps if s["participant"])
        online_people += sum(1 for s in snaps if s["participant"])
        online_frame_ms = online_frame_ms + [float(np.median(online_ms))]

    total = kept + prov
    print(f"离线 计入 {kept:.0f} + 低置信 {prov:.0f} = {total:.0f} / 应有 {expected:.0f} -> 完整率 {total / max(1, expected):.3f}")
    print(f"实时链路 {online_total} 次, 在跳绳 {online_people} 人;  剔除但仍数到 {lost:.0f}, 虚高(>1.05) {over} 条")
    print(f"回捞 {recovered} 次; 双计特征: 间隔 < 0.7 周期的相邻峰 {tight}/{double_check} = {tight / max(1, double_check):.2%}"
          f"; 回捞峰自身双计 {rec_tight}/{rec_check} = {rec_tight / max(1, rec_check):.2%}")
    print(f"节奏离散度 CV(越低越稳, 均值): {float(np.mean(cvs)) if cvs else 0:.3f}")
    print(f"耗时: 跟踪 {track_ms:.1f} ms/帧(中位), 离线计数 {count_ms:.1f} ms/帧, "
          f"实时链路 {float(np.mean(online_frame_ms)):.1f} ms/帧(含检测输入)")


if __name__ == "__main__":
    main()
