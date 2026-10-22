"""低抬脚轨迹台账: 站着没跳的人为什么被"低置信计入", 用哪一列能把真人和站着的人分开。

用法: .venv/bin/python out/tune/probe_lowlift.py [最低次数]
已肉眼确认(1790234843854 审核表): 计数最少的十几条轨迹几乎都是站着/走动的观众,
它们的共同点是 amplitude_threshold 塌到 1~2px(等于噪声), 节奏像"清楚"其实是亚像素抖动。
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
    min_total = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cfg = Config()
    started = time.time()
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
            st = res.stats
            total = res.count + st.get("filled", 0)
            if total < min_total:
                continue
            verdict, reason = participation(st, cfg)
            heights = np.array([s[2][3] - s[2][1] for s in slot.samples], dtype=float)
            rows.append({
                "video": os.path.basename(video)[:12], "tid": slot.track_id, "verdict": verdict,
                "total": total, "lift": max(float(st.get("lift_ratio", 0) or 0), float(st.get("lift_ratio_best", 0) or 0)),
                "lift_main": float(st.get("lift_ratio", 0) or 0), "snr": st.get("snr") or res.snr,
                "noise": st.get("noise", 0), "thr": st.get("amplitude_threshold", 0),
                "p95": st.get("peak_level_p95", 0), "rhythm": st.get("rhythm_strength", 0),
                "psrc": st.get("period_source", ""), "rate": res.rate_hz, "cv": st.get("height_cv", 0),
                "h": float(np.median(heights)), "dur": st.get("duration_sec", 0),
                "vis": st.get("visibility", 0), "src": res.signal_source, "reason": reason,
            })
    rows.sort(key=lambda r: r["lift"])
    print(f"用时 {time.time() - started:.1f}s, 共 {len(rows)} 条数到>={min_total}次的轨迹, 按抬脚比升序")
    print(f"{'视频':<13}{'轨迹':>4}{'判定':>5}{'次数':>5}{'抬脚比':>8}{'信噪':>7}{'噪声':>7}{'阈值':>7}{'p95':>7}"
          f"{'节奏':>7}{'周期源':>10}{'次/秒':>7}{'框高':>6}{'时长':>7}{'可见':>6}  理由")
    for r in rows[:60]:
        print(f"{r['video']:<13}#{r['tid']:<3}{r['verdict'][:4]:>5}{r['total']:>5}{r['lift']:>8.4f}{float(r['snr']):>7.1f}"
              f"{r['noise']:>7.2f}{r['thr']:>7.2f}{r['p95']:>7.2f}{r['rhythm']:>7.2f}{r['psrc'][:9]:>10}"
              f"{r['rate']:>7.2f}{r['h']:>6.0f}{r['dur']:>7.1f}{r['vis']:>6.2f}  {r['reason'][:24]}")
    print("\n分位数对照(全部轨迹):")
    for key in ("lift", "snr", "thr", "p95", "rhythm"):
        arr = np.array([float(r[key]) for r in rows])
        print(f"  {key:<6} p10={np.percentile(arr, 10):.3f} p25={np.percentile(arr, 25):.3f} "
              f"中位={np.median(arr):.3f} p75={np.percentile(arr, 75):.3f}")
    for cut_lift in (0.010, 0.015, 0.020):
        for cut_snr in (3.0, 4.0, 5.0, 6.0):
            hit = [r for r in rows if r["lift"] < cut_lift and float(r["snr"]) < cut_snr and r["verdict"] != "out"]
            print(f"  若加护栏「抬脚比<{cut_lift:.3f} 且 信噪<{cut_snr:.0f}」: 影响 {len(hit)} 条 / "
                  f"{sum(r['total'] for r in hit)} 次 (占计入 {sum(r['total'] for r in rows if r['verdict'] != 'out')} 的 "
                  f"{sum(r['total'] for r in hit) / max(1, sum(r['total'] for r in rows if r['verdict'] != 'out')) * 100:.1f}%)")


if __name__ == "__main__":
    main()
