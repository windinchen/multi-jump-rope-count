"""量化"倍频(2:1 半拍)锁相"还剩多少: 同一条信号, 分别用现行最小间隔和不设限的最小间隔取峰。

用法: .venv/bin/python out/tune/probe_octave.py [关键字]
    strict = min_distance = refractory_ratio × 估计周期 (现行做法)
    free   = min_distance = hard_min (只受 max_rate_hz 约束)
判读: free 明显多于 strict 且粗取峰间隔中位数 ≈ 估计周期的一半 → 周期被锁成了两倍, 真起跳被最小间隔吞掉。
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
from jumpcount import signal as sg  # noqa: E402
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import _choose_period, amplitude_threshold, channel_cache, pick_source  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402
from probe_period import detrended_series  # noqa: E402


def main() -> None:
    kw = sys.argv[1] if len(sys.argv) > 1 else ""
    cfg = Config()
    rows = []
    for video in VIDEOS:
        if kw and kw not in video:
            continue
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
            channels = channel_cache(slot.samples, cfg)
            source = pick_source(slot.samples, cfg, fps, channels)
            axis, det, observed, median_h = detrended_series(slot, cfg, fps, source)
            if det.size < 30:
                continue
            noise = sg.hf_sigma_masked(det, observed)
            peak_level = float(np.percentile(det, 95))
            amp_thr = amplitude_threshold(noise, peak_level, median_h, cfg)
            hard_min = max(2, int(fps / cfg.max_rate_hz))
            coarse = sg.find_peaks(det, hard_min, amp_thr, 0.8 * amp_thr)
            ac_period, strength = sg.autocorr_period(det, fps, cfg.min_rate_hz, cfg.max_rate_hz)
            period, src = _choose_period(coarse, ac_period, fps, cfg)
            min_distance = max(hard_min, int(round(cfg.refractory_ratio * period)))
            strict = sg.find_peaks(det, min_distance, amp_thr, 0.8 * amp_thr)
            if len(coarse) < 3:
                continue
            spacing = float(np.median(np.diff(coarse)))
            rows.append({
                "video": os.path.basename(video)[:12], "tid": slot.track_id,
                "strict": len(strict), "free": len(coarse), "gain": len(coarse) - len(strict),
                "period": round(period, 2), "src": src, "spacing": round(spacing, 2),
                "ac": round(ac_period * fps, 2) if ac_period else -1,
                "ratio": round(period / max(1e-6, ac_period * fps), 2) if ac_period else -1,
                "min_dist": min_distance, "rhy": round(strength, 3),
                "rate_free": round(fps / max(1e-6, spacing), 2),
                "rate_strict": round(fps / max(1e-6, period), 2),
            })
    rows.sort(key=lambda r: -r["gain"])
    print(f"{'视频':<14}{'轨迹':>5}{'现行':>6}{'不设限':>7}{'可多':>6}{'周期帧':>8}{'粗间隔':>8}{'自相关':>8}"
          f"{'周期/自相关':>12}{'最小间隔':>9}{'节律':>7}{'来源':>14}")
    for r in rows[:25]:
        print(f"{r['video']:<14}{r['tid']:>5}{r['strict']:>6}{r['free']:>7}{r['gain']:>6}{r['period']:>8.2f}"
              f"{r['spacing']:>8.2f}{r['ac']:>8.2f}{r['ratio']:>12.2f}{r['min_dist']:>9}{r['rhy']:>7.2f}{r['src']:>14}")
    total_strict = sum(r["strict"] for r in rows)
    total_free = sum(r["free"] for r in rows)
    print(f"\n轨迹 {len(rows)} 条: 现行共 {total_strict} 个峰, 不设限共 {total_free} 个 (+{total_free - total_strict}, "
          f"{100 * (total_free - total_strict) / max(1, total_strict):.1f}%)")
    octaves = [r for r in rows if r["gain"] >= 5 and 1.6 <= r["ratio"] <= 2.5]
    print(f"疑似倍频锁相(可多≥5 且 周期/自相关 ∈[1.6,2.5]): {len(octaves)} 条, 可多 {sum(r['gain'] for r in octaves)} 个峰")
    half = [r for r in rows if r["gain"] >= 5 and abs(r["spacing"] - r["period"] / 2) <= 1.0]
    print(f"粗取峰间隔≈周期一半: {len(half)} 条, 可多 {sum(r['gain'] for r in half)} 个峰")
    print("周期来源分布:", {s: sum(1 for r in rows if r["src"] == s) for s in {r["src"] for r in rows}})
    print("现行频率估计 vs 不设限频率(前 12 条):")
    for r in rows[:12]:
        print(f"  {r['video']}#{r['tid']:<4} 现行 {r['rate_strict']:.2f}Hz → 不设限 {r['rate_free']:.2f}Hz")


if __name__ == "__main__":
    main()
