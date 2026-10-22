"""补出来的次数有没有旁证: 逐条比对"实测起跳"与"按节奏补的数"在别的身体通道上能不能对上。

用法: .venv/bin/python out/tune/probe_fillproof.py [key=value ...]
口径:
  对每条轨迹, 用踝/髋/头/框底边里除主通道以外的通道各自取峰(自己的阈值),
  再看某一类事件附近 ±0.35 周期内有没有别的通道的峰 —— 这就是"他道作证率"。
  若补数的作证率明显低于实测起跳, 说明补出来的数里有一部分是猜的。
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
from jumpcount.counter import (amplitude_threshold, channel_cache, count_track, participation,
                               pick_source)  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402

CH = ("ankle", "hip", "head", "box_bottom")


def other_peaks(samples, cfg, fps, primary, axis0, size):
    """每条非主通道自己的峰(帧轴)。与 channel_quality 同一套阈值口径。"""
    ch = channel_cache(samples, cfg)
    frames = np.array([int(s[0]) for s in samples], dtype=np.int64)
    heights = np.array([float(s[2][3]) - float(s[2][1]) for s in samples], dtype=np.float64)
    median_h = float(np.median(heights)) if heights.size else 0.0
    out = {}
    for name in CH:
        if name == primary or name not in ch:
            continue
        values = -np.where(np.isfinite(ch[name]), ch[name], np.nan)
        keep = np.isfinite(values)
        if keep.sum() < 20:
            continue
        ax, dense, _ = sg.fill_gaps(frames[keep], values[keep], cfg.max_gap)
        observed = np.zeros(ax.size, dtype=bool)
        observed[frames[keep] - int(ax[0])] = True
        win = cfg.baseline_win if dense.size >= cfg.baseline_win else max(5, (dense.size // 2) | 1)
        det = sg.detrend(dense, cfg.smooth_win, win)
        noise = sg.hf_sigma_masked(det, observed)
        p95 = float(np.percentile(det, 95)) if det.size else 0.0
        thr = amplitude_threshold(noise, p95, median_h, cfg)
        hard_min = max(2, int(fps / cfg.max_rate_hz))
        idx = sg.find_peaks(det, hard_min, thr, 0.8 * thr)
        out[name] = [int(ax[0]) + int(i) for i in idx]
    return out


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

    cls = {"jump": [0, 0], "bump": [0, 0], "interpolated": [0, 0]}
    per_track = []
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
            fills = [e.frame for e in res.events if e.kind == "interpolated"]
            if not fills:
                continue
            period = res.period_sec * fps
            peaks = other_peaks(slot.samples, cfg, fps, res.signal_source, 0, 0)
            if not peaks:
                continue
            tol = 0.35 * period

            def rate(kind_frames):
                hit = sum(1 for f in kind_frames
                          if any(abs(f - p) <= tol for lst in peaks.values() for p in lst))
                return hit, len(kind_frames)

            for kind in ("jump", "bump", "interpolated"):
                frames_k = [e.frame for e in res.events if e.kind == kind]
                h, n = rate(frames_k)
                cls[kind][0] += h
                cls[kind][1] += n
            h, n = rate(fills)
            per_track.append((video[:12], slot.track_id, n, h, h / max(1, n),
                              res.stats.get("fill_capped", 0)))
    print("他道作证率(事件附近 ±0.35 周期内有其它通道的峰):")
    for kind, label in (("jump", "实测起跳"), ("bump", "回捞鼓包"), ("interpolated", "按节奏补")):
        h, n = cls[kind]
        print(f"  {label:<10} {h}/{n} = {h / max(1, n):.1%}")
    per_track.sort(key=lambda r: r[4])
    print("\n补数作证率最低的 10 条:")
    for r in per_track[:10]:
        print(f"  {r[0]} #{r[1]:<5} 补 {r[2]:<3} 作证 {r[3]:<3} {r[4]:.0%} 被护栏砍掉 {r[5]}")


if __name__ == "__main__":
    main()
