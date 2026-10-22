"""遮挡空档补出来的次数有没有旁证: 按"空档是遮挡造成还是一直在画面里"分开算他道作证率。

用法: .venv/bin/python out/tune/probe_phaseproof.py [key=value ...]
口径同 probe_fillproof: 对每条轨迹, 踝/髋/头/框底边各自取峰(各自的阈值),
看某类事件 ±0.35 周期内有没有别的通道的峰落进来。
分类: 事件前后 ±0.6 周期内的观测覆盖率 < min_fill_coverage = 遮挡造成的补数(走相位旁路),
     否则是"人在画面里但起跳幅度掉到阈值以下"的常规补数, 两者分开统计作证率。
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


def other_peaks(samples, cfg, fps, primary):
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
        out[name] = (int(ax[0]), set(int(ax[0]) + int(i) for i in sg.find_peaks(det, hard_min, thr, 0.8 * thr)))
    return out


def main() -> None:
    cfg = Config()
    for arg in sys.argv[1:]:
        k, _, v = arg.partition("=")
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true"))
    stat = {"实测起跳": [0, 0], "回捞峰": [0, 0], "常规补数": [0, 0], "遮挡补数": [0, 0]}
    tol_hit = [0, 0]
    for video in VIDEOS:
        path = os.path.join(CACHE, os.path.basename(video) + ".pkl")
        if not os.path.exists(path):
            continue
        data = pickle.load(open(path, "rb"))
        fps = data["fps"]
        tracker = Tracker(cfg)
        samples = {}
        for index in range(data["frames"]):
            for tid, det in tracker.update(index, _detections(data, index, cfg)):
                samples.setdefault(tid, []).append((index, index / fps, det.box, None, det.keypoints))
        slots = [build_slot(t, samples[t], [((d[2][0] + d[2][2]) / 2, (d[2][1] + d[2][3]) / 2)
                                            for d in samples[t]]) for t in sorted(samples) if len(samples[t]) >= 2]
        slots, _ = consolidate(slots, cfg)
        for slot in slots:
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            verdict, _ = participation(res.stats, cfg)
            if verdict == "out" or not res.rate_hz:
                continue
            period = fps / res.rate_hz
            opp = other_peaks(slot.samples, cfg, fps, res.signal_source)
            frames = np.array([int(s[0]) for s in slot.samples], dtype=np.int64)
            obs = np.zeros(int(frames[-1]) - int(frames[0]) + 1, dtype=bool)
            obs[frames - int(frames[0])] = True

            def cover(pos):
                lo = int(max(0, pos - 0.6 * period - frames[0]))
                hi = int(min(obs.size, pos + 0.6 * period - frames[0] + 1))
                return float(obs[lo:hi].mean()) if hi > lo else 0.0

            for ev in res.events:
                if ev.kind == "bump":
                    continue
                pos = ev.frame
                band = int(round(0.35 * period))
                hit = any(abs(pos - (p0 + p)) <= band for p0, pk in opp.values() for p in pk)
                if ev.kind == "interpolated":
                    key = "遮挡补数" if cover(pos) < cfg.min_fill_coverage else "常规补数"
                elif ev.recovered:
                    key = "回捞峰"
                else:
                    key = "实测起跳"
                stat[key][0] += 1
                stat[key][1] += int(hit)
                if key == "遮挡补数":
                    tol_hit[int(ev.confidence >= 1.0 - cfg.fill_hole_phase_tol)] += 1
    print(f"{'事件类别':<12}{'次数':>7}{'他道作证':>9}{'作证率':>8}")
    for k, (n, h) in stat.items():
        print(f"{k:<12}{n:>7}{h:>9}{100 * h / max(1, n):>7.1f}%")
    print(f"其中遮挡补数按相位残差分组: 残差<=tol(0.75以上置信) {tol_hit[1]} 次 / 更低 {tol_hit[0]} 次")


if __name__ == "__main__":
    main()
