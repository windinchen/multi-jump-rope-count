"""绕开所有阈值上限, 直接问信号: 这条轨迹的真实节奏周期到底是几帧。

用法: .venv/bin/python out/tune/probe_period.py <视频关键字> <轨迹ID> [轨迹ID...]
输出: 原始自相关曲线(滞后 3~30 帧) + 峰间隔直方图 + "计数 vs 最小间隔"曲线。
用途: 判断估计周期是真锁相(减半)还是真跳这么快; 只有曲率明显的第一峰才算真周期。
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
from jumpcount.counter import amplitude_threshold, channel_cache, count_track, extract_y_detail  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def detrended_series(slot, cfg, fps, source):
    frames = np.array([int(s[0]) for s in slot.samples], dtype=np.int64)
    heights = np.array([float(s[2][3] - s[2][1]) for s in slot.samples])
    channels = channel_cache(slot.samples, cfg)
    y, _ = extract_y_detail(slot.samples, source, cfg, channels=channels)
    raw = -np.where(np.isfinite(y), y, np.nan)
    keep = np.isfinite(raw)
    obs = frames[keep]
    axis, dense, _ = sg.fill_gaps(obs, raw[keep], cfg.max_gap)
    observed = np.zeros(axis.size, dtype=bool)
    observed[obs - int(axis[0])] = True
    det = sg.detrend(dense, cfg.smooth_win, cfg.baseline_win)
    return axis, det, observed, float(np.median(heights))


def main() -> None:
    vid_kw = sys.argv[1]
    tids = [int(a) for a in sys.argv[2:]] or [1, 4]
    cfg = Config()
    video = next((v for v in VIDEOS if vid_kw in v), None)
    if not video:
        print("找不到视频")
        return
    data = pickle.load(open(os.path.join(CACHE, os.path.basename(video) + ".pkl"), "rb"))
    fps = data["fps"]
    tracker = Tracker(cfg)
    samples, centers = {}, {}
    for index in range(data["frames"]):
        for tid, det in tracker.update(index, _detections(data, index, cfg)):
            samples.setdefault(tid, []).append((index, index / fps, det.box, None, det.keypoints))
            centers.setdefault(tid, []).append(((det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2))
    slots = {s.track_id: s for s in
             [build_slot(t, samples[t], centers[t]) for t in sorted(samples) if len(samples[t]) >= 2]}
    slots_list, _ = consolidate([slots[t] for t in sorted(slots)], cfg)
    slots = {s.track_id: s for s in slots_list}
    for tid in tids:
        if tid not in slots:
            print(f"#{tid} 不存在")
            continue
        res = count_track(tid, slots[tid].samples, cfg, fps)
        axis, det, observed, median_h = detrended_series(slots[tid], cfg, fps, res.signal_source)
        x = det - det.mean()
        spec = np.fft.rfft(x, n=2 * x.size)
        ac = np.fft.irfft(spec * np.conj(spec))[:x.size]
        ac = ac / max(1e-9, ac[0])
        amp_thr = amplitude_threshold(sg.hf_sigma_masked(det, observed), float(np.percentile(det, 95)),
                                      median_h, cfg)
        print(f"\n=== {os.path.basename(video)[:12]} #{tid} 源={res.signal_source} 计数={res.stats.get('total_count')} "
              f"估计周期={res.period_sec}s({(res.period_sec or 0) * fps:.1f}帧) 阈值={amp_thr:.2f}px 框高={median_h:.0f}px")
        print("自相关(滞后帧: 值), 只看 3~20:")
        line = []
        for lag in range(3, 21):
            line.append(f"{lag}:{ac[lag]:+.2f}")
        print("  " + "  ".join(line))
        local = sg.find_peaks(det, 2, amp_thr, 0.8 * amp_thr)
        gaps = np.diff(local)
        if gaps.size:
            hist = {int(k): int((gaps == k).sum()) for k in sorted(set(gaps.tolist())) if k <= 20}
            print(f"  全部局部峰的间隔分布(最小间隔=2帧): {hist}")
        print("  计数随最小间隔的变化(最小间隔: 峰数 -> 折算Hz):")
        row = []
        for md in range(3, 16):
            n = len(sg.find_peaks(det, md, amp_thr, 0.8 * amp_thr))
            row.append(f"{md}:{n}({fps * n / det.size:.2f}Hz)")
        print("    " + " ".join(row))


if __name__ == "__main__":
    main()
