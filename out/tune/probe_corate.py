"""快通道到底是真快跳还是关键点抖动: 数到的每一次起跳, 别的通道同一时刻有没有鼓包。

通道择优改成"同清晰度档内取数得多的通道"后, 快跳者会从平滑的头部慢摆换到高频通道。
高频通道也可能是姿态关键点的抖动, 抖动只出现在它自己那一维。
所以按选中的起跳时刻去问另外两条通道: 你这里在同一时刻(±0.25 周期)有没有一个鼓包?
作证率与"慢跳人群"的作证率持平 = 换过来的峰是真的; 明显更低 = 换来了抖动。
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
from jumpcount.counter import (_odd, amplitude_threshold, channel_cache, count_track,  # noqa: E402
                               extract_y_detail, participation)
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402

CHANNELS = ("ankle", "hip", "head")
VOTE_FRAC = 0.5      # 佐证鼓包下限 = 0.5 倍该通道自己的计数阈值


def channel_peaks(samples, source, cfg, fps, median_h):
    """某通道在它自己阈值下能数出的峰(帧号)与阈值。"""
    frames = np.array([int(s[0]) for s in samples], dtype=np.int64)
    y, _ = extract_y_detail(samples, source, cfg, channels=channel_cache(samples, cfg))
    keep = np.isfinite(y)
    if keep.sum() < 20:
        return np.array([]), 0.0, np.array([])
    axis, dense, _ = sg.fill_gaps(frames[keep], -y[keep], cfg.max_gap)
    observed = np.zeros(axis.size, dtype=bool)
    observed[frames[keep] - int(axis[0])] = True
    win = cfg.baseline_win if dense.size >= cfg.baseline_win else _odd(dense.size // 2)
    det = sg.detrend(dense, cfg.smooth_win, win)
    noise = sg.hf_sigma_masked(det, observed)
    p95 = float(np.percentile(det, 95)) if det.size else 0.0
    thr = amplitude_threshold(noise, p95, median_h, cfg)
    hard_min = max(2, int(fps / cfg.max_rate_hz))
    idx = sg.find_peaks(det, hard_min, thr * VOTE_FRAC, 0.4 * thr)
    return np.array([float(axis[i]) for i in idx], dtype=np.float64), thr, axis


def main() -> None:
    cfg = Config()
    margin = 0.0
    for item in sys.argv[1:]:
        k, v = item.split("=", 1)
        setattr(cfg, k, float(v))
    groups = {"快跳(>=2.4Hz)": [0, 0, 0], "慢跳": [0, 0, 0]}
    detail = []
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
            if st.get("reject") or not participation(st, cfg)[0]:
                continue
            events = [e for e in res.events if e.kind in ("jump", "interpolated")]
            if len(events) < 8 or not res.period_sec:
                continue
            heights = np.array([float(s[2][3]) - float(s[2][1]) for s in slot.samples])
            median_h = float(np.median(heights)) if heights.size else 0.0
            other = {}
            for name in CHANNELS:
                if name == res.signal_source:
                    continue
                peaks, _, _ = channel_peaks(slot.samples, name, cfg, fps, median_h)
                other[name] = peaks
            tol = 0.25 * res.period_sec * fps
            votes = sum(1 for e in events
                        if any(np.any(np.abs(peaks - e.frame) <= tol) for peaks in other.values()))
            key = "快跳(>=2.4Hz)" if res.rate_hz >= 2.4 else "慢跳"
            g = groups[key]
            g[0] += len(events)
            g[1] += votes
            g[2] += 1
            detail.append((os.path.basename(video)[:12], slot.track_id, key, res.signal_source,
                           len(events), votes, res.rate_hz))
    print(f"{'分组':<16}{'轨迹':>5}{'起跳':>7}{'他道佐证':>9}{'作证率':>9}")
    for key, (n, v, t) in groups.items():
        print(f"{key:<16}{t:>5}{n:>7}{v:>9}{v / max(1, n):>9.1%}")
    print()
    fast = [d for d in detail if d[2].startswith("快跳")]
    fast.sort(key=lambda d: d[5] / max(1, d[4]))
    print("作证率最低的快跳轨迹(前 12):")
    for video, tid, _, src, n, v, rate in fast[:12]:
        print(f"  {video} #{tid:<4} src={src:<6} 起跳 {n:>4} 佐证 {v:>4} 作证率 {v / max(1, n):>5.1%} 频率 {rate}")


if __name__ == "__main__":
    main()
