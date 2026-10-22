"""逐条轨迹打印各身体通道的"一起动"程度: 真跳绳时头/髋/踝等幅同相, 被前景行人或框抖动带出来的
假波纹往往只有一条通道动。用来找"站着的人被数了一百次"那类假计数的判别量。

用法: .venv/bin/python out/tune/probe_coherence.py <视频关键字> [轨迹ID ...]
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
from jumpcount.counter import channel_cache, count_track, pick_source  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount import signal as sg  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def detrend(ch, cfg):
    v = np.where(np.isfinite(ch), ch, np.nan)
    return sg.detrend(-v, cfg.smooth_win, cfg.baseline_win)


def main() -> None:
    key = sys.argv[1]
    tids = [int(x) for x in sys.argv[2:]]
    cfg = Config()
    video = next(v for v in VIDEOS if key in os.path.basename(v))
    data = pickle.load(open(os.path.join(CACHE, os.path.basename(video) + ".pkl"), "rb"))
    fps = data["fps"]
    tracker = Tracker(cfg)
    samples, centers = {}, {}
    for index in range(data["frames"]):
        for tid, det in tracker.update(index, _detections(data, index, cfg)):
            samples.setdefault(tid, []).append((index, index / fps, det.box, None, det.keypoints))
            centers.setdefault(tid, []).append(((det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2))
    slots = [build_slot(t, samples[t], centers[t]) for t in sorted(samples) if len(samples[t]) >= 2]
    slots, _ = consolidate(slots, cfg)
    print(f"{'轨迹':<10}{'框高':>6}{'计入':>6}{'频率':>7}{'选中':>11} | 各通道 p95幅度(px) 与 覆盖率 | 通道间相关(踝-头/踝-髋/踝-框底)")
    for slot in slots:
        tid = slot.track_id
        if tids and tid not in tids:
            continue
        s = slot.samples
        ch = channel_cache(s, cfg)
        dts = {}
        for name, arr in ch.items():
            if np.isfinite(arr).sum() >= 12:
                dts[name] = detrend(arr, cfg)
        if not dts:
            continue
        res = count_track(tid, s, cfg, fps)
        total = res.count + res.stats.get("filled", 0)
        heights = np.array([float(x[2][3]) - float(x[2][1]) for x in s])
        picked = pick_source(s, cfg, fps, ch)
        amps = {k: float(np.nanpercentile(v, 95)) if np.isfinite(v).any() else 0.0 for k, v in dts.items()}
        cov = {k: float(np.isfinite(v).mean()) for k, v in dts.items()}

        def corr(a, b):
            if a not in dts or b not in dts:
                return float("nan")
            x, y = dts[a], dts[b]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 20 or np.std(x[m]) < 1e-6 or np.std(y[m]) < 1e-6:
                return float("nan")
            return float(np.corrcoef(x[m], y[m])[0, 1])

        cells = " ".join(f"{k}:{amps[k]:>5.1f}/{cov[k]:.2f}" for k in ("head", "hip", "ankle", "box_bottom") if k in dts)
        cs = f"{corr('ankle', 'head'):>7.2f} {corr('ankle', 'hip'):>7.2f} {corr('ankle', 'box_bottom'):>7.2f}"
        print(f"#{tid:<9}{np.median(heights):>6.0f}{total:>6}{res.rate_hz:>7.2f}{picked:>11} | {cells} | {cs}")


if __name__ == "__main__":
    main()
