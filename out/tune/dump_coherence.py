"""全素材逐条轨迹导出"身体一致性"台账, 为新增判据定阈值。

判据假设: 真起跳是整个身体离地, 头必须跟着抬; 被前景行人/框抖出来的假波纹只动框底边或踝, 头不动。
列: 视频 轨迹 计入 频率 判定 选中通道 框高 lift_ratio 头位移占框高 踝-头相关 头通道节律 头p95 踝p95
用法: .venv/bin/python out/tune/dump_coherence.py > out/tune/coherence.tsv
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
from jumpcount.counter import channel_cache, count_track, participation, pick_source  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount import signal as sg  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def main() -> None:
    cfg = Config()
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
                centers.setdefault(tid, []).append(((det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2))
        slots = [build_slot(t, samples[t], centers[t]) for t in sorted(samples) if len(samples[t]) >= 2]
        slots, _ = consolidate(slots, cfg)
        for slot in slots:
            s = slot.samples
            ch = channel_cache(s, cfg)
            dts = {k: sg.detrend(-np.where(np.isfinite(v), v, np.nan), cfg.smooth_win, cfg.baseline_win)
                   for k, v in ch.items() if np.isfinite(v).sum() >= 12}
            if "head" not in dts:
                continue
            res = count_track(slot.track_id, s, cfg, fps)
            verdict = participation(res.stats, cfg)[0]
            total = res.count + res.stats.get("filled", 0)
            median_h = float(np.median([float(x[2][3]) - float(x[2][1]) for x in s]))
            picked = pick_source(s, cfg, fps, ch)
            head = dts["head"]
            head_p95 = float(np.nanpercentile(head, 95)) if np.isfinite(head).any() else 0.0
            ankle_p95 = float(np.nanpercentile(dts["ankle"], 95)) if "ankle" in dts and np.isfinite(dts["ankle"]).any() else 0.0
            x, y = dts["ankle"], head
            m = np.isfinite(x) & np.isfinite(y)
            corr = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() >= 20 and np.std(x[m]) > 1e-6 and np.std(y[m]) > 1e-6 else float("nan")
            hr = float(res.stats.get("lift_ratio", 0) or 0)
            print(f"{os.path.basename(video)[:12]}\t{slot.track_id}\t{total:.0f}\t{res.rate_hz}\t{verdict}\t{picked}\t"
                  f"{median_h:.0f}\t{hr:.4f}\t{head_p95 / max(median_h, 1e-6):.4f}\t{corr:.3f}\t"
                  f"{res.stats.get('rhythm_strength', 0):.3f}\t{head_p95:.2f}\t{ankle_p95:.2f}")


if __name__ == "__main__":
    main()
