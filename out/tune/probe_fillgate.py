"""列出被补齐证据闸门拦下的空档: 供人眼核对"拦掉的是人停下了, 还是被挡糊了的真拍"。

用法: .venv/bin/python out/tune/probe_fillgate.py > out/tune/DOC_fillgate.log
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
from jumpcount.counter import (count_track, gap_lift_evidence, participation,  # noqa: E402
                               pick_smooth_pass)
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def rebuild(slot, cfg, fps):
    """按 count_track 同一套前段, 拿到 detrended / amp_thr / axis, 好逐段量证据。"""
    from jumpcount.counter import channel_cache, extract_y_detail, _odd  # noqa: WPS433
    ch = channel_cache(slot.samples, cfg)
    res = count_track(slot.track_id, slot.samples, cfg, fps)
    y, _ = extract_y_detail(slot.samples, res.signal_source, cfg, channels=ch)
    raw = -np.where(np.isfinite(y), y, np.nan)
    frames = np.array([int(s[0]) for s in slot.samples], dtype=np.int64)
    heights = np.array([float(s[2][3]) - float(s[2][1]) for s in slot.samples])
    median_h = float(np.median(heights))
    keep = np.isfinite(raw)
    axis, dense, _ = sg.fill_gaps(frames[keep], raw[keep], cfg.max_gap)
    observed = np.zeros(axis.size, dtype=bool)
    observed[frames[keep] - int(axis[0])] = True
    bw = cfg.baseline_win if dense.size >= cfg.baseline_win else _odd(dense.size // 2)
    hard_min = max(2, int(fps / cfg.max_rate_hz))
    used = pick_smooth_pass(dense, observed, bw, median_h, fps, cfg, hard_min)
    return res, used, axis, observed


def main() -> None:
    cfg = Config()
    rows = []
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
            res, used, axis, observed = rebuild(slot, cfg, fps)
            if res.stats.get("reject") or res.stats.get("fill_no_lift", 0) <= 0:
                continue
            verdict, _ = participation(res.stats, cfg)
            det = used["detrended"]
            amp_thr = used["amp_thr"]
            period = used["period_frames"]
            jumps = [e.frame for e in res.events if e.kind == "jump"]
            for a, b in zip(jumps, jumps[1:]):
                lo, hi = int(round(a - axis[0])), int(round(b - axis[0]))
                gap = b - a
                if gap <= cfg.gap_ratio * period or lo < 0 or hi > det.size:
                    continue
                ev = gap_lift_evidence(det, lo, hi, period)
                if ev < cfg.fill_min_lift_frac * amp_thr:
                    rows.append(dict(v=os.path.basename(video)[:12], tid=slot.track_id, verdict=verdict,
                                     t0=round(a / fps, 2), t1=round(b / fps, 2), gap=round(gap, 1),
                                     per=round(period, 1), miss=int(round(gap / period)) - 1,
                                     ev=round(ev, 2), need=round(cfg.fill_min_lift_frac * amp_thr, 2),
                                     thr=round(amp_thr, 2), rec=int(res.stats.get("recovered", 0))))
    rows.sort(key=lambda r: -r["miss"])
    print(f"被拦下的空档 {len(rows)} 处, 少补 {sum(r['miss'] for r in rows)} 次")
    print(f"{'video':<14}{'tid':>4}{'判定':>13}{'起':>8}{'止':>8}{'档长':>6}{'周期':>6}{'少补':>6}{'证据':>7}{'需':>7}{'阈值':>7}{'回捞':>6}")
    for r in rows[:40]:
        print(f"{r['v']:<14}{r['tid']:>4}{r['verdict']:>13}{r['t0']:>8}{r['t1']:>8}{r['gap']:>6}{r['per']:>6}"
              f"{r['miss']:>6}{r['ev']:>7}{r['need']:>7}{r['thr']:>7}{r['rec']:>6}")
    inside = [r for r in rows if r["verdict"] != "out"]
    print(f"\n其中判定非 out 的 {len(inside)} 处 / 少补 {sum(r['miss'] for r in inside)} 次")


if __name__ == "__main__":
    main()
