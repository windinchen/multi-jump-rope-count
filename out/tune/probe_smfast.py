"""窄窗(不做中值滤波)多取到的峰, 是"被滤波抹掉的真拍"还是"把一次起跳劈成两半"?

判据: 真拍应该踩在宽窗那条轨迹自己的节拍格子上(与左右实测峰都差一个周期、且落在中点);
     拆分出来的假峰贴着某个实测峰, 或落在格子外。
用法: .venv/bin/python out/tune/probe_smfast.py > out/tune/DOC_smfast.log
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
from jumpcount.counter import (analyze_signal, channel_cache, count_track,  # noqa: E402
                               extract_y_detail, participation, pick_smooth_pass, _odd)
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def grid_ok(pos, mains, period, tol):
    """这个额外峰是不是踩在实测峰的节拍格子上: 与最近的实测峰差整数个周期, 且在容差内。"""
    if not mains or period <= 0:
        return False
    nearest = min(mains, key=lambda m: abs(m - pos))
    dist = abs(nearest - pos)
    if dist < 0.6 * period:
        return False            # 贴着实测峰 = 把它劈成了两半
    k = int(round(dist / period))
    return abs(dist - k * period) <= tol * period


def main() -> None:
    cfg = Config()
    tol = cfg.recover_center_tol
    stats = {"tracks_alt": 0, "extra": 0, "grid": 0, "near": 0, "off": 0}
    detail = []
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
            st = res.stats
            if st.get("reject") or not st.get("smooth_fast_tried") or st.get("smooth_win_used") != cfg.smooth_win_fast:
                continue
            ch = channel_cache(slot.samples, cfg)
            y, _ = extract_y_detail(slot.samples, res.signal_source, cfg, channels=ch)
            raw = -np.where(np.isfinite(y), y, np.nan)
            frames = np.array([int(s[0]) for s in slot.samples], dtype=np.int64)
            heights = np.array([float(s[2][3]) - float(s[2][1]) for s in slot.samples])
            keep = np.isfinite(raw)
            axis, dense, _ = sg.fill_gaps(frames[keep], raw[keep], cfg.max_gap)
            observed = np.zeros(axis.size, dtype=bool)
            observed[frames[keep] - int(axis[0])] = True
            bw = cfg.baseline_win if dense.size >= cfg.baseline_win else _odd(dense.size // 2)
            hm = max(2, int(fps / cfg.max_rate_hz))
            wide = analyze_signal(dense, observed, bw, float(np.median(heights)), fps, cfg, hm, 3)
            verdict, _ = participation(st, cfg)
            if verdict == "out":
                continue
            mains = list(wide["coarse"])
            period = wide["period_frames"]
            alt = analyze_signal(dense, observed, bw, float(np.median(heights)), fps, cfg, hm, cfg.smooth_win_fast)
            extras = [p for p in alt["coarse"] if all(abs(p - m) > 0.6 * max(1.0, period) for m in mains)]
            g = sum(1 for p in extras if grid_ok(p, mains, period, tol))
            stats["tracks_alt"] += 1
            stats["extra"] += len(extras)
            stats["grid"] += g
            stats["near"] += sum(1 for p in alt["coarse"] if any(abs(p - m) <= 0.6 * max(1.0, period) for m in mains)) - \
                sum(1 for p in mains if all(abs(p - m) <= 0.6 * max(1.0, period) for m in mains))
            stats["off"] += len(extras) - g
            detail.append((os.path.basename(video)[:12], slot.track_id, verdict, len(mains), len(alt["coarse"]),
                           len(extras), g, round(period, 1), round(st["rhythm_clarity"], 3)))
    print(f"启用窄窗的轨迹 {stats['tracks_alt']} 条: 额外峰 {stats['extra']} 个, "
          f"踩在节拍格子上 {stats['grid']} ({stats['grid'] / max(1, stats['extra']):.0%}), 格子外 {stats['off']}")
    print(f"\n{'video':<14}{'tid':>4}{'判定':>13}{'宽窗峰':>8}{'窄窗峰':>8}{'额外':>6}{'踩格':>6}{'周期':>6}{'清晰度':>8}")
    for row in sorted(detail, key=lambda r: -(r[5] - r[6])):
        print(f"{row[0]:<14}{row[1]:>4}{row[2]:>13}{row[3]:>8}{row[4]:>8}{row[5]:>6}{row[6]:>6}{row[7]:>6}{row[8]:>8}")


if __name__ == "__main__":
    main()
