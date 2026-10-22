"""漏计归因: 空档里到底是"根本没有信号"还是"有真实局部峰但没过阈值"。

用法: .venv/bin/python out/tune/probe_missed.py [key=value ...]
如果是后者, 就该做"节奏槽位内的次阈值提升"; 如果是前者, 算法已无解, 只能等更好的检测。
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from harness import CACHE, VIDEOS, _detections  # noqa: E402
from jumpcount import signal as sg  # noqa: E402
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import channel_cache, count_track, extract_y_detail, participation  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


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

    started = time.time()
    bucket = Counter()
    near_miss_ratio = []
    near_miss_period_fit = []
    per_track_gain = []
    for video in VIDEOS:
        path = os.path.join(CACHE, os.path.basename(video) + ".pkl")
        if not path.endswith(".pkl") or not os.path.exists(path):
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
            verdict, _ = participation(res.stats, cfg)
            if verdict == "out" or not res.period_sec or res.stats.get("period_source") == "fallback":
                continue
            amp_thr = float(res.stats.get("amplitude_threshold", 0.0))
            if amp_thr <= 0:
                continue
            frames = np.array([int(s[0]) for s in slot.samples], dtype=np.int64)
            channels = channel_cache(slot.samples, cfg)
            y_series, _ = extract_y_detail(slot.samples, res.signal_source, cfg, channels=channels)
            raw = -np.where(np.isfinite(y_series), y_series, np.nan)
            keep = np.isfinite(raw)
            obs_frames = frames[keep]
            axis, dense, _ = sg.fill_gaps(obs_frames, raw[keep], cfg.max_gap)
            det = sg.detrend(dense, cfg.smooth_win, cfg.baseline_win)
            observed = np.zeros(axis.size, dtype=bool)
            observed[obs_frames - int(axis[0])] = True
            period = res.period_sec * fps
            measured = [e.frame for e in res.events if e.kind in ("jump", "bump")]
            if len(measured) < 2:
                continue
            gain = 0
            for i in range(len(measured) - 1):
                lo = measured[i] + 0.6 * period
                hi = measured[i + 1] - 0.6 * period
                if hi <= lo:
                    continue
                seg = det[((axis >= lo) & (axis <= hi))]
                seg_obs = observed[((axis >= lo) & (axis <= hi))]
                expected = int(round((measured[i + 1] - measured[i]) / period)) - 1
                if expected <= 0 or seg.size == 0:
                    continue
                coverage = float(seg_obs.mean()) if seg_obs.size else 0.0
                if coverage < cfg.min_fill_coverage:
                    bucket["A 未观测(遮挡/出画)"] += expected
                    continue
                peaks = sg.find_peaks(seg, max(2, int(round(0.45 * period))), 0.15 * amp_thr, 0.1 * amp_thr)
                if not peaks:
                    bucket["C 空档里连局部鼓包都没有"] += expected
                    continue
                best = max(float(seg[p]) for p in peaks)
                ratio = best / amp_thr
                near_miss_ratio.append(ratio)
                n_fit = 0
                for p in peaks:
                    r = float(seg[p]) / amp_thr
                    near_miss_period_fit.append(min(abs(float(axis[p]) - measured[i]) / period - 1.0,
                                                    abs(measured[i + 1] - float(axis[p])) / period - 1.0))
                    if r >= 0.5:
                        n_fit += 1
                if ratio >= 0.7:
                    label = "B1 有真实鼓包 >=0.7x阈值(可提升)" if ratio < 1.0 else "B0 鼓包已过阈值却漏了"
                    bucket[label] += min(expected, max(1, n_fit))
                    gain += min(expected, max(1, n_fit))
                elif ratio >= 0.4:
                    bucket["B2 鼓包 0.4~0.7x阈值(弱但有形)"] += expected
                else:
                    bucket["B3 鼓包 <0.4x阈值(基本是噪声)"] += expected
            if gain:
                per_track_gain.append((os.path.basename(video)[:12], slot.track_id, gain,
                                       res.stats.get("total_count"), amp_thr, res.stats.get("lift_ratio")))
    total_expected = sum(bucket.values())
    print(f"用时 {time.time() - started:.1f}s, 配置: {sys.argv[1:] or '默认'}")
    print(f"空档中按节奏应有但未数到的次数合计 {total_expected}, 归类:")
    for k in sorted(bucket):
        print(f"  {k:<28} {bucket[k]:>5} 次  占 {bucket[k] / max(1, total_expected) * 100:>5.1f}%")
    if near_miss_ratio:
        arr = np.array(near_miss_ratio)
        print(f"\n空档内最强鼓包 / 阈值: p25={np.percentile(arr, 25):.2f} 中位={np.median(arr):.2f} "
              f"p75={np.percentile(arr, 75):.2f} p95={np.percentile(arr, 95):.2f}")
    if near_miss_period_fit:
        fit = np.array(near_miss_period_fit)
        print(f"鼓包到相邻实测峰的「距一个周期」偏差: 偏差<0.25 的占 {(fit < 0.25).mean() * 100:.1f}% "
              f"(偏差小说明鼓包正好落在节奏槽位上, 是真起跳的可能性高)")
    print(f"\n若把 B1 全部提升, 各轨迹增益(前 12 条): 视频 轨迹 增益 现有计数 阈值 抬脚比")
    for row in sorted(per_track_gain, key=lambda r: -r[2])[:12]:
        print(f"  {row[0]:<13}#{row[1]:<4} +{row[2]:<4}现有 {row[3]:<5} 阈值 {row[4]:.1f}px 抬脚比 {row[5]}")
    print(f"\nB1 合计可提升 {sum(v for k, v in bucket.items() if k.startswith('B0') or k.startswith('B1'))} 次")


if __name__ == "__main__":
    main()
