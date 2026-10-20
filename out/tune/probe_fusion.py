"""多通道佐证: 阈值以下的鼓包是"真起跳没数到"还是"这条通道的噪声"。

真起跳时踝、髋、头会一起动; 某个通道的抖动只在它自己那一维出现。
所以拿另外两条通道当独立证人: 主通道阈值压到 1/4 找出的鼓包, 若另一条通道
在同一时刻附近也有鼓包, 那就是真起跳; 再用随机平移做零假设, 排除"时刻本来就容易撞车"。
"""
from __future__ import annotations

import json
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
from jumpcount.counter import (_choose_period, _odd, amplitude_threshold, channel_cache,
                               extract_y_detail, pick_source)  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402

CHANNELS = ("ankle", "hip", "head")
LOOSE = 0.25


def channel_signal(frames, values, median_h, cfg, fps):
    keep = np.isfinite(values)
    if keep.sum() < 20:
        return None
    f, v = frames[keep], values[keep]
    axis, dense, _ = sg.fill_gaps(f, v, cfg.max_gap)
    observed = np.zeros(axis.size, dtype=bool)
    observed[f - int(axis[0])] = True
    win = cfg.baseline_win if dense.size >= cfg.baseline_win else _odd(dense.size // 2)
    det = sg.detrend(dense, cfg.smooth_win, win)
    noise = sg.hf_sigma_masked(det, observed)
    peak_level = float(np.percentile(det, 95)) if det.size else 0.0
    thr = amplitude_threshold(noise, peak_level, median_h, cfg)
    hard_min = max(2, int(round(fps / cfg.max_rate_hz)))
    coarse = sg.find_peaks(det, hard_min, thr, 0.8 * thr)
    ac, _ = sg.autocorr_period(det, fps, cfg.min_rate_hz, cfg.max_rate_hz)
    period, src = _choose_period(coarse, ac, fps, cfg)
    if src == "fallback" or period < 3:
        return None
    def pos(factor):
        idx = sg.find_peaks(det, max(hard_min, int(round(cfg.refractory_ratio * period))),
                            thr * factor, 0.8 * thr * factor)
        return np.array([float(axis[i]) for i in idx], dtype=np.float64)
    return dict(pos=pos, period=period, span=(float(axis[0]), float(axis[-1])))


def main():
    cfg = Config()
    rng = np.random.default_rng(20261021)
    rows = []
    for video in VIDEOS:
        path = os.path.join(CACHE, os.path.basename(video) + ".pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            data = pickle.load(fh)
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
            frames = np.array([int(s[0]) for s in slot.samples], dtype=np.int64)
            heights = np.array([float(s[2][3]) - float(s[2][1]) for s in slot.samples], dtype=np.float64)
            median_h = float(np.median(heights)) if heights.size else 0.0
            channels = channel_cache(slot.samples, cfg)
            main_src = pick_source(slot.samples, cfg, fps, channels)
            main_vals = -np.where(np.isfinite(channels.get(main_src, np.full(frames.size, np.nan))),
                                  channels.get(main_src, np.nan), np.nan)
            main_sig = channel_signal(frames, main_vals, median_h, cfg, fps)
            if not main_sig:
                continue
            others = {}
            for name in CHANNELS:
                if name == main_src or name not in channels:
                    continue
                vals = -np.where(np.isfinite(channels[name]), channels[name], np.nan)
                sig = channel_signal(frames, vals, median_h, cfg, fps)
                if sig:
                    others[name] = sig
            if not others:
                continue
            period = main_sig["period"]
            tol = 0.3 * period
            strong = main_sig["pos"](1.0)
            loose = main_sig["pos"](LOOSE)
            extra = np.array([t for t in loose if strong.size == 0 or np.min(np.abs(strong - t)) > tol * 2],
                             dtype=np.float64)
            if extra.size == 0:
                continue
            lo = main_sig["span"][0]
            hi = main_sig["span"][1]

            loose_others = {name: sig["pos"](LOOSE) for name, sig in others.items()}

            def corroborated(bumps, shifts):
                hit = 0
                for t in bumps:
                    for name, shifted in loose_others.items():
                        if shifted.size == 0:
                            continue
                        moved = shifted + shifts.get(name, 0.0)
                        if np.min(np.abs(moved - t)) <= tol:
                            hit += 1
                            break
                return hit

            hits = corroborated(extra, {})
            nulls = []
            for _ in range(60):
                shifts = {name: float(rng.uniform(-hi + lo, hi - lo)) for name in others}
                nulls.append(corroborated(extra, shifts) / extra.size)
            # 相位一致性及其零假设
            def phase_ok(bumps):
                if strong.size == 0 or bumps.size == 0:
                    return 0
                k = (bumps[:, None] - strong[None, :]) / period
                resid = np.abs(k - np.round(k)).min(axis=1)
                return int(np.sum(resid <= 0.3))

            phase = phase_ok(extra)
            phase_null = []
            for _ in range(60):
                rand = np.sort(rng.uniform(lo, hi, size=extra.size))
                if strong.size:
                    dist = np.abs(rand[:, None] - strong[None, :]).min(axis=1)
                    rand = rand[dist > tol * 2]
                if rand.size:
                    phase_null.append(phase_ok(rand) / rand.size)
            rows.append({
                "video": os.path.basename(video)[:8], "track": slot.track_id, "src": main_src,
                "period": round(period, 2), "extra": int(extra.size), "corroborated": hits,
                "corroborated_rate": round(hits / extra.size, 3),
                "null_rate": round(float(np.mean(nulls)), 3),
                "phase_rate": round(phase / extra.size, 3),
                "phase_null_rate": round(float(np.mean(phase_null)), 3),
            })

    with open(os.path.join(HERE, "fusion.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    extra = sum(r["extra"] for r in rows)
    hits = sum(r["corroborated"] for r in rows)
    null = float(np.mean([r["null_rate"] for r in rows]))
    phase = sum(round(r["phase_rate"] * r["extra"]) for r in rows)
    phase_null = float(np.mean([r["phase_null_rate"] for r in rows]))
    print(f"可佐证轨迹 {len(rows)} 条, 主通道阈值压到 1/4 多出的鼓包 {extra} 个")
    print(f"  另一条通道同时刻也有鼓包: {hits} 个 ({hits / max(1, extra):.1%}); 随机平移零假设 {null:.1%}")
    print(f"  落在既有节奏相位上: {phase} 个 ({phase / max(1, extra):.1%}); 随机时刻零假设 {phase_null:.1%}")
    print(f"  提升到真实个数的倍数: 通道佐证 {hits / max(1e-6, null * extra):.1f}x, 相位 {phase / max(1e-6, phase_null * extra):.1f}x")
    both = sum(min(r["corroborated"], round(r["phase_rate"] * r["extra"])) for r in rows)
    print(f"  同时满足两者(保守可挽回量): {both} 个")
    print("\n样例(主通道 / 鼓包 / 被佐证 / 佐证率 / 零假设):")
    for r in sorted(rows, key=lambda r: -r["extra"])[:8]:
        print(f"  {r['video']} #{r['track']:<3} {r['src']:<6} 鼓包 {r['extra']:>3} 佐证 {r['corroborated']:>3} "
              f"{r['corroborated_rate']:.2f} vs {r['null_rate']:.2f} | 相位 {r['phase_rate']:.2f} vs {r['phase_null_rate']:.2f}")


if __name__ == "__main__":
    main()
