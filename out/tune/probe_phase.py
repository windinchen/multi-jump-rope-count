"""相位普查: 阈值以下隐藏的到底是真起跳还是噪声。

做法: 用主链路同一套信号与阈值取"实测峰", 再把阈值压到 1/4 取"鼓包",
逐个检查只出现在低阈值里的鼓包, 它到最近实测峰的距离是不是接近整数倍周期。
落在相位上 = 有节奏背书的漏计; 随机分布 = 噪声, 放宽阈值只会虚高。
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
                               extract_y_detail, participation, pick_source, count_track)  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402

PHASE_TOL = 0.3  # 偏离整数倍周期 30% 以内算"落在相位上"


def prep(samples, cfg, fps):
    frames = np.array([int(s[0]) for s in samples], dtype=np.int64)
    heights = np.array([float(s[2][3]) - float(s[2][1]) for s in samples], dtype=np.float64)
    median_h = float(np.median(heights)) if heights.size else 0.0
    channels = channel_cache(samples, cfg)
    resolved = pick_source(samples, cfg, fps, channels)
    y, _ = extract_y_detail(samples, resolved, cfg, channels=channels)
    raw = -np.where(np.isfinite(y), y, np.nan)
    keep = np.isfinite(raw)
    frames, raw = frames[keep], raw[keep]
    if frames.size < 12:
        return None
    axis, dense, _ = sg.fill_gaps(frames, raw, cfg.max_gap)
    observed = np.zeros(axis.size, dtype=bool)
    observed[frames - int(axis[0])] = True
    win = cfg.baseline_win if dense.size >= cfg.baseline_win else _odd(dense.size // 2)
    det = sg.detrend(dense, cfg.smooth_win, win)
    noise = sg.hf_sigma_masked(det, observed)
    peak_level = float(np.percentile(det, 95)) if det.size else 0.0
    thr = amplitude_threshold(noise, peak_level, median_h, cfg)
    hard_min = max(2, int(round(fps / cfg.max_rate_hz)))
    coarse = sg.find_peaks(det, hard_min, thr, 0.8 * thr)
    ac, _ = sg.autocorr_period(det, fps, cfg.min_rate_hz, cfg.max_rate_hz)
    period, src = _choose_period(coarse, ac, fps, cfg)
    return dict(axis=axis, det=det, thr=thr, period=period, src=src, observed=observed,
                min_distance=max(hard_min, int(round(cfg.refractory_ratio * period))))


def positions(p, factor):
    idx = sg.find_peaks(p["det"], p["min_distance"], p["thr"] * factor, 0.8 * p["thr"] * factor)
    return np.array([float(p["axis"][i]) for i in idx], dtype=np.float64)


def main():
    cfg = Config()
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
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            verdict, _ = participation(res.stats, cfg)
            if verdict == "out":
                continue
            p = prep(slot.samples, cfg, fps)
            if not p or p["src"] == "fallback" or p["period"] < 3:
                continue
            strong = positions(p, 1.0)
            loose = positions(p, 0.25)
            extra = [t for t in loose if not np.any(np.abs(strong - t) <= p["min_distance"])] if strong.size else list(loose)
            consistent = 0
            for t in extra:
                if strong.size == 0:
                    break
                # 与整条节奏对齐: 到任一实测峰的相位残差接近 0 才算有节奏背书
                resid = np.min(np.abs((t - strong) / p["period"] - np.round((t - strong) / p["period"])))
                if resid <= PHASE_TOL:
                    consistent += 1
            observed_frames = int(p["observed"].sum())
            rows.append({
                "video": os.path.basename(video)[:8], "track": slot.track_id, "verdict": verdict,
                "count": res.count, "filled": int(res.stats.get("filled", 0)),
                "total": res.count + int(res.stats.get("filled", 0)),
                "expected": float(res.stats.get("expected_count", 0.0)),
                "observed_sec": round(observed_frames / fps, 2),
                "rate_hz": res.rate_hz, "rhythm": res.stats.get("rhythm_strength"),
                "peaks_strong": int(strong.size), "peaks_loose": int(loose.size),
                "extra": len(extra), "extra_phase_ok": consistent,
            })
    with open(os.path.join(HERE, "phase.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)

    total = sum(r["total"] for r in rows)
    exp_obs = sum(r["rate_hz"] * r["observed_sec"] for r in rows if r["rate_hz"])
    extra = sum(r["extra"] for r in rows)
    ok = sum(r["extra_phase_ok"] for r in rows)
    print(f"可计轨迹 {len(rows)} 条")
    print(f"数到 {total} / 观测时间内按节奏应有 {exp_obs:.0f} -> 观测内完整率 {total / max(1, exp_obs):.3f}, 观测内缺口 {exp_obs - total:.0f}")
    print(f"阈值压到 1/4 多出的鼓包 {extra} 个, 其中落在既有节奏相位上 {ok} 个 ({ok / max(1, extra):.1%})")
    print(f"  相位一致的鼓包 ≈ 可挽回的漏计; 相对观测内缺口 = {ok / max(1.0, exp_obs - total):.1%}")
    print(f"  其余 {extra - ok} 个鼓包在相位上随机, 放宽阈值会把它们数成起跳")
    rows2 = sorted(rows, key=lambda r: -(r["rate_hz"] * r["observed_sec"] - r["total"]))[:10]
    print("缺口最大 10 条:")
    for r in rows2:
        gap = r["rate_hz"] * r["observed_sec"] - r["total"]
        print(f"  {r['video']} #{r['track']:<3} 数到 {r['total']:>4}/{r['rate_hz'] * r['observed_sec']:>6.1f} "
              f"缺口 {gap:>5.1f} 实测峰 {r['peaks_strong']:>4} 低阈值鼓包 +{r['extra']:>3} 相位一致 {r['extra_phase_ok']:>3} 节律 {r['rhythm']}")


if __name__ == "__main__":
    main()
