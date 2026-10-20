"""缺口归因探针: 把"节奏应有 - 实际数到"的缺口按原因拆开.

口径与 harness.score 一致(只统计计入/低置信轨迹), 但把缺口拆成:
  未观测   人不在画面里的时间, 算力/检测问题, 阈值再松也补不出来
  补数封顶 节奏判定有起跳但受 max_fill_per_gap / min_fill_coverage 限制没补
  阈值以下 把阈值压低就能多找到的鼓包(上限, 含误计)
  残余     热身期/不应期/双摇判定等其它开销
同时量一下"周期取整"带来的误差: 现在周期取实测峰间隔的中位数(整数帧), 5% 的误差
就会同时带偏最小间隔、空档判定、补数个数和"应有"分母。
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
from jumpcount.counter import (  # noqa: E402
    _choose_period, amplitude_threshold, channel_cache, count_track,
    extract_y_detail, participation, pick_source,
)
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def signal_of(samples, cfg, fps):
    """复刻 count_track 的信号准备, 额外返回逐帧鼓包普查需要的量。"""
    frames = np.array([int(s[0]) for s in samples], dtype=np.int64)
    heights = np.array([float(s[2][3]) - float(s[2][1]) for s in samples], dtype=np.float64)
    median_h = float(np.median(heights)) if heights.size else 0.0
    channels = channel_cache(samples, cfg)
    resolved = pick_source(samples, cfg, fps, channels)
    y, _ = extract_y_detail(samples, resolved, cfg, channels=channels)
    raw = -np.where(np.isfinite(y), y, np.nan)
    keep = np.isfinite(raw)
    frames, raw = frames[keep], raw[keep]
    if frames.size < 6:
        return None
    axis, dense, _ = sg.fill_gaps(frames, raw, cfg.max_gap)
    observed = np.zeros(axis.size, dtype=bool)
    observed[frames - int(axis[0])] = True
    from jumpcount.counter import _odd
    win = cfg.baseline_win if dense.size >= cfg.baseline_win else _odd(dense.size // 2)
    det = sg.detrend(dense, cfg.smooth_win, win)
    noise = sg.hf_sigma_masked(det, observed)
    peak_level = float(np.percentile(det, 95)) if det.size else 0.0
    thr = amplitude_threshold(noise, peak_level, median_h, cfg)
    hard_min = max(2, int(round(fps / cfg.max_rate_hz)))
    coarse = sg.find_peaks(det, hard_min, thr, 0.8 * thr)
    ac_period, _ = sg.autocorr_period(det, fps, cfg.min_rate_hz, cfg.max_rate_hz)
    period_frames, period_src = _choose_period(coarse, ac_period, fps, cfg)
    min_distance = max(hard_min, int(round(cfg.refractory_ratio * period_frames)))
    return dict(axis=axis, det=det, thr=thr, period_frames=period_frames, period_src=period_src,
                min_distance=min_distance, observed=observed, median_h=median_h,
                ac_period_frames=(ac_period * fps) if ac_period else None)


def census(sig, cfg):
    """同一周期/最小间隔下, 阈值放宽若干倍时能数到多少个鼓包。"""
    out = {}
    for factor in (1.0, 0.5, 0.25):
        thr = sig["thr"] * factor
        out[f"peaks_x{factor:g}"] = len(sg.find_peaks(sig["det"], sig["min_distance"], thr, 0.8 * thr))
    return out


def continuous_period(sig):
    """同一批峰的亚帧位置取相邻间隔中位数: 与主链路同门槛, 只多算了亚帧偏移。"""
    det, axis = sig["det"], sig["axis"]
    peaks = sg.find_peaks(det, sig["min_distance"], sig["thr"], 0.8 * sig["thr"])
    sig["peaks"] = len(peaks)
    if len(peaks) < 4:
        return None
    pos = np.array([float(axis[p]) + (sg.parabolic_peak(det, p) - p) for p in peaks])
    step = float(np.median(np.diff(pos)))
    return step if step > 0 else None


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
        slots = [build_slot(tid, samples[tid], centers[tid]) for tid in sorted(samples) if len(samples[tid]) >= 2]
        slots, _ = consolidate(slots, cfg)
        for slot in slots:
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            verdict, reason = participation(res.stats, cfg)
            st = res.stats
            sig = signal_of(slot.samples, cfg, fps)
            row = {
                "video": os.path.basename(video)[:8], "track": slot.track_id, "verdict": verdict, "reason": reason,
                "count": res.count, "filled": int(st.get("filled", 0)), "total": res.count + int(st.get("filled", 0)),
                "expected": float(st.get("expected_count", 0.0)), "rate_hz": res.rate_hz,
                "duration": float(st.get("duration_sec", 0.0)), "visibility": float(res.visibility or 0.0),
                "period_frames": float(res.period_sec * fps) if res.period_sec else 0.0,
                "period_src": st.get("period_source"), "rhythm": st.get("rhythm_strength"),
                "source": res.signal_source, "estimated_misses": int(st.get("estimated_misses", 0)),
                "misses_unobserved": int(st.get("misses_unobserved", 0)),
                "median_h": float(res.median_height),
            }
            if sig:
                row.update(census(sig, cfg))
                cp = continuous_period(sig)
                row["period_cont"] = round(cp, 3) if cp else None
                row["period_ac"] = round(sig["ac_period_frames"], 3) if sig["ac_period_frames"] else None
            rows.append(row)

    with open(os.path.join(HERE, "deficit.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)

    kept = [r for r in rows if r["verdict"] in ("in", "provisional")]
    out_rows = [r for r in rows if r["verdict"] == "out"]
    exp_all = sum(r["expected"] for r in kept)
    total = sum(r["total"] for r in kept)
    # 只按"真正观测到的时间"折算应有数: 人不在画面里时不该指望数得到
    exp_obs = sum(r["rate_hz"] * r["visibility"] * r["duration"] for r in kept if r["rate_hz"])
    unobserved = exp_all - exp_obs
    capped = sum(max(0, r["estimated_misses"] - r["filled"]) for r in kept)
    threshold = sum(max(0, r.get("peaks_x0.25", r["total"]) - r.get("peaks_x1", r["total"])) for r in kept)
    print(f"[复算校验] 主链路数到 {total}, 探针复算峰数(含补数前) {sum(r.get('peaks_x1', 0) for r in kept)}")
    fallback_exp = sum(r["expected"] for r in kept if r["period_src"] == "fallback")
    lost = sum(r["total"] for r in out_rows)

    print(f"轨迹 {len(rows)} 条, 计入/低置信 {len(kept)} 条")
    print(f"合计 数到 {total} / 节奏应有 {exp_all:.0f} = 完整率 {total / max(1, exp_all):.3f}")
    print(f"  按观测时间折算应有 {exp_obs:.0f} -> 观测内完整率 {total / max(1, exp_obs):.3f}")
    print("缺口拆分(次):")
    print(f"  未观测(人不在画面)   {unobserved:8.1f}  {unobserved / max(1, exp_all - total):7.1%}")
    print(f"  补数封顶/空档未补    {capped:8.1f}  {capped / max(1, exp_obs - total):7.1%} (占观测内缺口)")
    print(f"  阈值以下鼓包(上限)   {threshold:8.1f}  (阈值压到 1/4 才多出来的峰)")
    print(f"  整条剔除但仍数到     {lost:8.1f}")
    print(f"  「应有」分母里来自兜底周期的轨迹 {fallback_exp:.0f} 次 ({fallback_exp / max(1, exp_all):.1%})")

    devs = [(r["period_frames"] / r["period_cont"] - 1) for r in kept
            if r.get("period_cont") and r["period_frames"] and r["period_cont"] > 1]
    if devs:
        arr = np.array(devs)
        print(f"周期取整偏差: 平均 {arr.mean():+.2%} 绝对均值 {np.abs(arr).mean():.2%} 最大 {np.abs(arr).max():.2%} (样本 {arr.size})")
    devs_ac = [(r["period_frames"] / r["period_ac"] - 1) for r in kept
               if r.get("period_ac") and r["period_frames"] and r["period_ac"] > 1]
    if devs_ac:
        arr = np.array(devs_ac)
        print(f"周期与亚帧自相关比: 平均 {arr.mean():+.2%} 绝对均值 {np.abs(arr).mean():.2%} (样本 {arr.size})")

    by_src = {}
    for r in kept:
        by_src.setdefault(r["source"], [0, 0.0, 0])
        by_src[r["source"]][0] += 1
        by_src[r["source"]][1] += r["total"] / max(1e-6, r["rate_hz"] * r["visibility"] * r["duration"])
        by_src[r["source"]][2] += 1
    print("按信号通道: " + " | ".join(
        f"{k} n={v[0]} 观测内完整率 {v[1] / max(1, v[2]):.3f}" for k, v in sorted(by_src.items())))

    worst = sorted(kept, key=lambda r: -(r["expected"] - r["total"]))[:12]
    print("最大缺口轨迹:")
    for r in worst:
        print(f"  {r['video']} #{r['track']:<3} 数到 {r['total']:>4}/{r['expected']:>6.1f} 可见 {r['visibility']:.2f} "
              f"估漏 {r['estimated_misses']:>3} 已补 {r['filled']:>3} 压阈可捞 {r.get('peaks_x0.25', 0) - r.get('peaks_x1', 0):>3} "
              f"源 {r['source']:<10} 节律 {r['rhythm']}")


if __name__ == "__main__":
    main()
