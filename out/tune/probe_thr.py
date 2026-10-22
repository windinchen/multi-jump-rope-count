"""阈值到底是哪一项在把关: 绝对下限 / 噪声倍数 / 峰值比例, 以及按周期折叠后的残差噪声能低多少。

假设: 快跳时 hf_sigma(一阶差分) 把起跳本身当噪声, snr_k*noise 顶到信号身上, 真峰整段被压掉。
用法: .venv/bin/python out/tune/probe_thr.py
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
from jumpcount.counter import absolute_floor, count_track, pick_smooth_pass  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def fold_residual(det, observed, period):
    """减去"按周期折叠取平均"得到的模板后的残差噪声; 周期成分被模板吃掉, 剩下的是抖动。"""
    p = int(round(period))
    if p < 2 or det.size < 4 * p:
        return None
    usable = det.size - (det.size % p)
    folded = det[:usable].reshape(-1, p)
    template = np.nanmean(np.where(np.isfinite(folded), folded, np.nan), axis=0)
    resid = det[:usable] - np.tile(template, (folded.shape[0], 1)).reshape(-1)
    med = float(np.median(resid))
    cycles = folded.shape[0]
    return float(1.4826 * np.median(np.abs(resid - med)) * np.sqrt(cycles / max(1.0, cycles - 1.0)))


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
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            st = res.stats
            if not res.rate_hz or st.get("reject"):
                continue
            # 复算前段, 拆开阈值三项
            from jumpcount.counter import channel_cache, extract_y_detail, _odd  # noqa: WPS433
            ch = channel_cache(slot.samples, cfg)
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
            det, noise, peak_level = used["detrended"], used["noise"], used["peak_level"]
            period = used["period_frames"]
            fold = fold_residual(det, observed, period)
            floor = absolute_floor(median_h, cfg) * max(0.05, cfg.floor_relief)
            terms = {"floor": floor, "snr": cfg.snr_k * noise, "pfrac": cfg.peak_frac * peak_level}
            bind = max(terms, key=lambda k: terms[k])
            rows.append(dict(video=os.path.basename(video)[:12], tid=slot.track_id,
                             total=st["total_count"], rate=res.rate_hz, period=round(period, 1),
                             peak95=round(peak_level, 2), noise=round(noise, 2),
                             fold=None if fold is None else round(fold, 2),
                             thr=st["amplitude_threshold"], bind=bind,
                             thr_fold=None if fold is None else round(min(
                                 max(floor, cfg.snr_k * fold, cfg.peak_frac * peak_level),
                                 max(cfg.amplitude_ratio * median_h, 0.45 * peak_level)), 2),
                             coarse=st.get("coarse_peaks", 0)))
    print(f"{'video':<14}{'tid':>4}{'总':>5}{'Hz':>6}{'周期':>6}{'p95':>7}{'噪声':>7}{'折叠残差':>9}"
          f"{'阈值':>7}{'新阈值':>7}{'把关项':>7}{'粗峰':>6}")
    for r in rows:
        print(f"{r['video']:<14}{r['tid']:>4}{r['total']:>5}{r['rate']:>6}{r['period']:>6}{r['peak95']:>7}"
              f"{r['noise']:>7}{str(r['fold']):>9}{r['thr']:>7}{str(r['thr_fold']):>7}{r['bind']:>7}{r['coarse']:>6}")
    binds = {}
    for r in rows:
        binds[r["bind"]] = binds.get(r["bind"], 0) + 1
    print("\n把关项分布:", binds)
    ok = [r for r in rows if r["thr_fold"]]
    if ok:
        gain = [(r["thr"] - r["thr_fold"]) / max(1e-6, r["thr"]) for r in ok]
        print(f"折叠残差作噪声可把阈值平均压低 {100*np.mean(gain):.1f}% (n={len(ok)})")


if __name__ == "__main__":
    main()
