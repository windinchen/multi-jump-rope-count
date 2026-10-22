"""被闸门拦下的空档到底有没有"离地": 用全部身体通道重新取证, 找最划算的闸门口径。

用法: .venv/bin/python out/tune/probe_ev.py
背景: probe_hole 显示缺口里约 961 次落在"观测覆盖率≈100% 却被 fill_min_lift_frac 拒了"的空档,
     不是人不在画面里。这些空档里到底有没有起跳, 决定了要不要放宽、放宽哪一条。
口径: 对每条轨迹, 踝/髋/头/框底边各自去基线+各自阈值(与 channel_quality 同一套),
     对每个"节奏说该有跳"的空档记录主通道证据比、最大证据比、超过各门槛的通道数。
     同时记录该空档里其它通道的峰能不能落进来(旁证), 用来估计补数的真实命中率。
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from harness import CACHE, VIDEOS, _detections  # noqa: E402
from jumpcount import signal as sg  # noqa: E402
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import (amplitude_threshold, channel_cache, count_track, gap_lift_evidence,
                               participation, pick_source)  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402

CH = ("ankle", "hip", "head", "box_bottom")


def chan_det(samples, cfg, fps, primary):
    """每个通道一套 (axis0, detrended, thr), 口径同 channel_quality / probe_fillproof。"""
    ch = channel_cache(samples, cfg)
    frames = np.array([int(s[0]) for s in samples], dtype=np.int64)
    heights = np.array([float(s[2][3]) - float(s[2][1]) for s in samples], dtype=np.float64)
    median_h = float(np.median(heights)) if heights.size else 0.0
    out = {}
    for name in CH:
        if name not in ch:
            continue
        values = -np.where(np.isfinite(ch[name]), ch[name], np.nan)
        keep = np.isfinite(values)
        if keep.sum() < 20:
            continue
        ax, dense, _ = sg.fill_gaps(frames[keep], values[keep], cfg.max_gap)
        observed = np.zeros(ax.size, dtype=bool)
        observed[frames[keep] - int(ax[0])] = True
        win = cfg.baseline_win if dense.size >= cfg.baseline_win else max(5, (dense.size // 2) | 1)
        det = sg.detrend(dense, cfg.smooth_win, win)
        noise = sg.hf_sigma_masked(det, observed)
        p95 = float(np.percentile(det, 95)) if det.size else 0.0
        out[name] = (int(ax[0]), det, max(float(amplitude_threshold(noise, p95, median_h, cfg)), 1e-6))
    return out, frames


def main() -> None:
    cfg = Config()
    for arg in sys.argv[1:]:
        k, _, v = arg.partition("=")
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true"))
    recs = []
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
            verdict, _ = participation(st, cfg)
            if verdict == "out" or not res.rate_hz:
                continue
            period = fps / res.rate_hz
            dets, frames = chan_det(slot.samples, cfg, fps, res.signal_source)
            if res.signal_source not in dets:
                continue
            other = {k: v for k, v in dets.items() if k != res.signal_source}
            # 其它通道各自的峰, 用来做旁证
            opp = {}
            for k, (a0, det, thr) in other.items():
                hard_min = max(2, int(fps / cfg.max_rate_hz))
                opp[k] = set(a0 + int(i) for i in sg.find_peaks(det, hard_min, thr, 0.8 * thr))
            ax0, det_main, thr_main = dets[res.signal_source]
            js = sorted(int(round(e.frame)) for e in res.events if e.kind == "jump")
            for a, b in zip(js, js[1:]):
                if b - a <= cfg.gap_ratio * period:
                    continue
                want = max(0, int(round((b - a) / period)) - 1)
                if want <= 0:
                    continue
                lo, hi = a - ax0, b - ax0
                if hi <= lo or lo < 0 or hi > det_main.size:
                    continue
                main_ev = gap_lift_evidence(det_main, lo, hi, period) / thr_main
                evs = []
                for k, (o0, odet, othr) in other.items():
                    l2, h2 = a - o0, b - o0
                    if h2 <= l2 or l2 < 0 or h2 > odet.size:
                        continue
                    evs.append(gap_lift_evidence(odet, l2, h2, period) / othr)
                maxev = max([main_ev] + evs) if evs else main_ev
                votes = sum(1 for e in evs if e >= 0.35) + (1 if main_ev >= 0.35 else 0)
                votes25 = sum(1 for e in evs if e >= 0.25) + (1 if main_ev >= 0.25 else 0)
                witness = 0
                for k, pk in opp.items():
                    if any(a - 0.35 * period <= p <= b + 0.35 * period for p in pk):
                        witness += 1
                recs.append(dict(video=os.path.basename(video)[:12], tid=slot.track_id, want=want,
                                 main=main_ev, maxev=maxev, votes=votes, votes25=votes25,
                                 other_n=len(evs), witness=witness, span=(b - a) / period))
    r = {k: np.asarray([x[k] for x in recs], dtype=np.float64)
         for k in ("want", "main", "maxev", "votes", "votes25", "other_n", "witness", "span")}
    blocked = r["main"] < cfg.fill_min_lift_frac
    tot_w = r["want"].sum()
    print(f"空档 {len(recs)} 段 / 应补 {tot_w:.0f} 次; 当前闸门放行 "
          f"{r['want'][~blocked].sum():.0f} 次({100 * (~blocked).mean():.0f}% 段)")
    print(f"被拦 {r['want'][blocked].sum():.0f} 次, 其最大证据比分位 "
          f"{np.percentile(r['maxev'][blocked], [25, 50, 75, 90]).round(2).tolist()}")
    print("\n候选闸门: 可及的缺失次数 / 旁证率(其它通道有峰的比例, 越高越可信)")
    base_w, base_wit = r["want"][~blocked].sum(), r["witness"][~blocked]
    print(f"  {'口径':<26}{'段':>6}{'新增次数':>9}{'累计次数':>9}{'旁证率':>8}")
    print(f"  {'当前(主通道>=0.35)':<26}{'-':>6}{'-':>9}{base_w:>9.0f}"
          f"{100 * np.mean(base_wit >= 1) if base_wit.size else 0:>7.0f}%")
    for name, mask in [
        ("最大通道>=0.35", r["maxev"] >= 0.35),
        ("最大通道>=0.25", r["maxev"] >= 0.25),
        ("最大通道>=0.20", r["maxev"] >= 0.20),
        (">=2通道>=0.25", r["votes25"] >= 2),
        (">=2通道>=0.20", np.asarray([sum(1 for _ in [0]) for _ in recs], dtype=np.float64) >= 0),
        ("旁证>=1通道", r["witness"] >= 1),
        ("旁证>=2通道", r["witness"] >= 2),
    ]:
        sel = mask & blocked
        w = r["want"][sel].sum()
        wit = r["witness"][sel]
        rate = 100 * np.mean(wit >= 1) if wit.size else 0.0
        print(f"  {name:<26}{sel.sum():>6}{w:>9.0f}{base_w + w:>9.0f}{rate:>7.0f}%")


if __name__ == "__main__":
    main()
