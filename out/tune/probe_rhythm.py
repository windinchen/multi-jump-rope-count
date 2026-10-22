"""节律强度到底测的是什么: 现行取带内自相关最大值(实测常被最低滞后钉住), 换成"实测周期处的自相关"再比。

用法: .venv/bin/python out/tune/probe_rhythm.py [视频关键字]
判据: keep_rhythm_strength=0.35 决定"幅度弱但节奏清楚"能不能低置信计入, 还决定在线要不要按节奏补数。
     如果这个数其实跟"跳得有没有节奏"无关, 那这道闸就是在拿噪声发资格。
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
from jumpcount.counter import channel_cache, count_track, extract_y_detail  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def ac_curve(values: np.ndarray) -> np.ndarray:
    x = values - values.mean()
    n = x.size
    spec = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    return ac / ac[0] if ac[0] > 0 else np.zeros(n)


def main() -> None:
    kw = sys.argv[1] if len(sys.argv) > 1 else ""
    cfg = Config()
    rows = []
    for video in VIDEOS:
        if kw and kw not in video:
            continue
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
                centers.setdefault(tid, []).append(((det.box[0] + det.box[2]) / 2,
                                                    (det.box[1] + det.box[3]) / 2))
        slots = [build_slot(t, samples[t], centers[t]) for t in sorted(samples) if len(samples[t]) >= 2]
        slots, _ = consolidate(slots, cfg)
        for slot in slots:
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            st = res.stats
            if st.get("reject") or res.period_sec is None:
                continue
            frames = np.array([int(s[0]) for s in slot.samples], dtype=np.int64)
            channels = channel_cache(slot.samples, cfg)
            y, _ = extract_y_detail(slot.samples, res.signal_source, cfg, channels=channels)
            raw = -np.where(np.isfinite(y), y, np.nan)
            keep = np.isfinite(raw)
            axis, dense, _ = sg.fill_gaps(frames[keep], raw[keep], cfg.max_gap)
            det = sg.detrend(dense, cfg.smooth_win,
                             cfg.baseline_win if dense.size >= cfg.baseline_win else 45)
            if det.size < 40:
                continue
            ac = ac_curve(det)
            period = res.period_sec * fps
            lag = int(round(period))
            at_period = float(ac[lag]) if 0 < lag < ac.size else 0.0
            band_lo = max(2, int(fps / cfg.max_rate_hz))
            band_hi = min(ac.size - 1, int(fps / cfg.min_rate_hz))
            band = ac[band_lo:band_hi + 1] if band_hi > band_lo else np.array([0.0])
            # 峰间隔一致性: 完全不依赖自相关的节律度量(实测起跳间隔的离散程度)
            gaps = np.diff([e.frame for e in res.events if e.kind in ("jump", "bump")])
            consist = (float(np.clip(1.0 - np.median(np.abs(gaps - period)) / max(period, 1e-6), 0, 1))
                       if gaps.size >= 4 else None)
            rows.append({
                "video": os.path.basename(video)[:12], "tid": slot.track_id,
                "count": st.get("total_count", 0), "verdict": st.get("verdict"),
                "cur": st.get("rhythm_strength"), "at_period": round(at_period, 3),
                "band_max": round(float(band.max()), 3), "band_med": round(float(np.median(band)), 3),
                "consist": round(consist, 3) if consist is not None else None,
                "period": round(period, 1), "src": st.get("period_source"), "lag": lag,
            })
    print(f"{'视频':<14}{'轨迹':>5}{'计数':>6}{'判定':>13}{'现行rhy':>9}{'周期处ac':>10}{'带内最大':>10}"
          f"{'带内中位':>10}{'间隔一致':>10}{'周期帧':>8}{'来源':>14}")
    for r in sorted(rows, key=lambda r: -(r["cur"] or 0))[:20]:
        print(f"{r['video']:<14}{r['tid']:>5}{r['count']:>6}{r['verdict']:>13}{(r['cur'] or 0):>9.2f}"
              f"{r['at_period']:>10.2f}{r['band_max']:>10.2f}{r['band_med']:>10.2f}"
              f"{(r['consist'] if r['consist'] is not None else -1):>10.2f}{r['period']:>8.1f}{r['src']:>14}")
    cur = np.array([r["cur"] or 0.0 for r in rows])
    newp = np.array([r["at_period"] for r in rows])
    cons = np.array([-1.0 if r["consist"] is None else r["consist"] for r in rows])
    print(f"\n轨迹 {len(rows)} 条")
    print(f"  现行 rhythm_strength ≥0.35: {int((cur >= 0.35).sum())} 条")
    print(f"  周期处 ac ≥0.35:        {int((newp >= 0.35).sum())} 条")
    print(f"  间隔一致性 ≥0.35:       {int((cons >= 0.35).sum())} 条")
    def corr(a, b):
        m = (a >= 0) & (b >= 0)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else float("nan")
    print(f"  相关: 现行 vs 周期处 ac = {corr(cur, newp):.3f}; 现行 vs 间隔一致性 = {corr(cur, cons):.3f}; "
          f"周期处 ac vs 间隔一致性 = {corr(newp, cons):.3f}")
    both = [r for r in rows if (r["cur"] or 0) >= 0.35 and r["at_period"] < 0.35]
    print(f"  现行给了资格、按周期处 ac 不该给的: {len(both)} 条 / {sum(r['count'] for r in both)} 次")
    gain = [r for r in rows if (r["cur"] or 0) < 0.35 <= r["at_period"]]
    print(f"  现行卡掉、按周期处 ac 该给的: {len(gain)} 条 / {sum(r['count'] for r in gain)} 次")


if __name__ == "__main__":
    main()
