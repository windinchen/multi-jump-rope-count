"""查"双摇(bump)"到底吃掉了多少次真起跳: 逐条统计事件类型, 并量相邻两峰之间有没有真的落回地面。

用法: .venv/bin/python out/tune/probe_bump.py [视频关键字]
判读: 若被判成 bump 的峰对之间"落回了地面"(谷底/较小峰 很低), 那是两次真起跳被当成双摇扣掉了;
     谷底一直是高位才说明真是同一次起跳里绳过了两次。
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


def detrend_of(slot, cfg, fps, source):
    frames = np.array([int(s[0]) for s in slot.samples], dtype=np.int64)
    channels = channel_cache(slot.samples, cfg)
    y, _ = extract_y_detail(slot.samples, source, cfg, channels=channels)
    raw = -np.where(np.isfinite(y), y, np.nan)
    keep = np.isfinite(raw)
    axis, dense, _ = sg.fill_gaps(frames[keep], raw[keep], cfg.max_gap)
    return axis, sg.detrend(dense, cfg.smooth_win, cfg.baseline_win if dense.size >= cfg.baseline_win else 45)



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
            if res.stats.get("verdict") == "out":
                continue
            axis, det = detrend_of(slot, cfg, fps, res.signal_source)
            jumps = [e for e in res.events if e.kind in ("jump", "bump")]
            ratios = []
            for a, b in zip(jumps, jumps[1:]):
                if b.kind != "bump":
                    continue
                lo, hi = int(round(a.frame)), int(round(b.frame))
                if hi - lo < 2 or hi >= axis.size or lo < 0:
                    continue
                seg = det[lo:hi + 1]
                if seg.size < 3:
                    continue
                base = min(det[lo], det[hi])
                if base <= 0.5:
                    continue
                ratios.append(float(np.min(seg[1:-1])) / base)
            rows.append({
                "video": os.path.basename(video)[:12], "tid": slot.track_id,
                "jump": sum(1 for e in res.events if e.kind == "jump"),
                "bump": sum(1 for e in res.events if e.kind == "bump"),
                "filled": int(res.stats.get("filled", 0)),
                "rec": int(res.stats.get("recovered", 0)),
                "rate": res.rate_hz or 0.0,
                "ratios": ratios,
            })
    all_ratios = np.array([r for row in rows for r in row["ratios"]])
    total_jump = sum(r["jump"] for r in rows)
    total_bump = sum(r["bump"] for r in rows)
    print(f"计入轨迹 {len(rows)} 条: 起跳 {total_jump}, 双摇扣掉 {total_bump} "
          f"({100 * total_bump / max(1, total_jump + total_bump):.1f}%), 补齐 {sum(r['filled'] for r in rows)}, "
          f"回捞 {sum(r['rec'] for r in rows)}")
    if all_ratios.size:
        qs = [5, 10, 25, 50, 75, 90, 95]
        line = "  ".join(f"p{q}={np.percentile(all_ratios, q):.2f}" for q in qs)
        print(f"双摇峰对的 谷底/较小峰 分布 (n={all_ratios.size}): {line}")
        for thr in (0.2, 0.3, 0.4, 0.5):
            n = int((all_ratios <= thr).sum())
            print(f"  谷底 ≤{thr:.1f}(真落回地面 → 应算两次起跳): {n} 对 / {all_ratios.size}")
    print("\n按频率分组的双摇占比:")
    groups = [(0, 1.5), (1.5, 2.5), (2.5, 3.2), (3.2, 9)]
    for lo, hi in groups:
        g = [r for r in rows if lo <= r["rate"] < hi]
        if not g:
            continue
        j = sum(r["jump"] for r in g)
        b = sum(r["bump"] for r in g)
        rr = np.array([x for r in g for x in r["ratios"]])
        med = f"{np.median(rr):.2f}" if rr.size else "-"
        print(f"  {lo:.1f}~{hi:.1f}Hz: {len(g)}条 起跳{j} 双摇{b} ({100 * b / max(1, j + b):.1f}%) 谷底中位{med}")


if __name__ == "__main__":
    main()
