"""框高波动判据归因: 被 height_cv 剔除的轨迹, 到底是 ID 串人, 还是同一个人姿态/遮挡变化。

用法: .venv/bin/python out/tune/probe_hcv.py [key=value ...]
若结论是"同一个人", 说明这条判据在误杀, 应该换成真正检测"轨迹在不同人之间跳"的信号。
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from harness import CACHE, VIDEOS, _detections  # noqa: E402
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import count_track, participation  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402

SPARK = " .:-=+*#%@"


def sparkline(values: np.ndarray, buckets: int = 48, lo: float = 0.0, hi: float = 1.0) -> str:
    """把序列按时间分成 buckets 段, 每段取中位数画成字符图。"""
    if values.size == 0:
        return ""
    edges = np.linspace(0, values.size, buckets + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            out.append(" ")
            continue
        v = float(np.median(values[a:b]))
        ratio = 0.0 if hi <= lo else (v - lo) / (hi - lo)
        out.append(SPARK[min(len(SPARK) - 1, max(0, int(ratio * (len(SPARK) - 1))))])
    return "".join(out)


def level_shifts(heights: np.ndarray, median_h: float, step: float = 0.12) -> tuple:
    """相邻观测的高度相对跳变: 次数 / 最大幅度 / 跳变后新水平维持的帧数中位数。"""
    if heights.size < 3 or median_h <= 0:
        return 0, 0.0, 0.0
    rel = np.abs(np.diff(heights)) / median_h
    idx = np.flatnonzero(rel > step)
    if idx.size == 0:
        return 0, 0.0, 0.0
    holds = []
    for i in idx:
        after = heights[i + 1:i + 1 + int(25)]
        before = heights[max(0, i - int(25)):i + 1]
        if after.size == 0 or before.size == 0:
            continue
        shift = abs(float(np.median(after)) - float(np.median(before))) / median_h
        if shift < 0.06:
            continue
        # 新水平维持多久: 从中位数改变处开始, 数到中位数又转回去
        keep = 0
        for j in range(i + 1, heights.size):
            if abs(float(heights[j]) - float(np.median(after))) / median_h < 0.06:
                keep += 1
            else:
                break
        holds.append(keep)
    return int(idx.size), float(rel.max()), float(np.median(holds)) if holds else 0.0


def two_cluster(heights: np.ndarray, median_h: float) -> tuple:
    """把高度分成高低两档: 返回(两档中位数差 / 中位身高, 低档持续时长占比, 高档持续时长占比, 切换次数)。"""
    if heights.size < 10 or median_h <= 0:
        return 0.0, 0.0, 0.0, 0
    lo, hi = np.percentile(heights, 20), np.percentile(heights, 80)
    cut = (lo + hi) / 2
    mask = heights > cut
    segs = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    runs = np.diff(segs)
    low_ratio = float((~mask).mean())
    high_ratio = float(mask.mean())
    long_runs = [r for r in runs if r >= 25]
    stable = len(long_runs) >= 2 and low_ratio > 0.15 and high_ratio > 0.15
    return abs(hi - lo) / median_h, low_ratio, high_ratio, (len(runs) - 1) if stable else 0


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
                centers.setdefault(tid, []).append(((det.box[0] + det.box[2]) / 2,
                                                    (det.box[1] + det.box[3]) / 2))
        slots = [build_slot(t, samples[t], centers[t]) for t in sorted(samples) if len(samples[t]) >= 2]
        slots, _ = consolidate(slots, cfg)
        for slot in slots:
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            verdict, reason = participation(res.stats, cfg)
            total = res.count + res.stats.get("filled", 0)
            cv = res.stats.get("height_cv")
            if cv is None or cv <= 0.10 or total < 3:
                continue
            heights = np.array([float(s[2][3]) - float(s[2][1]) for s in slot.samples])
            cxs = np.array([float(s[2][0] + s[2][2]) / 2 for s in slot.samples])
            med_h = float(np.median(heights))
            shifts, max_rel, hold = level_shifts(heights, med_h)
            gap, low_r, high_r, switches = two_cluster(heights, med_h)
            rows.append({
                "video": os.path.basename(video)[:12], "tid": slot.track_id, "cv": cv,
                "verdict": verdict, "total": total, "period": res.period_sec or 0.0,
                "rate": total / max(0.1, res.stats.get("duration_sec", 0.1)),
                "dur": res.stats.get("duration_sec", 0), "n": heights.size,
                "hspark": sparkline(heights / med_h, lo=0.7, hi=1.3),
                "xspark": sparkline(cxs, lo=float(cxs.min()), hi=float(cxs.max())),
                "shifts": shifts, "max_rel": max_rel, "hold": hold,
                "gap": gap, "low_r": low_r, "high_r": high_r, "switches": switches,
                "reason": reason,
            })

    rows.sort(key=lambda r: (-r["total"]))
    print(f"用时 {time.time() - started:.1f}s, 配置差异: {sys.argv[1:] or '默认'}, 阈值 max_height_cv={cfg.max_height_cv}")
    print(f"共 {len(rows)} 条 height_cv>0.10 且数到 >=3 次的轨迹\n")
    print("高度字符图: ' '=矮 .- 低 += 中 *# 高 @=超高 (横轴为整条轨迹时间, 每格约等宽)")
    print("标志: shifts=相邻观测高度突变次数 hold=突变后新水平维持帧数 sw=高低两档切换次数 gap=两档差/身高\n")
    print(f"{'视频':<13}{'轨迹':>4}{'判定':>5}{'次数':>5}{'次/秒':>7}{'cv':>6}{'时长':>7}{'突变':>5}{'最大跳':>7}{'维持':>6}{'两档差':>7}{'切换':>5}  高度图 / 横向图")
    for r in rows:
        print(f"{r['video']:<13}#{r['tid']:<3}{r['verdict'][:4]:>5}{r['total']:>5}{r['rate']:>7.2f}{r['cv']:>6.2f}"
              f"{r['dur']:>7.1f}{r['shifts']:>5}{r['max_rel']:>7.2f}{r['hold']:>6.0f}{r['gap']:>7.2f}{r['switches']:>5}  {r['hspark']}")
        print(f"{'':<13}{'':<4}{'':<5}{'':<5}{'':<7}{'':<6}{'':<7}{'':<5}{'':<7}{'':<6}{'':<7}{'':<5}  {r['xspark']}  <- {r['reason']}")
    print(f"\n注: 下方每行第二行是框中心横向位置图, 若高度突变处横向同时跳, 才是真的串到另一个人身上。")


if __name__ == "__main__":
    main()
