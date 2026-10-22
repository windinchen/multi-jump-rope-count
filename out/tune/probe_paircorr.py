"""同人重复判据实验: 两条轨迹若在同帧的纵向位移曲线高度相关, 就是同一个人数了两遍。

用法: .venv/bin/python out/tune/probe_paircorr.py [视频关键字]
理由: 位置/尺寸会被前后遮挡误导, 但两个人的起跳时刻曲线几乎不可能同步到 r>0.9;
      先测分布, 再决定要不要把它做成 consolidate 的合并判据。
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
from jumpcount import signal as sg  # noqa: E402
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import channel_cache, count_track, extract_y_detail, participation  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def series_for(slot, cfg, fps, source=None):
    """返回 {frame: 去基线后的纵向位移}, 与人无关的相对量, 便于跨轨迹比较。"""
    frames = np.array([int(s[0]) for s in slot.samples], dtype=np.int64)
    channels = channel_cache(slot.samples, cfg)
    src = source or "head"
    y, _ = extract_y_detail(slot.samples, src, cfg, channels=channels)
    raw = -np.where(np.isfinite(y), y, np.nan)
    keep = np.isfinite(raw)
    if keep.sum() < 12:
        return {}
    obs = frames[keep]
    axis, dense, _ = sg.fill_gaps(obs, raw[keep], cfg.max_gap)
    det = sg.detrend(dense, cfg.smooth_win, cfg.baseline_win)
    return {int(f): float(v) for f, v in zip(axis, det)}


def main() -> None:
    args = [a for a in sys.argv[1:] if "=" not in a]
    keyword = args[0] if args else ""
    cfg = Config()
    started = time.time()
    all_corr = []
    for video in VIDEOS:
        if keyword and keyword not in video:
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
        infos = []
        for slot in slots:
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            meta = {"tid": slot.track_id, "source": res.signal_source,
                    "total": res.count + res.stats.get("filled", 0),
                    "verdict": participation(res.stats, cfg)[0]}
            infos.append((meta, series_for(slot, cfg, fps, res.signal_source)))
        rows = []
        for i in range(len(infos)):
            for j in range(i + 1, len(infos)):
                a, b = infos[i][1], infos[j][1]
                common = sorted(set(a) & set(b))
                if len(common) < 40:
                    continue
                va = np.array([a[f] for f in common])
                vb = np.array([b[f] for f in common])
                if np.std(va) < 1e-6 or np.std(vb) < 1e-6:
                    continue
                r = float(np.corrcoef(va, vb)[0, 1])
                rows.append((r, len(common), infos[i][0], infos[j][0]))
        rows.sort(key=lambda r: -r[0])
        hi = [r for r in rows if r[0] >= 0.8]
        arr = np.array([r[0] for r in rows]) if rows else np.zeros(0)
        print(f"\n=== {os.path.basename(video)} 可比对轨迹对 {len(rows)} 组, 相关系数 r>=0.8 的 {len(hi)} 组")
        if arr.size:
            print(f"  r 分布: 中位={np.median(arr):.2f} p90={np.percentile(arr, 90):.2f} "
                  f"p99={np.percentile(arr, 99):.2f} max={arr.max():.2f}")
        for r, n, sa, sb in rows[:8]:
            dup = min(sa["total"], sb["total"])
            print(f"  r={r:.2f} 共视{n}帧  #{sa['tid']}({sa['verdict']},{sa['total']}次,{sa['source']})"
                  f" <-> #{sb['tid']}({sb['verdict']},{sb['total']}次,{sb['source']})  重复量约 {dup}")
        all_corr.extend(hi)
    gain = sum(min(a[2]["total"], a[3]["total"]) for a in all_corr
               if a[2]["verdict"] != "out" and a[3]["verdict"] != "out")
    print(f"\nr>=0.8 的组数 {len(all_corr)}, 若都是同人重复, 多计入约 {gain} 次, 用时 {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
