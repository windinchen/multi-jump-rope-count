"""同组同跳证据: 多人跳绳里同一根绳上的人必须同跳, 量"群体节拍共识"能补回多少漏计。

用法: .venv/bin/python out/tune/probe_group.py [key=value ...] [--tsv]
口径:
  锁相 R = |mean(exp(i*2*pi*Δ/T))|, Δ = 两人起跳时刻之差, T = 共识周期。
  R→1 表示两人每一拍都对齐(同一根绳), R→0 表示互不相干。
  机会 = 簇内共识速率 × 该人观测时长 - 该人数到的次数(只计正数)。
"""
from __future__ import annotations

import math
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from harness import CACHE, VIDEOS, _detections  # noqa: E402
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import count_track, participation  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def lock(beats_a, beats_b, period_frames):
    """返回 (R, 命中率)。beats 是帧号数组, period_frames 是共识周期(帧)。"""
    if len(beats_a) < 4 or len(beats_b) < 4 or period_frames <= 0:
        return 0.0, 0.0
    b = np.asarray(beats_b, dtype=float)
    ang = []
    hit = 0
    for a in beats_a:
        d = b - float(a)
        i = int(np.argmin(np.abs(d)))
        if abs(d[i]) > 1.5 * period_frames:
            continue
        hit += 1
        phase = (d[i] % period_frames) / period_frames
        ang.append(2 * math.pi * phase)
    if hit < 4:
        return 0.0, hit / max(1, len(beats_a))
    r = abs(complex(*[sum(math.cos(x) for x in ang) / len(ang),
                      sum(math.sin(x) for x in ang) / len(ang)]))
    return float(r), hit / max(1, len(beats_a))


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--tsv"]
    tsv = "--tsv" in sys.argv[1:]
    cfg = Config()
    for item in args:
        key, value = item.split("=", 1)
        low = value.lower()
        cast = low == "true" if low in ("true", "false") else None
        if cast is None:
            try:
                cast = int(value)
            except ValueError:
                cast = float(value)
        setattr(cfg, key, cast)

    tot_opp = tot_kept = 0.0
    n_clusters = 0
    for video in VIDEOS:
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
        recs = []
        for slot in slots:
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            verdict = participation(res.stats, cfg)[0]
            total = res.count + res.stats.get("filled", 0)
            beats = [e.frame for e in res.events if e.kind in ("jump", "bump")]
            if verdict == "out" or total < 8 or not res.period_sec or len(beats) < 8:
                continue
            cs = centers.get(slot.track_id) or [(0.0, 0.0)]
            recs.append({
                "tid": slot.track_id, "count": float(total), "period": res.period_sec * fps,
                "beats": beats, "cx": float(np.median([c[0] for c in cs])),
                "obs": len(beats) and (beats[-1] - beats[0]) / fps or 0.0,
                "verdict": verdict,
            })
            tot_kept += total
        # 两两锁相判定
        pairs = []
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                if max(a["period"], b["period"]) / min(a["period"], b["period"]) > 1.15:
                    continue
                r, hit = lock(a["beats"], b["beats"], min(a["period"], b["period"]))
                if r >= 0.5 and hit >= 0.6:
                    pairs.append((a["tid"], b["tid"], r, hit))
        # 并查集聚簇
        parent = {r["tid"]: r["tid"] for r in recs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for x, y, _r, _h in pairs:
            if find(x) != find(y):
                parent[find(x)] = find(y)
        by = defaultdict(list)
        for r in recs:
            by[find(r["tid"])].append(r)
        print(f"\n=== {os.path.basename(video)[:12]}  fps={fps}  可参与聚类 {len(recs)} 条  锁相边 {len(pairs)} ===")
        for _k, members in sorted(by.items(), key=lambda kv: -len(kv[1])):
            if len(members) < 2:
                continue
            n_clusters += 1
            cons = float(np.median([m["period"] for m in members]))
            cx = sorted(m["cx"] for m in members)
            opp = 0.0
            print(f"簇 {len(members)} 人  共识周期 {cons:.2f}f={fps/cons:.2f}Hz  "
                  f"横向跨度 {cx[-1]-cx[0]:.0f}px  共识观测 {(min(m['obs'] for m in members)):.1f}s")
            for m in sorted(members, key=lambda m: -m["count"]):
                want = m["obs"] * fps / cons
                gap = max(0.0, want - m["count"])
                opp += gap
                print(f"   #{m['tid']:<4} {m['verdict']:<12} 数到 {m['count']:.0f}  "
                      f"周期 {m['period']:.2f}f  共识应数 {want:.0f}  缺口 {gap:.0f}")
            tot_opp += opp
            print(f"   簇内可补 {opp:.0f} 次")
    print(f"\n合计: 锁相簇 {n_clusters} 个, 共识口径可补 {tot_opp:.0f} 次 (当前计入 {tot_kept:.0f})")


if __name__ == "__main__":
    main()
