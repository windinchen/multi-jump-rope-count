"""多通道并集/投票计数: 同一条轨迹用踝/髋/头/框底边各数一遍, 再按节拍合并。

用法: .venv/bin/python out/tune/probe_union.py [key=value ...] [--top N]
口径:
  any  = 任一通道数到的事件都算(召回上限, 噪声也会进来)
  vote = 至少 N 个通道在同一拍(容差 0.35 周期)都数到才算(召回与精度的折中)
  合并: 事件间隔 < 0.7 周期的只留一个, 用主通道的周期做尺子。
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
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import count_track, participation  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402

KINDS = ("jump", "bump", "interpolated")
CH = ("ankle", "hip", "head", "box_bottom")


def merge(events, period, merge_ratio=0.7):
    out = []
    for f in sorted(events):
        if out and f - out[-1] < merge_ratio * period:
            continue
        out.append(f)
    return out


def vote_events(ch_ev, period, need, tol_ratio, merge_ratio=0.7):
    allf = sorted({f for ev in ch_ev.values() for f in ev})
    out = []
    for f in allf:
        agree = sum(1 for ev in ch_ev.values() if any(abs(g - f) <= tol_ratio * period for g in ev))
        if agree >= need:
            out.append(f)
    return merge(out, period, merge_ratio)


def stats_of(events, period):
    if len(events) < 5 or period <= 0:
        return 0.0, 0.0
    d = np.diff(events)
    return float((d < 0.7 * period).sum()) / max(1, len(d)), float(np.std(d) / max(1e-6, np.mean(d)))


def parse(argv):
    cfg = Config()
    for item in argv:
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
    return cfg


def main() -> None:
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 0
    skip = {sys.argv.index("--top") + 1} if "--top" in sys.argv else set()
    args = [a for i, a in enumerate(sys.argv[1:], start=1) if not a.startswith("--") and i not in skip]
    cfg = parse(args)
    rows = []
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
        for slot in slots:
            base = count_track(slot.track_id, slot.samples, cfg, fps)
            verdict = participation(base.stats, cfg)[0]
            if verdict == "out":
                continue
            base_total = base.count + base.stats.get("filled", 0)
            period = (base.period_sec or 0.0) * fps
            if period <= 0 or base_total < 5:
                continue
            ch_ev = {}
            for src in CH:
                if src == base.signal_source:
                    continue
                try:
                    r = count_track(slot.track_id, slot.samples, cfg, fps, source=src)
                except Exception:
                    continue
                ev = [e.frame for e in r.events if e.kind in KINDS]
                if len(ev) >= 3:
                    ch_ev[src] = ev
            main_ev = [e.frame for e in base.events if e.kind in KINDS]
            ch_ev[base.signal_source] = main_ev
            u_any = merge([f for ev in ch_ev.values() for f in ev], period)
            u_v2 = vote_events(ch_ev, period, 2, 0.35)
            b_dr, b_cv = stats_of(sorted(main_ev), period)
            a_dr, a_cv = stats_of(u_any, period)
            v_dr, v_cv = stats_of(u_v2, period)
            rows.append({"video": os.path.basename(video)[:12], "tid": slot.track_id,
                         "verdict": verdict, "period": period, "src": base.signal_source,
                         "n_ch": len(ch_ev), "base": base_total, "any": len(u_any), "vote2": len(u_v2),
                         "b_dr": b_dr, "a_dr": a_dr, "v_dr": v_dr,
                         "b_cv": b_cv, "a_cv": a_cv, "v_cv": v_cv,
                         "detail": {k: len(v) for k, v in sorted(ch_ev.items())}})

    tot = lambda k: sum(r[k] for r in rows)
    print(f"条数 {len(rows)}  主通道口径 {tot('base')}  any并集 {tot('any')} ({tot('any')-tot('base'):+d}) "
          f" 2/投票 {tot('vote2')} ({tot('vote2')-tot('base'):+d})")
    for tag, name in (("b", "基础"), ("a", "any "), ("v", "投票")):
        dr = [r[f"{tag}_dr"] for r in rows if r[f"{tag}_cv"]]
        cv = [r[f"{tag}_cv"] for r in rows if r[f"{tag}_cv"]]
        print(f"  {name}: 双计特征均值 {np.mean(dr):.3%}  CV均值 {np.mean(cv):.3f}")
    gain = sorted(rows, key=lambda r: -(r["vote2"] - r["base"]))
    k = top or 12
    print(f"\n投票口径增益最大的 {k} 条:")
    for r in gain[:k]:
        print(f"  {r['video']} #{r['tid']:<4} {r['verdict']:<12} 主{r['src']:<11}"
              f"基础 {r['base']:<4} any {r['any']:<4} 投票 {r['vote2']:<4} 各通道 {r['detail']} "
              f"双计 {r['b_dr']:.2%}->{r['v_dr']:.2%}")
    print(f"\n投票口径损失最大的 {k} 条:")
    for r in gain[::-1][:k]:
        print(f"  {r['video']} #{r['tid']:<4} {r['verdict']:<12} 主{r['src']:<11}"
              f"基础 {r['base']:<4} any {r['any']:<4} 投票 {r['vote2']:<4} 各通道 {r['detail']}")


if __name__ == "__main__":
    main()
