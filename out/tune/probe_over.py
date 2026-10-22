""""虚高"的到底是补过头, 还是应有次数被算少了: 逐条列出 total > 1.05×expected 的轨迹。

用法: .venv/bin/python out/tune/probe_over.py [key=value ...]
关键区分: 若这些轨迹的实测频率明显高于按周期算出来的节奏(周期被锁成两倍), 那是 expected 错了;
若频率正常而 filled 很多, 才是补数补过了头。
"""
from __future__ import annotations

import os
import pickle
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from harness import CACHE, VIDEOS, _detections  # noqa: E402
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import count_track, participation  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def main() -> None:
    cfg = Config()
    for arg in sys.argv[1:]:
        k, _, v = arg.partition("=")
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true"))
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
            verdict, _ = participation(st, cfg)
            if verdict == "out":
                continue
            total = res.count + int(st.get("filled", 0))
            exp = float(st.get("expected_count", 0.0))
            if exp <= 0:
                continue
            gaps = [e2 - e1 for e1, e2 in zip([e.frame for e in res.events if e.kind == "jump"],
                                              [e.frame for e in res.events if e.kind == "jump"][1:])]
            rows.append(dict(video=os.path.basename(video)[:12], tid=slot.track_id, ratio=total / exp,
                             total=total, exp=exp, direct=res.count, filled=int(st.get("filled", 0)),
                             rate=res.rate_hz or 0.0, src=st.get("period_source", ""),
                             gap_med=float(np.median(gaps)) if gaps else 0.0,
                             period=(fps / res.rate_hz) if res.rate_hz else 0.0,
                             capped=int(st.get("fill_capped", 0)), nch=int(st.get("evidence_channels", 0))))
    rows.sort(key=lambda r: -r["ratio"])
    print(f"{'video':<14}{'tid':>4}{'比值':>7}{'实数':>6}{'应有':>7}{'实测':>6}{'补':>5}{'Hz':>7}"
          f"{'实测峰隔':>9}{'周期':>8}{'截断':>5}{'取证道':>7}")
    for r in rows[:14]:
        print(f"{r['video']:<14}{r['tid']:>4}{r['ratio']:>7.2f}{r['total']:>6}{r['exp']:>7.0f}{r['direct']:>6}"
              f"{r['filled']:>5}{r['rate']:>7}{r['gap_med']:>9.1f}{r['period']:>8.1f}{r['capped']:>5}{r['nch']:>7}")
    over = [r for r in rows if r["ratio"] > 1.05]
    print(f"\n虚高(>1.05) {len(over)} 条 / 共 {len(rows)} 条; 其中补数占比 "
          f"{sum(r['filled'] for r in over)}/{sum(r['total'] for r in over)}")
    print("周期 vs 实测峰间隔(锁相时峰隔≈2×周期):",
          Counter(round(r["gap_med"] / max(r["period"], 1e-6)) for r in over if r["period"] > 0))
    print("补出来的次数总计", sum(r["filled"] for r in over), "; 被上限截断总计", sum(r["capped"] for r in rows))


if __name__ == "__main__":
    main()
