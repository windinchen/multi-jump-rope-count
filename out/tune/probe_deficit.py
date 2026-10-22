"""缺口集中在哪: 逐条轨迹列出 实数/应有 与各种成因计数, 再按特征分桶看缺口分布。

用法: .venv/bin/python out/tune/probe_deficit.py > out/tune/DOC_deficit.log
分桶维度: 频率 / 可见率 / 轨迹时长 / 节奏来源。缺口 = 应有(expect) - (count + filled)。
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
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import count_track, participation  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


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
            verdict, _ = participation(st, cfg)
            total = res.count + int(st.get("filled", 0))
            exp = float(st.get("expected_count", 0.0))
            rows.append(dict(video=os.path.basename(video)[:12], tid=slot.track_id, verdict=verdict,
                             total=total, exp=exp, deficit=max(0.0, exp - total),
                             ratio=total / exp if exp > 0 else 0.0,
                             rate=res.rate_hz or 0.0, vis=float(st.get("visibility", 0.0)),
                             dur=float(st.get("duration_sec", 0.0)), src=st.get("period_source", ""),
                             bumps=int(st.get("bumps", 0)), filled=int(st.get("filled", 0)),
                             rec=int(st.get("recovered", 0)), est=int(st.get("estimated_misses", 0)),
                             unobs=int(st.get("misses_unobserved", 0)), holes=int(st.get("long_holes", 0)),
                             win=int(st.get("smooth_win_used", cfg.smooth_win))))

    counted = [r for r in rows if r["verdict"] != "out" and r["exp"] > 0]
    deficit_all = sum(r["deficit"] for r in counted)
    print(f"条数(计入判定) {len(counted)}  实数 {sum(r['total'] for r in counted):.0f}  "
          f"应有 {sum(r['exp'] for r in counted):.0f}  缺口 {deficit_all:.0f}")
    print(f"其中 estimated_misses 合计 {sum(r['est'] for r in counted)} / "
          f"未观测 {sum(r['unobs'] for r in counted)} / 已补 {sum(r['filled'] for r in counted)} / "
          f"bump {sum(r['bumps'] for r in counted)}")

    def bucket(title, key):
        groups = defaultdict(lambda: [0, 0.0, 0.0, 0])
        for r in counted:
            g = key(r)
            groups[g][0] += 1
            groups[g][1] += r["total"]
            groups[g][2] += r["exp"]
            groups[g][3] += 1 if r["ratio"] < 0.85 else 0
        print(f"\n## {title}")
        print(f"{'桶':<16}{'条数':>5}{'实数':>8}{'应有':>8}{'缺口':>7}{'占缺口%':>8}{'达成<85%条数':>12}")
        for g in sorted(groups):
            n, tot, exp, low = groups[g]
            d = max(0.0, exp - tot)
            print(f"{str(g):<16}{n:>5}{tot:>8.0f}{exp:>8.0f}{d:>7.0f}"
                  f"{100 * d / max(1.0, deficit_all):>8.1f}{low:>12}")

    bucket("按频率(Hz)", lambda r: "缺节奏" if r["rate"] <= 0 else f"{int(r['rate'] * 2) / 2:.1f}+")
    bucket("按可见率", lambda r: f"{int(r['vis'] * 10) / 10:.1f}+")
    bucket("按时长(s)", lambda r: f"{int(r['dur'] // 10) * 10}-{int(r['dur'] // 10) * 10 + 10}")
    bucket("按节奏来源", lambda r: r["src"])
    bucket("按滤波窗口", lambda r: f"win{r['win']}")

    counted.sort(key=lambda r: -r["deficit"])
    print("\n## 缺口最大的 20 条")
    print(f"{'video':<14}{'tid':>4}{'判定':>13}{'实数':>6}{'应有':>7}{'缺口':>7}{'达成':>7}{'Hz':>6}{'vis':>6}{'时长':>6}"
          f"{'bump':>5}{'补':>4}{'捞':>4}{'估漏':>5}{'未观':>5}{'长洞':>5}{'win':>4}")
    for r in counted[:20]:
        print(f"{r['video']:<14}{r['tid']:>4}{r['verdict']:>13}{r['total']:>6}{r['exp']:>7.0f}{r['deficit']:>7.0f}"
              f"{r['ratio']:>7.2f}{r['rate']:>6}{r['vis']:>6.2f}{r['dur']:>6.1f}{r['bumps']:>5}{r['filled']:>4}"
              f"{r['rec']:>4}{r['est']:>5}{r['unobs']:>5}{r['holes']:>5}{r['win']:>4}")


if __name__ == "__main__":
    main()
