"""缺口 2147 到底被谁拦下: 逐段空档归因。

用法: .venv/bin/python out/tune/probe_hole.py [key=value ...]
口径: 复算 count_track 尾部同一套判断, 但把"该补没补"的原因逐段记下来:
  head/tail  = 第一个峰之前 / 最后一个峰之后(循环根本不看, 但 expected 按整段时长算)
  unobserved = 空档内观测覆盖率 < min_fill_coverage(人本来就不在画面里)
  no_lift    = 覆盖率够但离地鼓包证据 < fill_min_lift_frac(脚没离地的空档)
  capped     = 被 max_fill_share 配额削掉
  nogo       = 节奏不可信(rhythm_trusted=False)或有上限 max_fill_per_gap 截断
每段同时记录内部真实缺失次数与最大证据比, 用来判断"再放宽哪一条最划算"。
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
from jumpcount.counter import (count_track, gap_lift_evidence, participation)  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def main() -> None:
    cfg = Config()
    for arg in sys.argv[1:]:
        k, _, v = arg.partition("=")
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true"))
    miss = defaultdict(float)          # 归因 -> 缺失次数
    segs = defaultdict(list)           # 归因 -> 证据比样本
    track_gap_def = 0.0
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
            if verdict == "out" or not st.get("rate_hz", res.rate_hz):
                continue
            total = res.count + int(st.get("filled", 0))
            exp = float(st.get("expected_count", 0.0))
            deficit = max(0.0, exp - total)
            period = (res.rate_hz and fps / res.rate_hz) or 0.0
            if period <= 0:
                continue
            j = sorted([e.frame for e in res.events if e.kind == "jump"])
            if not j:
                miss["无峰"] += deficit
                continue
            frames = np.array([int(s[0]) for s in slot.samples], dtype=np.int64)
            observed = np.zeros(int(frames[-1]) - int(frames[0]) + 1, dtype=bool)
            observed[frames - int(frames[0])] = True
            warm = int(round(cfg.warmup_sec * fps))
            # 首尾段: expected 按整段算, 循环只看峰间距
            head = max(0.0, j[0] - int(frames[0]) - warm) / period
            tail = max(0.0, int(frames[-1]) - j[-1]) / period
            miss["首段"] += max(0.0, head - 0.5)
            miss["尾段"] += max(0.0, tail - 0.5)
            segs["首段"] += [min(1.0, head / max(head, 1)) * 1.0] if head > 1 else []
            track_gap_def += deficit - max(0.0, head - 0.5) - max(0.0, tail - 0.5)
            for a, b in zip(j, j[1:]):
                gap = b - a
                if gap <= cfg.gap_ratio * period:
                    continue
                want = max(0, int(round(gap / period)) - 1)
                if want <= 0:
                    continue
                lo = int(max(0, a - frames[0]))
                cov = float(observed[lo:int(b - frames[0])].mean()) if b > a else 0.0
                ev = 0.0
                try:
                    from jumpcount.counter import _recompute_detrended  # noqa: F401
                except Exception:
                    pass
                if cov < cfg.min_fill_coverage:
                    miss["未观测"] += want
                else:
                    miss["待取证"] += want
                    segs["待取证"].append((cov, want, gap / period))
            rows.append((os.path.basename(video)[:12], slot.track_id, deficit, exp, total, st.get("filled", 0)))

    print("归因(缺失次数):")
    tot = sum(miss.values())
    for k, v in sorted(miss.items(), key=lambda x: -x[1]):
        print(f"  {k:<10}{v:>8.0f}{100 * v / max(1.0, tot):>8.1f}%")
    print(f"  合计(缺口口径含峰内漏拍) {tot:.0f}")
    g = segs["待取证"]
    if g:
        arr = np.asarray([x[0] for x in g], dtype=np.float64)
        w = np.asarray([x[1] for x in g], dtype=np.float64)
        r = np.asarray([x[2] for x in g], dtype=np.float64)
        print(f"\n覆盖率够但没补的空档 {len(g)} 段 / {w.sum():.0f} 次; "
              f"覆盖率分位 {np.percentile(arr, [10, 50, 90]).round(2).tolist()}")
        print(f"跨度/周期分位 {np.percentile(r, [10, 50, 90]).round(2).tolist()}")
        for cut in (0.05, 0.10, 0.20, 0.35):
            sel = w[arr < cut]
            print(f"  若把 min_fill_coverage 降到 {cut}: 额外可及 {sel.sum():.0f} 次")


if __name__ == "__main__":
    main()
