"""剩余误差归因: 被剔除却已数到的次数是按哪条判据剔的, 双计的对长什么样。

用法: .venv/bin/python out/tune/probe_errors.py [key=value ...]
目的不是再调一个阈值, 而是看清楚"还有多少误差集中在哪一条判据上", 决定该修判据还是该修信号。
"""
from __future__ import annotations

import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from harness import CACHE, VIDEOS, _detections  # noqa: E402
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import count_track, participation  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def bucket(reason: str) -> str:
    """把带数字的剔除理由归并成同一类, 例如"只出现 3.2s" -> "只出现 Ns"。"""
    text = re.sub(r"[-+]?[0-9]*\.?[0-9]+", "N", reason or "(空)")
    return text.split("，")[0].split(",")[0][:40]


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
    excluded = Counter()
    excluded_counts = Counter()
    excluded_detail = []
    pair_kind = Counter()
    pair_amp = []          # 双计对里"较弱一峰 / 本条轨迹中位幅度"
    pair_position = []     # 双计对落在轨迹观测区间内的相对位置
    pair_verdict = Counter()
    guard_dropped = 0      # 假设加一道"弱峰剔除"会掉多少计数
    total_kept = 0
    tight_total = 0
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
            if verdict != "out":
                total_kept += total
            if verdict == "out" and total > 0:
                excluded[bucket(reason)] += 1
                excluded_counts[bucket(reason)] += total
                excluded_detail.append((video, slot.track_id, total, reason,
                                        res.stats.get("duration_sec"), res.stats.get("visibility"),
                                        res.stats.get("height_cv"), res.stats.get("ankle_sync"),
                                        res.stats.get("lift_ratio"), res.stats.get("rhythm_strength"),
                                        res.stats.get("excursion_ratio")))
            jumps = [e for e in res.events if e.kind in ("jump", "bump")]
            if len(jumps) < 2 or not res.period_sec:
                continue
            period = res.period_sec * fps
            amps = np.array([e.amplitude for e in jumps], dtype=float)
            median_amp = float(np.median(amps)) if amps.size else 0.0
            first, last = int(jumps[0].frame), int(jumps[-1].frame)
            span = max(1, last - first)
            for i in range(len(jumps) - 1):
                a, b = jumps[i], jumps[i + 1]
                if (int(b.frame) - int(a.frame)) >= 0.7 * period:
                    continue
                tight_total += 1
                weaker = a if a.amplitude <= b.amplitude else b
                stronger = b if a.amplitude <= b.amplitude else a
                flags = "".join("R" if jumps[j].recovered else "M" for j in (i, i + 1))
                pair_kind[flags] += 1
                pair_verdict[verdict] += 1
                if median_amp > 0:
                    pair_amp.append(weaker.amplitude / median_amp)
                if len(jumps) > 2:
                    pair_position.append((int(b.frame) - first) / span)
                if median_amp > 0 and weaker.amplitude < 0.5 * median_amp and stronger.amplitude >= 0.5 * median_amp:
                    guard_dropped += 1

    print(f"素材 {len(VIDEOS)} 段, 用时 {time.time() - started:.1f}s, 配置差异: {sys.argv[1:] or '默认'}")
    print(f"\n计入 + 低置信合计 = {total_kept}")
    print(f"\n【A 被剔除但已数到】共 {sum(excluded_counts.values())} 次, 分布在 {sum(excluded.values())} 条轨迹")
    for reason, times in excluded.most_common():
        print(f"  {excluded_counts[reason]:>5} 次 / {times:>3} 条  {reason}")
    print("\n  每类判据的样本(前 6 条): 视频 轨迹 次数 | 时长s 可见率 框高CV 双脚同步 抬脚比 节奏强度 位移比")
    shown = Counter()
    for row in sorted(excluded_detail, key=lambda r: -r[2]):
        key = bucket(row[3])
        if shown[key] >= 6:
            continue
        shown[key] += 1
        v, tid, total, reason, dur, vis, hcv, sync, lift, rhythm, exc = row
        print(f"    {os.path.basename(v)[:12]} #{tid} {total:>4} | {dur} {vis} {hcv} {sync} {lift} {rhythm} {exc}  <- {reason}")

    print(f"\n【B 双计对】共 {tight_total} 对 (间隔 < 0.7 周期)")
    print("  按来源组合 (M=实测峰, R=包络回捞峰):")
    for kind, num in pair_kind.most_common():
        print(f"    {kind}: {num}")
    print("  按所在轨迹的判定:")
    for verdict, num in pair_verdict.most_common():
        print(f"    {verdict}: {num}")
    if pair_amp:
        arr = np.array(pair_amp)
        print("  较弱一峰 / 本条轨迹中位幅度: "
              f"p10={np.percentile(arr, 10):.2f} p25={np.percentile(arr, 25):.2f} "
              f"中位={np.median(arr):.2f} p75={np.percentile(arr, 75):.2f} max={arr.max():.2f}")
        for cut in (0.3, 0.5, 0.7):
            print(f"    若把「弱峰 < {cut}x 中位幅度」的弱峰剔掉: 涉及 {int((arr < cut).sum())} 个弱峰")
    if pair_position:
        pos = np.array(pair_position)
        print(f"  双计对落在轨迹前 10% 的比例: {(pos <= 0.1).mean() * 100:.1f}%, "
              f"落在后 10% 的比例: {(pos >= 0.9).mean() * 100:.1f}% (均匀分布应是 10%/10%)")
    print(f"\n  现成的假想护栏代价: 剔掉「弱峰 < 0.5x 中位幅度且另一峰正常」的弱峰, 涉及 {guard_dropped} 个计数")

if __name__ == "__main__":
    main()
