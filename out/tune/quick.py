"""改动前后的快速对照: 离线总数 + 跟踪/计数耗时 + 双计特征 + 虚高条数。

用法: .venv/bin/python out/tune/quick.py [key=value ...]
双计特征 = 相邻起跳间隔 < 0.7 倍周期的比例(一次起跳被数成两个的直接证据)。
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
from jumpcount.online import OnlineCounter  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def evaluate(cfg: Config) -> dict:
    """跑一遍 6 段素材的离线 + 实时链路, 返回第 8.2/9.3 节那套代理指标。"""
    caches = {}
    for video in VIDEOS:
        path = os.path.join(CACHE, os.path.basename(video) + ".pkl")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                caches[video] = pickle.load(fh)

    kept = prov = lost = expected = 0.0
    exp_vis = 0.0
    recovered = 0
    over = 0
    cvs: list = []
    tight = double_check = rec_tight = rec_check = 0
    track_ms = count_ms = 0.0
    online_total = online_people = 0
    online_frame_ms = []
    filled = narrow_tracks = fast_tracks = 0
    for video, data in caches.items():
        fps = data["fps"]
        t0 = time.perf_counter()
        tracker = Tracker(cfg)
        samples, centers = {}, {}
        per_frame = []
        for index in range(data["frames"]):
            a = time.perf_counter()
            pairs = tracker.update(index, _detections(data, index, cfg))
            per_frame.append((time.perf_counter() - a) * 1000.0)
            for tid, det in pairs:
                samples.setdefault(tid, []).append((index, index / fps, det.box, None, det.keypoints))
                centers.setdefault(tid, []).append(((det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2))
        track_ms += float(np.median(per_frame))
        b = time.perf_counter()
        slots = [build_slot(t, samples[t], centers[t]) for t in sorted(samples) if len(samples[t]) >= 2]
        slots, _actions = consolidate(slots, cfg)
        for slot in slots:
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            verdict, _reason = participation(res.stats, cfg)
            total = res.count + res.stats.get("filled", 0)
            recovered += int(res.stats.get("recovered", 0))
            filled += int(res.stats.get("filled", 0))
            fast_tracks += int(res.stats.get("smooth_fast_tried", 0))
            narrow_tracks += int(res.stats.get("smooth_win_used", cfg.smooth_win) < cfg.smooth_win)
            exp = res.stats.get("expected_count", 0.0)
            if verdict == "in":
                kept += total
                expected += exp
                exp_vis += float(res.stats.get("expected_visible", 0.0))
            elif verdict == "provisional":
                prov += total
                expected += exp
                exp_vis += float(res.stats.get("expected_visible", 0.0))
            else:
                lost += total
            if verdict != "out" and exp and total > 1.05 * exp:
                over += 1
            intervals = np.diff([e.frame for e in res.events if e.kind in ("jump", "bump")])
            if intervals.size >= 4:
                cvs.append(float(np.std(intervals) / max(1e-6, np.mean(intervals))))
            jumps = [e.frame for e in res.events if e.kind in ("jump", "bump")]
            marks = [bool(getattr(e, "recovered", False)) for e in res.events if e.kind in ("jump", "bump")]
            if res.period_sec:
                period = res.period_sec * fps
                for x, y in zip(jumps, jumps[1:]):
                    double_check += 1
                    if y - x < 0.7 * period:
                        tight += 1
                for i, flag in enumerate(marks):
                    if not flag:
                        continue
                    rec_check += 1
                    near = [j for j in (i - 1, i + 1) if 0 <= j < len(marks)
                            and abs(jumps[j] - jumps[i]) < 0.7 * period]
                    if near:
                        rec_tight += 1
        count_ms += (time.perf_counter() - b) * 1000.0 / data["frames"]

        # 实时链路
        tracker = Tracker(cfg)
        counter = OnlineCounter(cfg, fps)
        online_ms = []
        for index in range(data["frames"]):
            c0 = time.perf_counter()
            dets = _detections(data, index, cfg)
            counter.update(index, index / fps, [(t, d.box, d.keypoints, d.conf) for t, d in tracker.update(index, dets)])
            online_ms.append((time.perf_counter() - c0) * 1000.0)
        snaps = counter.snapshot(data["frames"])
        online_total += sum(s.get("total_count", s["count"]) for s in snaps if s["participant"])
        online_people += sum(1 for s in snaps if s["participant"])
        online_frame_ms = online_frame_ms + [float(np.median(online_ms))]

    total = kept + prov
    return {
        "kept": kept, "provisional": prov, "total": total, "expected": expected,
        "coverage": total / max(1, expected),
        "expected_visible": exp_vis,
        "coverage_visible": total / max(1, exp_vis),
        "online_total": online_total, "online_people": online_people,
        "lost": lost, "over": over, "recovered": recovered,
        "tight": tight, "double_check": double_check,
        "double_rate": tight / max(1, double_check),
        "rec_tight": rec_tight, "rec_check": rec_check,
        "rec_double_rate": rec_tight / max(1, rec_check),
        "cv": float(np.mean(cvs)) if cvs else 0.0,
        "filled": filled, "narrow_tracks": narrow_tracks, "fast_tracks": fast_tracks,
        "track_ms": track_ms, "count_ms": count_ms,
        "online_ms": float(np.mean(online_frame_ms)) if online_frame_ms else 0.0,
    }


def format_metrics(m: dict) -> str:
    return "\n".join([
        f"离线 计入 {m['kept']:.0f} + 低置信 {m['provisional']:.0f} = {m['total']:.0f} / 应有 {m['expected']:.0f}"
        f" -> 完整率 {m['coverage']:.3f}",
        f"  (可见段口径: 应有 {m['expected_visible']:.0f} -> 数到 {m['coverage_visible']:.1%};"
        f"  墙钟口径把遮挡与走路时间也算成应有, 只用于虚高判定)",
        f"实时链路 {m['online_total']} 次, 在跳绳 {m['online_people']} 人; "
        f" 剔除但仍数到 {m['lost']:.0f}, 虚高(>1.05) {m['over']} 条",
        f"回捞 {m['recovered']} 次; 双计特征: 间隔 < 0.7 周期的相邻峰 {m['tight']}/{m['double_check']}"
        f" = {m['double_rate']:.2%}; 回捞峰自身双计 {m['rec_tight']}/{m['rec_check']}"
        f" = {m['rec_double_rate']:.2%}",
        f"节奏离散度 CV(越低越稳, 均值): {m['cv']:.3f}",
        f"耗时: 跟踪 {m['track_ms']:.1f} ms/帧(中位), 离线计数 {m['count_ms']:.1f} ms/帧, "
        f"实时链路 {m['online_ms']:.1f} ms/帧(含检测输入)",
    ])


def parse_overrides(items) -> Config:
    cfg = Config()
    for item in items:
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
    print(format_metrics(evaluate(parse_overrides(sys.argv[1:]))))


if __name__ == "__main__":
    main()
