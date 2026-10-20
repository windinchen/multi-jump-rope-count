"""身份归并: 先把"同一个人被拆成多条轨迹"和"同一帧被两条轨迹重复计数"收拾干净再数。

多人跳绳的总量误差几乎都来自这两类:
  * 重复轨迹(时间重叠)  -> 同一个人被数了两遍, 总量偏高
  * 碎片轨迹(时间接续)  -> 每个人各损失一段 warmup 与断档跳跃, 总量偏低
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config


@dataclass
class TrackSlot:
    track_id: int
    samples: List[Tuple[int, float, Tuple[float, float, float, float], float]]
    first_frame: int
    last_frame: int
    start_center: Tuple[float, float]
    end_center: Tuple[float, float]
    height: float
    conf: float
    merged_from: List[int] = field(default_factory=list)


def _overlap_frac(a: TrackSlot, b: TrackSlot) -> float:
    lo = max(a.first_frame, b.first_frame)
    hi = min(a.last_frame, b.last_frame)
    if hi <= lo:
        return 0.0
    return (hi - lo) / max(1, min(a.last_frame - a.first_frame, b.last_frame - b.first_frame))


def _gap_frames(a: TrackSlot, b: TrackSlot) -> int:
    return min(abs(b.first_frame - a.last_frame), abs(a.first_frame - b.last_frame))


def _iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlap_iou(a: TrackSlot, b: TrackSlot) -> float:
    """两条轨迹在共同出现的帧上的框重叠中位数: 同一个人的两个号几乎完全压在一起,
    而并排站的两个人水平距离再近, 框也只是挨着而不是叠着。"""
    other = {int(s[0]): s[2] for s in b.samples}
    ious = [_iou(s[2], other[int(s[0])]) for s in a.samples if int(s[0]) in other]
    if len(ious) < 3:
        return 0.0
    return float(np.median(ious))


def _dx(a: TrackSlot, b: TrackSlot) -> float:
    """用相接处的水平位置判断是不是同一个人。"""
    if a.last_frame <= b.first_frame:
        return abs(a.end_center[0] - b.start_center[0])
    return abs(b.end_center[0] - a.start_center[0])


def build_slot(track_id: int, samples: List[Tuple], centers: List[Tuple[float, float]], conf: float = 0.0) -> TrackSlot:
    heights = [s[2][3] - s[2][1] for s in samples]
    return TrackSlot(
        track_id=track_id,
        samples=samples,
        first_frame=int(samples[0][0]),
        last_frame=int(samples[-1][0]),
        start_center=tuple(centers[0]),
        end_center=tuple(centers[-1]),
        height=float(np.median(heights)) if heights else 0.0,
        conf=float(conf),
        merged_from=[track_id],
    )


def consolidate(slots: List[TrackSlot], cfg: Config) -> Tuple[List[TrackSlot], List[dict]]:
    actions: List[dict] = []
    alive = {s.track_id: s for s in slots}

    # 1) 重复轨迹抑制: 时间高度重叠 + 位置几乎重合, 保留观测更多的第一条
    pairs = []
    ids = sorted(alive)
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1 :]:
            a, b = alive[a_id], alive[b_id]
            frac = _overlap_frac(a, b)
            if frac < cfg.dup_overlap:
                continue
            ref = max(1.0, min(a.height, b.height))
            dx = abs((a.start_center[0] + a.end_center[0]) / 2 - (b.start_center[0] + b.end_center[0]) / 2)
            if dx > cfg.dup_dist_ratio * ref:
                continue
            iou = overlap_iou(a, b)
            if iou < cfg.dup_iou_min:
                continue
            pairs.append((frac, a_id, b_id, dx, iou))
    pairs.sort(key=lambda item: (-item[4], -item[0]))
    for frac, a_id, b_id, dx, iou in pairs:
        if a_id not in alive or b_id not in alive:
            continue
        a, b = alive[a_id], alive[b_id]
        loser, winner = (a, b) if len(a.samples) < len(b.samples) else (b, a)
        alive.pop(loser.track_id)
        actions.append(
            {
                "action": "drop_duplicate",
                "dropped": loser.track_id,
                "kept": winner.track_id,
                "overlap": round(frac, 3),
                "dx": round(dx, 1),
                "iou": round(iou, 3),
                "dropped_counts_samples": len(loser.samples),
            }
        )

    # 2) 碎片缝合(并查集): 时间接续 + 位置连续 + 尺寸相近
    ids = sorted(alive)
    parent = {i: i for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1 :]:
            a, b = alive[a_id], alive[b_id]
            if _overlap_frac(a, b) > 0.25:
                continue
            gap = _gap_frames(a, b)
            if gap > cfg.merge_gap_frames:
                continue
            ref = max(1.0, min(a.height, b.height))
            dx = _dx(a, b)
            ratio = max(a.height, b.height) / max(1.0, min(a.height, b.height))
            if dx <= cfg.merge_dist_ratio * ref and ratio <= 1.8:
                union(a_id, b_id)
                actions.append(
                    {
                        "action": "stitch_fragment",
                        "tracks": [a_id, b_id],
                        "gap_frames": gap,
                        "dx": round(dx, 1),
                    }
                )

    groups: Dict[int, List[int]] = {}
    for tid in ids:
        groups.setdefault(find(tid), []).append(tid)

    merged: List[TrackSlot] = []
    for root, members in sorted(groups.items()):
        members = sorted(members, key=lambda t: alive[t].first_frame)
        base = alive[members[0]]
        if len(members) == 1:
            merged.append(base)
            continue
        seen: Dict[int, Tuple] = {}
        for tid in members:
            for s in alive[tid].samples:
                seen.setdefault(int(s[0]), s)
        samples = [seen[k] for k in sorted(seen)]
        slot = TrackSlot(
            track_id=root,
            samples=samples,
            first_frame=samples[0][0],
            last_frame=samples[-1][0],
            start_center=alive[members[0]].start_center,
            end_center=alive[members[-1]].end_center,
            height=float(np.median([alive[t].height for t in members])),
            conf=float(np.mean([alive[t].conf for t in members])),
            merged_from=members,
        )
        merged.append(slot)
    return merged, actions
