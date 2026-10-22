"""视频 -> 检测 -> 跟踪 -> 逐人计数 -> 准确性审计。"""

from __future__ import annotations

import csv
import json
import os
import time as _time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .audit import audit
from .config import Config
from .counter import OUT, PROVISIONAL, candidate_values, count_track, participation
from .detect import BaseDetector, build_detector
from .identity import build_slot, consolidate
from .render import draw
from .tracking import Tracker
from .types import TrackResult


def _max_iou_among(dets) -> float:
    best = 0.0
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            a, b = dets[i].box, dets[j].box
            iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = iw * ih
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            if ua > 0:
                best = max(best, inter / ua)
    return best


def run(
    video: str,
    cfg: Config,
    detector: BaseDetector,
    max_frames: Optional[int] = None,
    progress: bool = False,
) -> dict:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 1e-3 or fps > 240:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    frame_height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)

    tracker = Tracker(cfg)
    samples: Dict[int, List[Tuple[int, float, Tuple[float, float, float, float], float]]] = {}
    sources: Dict[int, str] = {}
    centers: Dict[int, List[Tuple[float, float]]] = {}
    frame_records: List[Tuple[int, float, List[Tuple[int, Tuple[float, float, float, float]]]]] = []
    crowd: List[float] = []

    index = 0
    started = _time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = index / fps
        dets = detector.detect(frame, index, t)
        pairs = tracker.update(index, dets)
        records: List[Tuple[int, Tuple[float, float, float, float]]] = []
        for track_id, det in pairs:
            cands = candidate_values(det.keypoints, det.box, cfg)
            if not cands:
                continue
            samples.setdefault(track_id, []).append(
                (index, t, tuple(float(v) for v in det.box), cands.get("box_bottom"), det.keypoints)
            )
            cx = (det.box[0] + det.box[2]) / 2.0
            cy = (det.box[1] + det.box[3]) / 2.0
            centers.setdefault(track_id, []).append((cx, cy))
            records.append((track_id, tuple(float(v) for v in det.box)))
        frame_records.append((index, t, records))
        if len(dets) > 1:
            crowd.append(_max_iou_among(dets))
        index += 1
        if progress and index % 100 == 0:
            print(f"  已处理 {index} 帧 (检测框 {len(dets)})", flush=True)
        if max_frames and index >= max_frames:
            break
    cap.release()

    meta_all = tracker.track_meta()
    slots = [
        build_slot(tid, samples[tid], centers[tid], meta_all.get(tid, {}).get("mean_conf", 0.0))
        for tid in sorted(samples)
        if len(samples[tid]) >= 2 and len(centers.get(tid, [])) >= 2
    ]
    slots, identity_actions = consolidate(slots, cfg)

    results: List[TrackResult] = []
    excluded: List[dict] = []
    provisional: List[dict] = []
    for slot in slots:
        res = count_track(slot.track_id, slot.samples, cfg, fps)
        sources[slot.track_id] = res.signal_source
        res.stats["merged_from"] = list(slot.merged_from)
        res.stats["fragments"] = len(slot.merged_from)
        res.stats["track_mean_conf"] = round(float(slot.conf), 4)
        boxes = np.array([s[2] for s in slot.samples], dtype=np.float64)
        res.stats["center_x_mean"] = round(float(((boxes[:, 0] + boxes[:, 2]) / 2).mean()), 2)
        verdict, reason = participation(res.stats, cfg)
        res.stats["verdict"] = verdict
        res.stats["participant"] = verdict != OUT
        if verdict == OUT:
            res.stats["excluded"] = reason
            excluded.append({"track_id": res.track_id, "count": res.count, "reason": reason})
        elif verdict == PROVISIONAL:
            res.stats["provisional_reason"] = reason
            provisional.append({"track_id": res.track_id, "count": res.count, "reason": reason})
        results.append(res)

    counted = [r for r in results if "excluded" not in r.stats]
    confident = [r for r in counted if r.stats.get("verdict") != PROVISIONAL]
    expected = round(float(sum(r.stats.get("expected_count", 0.0) for r in counted)), 1)
    # 另一个口径: 只算"人真在画面里的那段"应有的次数(见 counter.count_track 里的注释),
    # 墙钟口径的 expected_by_rhythm 继续用于虚高核对, 两个都写进 counts.json。
    expected_visible = round(float(sum(r.stats.get("expected_visible", 0.0) for r in counted)), 1)

    elapsed = _time.time() - started
    summary = {
        "video": os.path.abspath(video),
        "fps": round(float(fps), 4),
        "width": frame_width,
        "height": frame_height,
        "frames": index,
        "duration": round(index / fps, 3),
        "detector": detector.name,
        "policy": cfg.policy,
        "signal_sources": {str(k): v for k, v in sources.items()},
        "participants": len(counted),
        "participants_confident": len(confident),
        "total_jumps": int(sum(r.count + r.stats.get("filled", 0) for r in counted)),
        "total_jumps_direct": int(sum(r.count for r in counted)),
        "total_jumps_filled": int(sum(r.stats.get("filled", 0) for r in counted)),
        "total_jumps_confident": int(sum(r.count + r.stats.get("filled", 0) for r in confident)),
        "total_rope_passes": int(sum(r.count_rope for r in counted)),
        "expected_by_rhythm": expected,
        "rhythm_coverage": (round(int(sum(r.count + r.stats.get("filled", 0) for r in counted)) / expected, 3)
                            if expected > 0 else 0.0),
        "expected_by_rhythm_visible": expected_visible,
        "rhythm_coverage_visible": (round(int(sum(r.count + r.stats.get("filled", 0) for r in counted))
                                          / expected_visible, 3) if expected_visible > 0 else 0.0),
        "filled_by_phase": int(sum(r.stats.get("fill_phase", 0) for r in counted)),
        "fills_capped": int(sum(r.stats.get("fill_capped", 0) for r in counted)),
        "track_count": len(results),
        "excluded_tracks": excluded,
        "provisional_tracks": provisional,
        "identity_actions": identity_actions,
        "elapsed_sec": round(elapsed, 2),
        "fps_processed": round(index / elapsed, 2) if elapsed > 0 else 0.0,
        "crowding": {
            "mean_max_iou": round(float(np.mean(crowd)), 4) if crowd else 0.0,
            "p95_max_iou": round(float(np.percentile(crowd, 95)), 4) if crowd else 0.0,
        },
        "tracking_events": list(tracker.events),
        "config": cfg.to_dict(),
        "tracks": [_track_to_dict(r) for r in results],
    }
    summary["audit"] = audit(summary, cfg)
    summary["_frame_records"] = frame_records
    summary["_slot_samples"] = {slot.track_id: slot.samples for slot in slots}
    summary["_render_fps"] = float(fps)
    return summary


def _track_to_dict(res: TrackResult) -> dict:
    data = asdict(res)
    data["events"] = [asdict(e) for e in res.events]
    return data


def write_outputs(summary: dict, out_dir: str, save_video: bool = True) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    light = {k: v for k, v in summary.items() if not k.startswith("_")}
    paths["counts"] = os.path.join(out_dir, "counts.json")
    with open(paths["counts"], "w", encoding="utf-8") as fh:
        json.dump(light, fh, ensure_ascii=False, indent=2)

    paths["events"] = os.path.join(out_dir, "events.csv")
    with open(paths["events"], "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["track_id", "event_index", "time_sec", "frame", "amplitude", "kind", "confidence"])
        for trk in summary["tracks"]:
            for i, ev in enumerate(trk["events"]):
                writer.writerow([trk["track_id"], i, ev["time"], ev["frame"], ev["amplitude"], ev["kind"], ev["confidence"]])

    paths["audit"] = os.path.join(out_dir, "audit.json")
    with open(paths["audit"], "w", encoding="utf-8") as fh:
        json.dump(light["audit"], fh, ensure_ascii=False, indent=2)

    if save_video and summary.get("_frame_records"):
        paths["video"] = os.path.join(out_dir, "annotated.mp4")
        _render_video(summary, paths["video"])
    return paths


def counted_track_ids(summary: dict) -> set:
    """参与判定后仍然计数的轨迹号: 标注视频与 counts.json 必须按同一批人算总量。"""
    return {trk["track_id"] for trk in summary["tracks"] if trk.get("stats", {}).get("verdict") != "out"}


def _render_video(summary: dict, path: str) -> None:
    cap = cv2.VideoCapture(summary["video"])
    fps = summary.get("_render_fps") or summary["fps"] or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    running: Dict[int, int] = {}
    # 被参与判定剔除的路人也出现在画面里, 但不给他们排数字队列: 以前他们的数字一路涨,
    # 标注视频上看到的总次数比 counts.json 里报出来的多出一截, 复盘时反而以为计数丢了。
    counted = counted_track_ids(summary)
    queues = {
        trk["track_id"]: [ev["time"] for ev in trk["events"] if ev["kind"] in ("jump", "bump", "interpolated")]
        for trk in summary["tracks"] if trk["track_id"] in counted
    }
    pointers = {k: 0 for k in queues}
    for frame_index, t, records in summary["_frame_records"]:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, times in queues.items():
            p = pointers[tid]
            while p < len(times) and times[p] <= t:
                running[tid] = running.get(tid, 0) + 1
                p += 1
            pointers[tid] = p
        writer.write(draw(frame, records, running, fps, t, counted=counted))
    cap.release()
    writer.release()
