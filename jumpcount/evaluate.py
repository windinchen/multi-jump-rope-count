"""预测计数 vs 人工计数的准确性评估。"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config

GT_TEMPLATE = {
    "video": "clip01.mp4",
    "fps": 30,
    "persons": [
        {"name": "P1", "center_x": 0.22, "count": 62, "events": [0.41, 0.78, 1.12]},
        {"name": "P2", "center_x": 0.63, "count": 55},
    ],
}


def load_ground_truth(path: str) -> dict:
    """支持 JSON 与 CSV 两种人工标注。CSV 列: name,count[,center_x][,events(分号分隔秒)]。"""
    if path.endswith(".csv"):
        persons = []
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                name = (row.get("name") or row.get("person") or f"P{len(persons) + 1}").strip()
                count = int(float(row["count"])) if row.get("count") not in (None, "") else None
                cx = float(row["center_x"]) if row.get("center_x") not in (None, "") else None
                raw = (row.get("events") or "").strip()
                events = [float(x) for x in raw.replace(",", ";").split(";") if x.strip()] if raw else None
                persons.append({"name": name, "count": count, "center_x": cx, "events": events})
        return {"persons": persons}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        data = {"persons": data}
    for p in data.setdefault("persons", []):
        p.setdefault("center_x", None)
        p.setdefault("events", None)
        if p.get("count") is None and p.get("events") is not None:
            p["count"] = len(p["events"])
    return data


def blank_template(path: str, names: int = 3) -> str:
    data = {"persons": [{"name": f"P{i+1}", "center_x": None, "count": None, "events": None} for i in range(names)]}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return path


def _match(pred_tracks: List[dict], persons: List[dict], width: float, cfg: Config):
    """按水平位置把预测轨迹对上人工标注的人。位置缺失时退化为按计数排序对齐。"""
    usable_pos = width > 0 and all(p.get("center_x") is not None for p in persons) and all(
        t["stats"].get("center_x_mean") is not None for t in pred_tracks
    )
    pairs: List[Tuple[dict, dict, float]] = []
    used_p: set = set()
    used_t: set = set()
    if usable_pos:
        cand = []
        for ti, trk in enumerate(pred_tracks):
            fx = trk["stats"]["center_x_mean"] / width
            for pi, person in enumerate(persons):
                cand.append((abs(fx - float(person["center_x"])), ti, pi))
        cand.sort()
        for dist, ti, pi in cand:
            if ti in used_t or pi in used_p or dist > 0.22:
                continue
            used_t.add(ti)
            used_p.add(pi)
            pairs.append((pred_tracks[ti], persons[pi], dist))
    else:
        order_t = sorted(range(len(pred_tracks)), key=lambda i: -pred_tracks[i]["count"])
        order_p = sorted(range(len(persons)), key=lambda i: -(persons[i].get("count") or 0))
        for ti, pi in zip(order_t, order_p):
            used_t.add(ti)
            used_p.add(pi)
            pairs.append((pred_tracks[ti], persons[pi], 0.0))
    left_t = [t for i, t in enumerate(pred_tracks) if i not in used_t]
    left_p = [p for i, p in enumerate(persons) if i not in used_p]
    return pairs, left_t, left_p, usable_pos


def _event_metrics(pred_events: List[float], gt_events: List[float], tol: float):
    """贪心最近匹配(按 GT 顺序优先), 返回 TP/FP/FN 与时间偏差。"""
    remaining = list(pred_events)
    tp = 0
    biases: List[float] = []
    for g in sorted(gt_events):
        if not remaining:
            break
        deltas = [(abs(p - g), i) for i, p in enumerate(remaining)]
        deltas.sort()
        if deltas and deltas[0][0] <= tol:
            i = deltas[0][1]
            biases.append(remaining[i] - g)
            remaining.pop(i)
            tp += 1
    fn = len(gt_events) - tp
    return tp, len(remaining), fn, biases


def evaluate(pred: dict, gt: dict, cfg: Config) -> dict:
    tracks = [t for t in pred.get("tracks", [])]
    persons = [p for p in gt.get("persons", []) if (p.get("count") is not None) or p.get("events")]
    width = float(pred.get("width") or 0.0)
    pairs, extra_tracks, missing_persons, by_position = _match(tracks, persons, width, cfg)

    rows: List[dict] = []
    for trk, person, dist in pairs:
        gt_count = int(person["count"] if person["count"] is not None else len(person["events"] or []))
        pred_count = int(trk["count"])
        period = trk.get("period_sec") or (1.0 / max(cfg.min_rate_hz, 0.001))
        tol = max(cfg.tolerance_sec, cfg.tolerance_ratio * period)
        row = {
            "person": person["name"],
            "track_id": trk["track_id"],
            "gt_count": gt_count,
            "pred_count": pred_count,
            "error": pred_count - gt_count,
            "abs_error": abs(pred_count - gt_count),
            "rel_error": round(abs(pred_count - gt_count) / gt_count, 4) if gt_count else None,
            "match_distance": round(dist, 4),
            "tolerance_sec": round(tol, 3),
            "rate_hz": trk.get("rate_hz"),
            "snr": trk.get("snr"),
            "visibility": trk.get("visibility"),
            "estimated_misses": trk.get("stats", {}).get("estimated_misses", 0),
        }
        if person.get("events") and trk.get("events"):
            pe = [e["time"] for e in trk["events"] if e["kind"] in ("jump", "bump", "interpolated")]
            tp, fp, fn, biases = _event_metrics(pe, [float(x) for x in person["events"]], tol)
            row["event_tp"] = tp
            row["event_false"] = fp
            row["event_missed"] = fn
            row["precision"] = round(tp / (tp + fp), 4) if tp + fp else None
            row["recall"] = round(tp / (tp + fn), 4) if tp + fn else None
            row["f1"] = (
                round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else None
            )
            row["time_bias_ms"] = round(float(np.mean(biases)) * 1000, 1) if biases else None
        rows.append(row)

    gt_total = sum(int(p["count"] if p["count"] is not None else len(p["events"] or [])) for p in persons)
    pred_total = sum(int(t["count"]) for t in tracks)
    paired_gt = sum(r["gt_count"] for r in rows)
    paired_pred = sum(r["pred_count"] for r in rows)
    abs_err = sum(r["abs_error"] for r in rows)
    tp = sum(r.get("event_tp", 0) for r in rows)
    fp = sum(r.get("event_false", 0) for r in rows) + sum(int(t["count"]) for t in extra_tracks)
    fn = sum(r.get("event_missed", 0) for r in rows) + sum(
        int(p["count"] if p["count"] is not None else len(p["events"] or [])) for p in missing_persons
    )
    exact = sum(1 for r in rows if r["error"] == 0)
    within1 = sum(1 for r in rows if r["abs_error"] <= 1)
    rels = [r["rel_error"] for r in rows if r["rel_error"] is not None]

    return {
        "video": pred.get("video"),
        "matched_by": "水平位置" if by_position else "计数排序(位置信息缺失, 结果仅供参考)",
        "persons": len(persons),
        "tracks": len(tracks),
        "paired": len(rows),
        "gt_total": gt_total,
        "pred_total": pred_total,
        "total_abs_error_paired": abs_err,
        "total_error_signed": pred_total - gt_total,
        "count_accuracy": round(max(0.0, 1 - abs_err / paired_gt), 4) if paired_gt else None,
        "exact_match_rate": round(exact / len(rows), 4) if rows else None,
        "within_one_rate": round(within1 / len(rows), 4) if rows else None,
        "mae": round(abs_err / len(rows), 3) if rows else None,
        "mape": round(float(np.mean(rels)), 4) if rels else None,
        "event_precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "event_recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "event_f1": round(2 * tp / (2 * tp + fp + fn), 4) if (2 * tp + fp + fn) else None,
        "ghost_tracks": [{"track_id": t["track_id"], "count": t["count"]} for t in extra_tracks],
        "missing_persons": [p["name"] for p in missing_persons],
        "per_person": rows,
        "segments": _segments(rows, tracks, persons, pred),
    }


def _segments(rows: List[dict], tracks: List[dict], persons: List[dict], pred: dict) -> List[dict]:
    """按 10 秒分桶看误差, 定位精度是从第几秒开始崩的。"""
    duration = float(pred.get("duration") or 0.0)
    if duration <= 0 or not any(p.get("events") for p in persons):
        return []
    by_track = {t["track_id"]: t for t in tracks}
    size = 10.0
    buckets = int(np.ceil(duration / size))
    out = []
    for b in range(buckets):
        lo, hi = b * size, (b + 1) * size
        gt_c = 0
        pred_c = 0
        for row in rows:
            person = next((p for p in persons if p["name"] == row["person"]), None)
            trk = by_track.get(row["track_id"])
            if not person or not trk:
                continue
            if person.get("events"):
                gt_c += sum(1 for x in person["events"] if lo <= float(x) < hi)
            pred_c += sum(1 for e in trk.get("events", []) if e["kind"] in ("jump", "bump", "interpolated") and lo <= e["time"] < hi)
        out.append({"window": f"{lo:.0f}-{hi:.0f}s", "gt": gt_c, "pred": pred_c, "error": pred_c - gt_c})
    return out


def report_md(ev: dict) -> str:
    lines = ["# 计数准确性报告", ""]
    lines.append(f"- 视频: `{os.path.basename(str(ev.get('video') or ''))}`")
    lines.append(f"- 轨迹与标注对应方式: {ev['matched_by']}")
    lines.append(
        f"- 总体计数: 人工 **{ev['gt_total']}**, 算法 **{ev['pred_total']}**, 偏差 **{ev['total_error_signed']:+d}**"
    )
    acc = ev.get("count_accuracy")
    if acc is not None:
        lines.append(f"- 计数准确率: **{acc * 100:.1f}%** (按配对样本绝对误差计算)")
    if ev.get("mae") is not None:
        lines.append(f"- 单人平均误差: **{ev['mae']} 个** (MAE), MAPE {float(ev['mape']) * 100:.1f}%")
    if ev.get("event_f1") is not None:
        lines.append(
            f"- 逐次命中: 精确率 {float(ev['event_precision']) * 100:.1f}% / 召回率 "
            f"{float(ev['event_recall']) * 100:.1f}% / F1 {float(ev['event_f1']) * 100:.1f}%"
        )
    lines += ["", "## 明细", "", "| 人名 | 轨迹 | 人工 | 算法 | 误差 | 相对误差 | 漏计次 | 误计次 | 信噪比 | 可见率 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in ev["per_person"]:
        rel = "-" if r["rel_error"] is None else f"{r['rel_error'] * 100:.1f}%"
        lines.append(
            f"| {r['person']} | #{r['track_id']} | {r['gt_count']} | {r['pred_count']} | {r['error']:+d} | {rel} | "
            f"{r.get('event_missed', '-')} | {r.get('event_false', '-')} | {r.get('snr', '-')} | {r.get('visibility', '-')} |"
        )
    if ev.get("ghost_tracks") or ev.get("missing_persons"):
        lines += ["", "## 对齐问题", ""]
        for g in ev["ghost_tracks"]:
            lines.append(f"- 多余轨迹 #{g['track_id']} 计数 {g['count']}: 标注里没有人对应, 属于重复检出或幻影目标")
        for name in ev["missing_persons"]:
            lines.append(f"- 标注人员 {name} 没有被任何轨迹覆盖: 该人全部跳跃被漏掉")
    if ev.get("segments"):
        lines += ["", "## 分时段误差", "", "| 时间段 | 人工 | 算法 | 误差 |", "| --- | --- | --- | --- |"]
        for s in ev["segments"]:
            lines.append(f"| {s['window']} | {s['gt']} | {s['pred']} | {s['error']:+d} |")
    return "\n".join(lines) + "\n"
