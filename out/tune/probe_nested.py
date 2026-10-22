"""重复框归因: 同一时刻是否有两条轨迹锁在同一个人身上(小框套在大框里), 造成同一个人被数两遍。

用法: .venv/bin/python out/tune/probe_nested.py [视频关键字]
判法: 对时间重叠高的轨迹两两求"逐帧框重叠"——平均 IoU 与平均包含率(交集/较小框面积);
      包含率高 = 同一个人被拆成两条(重复计数); 同横向但纵向分层 = 前后两个人(正常)。
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
from jumpcount.tracking import Tracker  # noqa: E402


def iou_contain(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    return (inter / union if union > 0 else 0.0, inter / smaller if smaller > 0 else 0.0)


def main() -> None:
    args = [a for a in sys.argv[1:] if "=" not in a]
    keyword = args[0] if args else ""
    cfg = Config()
    started = time.time()
    grand_dup = 0
    for video in VIDEOS:
        if keyword and keyword not in video:
            continue
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
        infos = []
        for slot in slots:
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            verdict, reason = participation(res.stats, cfg)
            total = res.count + res.stats.get("filled", 0)
            if total < 3:
                continue
            boxes = {int(s[0]): np.asarray(s[2], dtype=float) for s in slot.samples}
            infos.append({"tid": slot.track_id, "verdict": verdict, "total": total,
                          "boxes": boxes, "reason": reason,
                          "h": float(np.median([b[3] - b[1] for b in boxes.values()])),
                          "y": float(np.median([b[3] for b in boxes.values()]))})
        counted = sum(r["total"] for r in infos if r["verdict"] != "out")
        print(f"\n=== {os.path.basename(video)} 计入 {counted} 次 (数到>=3次的轨迹 {len(infos)} 条)")
        print("疑似同一人(时间重叠>50% 且 平均包含率>0.6): 小框那条的次数就是重复计数")
        dup_small = 0
        rows = []
        for i in range(len(infos)):
            for j in range(i + 1, len(infos)):
                a, b = infos[i], infos[j]
                common = sorted(set(a["boxes"]) & set(b["boxes"]))
                ov_frames = min(infos[i]["total"], infos[j]["total"])
                if len(common) < 20:
                    continue
                pairs = np.array([iou_contain(a["boxes"][f], b["boxes"][f]) for f in common])
                mean_iou, mean_cont = float(pairs[:, 0].mean()), float(pairs[:, 1].mean())
                if mean_cont <= 0.6:
                    continue
                small, big = (a, b) if a["h"] <= b["h"] else (b, a)
                rows.append((mean_cont, mean_iou, len(common) / max(1, min(len(a["boxes"]), len(b["boxes"]))),
                             small, big))
        rows.sort(key=lambda r: -r[0])
        if not rows:
            print("  无")
        for cont, miou, cover, small, big in rows:
            penalty = small["total"] if small["verdict"] != "out" else 0
            dup_small += penalty
            print(f"  小#{small['tid']}({small['verdict']},{small['total']}次,h={small['h']:.0f},脚y={small['y']:.0f}) "
                  f"套在大#{big['tid']}({big['verdict']},{big['total']}次,h={big['h']:.0f},脚y={big['y']:.0f}) 内 "
                  f"包含率 {cont:.2f} IoU {miou:.2f} 同框帧占比 {cover * 100:.0f}% -> 重复 {penalty} 次")
        print(f"  本段疑似重复计数合计 {dup_small} 次 (占计入 {counted} 的 {dup_small / max(1, counted) * 100:.1f}%)")
        grand_dup += dup_small
    print(f"\n全部素材疑似重复计数合计 {grand_dup} 次, 用时 {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
