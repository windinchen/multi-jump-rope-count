"""逐条轨迹的肉眼审核表: 每条"计入"的轨迹取 3 帧(含 1 个被判定的峰), 拼成一行给人核对是不是真人在跳。

用法: .venv/bin/python out/tune/audit_tracks.py [视频关键字] [每视频最多条数]
目的: 自我一致指标报不出"把站着/坐着的人也数了", 只能靠这张表逐行看。
"""
from __future__ import annotations

import os
import pickle
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from harness import CACHE, VIDEOS, _detections  # noqa: E402
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import count_track, participation  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def main() -> None:
    args = [a for a in sys.argv[1:] if "=" not in a]
    keyword = args[0] if args else ""
    max_rows = int(args[1]) if len(args) > 1 else 12
    order = args[2] if len(args) > 2 else "desc"
    cfg = Config()
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
        rows = []
        for slot in slots:
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            verdict, reason = participation(res.stats, cfg)
            total = res.count + res.stats.get("filled", 0)
            if total < 2:
                continue
            peaks = [int(round(e.frame)) for e in res.events if e.kind == "jump"]
            by_frame = {int(s[0]): np.asarray(s[2], dtype=float) for s in slot.samples}
            pick = []
            if peaks:
                pick.append(peaks[len(peaks) // 2])
                pick.append(peaks[0])
            mid = sorted(by_frame)[len(by_frame) // 2]
            pick.append(mid)
            seen, order = set(), []
            for f in pick:
                if f in by_frame and f not in seen:
                    seen.add(f)
                    order.append(f)
            while len(order) < 3:
                cand = sorted(by_frame)
                extra = cand[len(cand) * (len(order) + 1) // 4]
                if extra not in seen:
                    seen.add(extra)
                    order.append(extra)
                else:
                    break
            rows.append({"tid": slot.track_id, "verdict": verdict, "total": total, "reason": reason,
                         "stats": res.stats, "frames": order[:3], "boxes": by_frame})
        rows.sort(key=lambda r: (-1 if r["verdict"] != "out" else 0,
                                (-1 if order == "asc" else 1) * r["total"]))
        rows = rows[:max_rows]
        cap = cv2.VideoCapture(os.path.join(ROOT, video))
        tiles_h, tiles = 220, []
        for r in rows:
            strip = np.full((tiles_h, 300 * 3 + 8, 3), 25, dtype=np.uint8)
            for k, f in enumerate(r["frames"]):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, frame = cap.read()
                if not ok:
                    continue
                b = r["boxes"][f]
                x1, y1, x2, y2 = [int(v) for v in b]
                pad = int(0.18 * max(x2 - x1, y2 - y1))
                cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                cx2, cy2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2].copy()
                cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 0, 255), 2)
                s = min(220.0 / crop.shape[0], 300.0 / crop.shape[1])
                crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))
                strip[0:crop.shape[0], k * 300:k * 300 + crop.shape[1]] = crop
                cv2.putText(strip, f"{f / fps:.1f}s", (k * 300 + 3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
            st = r["stats"]
            label = (f"#{r['tid']} {r['verdict']} {r['total']}c lift={st.get('lift_ratio')} "
                     f"cv={st.get('height_cv')} thr={st.get('amplitude_threshold')} "
                     f"h={int(np.median([b[3] - b[1] for b in r['boxes'].values()]))}px")
            cv2.putText(strip, label[:58], (4, tiles_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1,
                        cv2.LINE_AA)
            tiles.append(strip)
        canvas = np.vstack(tiles) if tiles else np.zeros((10, 10, 3), np.uint8)
        out = os.path.join(HERE, f"audit_{os.path.basename(video)[:12]}.png")
        cv2.imwrite(out, canvas)
        print(f"审核表: {out}  行数={len(tiles)}")


if __name__ == "__main__":
    main()
