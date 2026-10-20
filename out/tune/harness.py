"""调参台: YOLO 输出只跑一次并缓存, 抑制/跟踪/计数每次复算, 改动任何一层都能看到效果。"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from jumpcount.config import Config
from jumpcount.counter import classify, count_track, participation
from jumpcount.detect import suppress_duplicates
from jumpcount.identity import build_slot, consolidate
from jumpcount.tracking import Tracker
from jumpcount.types import Detection

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
WEIGHTS = os.path.join(ROOT, "跳绳素材", "yolo26s-pose.pt")
VIDEOS = ["跳绳素材/" + name for name in (
    "1790062131196.mp4",
    "1790234282676.mp4",
    "1790234713306.mp4",
    "1790234843854.mp4",
    "3c5e84ee19443e3fbb8fa26778b5bc6c.mp4",
    "f4b3e03db8a7f17a77191e8ee29dcf2d.mp4",
)]


def capture(video: str, cfg: Config) -> dict:
    """只做推理, 把每帧的原始框+关键点存下来(未做重复框抑制)。"""
    import cv2
    from ultralytics import YOLO

    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(os.path.join(ROOT, video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    rows = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=cfg.conf, imgsz=cfg.imgsz, classes=list(cfg.classes), verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))
        kps = res.keypoints.data.cpu().numpy() if getattr(res, "keypoints", None) is not None else None
        kept_kps = []
        for i in range(len(boxes)):
            if kps is None:
                kept_kps.append(None)
                continue
            one = kps[i]
            one = np.column_stack([one[:, :2], np.clip(one[:, 2], 0.0, 1.0)])
            kept_kps.append(one.astype(np.float32))
        item = [(tuple(float(v) for v in box), float(confs[i]), kept_kps[i]) for i, box in enumerate(boxes)]
        rows.append(item)
        index += 1
    cap.release()
    return {"video": video, "fps": float(fps), "frames": index, "rows": rows}


def ensure_cache(cfg: Config) -> None:
    os.makedirs(CACHE, exist_ok=True)
    for video in VIDEOS:
        path = os.path.join(CACHE, os.path.basename(video) + ".pkl")
        if os.path.exists(path):
            continue
        started = time.time()
        data = capture(video, cfg)
        with open(path, "wb") as fh:
            pickle.dump(data, fh)
        print(f"缓存 {os.path.basename(video)}: {data['frames']} 帧 / {time.time() - started:.1f}s", flush=True)


def _detections(data: dict, index: int, cfg: Config):
    fps = data["fps"]
    out = []
    for box, conf, kps in data["rows"][index]:
        out.append(Detection(frame=index, time=index / fps, box=box, conf=conf, keypoints=kps))
    return suppress_duplicates(out, cfg)


def replay(cfg: Config, video: str, data: dict):
    """用当前代码从缓存重跑: 抑制 -> 跟踪 -> 身份归并 -> 计数。"""
    fps = data["fps"]
    tracker = Tracker(cfg)
    samples, centers = {}, {}
    for index in range(data["frames"]):
        for tid, det in tracker.update(index, _detections(data, index, cfg)):
            samples.setdefault(tid, []).append((index, index / fps, det.box, None, det.keypoints))
            centers.setdefault(tid, []).append(((det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2))
    slots = [build_slot(tid, samples[tid], centers[tid], tracker.track_meta().get(tid, {}).get("mean_conf", 0.0))
             for tid in sorted(samples) if len(samples[tid]) >= 2]
    slots, actions = consolidate(slots, cfg)
    rows, kept, provisional, expected = [], 0, 0, 0.0
    for slot in slots:
        res = count_track(slot.track_id, slot.samples, cfg, fps)
        verdict, reason = participation(res.stats, cfg)
        exp = res.stats.get("expected_count", 0.0)
        total = res.count + res.stats.get("filled", 0)
        if verdict == "in":
            kept += total
            expected += exp
        elif verdict == "provisional":
            provisional += total
            expected += exp
        rows.append((slot.track_id, verdict, total, round(exp, 1), res.rate_hz, res.visibility,
                     res.signal_source, res.stats.get("rhythm_strength"), res.stats.get("height_cv"), reason))
    return rows, actions, kept, provisional, expected


def score(cfg: Config, verbose: bool = True, videos=None) -> dict:
    from jumpcount.online import OnlineCounter

    totals = {"in": 0, "provisional": 0, "expected": 0.0, "lost": 0, "tracks": 0, "online": 0,
              "online_people": 0, "online_provisional": 0}
    for video in (videos or VIDEOS):
        path = os.path.join(CACHE, os.path.basename(video) + ".pkl")
        if not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        rows, actions, kept, prov, expected = replay(cfg, video, data)
        totals["in"] += kept
        totals["provisional"] += prov
        totals["expected"] += expected
        totals["tracks"] += len(rows)
        totals["lost"] += sum(r[2] for r in rows if r[1] == "out")
        if verbose:
            print(f"\n{os.path.basename(video)}  {data['frames'] / data['fps']:.1f}s  轨迹 {len(rows)} 条  "
                  f"剔重 {sum(1 for a in actions if a['action'] == 'drop_duplicate')} 缝帧 {sum(1 for a in actions if a['action'] == 'stitch_fragment')}")
            for tid, verdict, count, exp, rate, vis, src, rhythm, hcv, reason in sorted(rows, key=lambda r: -r[2]):
                ratio = count / exp if exp else 0.0
                mark = {"in": "计入", "provisional": "低置信", "out": "剔除"}[verdict]
                print(f"  #{tid:<3} {mark:<6} 总数 {count:>4} 节奏应有 {exp:>6} 完整率 {ratio:>4.2f} "
                      f"{rate if rate else 0:>5.2f}Hz 可见 {vis:>4.2f} 节律 {str(rhythm):>5} 框高波动 {str(hcv):>5} {src:<10}{reason[:26]}")
        # 实时链路
        fps = data["fps"]
        tracker = Tracker(cfg)
        counter = OnlineCounter(cfg, fps)
        for index in range(data["frames"]):
            dets = _detections(data, index, cfg)
            counter.update(index, index / fps, [(tid, d.box, d.keypoints, d.conf) for tid, d in tracker.update(index, dets)])
        snaps = counter.snapshot(data["frames"])
        totals["online"] += sum(s.get("total_count", s["count"]) for s in snaps if s["participant"])
        totals["online_provisional"] += sum(s.get("total_count", s["count"]) for s in snaps if s.get("verdict") == "provisional")
        totals["online_people"] += sum(1 for s in snaps if s["participant"])
        if verbose:
            print("  实时链路: " + " | ".join(
                f"#{s['id']} {s.get('total_count', s['count'])}次({s.get('verdict')})" for s in snaps if s["count"])
                + f"  共 {sum(s.get('total_count', s['count']) for s in snaps if s['participant'])}")
    total = totals["in"] + totals["provisional"]
    totals["total"] = total
    totals["rhythm_coverage"] = round(total / totals["expected"], 3) if totals["expected"] else 0.0
    if verbose:
        print(f"\n== 离线 计入 {totals['in']} + 低置信 {totals['provisional']} = {total} 次 / 按节奏应有 {totals['expected']:.0f} -> 完整率 {totals['rhythm_coverage']}")
        print(f"== 被剔除但数到的 {totals['lost']} 次;  实时链路 {totals['online']} 次(其中低置信 {totals['online_provisional']}), 人数 {totals['online_people']}")
    return totals


def overrides(argv) -> Config:
    cfg = Config()
    for item in argv:
        key, value = item.split("=", 1)
        low = value.lower()
        if low in ("true", "false"):
            cast = low == "true"
        else:
            try:
                cast = int(value)
            except ValueError:
                try:
                    cast = float(value)
                except ValueError:
                    cast = value
        setattr(cfg, key, cast)
    return cfg


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        ensure_cache(overrides(sys.argv[2:]))
    else:
        score(overrides(sys.argv[2:]))
