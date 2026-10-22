"""列出"头部零位移"闸门的命中情况: 每条轨迹的头/踝/髋位移 p95 + 判定结论 + 出现时间。

用法: .venv/bin/python out/tune/probe_gate.py [min_head_move_px=N] [min_head_share=N] [min_head_coverage=N]
判读: 命中的应当都是"站着没离地、只有局部被前景带出波纹"的人; 真跳绳者头部必然跟着抬。
"""
from __future__ import annotations

import os
import pickle
import sys

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
    for item in sys.argv[1:]:
        key, value = item.split("=", 1)
        setattr(cfg, key, float(value))
    rows = []
    for video in VIDEOS:
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
        for slot in slots:
            res = count_track(slot.track_id, slot.samples, cfg, fps)
            st = res.stats
            head, body = st.get("head_p95_px"), st.get("body_p95_px")
            fired = (cfg.min_head_move_px > 0 and head is not None and body
                     and head < cfg.min_head_move_px and head < cfg.min_head_share * body)
            ch = st.get("channel_p95_px", {})
            rows.append({
                "video": os.path.basename(video)[:12], "tid": slot.track_id,
                "count": st.get("total_count", 0), "verdict": st.get("verdict"),
                "src": res.signal_source, "head": head, "body": body,
                "share": round(head / body, 3) if head is not None and body else None,
                "ankle": ch.get("ankle"), "hip": ch.get("hip"), "box": ch.get("box_bottom"),
                "rhy": st.get("rhythm_strength"), "dur": st.get("duration_sec"),
                "start": round(slot.samples[0][1], 1), "end": round(slot.samples[-1][1], 1),
                "fired": fired, "reason": st.get("excluded", ""),
            })
    hit = [r for r in rows if r["fired"]]
    kept = [r for r in rows if r["verdict"] != "out"]
    print(f"{'视频':<14}{'轨迹':>5}{'计数':>6}{'判定':>13}{'信号':>11}{'头px':>7}{'全身px':>8}{'占比':>7}"
          f"{'踝px':>7}{'髋px':>7}{'节律':>7}{'区间':>15}")
    for r in sorted(hit, key=lambda r: -r["count"]):
        print(f"{r['video']:<14}{r['tid']:>5}{r['count']:>6}{r['verdict']:>13}{r['src']:>11}"
              f"{r['head']:>7.2f}{r['body']:>8.2f}{r['share']:>7.3f}"
              f"{(r['ankle'] if r['ankle'] is not None else -1):>7.2f}"
              f"{(r['hip'] if r['hip'] is not None else -1):>7.2f}"
              f"{(r['rhy'] if r['rhy'] is not None else -1):>7.2f}"
              f"{r['start']:>6.1f}-{r['end']:<8.1f}")
    print(f"\n命中 {len(hit)} 条 / {sum(r['count'] for r in hit)} 次; "
          f"计入 {len(kept)} 条 / {sum(r['count'] for r in kept)} 次; 轨迹总数 {len(rows)}")
    near = [r for r in rows if not r["fired"] and r["head"] is not None and r["body"]
            and r["head"] < 2.0 and r["head"] < 0.6 * r["body"]]
    print(f"未命中但接近(头<2px 且占比<0.6): {len(near)} 条 / {sum(r['count'] for r in near)} 次")
    for r in sorted(near, key=lambda r: -r["count"])[:12]:
        print(f"  {r['video']}#{r['tid']:<4} 计数{r['count']:>4} 判定{r['verdict']:<12} 头{r['head']:.2f}"
              f" 全身{r['body']:.2f} 占比{r['share']:.2f} 节律{r['rhy']}")


if __name__ == "__main__":
    main()
