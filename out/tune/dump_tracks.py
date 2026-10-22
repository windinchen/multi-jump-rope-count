"""把每条轨迹的计数/估计频率导成一行, 便于两份配置做逐条差分。

用法: .venv/bin/python out/tune/dump_tracks.py [key=value ...] > out/tune/tracks_xxx.tsv
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
            verdict = participation(res.stats, cfg)[0]
            total = res.count + res.stats.get("filled", 0)
            print(f"{os.path.basename(video)[:12]}\t{slot.track_id}\t{total}\t{res.rate_hz}\t"
                  f"{res.period_sec}\t{verdict}\t{res.stats.get('rhythm_strength', 0)}\t"
                  f"{res.stats.get('period_source', '')}\t"
                  f"{max(float(res.stats.get('lift_ratio', 0) or 0), float(res.stats.get('lift_ratio_best', 0) or 0)):.4f}\t"
                  f"{res.stats.get('amplitude_threshold', 0)}\t"
                  f"{res.stats.get('visibility', 0)}\t{res.stats.get('duration_sec', 0)}")


if __name__ == "__main__":
    main()
