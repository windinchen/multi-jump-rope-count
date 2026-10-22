"""绳影通道: 绳过脚时在"落点条带"里留下一道快速移动的高对比纹理, 这是关键点通道拿不到的信息。

用法: .venv/bin/python out/tune/probe_rope.py [key=value ...]
做法:
  1) 用缓存的检测/跟踪拿到每条轨迹每一帧的框(与线上同一套判据, 不引入新变量);
  2) 逐帧在框的落点条带里算帧间灰度变化能量, 存成每条轨迹一条一维信号;
  3) 用同一套 find_peaks/autocorr 数绳影峰, 与现有计数逐条对照。
指标:
  作证率 = 现有起跳中, 绳影在 ±0.3 周期内也有峰的比例(绳影能解释多少现有计数)
  机会   = 空档(>1.5 周期)里出现绳影峰的次数(现在的漏计能不能被绳影补上)
  双计   = 绳影峰自身间隔 < 0.7 周期的比例
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from harness import CACHE, ROOT, VIDEOS, _detections  # noqa: E402
from jumpcount import signal as sg  # noqa: E402
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import count_track, participation  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402

DOWN = 2.0
OUT = os.path.join(HERE, "rope_signals.pkl")


def band_stats(gray, prev, box):
    """返回 (落点条带能量, 小腿条带能量, 整框能量, 落点条带列方差)。坐标按 DOWN 缩放后的框。"""
    h_img, w_img = gray.shape
    x1, y1, x2, y2 = box
    h = y2 - y1
    if h < 12 or x2 - x1 < 6:
        return None
    pad = 0.45 * h
    cx1, cx2 = int(max(0, x1 - pad)), int(min(w_img, x2 + pad))
    if cx2 - cx1 < 8:
        return None

    def energy(top, bot):
        t, b = int(max(0, top)), int(min(h_img, bot))
        if b - t < 2:
            return None, None
        cur = gray[t:b, cx1:cx2]
        d = np.abs(cur.astype(np.float32) - prev[t:b, cx1:cx2].astype(np.float32)) / 255.0
        cols = d.mean(axis=0)
        return float(d.mean()), float(cols.std())

    low = energy(y2 - 0.12 * h, y2 + 0.22 * h)
    mid = energy(y2 - 0.55 * h, y2 - 0.12 * h)
    allb = energy(y1, y2)
    if low[0] is None or mid[0] is None or allb[0] is None:
        return None
    return (round(low[0], 5), round(mid[0], 5), round(allb[0], 5), round(low[1], 5))


def dump(cfg: Config) -> dict:
    import cv2

    out: dict = {}
    for video in VIDEOS:
        path = os.path.join(CACHE, os.path.basename(video) + ".pkl")
        if not os.path.exists(path):
            continue
        data = pickle.load(open(path, "rb"))
        fps = data["fps"]
        tracker = Tracker(cfg)
        samples, centers = {}, {}
        boxes_by_frame: dict = {}
        for index in range(data["frames"]):
            pairs = tracker.update(index, _detections(data, index, cfg))
            for tid, det in pairs:
                samples.setdefault(tid, []).append((index, index / fps, det.box, None, det.keypoints))
                centers.setdefault(tid, []).append(((det.box[0] + det.box[2]) / 2,
                                                    (det.box[1] + det.box[3]) / 2))
                boxes_by_frame.setdefault(index, {})[tid] = det.box
        slots = [build_slot(t, samples[t], centers[t]) for t in sorted(samples) if len(samples[t]) >= 2]
        slots, _ = consolidate(slots, cfg)
        # 归并后的站位 → 原始轨迹号, 帧轴上的框要按归并后的身份取最新的那个
        owner = {}
        for slot in slots:
            for tid in slot.merged_from:
                owner[tid] = slot.track_id
        sig = {}
        cap = cv2.VideoCapture(os.path.join(ROOT, video))
        prev = None
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, None, fx=1.0 / DOWN, fy=1.0 / DOWN, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                for tid, box in boxes_by_frame.get(index, {}).items():
                    key = owner.get(tid, tid)
                    scaled = tuple(v / DOWN for v in box)
                    st = band_stats(gray, prev, scaled)
                    if st is not None:
                        sig.setdefault(key, []).append((index,) + st)
            prev = gray
            index += 1
        cap.release()
        out[os.path.basename(video)] = {"fps": fps, "sig": sig, "frames": data["frames"],
                                        "slots": {s.track_id: s.samples for s in slots}}
        print(f"{os.path.basename(video)[:12]}: {len(sig)} 条轨迹有绳影信号, "
              f"平均每条约 {np.mean([len(v) for v in sig.values()] or [0]):.0f} 帧", flush=True)
    with open(OUT, "wb") as fh:
        pickle.dump(out, fh)
    return out


def count_rope(values, fps, cfg: Config, period_frames):
    """绳影能量序列 -> 峰(用与线上一致的取峰与节奏口径)。"""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 20:
        return [], 0.0, 0
    v = v - sg.running_baseline(v, cfg.baseline_win if v.size >= cfg.baseline_win else _odd(v.size // 2))
    noise = sg.hf_sigma(v)
    peak_level = float(np.percentile(v, 95)) if v.size else 0.0
    thr = max(noise * cfg.snr_k, 0.05 * peak_level, 1e-4)
    hard_min = max(2, int(fps / cfg.max_rate_hz))
    peaks = sg.find_peaks(v, hard_min, thr, 0.8 * thr)
    _p, strength = sg.autocorr_period(v, fps, cfg.min_rate_hz, cfg.max_rate_hz)
    return peaks, float(strength), len(peaks)


def _odd(value, floor=5):
    value = max(floor, int(value))
    return value if value % 2 else value - 1


def evaluate(cfg: Config, cached: dict) -> None:
    print(f"\n{'视频':<13}{'轨迹':<6}{'判定':<13}{'现计':<6}{'绳影峰':<7}{'作证率':<8}{'机会':<6}{'绳双计':<8}{'绳率':<7}{'现有率'}")
    tot_now = tot_rope = 0
    agree = opp = 0
    checks = 0
    for video, blob in cached.items():
        fps = blob["fps"]
        sig, slot_samples = blob["sig"], blob["slots"]
        for tid, samples in sorted(slot_samples.items()):
            res = count_track(tid, samples, cfg, fps)
            verdict = participation(res.stats, cfg)[0]
            if verdict == "out" or not res.period_sec:
                continue
            now = res.count + res.stats.get("filled", 0)
            rows = sig.get(tid)
            if not rows or len(rows) < 30 or now < 5:
                continue
            period = res.period_sec * fps
            for col, name in ((1, "落点"), (2, "小腿"), (4, "落点列方差")):
                values = [r[col] for r in rows]
                frames = [r[0] for r in rows]
                peaks, strength, n = count_rope(values, fps, cfg, period)
                if col != 1:
                    continue
                tag = video[:12]
                if not peaks:
                    print(f"{tag:<13}#{tid:<5}{verdict:<13}{now:<6}{0:<7}{'-':<8}{'-':<6}{'-':<8}{'-':<7}{res.rate_hz}")
                    continue
                pf = [frames[i] for i in peaks]
                ok = sum(1 for e in res.events if e.kind in ("jump", "bump")
                         and any(abs(f - e.frame) <= 0.3 * period for f in pf))
                checks += sum(1 for e in res.events if e.kind in ("jump", "bump"))
                agree += ok
                jumps = sorted(e.frame for e in res.events if e.kind in ("jump", "bump"))
                chance = 0
                for a, b in zip(jumps, jumps[1:]):
                    if b - a > cfg.gap_ratio * period:
                        if any(a + 0.3 * period < f < b - 0.3 * period for f in pf):
                            chance += 1
                d = np.diff(pf)
                dbl = float((d < 0.7 * period).sum()) / max(1, len(d)) if len(d) >= 4 else 0.0
                opp += chance
                tot_now += now
                tot_rope += n
                print(f"{tag:<13}#{tid:<5}{verdict:<13}{now:<6}{n:<7}{ok / max(1, len(jumps)):<8.0%}"
                      f"{chance:<6}{dbl:<8.0%}{fps / max(1e-6, np.median(d) if len(d) else period):<7.2f}{res.rate_hz}")
    print(f"\n现有起跳 {checks} 次中绳影能作证的: {agree} ({agree / max(1, checks):.1%})")
    print(f"空档里出现绳影峰的次数(潜在漏计补回): {opp}")
    print(f"现有(在跳绳口径) {tot_now} vs 绳影峰 {tot_rope}")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--redump"]
    redump = "--redump" in sys.argv[1:]
    cfg = Config()
    for item in args:
        key, value = item.split("=", 1)
        low = value.lower()
        cast = low == "true" if low in ("true", "false") else None
        if cast is None:
            try:
                cast = int(value)
            except ValueError:
                cast = float(value)
        setattr(cfg, key, cast)
    if redump or not os.path.exists(OUT):
        cached = dump(cfg)
    else:
        cached = pickle.load(open(OUT, "rb"))
    evaluate(cfg, cached)


if __name__ == "__main__":
    main()
