"""把一条轨迹的计数账画出来: 实测峰 / 按节奏补 / 低于阈值漏掉的候选。"""
import os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "Songti SC"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from jumpcount.config import Config
from jumpcount import signal as sg
from jumpcount.counter import channel_cache, count_track, extract_y_detail, pick_source
from jumpcount.detect import suppress_duplicates
from jumpcount.identity import build_slot, consolidate
from jumpcount.tracking import Tracker
from jumpcount.types import Detection

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
key, want = sys.argv[1], int(sys.argv[2])
out = sys.argv[3]
cfg = Config()
data = pickle.load(open(os.path.join(CACHE, [f for f in os.listdir(CACHE) if f.startswith(key)][0]), "rb"))
fps = data["fps"]
tracker = Tracker(cfg)
samples, centers = {}, {}
for i in range(data["frames"]):
    dets = suppress_duplicates([Detection(frame=i, time=i / fps, box=b, conf=c, keypoints=k) for b, c, k in data["rows"][i]], cfg)
    for tid, d in tracker.update(i, dets):
        samples.setdefault(tid, []).append((i, i / fps, d.box, None, d.keypoints))
        centers.setdefault(tid, []).append(((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2))
slots = [build_slot(t, samples[t], centers[t], tracker.track_meta().get(t, {}).get("mean_conf", 0.0))
         for t in sorted(samples) if len(samples[t]) >= 2]
slots, _ = consolidate(slots, cfg)
slot = next(s for s in slots if s.track_id == want)
s = slot.samples
frames = np.array([int(x[0]) for x in s])
heights = np.array([x[2][3] - x[2][1] for x in s])
median_h = float(np.median(heights))
source = pick_source(s, cfg)
res = count_track(want, s, cfg, fps)
y, _ = extract_y_detail(s, source, cfg, channel_cache(s))
raw = -np.where(np.isfinite(y), y, np.nan)
axis, dense, _ = sg.fill_gaps(frames, raw, cfg.max_gap)
observed = np.zeros(axis.size, dtype=bool); observed[frames - int(axis[0])] = True
det = sg.detrend(dense, cfg.smooth_win, cfg.baseline_win if dense.size >= cfg.baseline_win else 3)
p95 = float(np.percentile(det, 95))
floor = max(cfg.min_amplitude, cfg.amplitude_ratio * median_h) * max(0.05, cfg.floor_relief)
thr = min(max(floor, cfg.snr_k * sg.hf_sigma_masked(det, observed), cfg.peak_frac * p95),
          max(cfg.amplitude_ratio * median_h * max(0.05, cfg.floor_relief), 0.45 * p95))
t = axis / fps
counted = np.array([e.frame for e in res.events if e.kind == "jump"])
filled = np.array([e.frame for e in res.events if e.kind == "interpolated"])
oracle = sg.find_peaks(det, 5, 0.18 * p95, 0.1 * p95)
missed = [float(axis[p]) for p in oracle
          if not any(abs(axis[p] - c) <= 3.0 for c in np.concatenate([counted, filled]))]

fig, ax = plt.subplots(figsize=(13.5, 4.4))
ax.plot(t, det, lw=0.9, color="#4b6cb7", label="去基线后的脚踝垂直位移")
ax.axhline(thr, ls="--", lw=1.2, color="#c0392b", label=f"计数阈值 {thr:.2f}px")
ax.plot(t[observed], det[observed], ".", ms=1.6, color="#9aa5b1", alpha=.45)
ax.scatter(counted / fps, [det[np.argmin(np.abs(axis - f))] for f in counted], s=46, color="#1e8449",
           zorder=5, label=f"实测到 {len(counted)} 个")
ax.scatter(filled / fps, [det[np.argmin(np.abs(axis - f))] for f in filled], s=46, marker="x",
           color="#e67e22", zorder=5, label=f"按节奏补 {len(filled)} 个")
ax.scatter(np.array(missed) / fps, [det[np.argmin(np.abs(axis - f))] for f in missed], s=22, marker="v",
           color="#c0392b", alpha=.85, zorder=4, label=f"低于阈值漏掉 {len(missed)} 个")
ax.set_xlabel("秒"); ax.set_ylabel("垂直位移 (px)")
ax.set_title(f"{key}… 轨迹 #{want}：数到 {len(counted) + len(filled)}，漏掉的都在阈值线下方一点点（阈值 {thr:.2f}px，噪声 0.37px，人只有 {median_h:.0f}px 高）")
ax.legend(loc="upper right", fontsize=9, ncol=2, framealpha=.92)
ax.grid(alpha=.25)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, out), dpi=135)
print(out)
