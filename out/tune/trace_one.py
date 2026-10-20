"""单条轨迹复盘: 把"该数到但没数到"的起跳逐个定位出来。"""
import os, pickle, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from jumpcount.config import Config
from jumpcount import signal as sg
from jumpcount.counter import (channel_cache, count_track, drift_spread, extract_y_detail,
                               pick_source, robust_spread)
from jumpcount.detect import suppress_duplicates
from jumpcount.identity import build_slot, consolidate
from jumpcount.tracking import Tracker
from jumpcount.types import Detection

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
cfg = Config()
for item in os.environ.get("SET", "").split():
    k, v = item.split("=", 1)
    try:
        cast = int(v)
    except ValueError:
        try:
            cast = float(v)
        except ValueError:
            cast = v
    setattr(cfg, k, cast)
key = sys.argv[1] if len(sys.argv) > 1 else "3c5e84"
want = int(sys.argv[2]) if len(sys.argv) > 2 else 6

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
channels = channel_cache(s)
y, substituted = extract_y_detail(s, source, cfg, channels)
raw = -np.where(np.isfinite(y), y, np.nan)
res = count_track(want, s, cfg, fps)
st = res.stats
axis, dense, long_holes = sg.fill_gaps(frames, raw, cfg.max_gap)
observed = np.zeros(axis.size, dtype=bool); observed[frames - int(axis[0])] = True
baseline_win = cfg.baseline_win if dense.size >= cfg.baseline_win else 3
det = sg.detrend(dense, cfg.smooth_win, baseline_win)
noise = sg.hf_sigma_masked(det, observed)
p95 = float(np.percentile(det, 95))
floor = max(cfg.min_amplitude, cfg.amplitude_ratio * median_h) * max(0.05, cfg.floor_relief)
rel_floor = cfg.amplitude_ratio * median_h * max(0.05, cfg.floor_relief)
thr = min(max(floor, cfg.snr_k * noise, cfg.peak_frac * p95), max(rel_floor, 0.45 * p95))
period = res.period_sec * fps
print(f"轨迹 #{want}  源 {source}  帧 {frames[0]}..{frames[-1]} ({len(frames)} 个观测 / {(frames[-1]-frames[0])/fps:.1f}s 跨度)  中位框高 {median_h:.0f}px")
print(f"  计数 count={res.count} 补={st['filled']} 双摇={st['bumps']} 总数={res.count+st['filled']}  节奏周期={period:.1f}帧({res.rate_hz}Hz) 来源 {st['period_source']} 节律 {st['rhythm_strength']}")
print(f"  阈值 thr={thr:.2f}px  噪声 {noise:.2f}  下限(floor={floor:.2f}, 框高比例下限={rel_floor:.2f})  p95={p95:.2f}  信噪比 {res.snr}")
print(f"  估计漏计 {st['estimated_misses']} 次 / 其中因不在画面无法补 {st['misses_unobserved']}  补齐 {st['filled']}  长空白 {long_holes} 段  换源补齐 {substituted} 帧")

# 用极宽阈值找"真实存在的向上鼓包", 再逐个解释为什么没进计数
counted = [e.frame for e in res.events if e.kind == "jump"]
bumps = [e.frame for e in res.events if e.kind == "bump"]
oracle = sg.find_peaks(det, 5, 0.18 * p95, 0.1 * p95)
print(f"\n宽阈值候选鼓包 {len(oracle)} 个; 已计 {len(counted)} + 判双摇 {len(bumps)}")
missing = []
for p in oracle:
    at = float(axis[p])
    if any(abs(at - c) <= 2.0 for c in counted):
        continue
    tag = "判为双摇" if any(abs(at - b) <= 2.0 for b in bumps) else None
    if tag is None:
        near = min(counted, key=lambda c: abs(c - at)) if counted else None
        if near is not None and abs(near - at) < cfg.refractory_ratio * period:
            tag = f"被最小间隔{cfg.refractory_ratio * period:.1f}帧压掉(距上个峰{abs(near - at):.1f}帧)"
        elif det[p] < thr:
            tag = f"低于阈值({det[p]:.2f} < {thr:.2f})"
        elif not observed[p]:
            tag = "该帧未观测(遮挡/跟丢)"
        else:
            tag = "其它"
    if tag.startswith("判为双摇") or tag.startswith("低于阈值") or tag.startswith("该帧") or tag == "其它":
        missing.append((at, float(det[p]), tag, bool(observed[p])))
print(f"未计入的候选 {len(missing)} 个:")
for at, amp, tag, obs in missing:
    print(f"  帧 {at:7.1f} ({at / fps:5.2f}s) 幅度 {amp:6.2f}px 观测 {'是' if obs else '否'}  {tag}")
print(f"\n结论粗算: 真实约 {len(counted) + len(missing)} 个鼓包, 计到 {res.count + st['filled']}")

# 缺口级复盘: 每个空档期望补几个、实际补了几个、被哪条护栏挡了
print("\n空档明细(已计峰之间的长空档):")
lost_cap = 0
prev = None
for c in sorted(counted):
    if prev is not None:
        gap = c - prev
        if gap > cfg.gap_ratio * period:
            expected = int(round(gap / period)) - 1
            gave = min(expected, cfg.max_fill_per_gap)
            lost_cap += max(0, expected - cfg.max_fill_per_gap)
            print(f"  帧 {prev:6.1f}->{c:6.1f} 间隔 {gap:5.1f}帧 = {gap / period:4.2f}个周期  期望补 {expected}  实补 {gave}"
                  + ("  <- 被每档封顶 2 砍掉" if expected > cfg.max_fill_per_gap else ""))
    prev = c
print(f"  合计: 封顶丢掉 {lost_cap} 个")

ivs = np.diff(sorted(counted))
print(f"\n已计峰间隔: 中位 {np.median(ivs):.1f}帧 直方图 {np.bincount(np.round(ivs).astype(int))[2:20].tolist()}")
ac, strength = sg.autocorr_period(det, fps, cfg.min_rate_hz, cfg.max_rate_hz)
print(f"自相关周期 {ac * fps if ac else 0:.1f}帧 ({ac and round(1 / ac, 2)}Hz) 强度 {strength:.3f}  vs 采用的 {period:.1f}帧")
print(f"按 {cfg.max_rate_hz}Hz 上限推出的硬最小间距 = {max(2, int(round(fps / cfg.max_rate_hz)))} 帧")
for cap in (2, 3, 4, 6):
    got = 0
    prev = None
    for c in sorted(counted):
        if prev is not None and c - prev > cfg.gap_ratio * period:
            got += min(max(0, int(round((c - prev) / period)) - 1), cap)
        prev = c
    print(f"  若每档最多补 {cap} 个: 总数 {len(counted) + got}")
