"""复盘实时页面上编号 N 的那个人: 实时数到多少 / 同样样本离线能数到多少 / 信号里到底有多少个鼓包。
用法: trace_live.py <素材前缀> <实时编号>   可用 SET="k=v ..." 覆盖配置"""
import os, pickle, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
from jumpcount.config import Config
from jumpcount import signal as sg
from jumpcount.counter import (amplitude_threshold, channel_cache, count_track, extract_y_detail,
                               participation)
from jumpcount.online import OnlineCounter, TrackCounter
from jumpcount.tracking import Tracker

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

key = sys.argv[1] if len(sys.argv) > 1 else "1790234282676"
want = int(sys.argv[2]) if len(sys.argv) > 2 else 6
name = key if key.endswith(".pkl") else (key if key.endswith(".mp4") else key + ".mp4") + ".pkl"
data = pickle.load(open(os.path.join(harness.CACHE, name), "rb"))
fps, n = data["fps"], data["frames"]

samples = []
observe = TrackCounter.observe


def traced(self, frame, box, kps, conf):
    observe(self, frame, box, kps, conf)
    if self.track_id == want:
        samples.append((int(frame), frame / fps, tuple(float(v) for v in box), None, kps))


TrackCounter.observe = traced
tracker, counter = Tracker(cfg), OnlineCounter(cfg, fps)
for index in range(n):
    items = [(tid, d.box, d.keypoints, d.conf) for tid, d in tracker.update(index, harness._detections(data, index, cfg))]
    counter.update(index, index / fps, items)

snaps = counter.snapshot(n)
print(f"实时链路总览: 共 {sum(s.get('total_count', s['count']) for s in snaps if s['participant'])} 次 / "
      f"{sum(1 for s in snaps if s['participant'])} 人  "
      + " ".join(f"#{s['id']}:{s.get('total_count', s['count'])}({s['signal_source']})"
                 for s in sorted(snaps, key=lambda z: z['id']) if s.get('total_count', s['count'])))
snap = next((s for s in snaps if s["id"] == want), None)
if snap is None:
    sys.exit(f"实时链路里没有 #{want}")
print(f"实时 #{want}: 数到 {snap.get('total_count', snap['count'])} (实测 {snap['count']} + 补 {snap['filled']}) "
      f"源 {snap['signal_source']} 节律 {snap['rhythm_strength']} 判定 {snap['verdict']} {snap['reason']}")
print(f"  样本 {len(samples)} 帧 / 跨度 {(samples[-1][0]-samples[0][0])/fps:.1f}s  "
      f"中位框高 {np.median([s[2][3]-s[2][1] for s in samples]):.0f}px")

res = count_track(want, samples, cfg, fps)
verdict, reason = participation(res.stats, cfg)
print(f"离线同样本: 数到 {res.count + res.stats.get('filled', 0)} (实测 {res.count} + 补 {res.stats.get('filled', 0)}) "
      f"源 {res.signal_source} 节律 {res.stats['rhythm_strength']} 周期 {res.stats.get('period_estimated')} {verdict} {reason}")

med_h = float(np.median([s[2][3] - s[2][1] for s in samples]))
frames = np.array([s[0] for s in samples], dtype=np.int64)
ch = channel_cache(samples, cfg)
y, sub = extract_y_detail(samples, res.signal_source, cfg, ch)
keep = np.isfinite(y)
axis, dense, _ = sg.fill_gaps(frames[keep], -y[keep], cfg.max_gap)
det = sg.detrend(dense, cfg.smooth_win, cfg.baseline_win)
noise = sg.hf_sigma(det)
p95 = float(np.percentile(det, 95))
thr = amplitude_threshold(noise, p95, med_h, cfg)
hard_min = max(2, int(round(fps / cfg.max_rate_hz)))
print(f"信号: 噪声 {noise:.2f}px  p95 {p95:.2f}px  阈值 {thr:.2f}px "
      f"(噪声项 {cfg.snr_k*noise:.2f} / 峰值项 {cfg.peak_frac*p95:.2f} / 下限 {max(cfg.min_amplitude, cfg.amplitude_ratio*med_h)*max(0.05, cfg.floor_relief):.2f} / 封顶 {0.45*p95:.2f})")
print("鼓包普查(阈值越低看得越全):")
for tag, t in (("当前", thr), ("0.6x", 0.6 * thr), ("0.35x", 0.35 * thr), ("0.2x", 0.2 * thr), ("0.1x", 0.1 * thr)):
    pk = sg.find_peaks(det, hard_min, t, 0.8 * t)
    gap = np.median(np.diff(pk)) if len(pk) > 2 else 0
    print(f"  阈值 {t:5.2f}px ({tag:<5}) -> 鼓包 {len(pk):>3} 个  中位间隔 {gap:5.1f}帧 ({fps/gap if gap else 0:.2f}Hz)")
ac, strength = sg.autocorr_period(det, fps, cfg.min_rate_hz, cfg.max_rate_hz)
print(f"自相关周期 {ac:.1f}帧 ({fps/ac if ac else 0:.2f}Hz) 强度 {strength:.3f}")
