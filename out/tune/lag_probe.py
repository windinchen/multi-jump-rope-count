"""诊断: 人群素材漏计 + 实时链路耗时. 用法 lag_probe.py [key=val ...]"""
import os, pickle, sys, time
sys.path.insert(0, "out/tune")
from harness import CACHE, VIDEOS, _detections, overrides, replay
from jumpcount.tracking import Tracker
from jumpcount.online import OnlineCounter

NAME = "1790234282676.mp4"
cfg = overrides(sys.argv[1:])
data = pickle.load(open(os.path.join(CACHE, NAME + ".pkl"), "rb"))
fps, n = data["fps"], data["frames"]

print("== 配置:", {k: getattr(cfg, k) for k in
      ("snr_k", "peak_frac", "floor_relief", "min_lift_ratio", "keep_rhythm_strength",
       "channel_select", "poll_hz", "online_window_sec", "max_height_cv")})

rows, _, kept, prov, expected = replay(cfg, NAME, data)
print(f"\n== 离线  {NAME}  {n/fps:.1f}s")
for r in sorted(rows, key=lambda r: -r[2]):
    tid, verdict, count, exp, rate, vis, src, rhythm, hcv, reason = r
    print(f"  #{tid:<3} {verdict:<11} 数到 {count:>4} 应有 {exp:>6} 完整率 {count/exp if exp else 0:>4.2f} "
          f"{rate or 0:>5.2f}Hz 节律 {str(rhythm):>5} 源 {src:<10}{reason[:24]}")
print(f"  离线合计 计入 {kept} + 低置信 {prov} = {kept+prov} / 应有 {expected:.0f} -> {round((kept+prov)/expected,3)}")

# ---- 实时链路: 单独计耗时 ----
dets = [(i, _detections(data, i, cfg)) for i in range(n)]
for tag, sel in (("节律选源(现默认)", "rhythm"), ("旧策略 coverage", "coverage")):
    cfg.channel_select = sel
    tracker, counter = Tracker(cfg), OnlineCounter(cfg, fps)
    t_track = t_count = 0.0
    for i, dd in dets:
        t0 = time.perf_counter(); items = [(tid, d.box, d.keypoints, d.conf) for tid, d in tracker.update(i, dd)]
        t1 = time.perf_counter(); counter.update(i, i / fps, items); t2 = time.perf_counter()
        t_track += t1 - t0; t_count += t2 - t1
    snaps = counter.snapshot(n)
    tot = sum(s.get("total_count", s["count"]) for s in snaps if s["participant"])
    print(f"\n== 实时 {tag}: 跟踪 {t_track*1000/n:.1f}ms/帧  计数 {t_count*1000/n:.1f}ms/帧  "
          f"合计 {(t_track+t_count)*1000/n:.1f}ms/帧 -> 上限 {36/((t_track+t_count)/n):.0f}fps  总数 {tot}")
