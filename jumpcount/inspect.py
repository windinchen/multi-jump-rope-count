"""多信号对照核查: 同一个人分别用脚踝/髋/头/框底边计数, 看差异出在哪。

脚踝关键点最容易抖出多余的鼓包(把一次起跳数成两次), 而髋和头代表整体离地,
更接近"真的跳了一下"。三条曲线画在一起就能判断该用哪条, 以及是不是在重复计数。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .config import Config
from .counter import count_track, feature_y
from .types import Detection


def series_for(samples: List[Tuple], source: str, cfg: Config) -> List[Tuple]:
    out: List[Tuple] = []
    for sample in samples:
        frame, time, box = sample[0], sample[1], sample[2]
        kps = sample[4] if len(sample) > 4 else None
        det = Detection(frame=int(frame), time=float(time), box=tuple(float(v) for v in box), conf=1.0, keypoints=kps)
        value = feature_y(det, source)
        if value is None:
            value = feature_y(det, "box_bottom")
        out.append((int(frame), float(time), tuple(float(v) for v in box), float(value), kps))
    return out


def compare(samples: List[Tuple], cfg: Config, fps: float, track_id: int = 1) -> Dict[str, dict]:
    report: Dict[str, dict] = {}
    for source in ("ankle", "hip", "head", "box_bottom"):
        data = series_for(samples, source, cfg)
        if len(data) < 10:
            continue
        res = count_track(track_id, data, cfg, fps, source)
        if res.stats.get("reject"):
            continue
        frames, lift = [], []
        for item in data:
            frames.append(item[0])
            lift.append(item[3])
        report[source] = {
            "count": res.count,
            "count_rope": res.count_rope,
            "rate_hz": res.rate_hz,
            "period_sec": res.period_sec,
            "snr": res.snr,
            "lift_ratio": res.stats.get("lift_ratio"),
            "period_source": res.stats.get("period_source"),
            "coverage": res.stats.get("signal_coverage"),
            "peak_times": [round(e.time, 3) for e in res.events],
            "frames": frames,
            "lift": lift,
        }
    return report


CJK_CANDIDATES = ("PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei")


def pick_font() -> bool:
    """matplotlib 默认字体没有中文, 找到中文字体就用, 否则退回英文标签。"""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_CANDIDATES:
        if name in installed:
            import matplotlib

            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


def plot(report: Dict[str, dict], out_png: str, fps: float = 25.0, title: str = "") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cjk = pick_font()
    lab_curve = "向上位移(去基线)" if cjk else "upward lift (detrended)"
    lab_peak = "判为起跳 {} 次" if cjk else "counted jumps: {}"
    title = title or ("同一人的多信号计数对照" if cjk else "Per-signal counting comparison")

    sources = list(report)
    fig, axes = plt.subplots(len(sources), 1, figsize=(13, 2.3 * len(sources) + 1.4), sharex=True)
    if len(sources) == 1:
        axes = [axes]
    for axis, source in zip(axes, sources):
        item = report[source]
        frames = np.array(item["frames"], dtype=float)
        raw = np.array(item["lift"], dtype=float)
        baseline = np.median(raw)
        axis.plot(frames, baseline - raw, lw=1.1, color="#2f6fb5", label=lab_curve)
        peak_frames = [float(t) * fps for t in item["peak_times"]]
        if peak_frames:
            vals = [baseline - raw[min(len(raw) - 1, int(round(p - frames[0])))] for p in peak_frames]
            axis.plot(peak_frames, vals, "x", color="#d1495b", ms=7, mew=2, label=lab_peak.format(len(peak_frames)))
        axis.set_ylabel(source)
        axis.grid(alpha=.25)
        axis.set_title(f"{source}: 计数 {item['count']}  节奏 {item['rate_hz']}Hz  信噪比 {item['snr']}  幅度/身高 {item['lift_ratio']}  信号覆盖 {item.get('coverage', '-')}", fontsize=10)
        axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("帧号")
    fig.suptitle(title or "同一人的多信号计数对照", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png

