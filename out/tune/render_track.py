"""把一条轨迹渲染成胶片图 + 信号图, 用肉眼核对"算法数到的峰"和"真人离地"是否一一对应。

用法: .venv/bin/python out/tune/render_track.py <视频关键字> <轨迹ID> [起始秒] [时长秒] [采样fps]
     .venv/bin/python out/tune/render_track.py <视频关键字> --overview <帧号>
这是给人眼看的真值定标工具: 自我一致指标永远报不出"数过头", 只能靠看图。
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import cv2  # noqa: E402

from harness import CACHE, VIDEOS, _detections  # noqa: E402
from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import channel_cache, count_track, extract_y_detail, pick_source  # noqa: E402
from jumpcount.identity import build_slot, consolidate  # noqa: E402
from jumpcount.tracking import Tracker  # noqa: E402


def track_boxes(video: str, data: dict, cfg: Config):
    """重跑跟踪, 返回 {frame: [(tid, box)]} 和 {tid: samples}。"""
    tracker = Tracker(cfg)
    per_frame: dict = {}
    samples: dict = {}
    centers: dict = {}
    for index in range(data["frames"]):
        dets = _detections(data, index, cfg)
        kept = []
        for tid, det in tracker.update(index, dets):
            kept.append((tid, det.box))
            samples.setdefault(tid, []).append((index, index / data["fps"], det.box, None, det.keypoints))
            centers.setdefault(tid, []).append(((det.box[0] + det.box[2]) / 2, (det.box[1] + det.box[3]) / 2))
        per_frame[index] = kept
    return per_frame, samples, centers


def draw(frame, box, color, text="", thick=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
    if text:
        cv2.putText(frame, text, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def overview(video: str, frame_index: int, cfg: Config) -> None:
    data = pickle.load(open(os.path.join(CACHE, os.path.basename(video) + ".pkl"), "rb"))
    per_frame, samples, centers = track_boxes(video, data, cfg)
    cap = cv2.VideoCapture(os.path.join(ROOT, video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        print("读帧失败")
        return
    h0 = float(np.median([np.median([b[3] - b[1] for _, b in per_frame.get(frame_index, [])] or [0])]))
    for tid, box in sorted(per_frame.get(frame_index, [])):
        draw(frame, box, (0, 255, 0), f"#{tid}")
    out = os.path.join(HERE, f"overview_{os.path.basename(video)[:12]}_{frame_index}.png")
    cv2.imwrite(out, frame)
    print(f"整帧标注图(绿框带轨迹号): {out}  该帧 {len(per_frame.get(frame_index, []))} 个轨迹, 中位框高 {h0:.0f}px")


def filmstrip(video: str, tid: int, start_sec: float, dur_sec: float, sample_fps: float, cfg: Config) -> None:
    data = pickle.load(open(os.path.join(CACHE, os.path.basename(video) + ".pkl"), "rb"))
    fps = data["fps"]
    per_frame, samples, centers = track_boxes(video, data, cfg)
    if tid not in samples:
        print(f"轨迹 #{tid} 不存在, 现有: {sorted(samples)}")
        return
    slot = build_slot(tid, samples[tid], centers[tid])
    res = count_track(tid, slot.samples, cfg, fps)
    peak_frames = {int(e.frame): e for e in res.events}
    by_frame = {}
    for s in slot.samples:
        by_frame[int(s[0])] = s[2]
    source = res.signal_source
    channels = channel_cache(slot.samples, cfg)
    y_series, _ = extract_y_detail(slot.samples, source, cfg, channels=channels)
    sample_frames = [int(s[0]) for s in slot.samples]
    cap = cv2.VideoCapture(os.path.join(ROOT, video))
    start, end = start_sec, start_sec + dur_sec
    want = [f for f in sorted(by_frame)
            if start <= f / fps <= end and (f - start * fps) % max(1, round(fps / sample_fps)) < 1]
    want = want[:60]
    tiles, labels = [], []
    tile_h = 0
    for f in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        box = by_frame[f]
        x1, y1, x2, y2 = [int(v) for v in box]
        pad = int(0.12 * max(x2 - x1, y2 - y1))
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2].copy()
        draw(crop, [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1], (0, 0, 255), thick=2)
        for other_tid, ob in per_frame.get(f, []):
            if other_tid != tid:
                draw(crop, [ob[0] - cx1, ob[1] - cy1, ob[2] - cx1, ob[3] - cy1], (200, 200, 200), thick=1)
        mark = f"{f / fps:.2f}s"
        if f in peak_frames:
            e = peak_frames[f]
            mark += " PEAK" + ("R" if e.recovered else "M")
            cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (0, 255, 0), 4)
        idx = sample_frames.index(f) if f in sample_frames else -1
        if idx >= 0 and np.isfinite(y_series[idx]):
            mark += f" y={-y_series[idx]:+.0f}"
        h, w = crop.shape[:2]
        scale = 200.0 / max(h, w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)))
        cv2.putText(crop, mark, (2, crop.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(crop)
        tile_h = max(tile_h, crop.shape[0])
    if not tiles:
        print("没有可用的帧")
        return
    cols = 10
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * (tile_h + 2), cols * 210, 3), 30, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        y0 = r * (tile_h + 2)
        canvas[y0:y0 + tile.shape[0], c * 210:c * 210 + tile.shape[1]] = tile
    out = os.path.join(HERE, f"strip_{os.path.basename(video)[:12]}_{tid}_{int(start_sec)}-{int(end)}s.png")
    cv2.imwrite(out, canvas)
    peaks_in = [f for f in want if f in peak_frames]
    print(f"胶片图: {out}")
    print(f"轨迹 #{tid} 判定信号源={source} 总计数={res.count}+补齐{res.stats.get('filled', 0)} 周期={res.period_sec}")
    print(f"窗口 {start}~{end}s 采样 {len(want)} 帧(约 {sample_fps}fps), 其中标绿框=算法判定的峰 {len(peaks_in)} 个: "
          f"{[round(p / fps, 2) for p in peaks_in]}")



def spacetime(video: str, tid: int, cfg: Config, start: float, dur: float) -> None:
    """时空展开图: 沿时间轴逐帧取一条竖向窄带堆叠成图, 离地/落地变成明显的波纹, 便于肉眼逐次核对。

    左侧绿线 = 算法判定的峰(实测实线/回捞虚线), 灰线 = 该帧没有这个轨迹。
    """
    data = pickle.load(open(os.path.join(CACHE, os.path.basename(video) + ".pkl"), "rb"))
    fps = data["fps"]
    per_frame, samples, centers = track_boxes(video, data, cfg)
    if tid not in samples:
        print(f"轨迹 #{tid} 不存在, 现有: {sorted(samples)}")
        return
    slot = build_slot(tid, samples[tid], centers[tid])
    res = count_track(tid, slot.samples, cfg, fps)
    by_frame = {int(s[0]): np.asarray(s[2], dtype=float) for s in slot.samples}
    peak_frames = {int(round(e.frame)): e.kind for e in res.events}
    cap = cv2.VideoCapture(os.path.join(ROOT, video))
    band_w, row_h = 96, 3
    rows, marks = [], []
    f0, f1 = int(start * fps), int((start + dur) * fps)
    for f in range(f0, min(f1, data["frames"])):
        box = by_frame.get(f)
        if box is None:
            rows.append(np.full((row_h, band_w, 3), 40, dtype=np.uint8))
            marks.append(2)
            continue
        cx = (box[0] + box[2]) / 2.0
        x1 = int(max(0, cx - band_w / 2))
        x2 = int(min(cap.get(cv2.CAP_PROP_FRAME_WIDTH), x1 + band_w))
        y1, y2 = int(box[1]), int(min(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), box[3] + 0.12 * (box[3] - box[1])))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if crop.ndim != 3 or crop.shape[0] < 2 or crop.shape[1] < 2:
            continue   # 框贴到画面边缘时裁出来只有一两行, resize 会退化成二维数组, 跳过
        crop = cv2.resize(crop, (band_w, row_h), interpolation=cv2.INTER_AREA)
        rows.append(crop)
        marks.append(0 if f in peak_frames and peak_frames[f] == "jump" else
                     (1 if f in peak_frames else 3))
    if not rows:
        print("没有可用的帧")
        return
    h = int(np.median([r.shape[0] for r in rows]))
    canvas = np.zeros((len(rows) * h, band_w + 12, 3), dtype=np.uint8)
    for i, r in enumerate(rows):
        canvas[i * h:i * h + r.shape[0], 12:12 + r.shape[1]] = r
        color = {(0): (0, 255, 0), (1): (0, 200, 255), (2): (90, 90, 90), (3): (255, 255, 255)}[marks[i]]
        cv2.line(canvas, (0, i * h), (10, i * h), color, 1)
    scale = 4
    canvas = cv2.resize(canvas, (canvas.shape[1] * scale, canvas.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    out = os.path.join(HERE, f"st_{os.path.basename(video)[:12]}_{tid}_{int(start)}s.png")
    cv2.imwrite(out, canvas)
    n_peak = sum(1 for m in marks if m in (0, 1))
    print(f"时空图: {out}  行=帧(自上而下 {start}~{start + dur:.0f}s), 窗口内标记峰 {n_peak} 个 "
          f"(绿=实测, 黄=其它事件), 灰条=该帧无观测")

def data_span(video: str) -> float:
    data = pickle.load(open(os.path.join(CACHE, os.path.basename(video) + ".pkl"), "rb"))
    return data["frames"] / data["fps"]


def plot_track(video: str, tid: int, cfg: Config, start: float, end: float) -> None:
    """画去基线后的位移信号 + 阈值 + 每个峰的来源标记, 用肉眼判断"一次起跳被数成两次"和漏计形态。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = pickle.load(open(os.path.join(CACHE, os.path.basename(video) + ".pkl"), "rb"))
    fps = data["fps"]
    from jumpcount import signal as sg

    _pf, samples, centers = track_boxes(video, data, cfg)
    if tid not in samples:
        print(f"轨迹 #{tid} 不存在, 现有: {sorted(samples)}")
        return
    slot = build_slot(tid, samples[tid], centers[tid])
    res = count_track(tid, slot.samples, cfg, fps)
    frames = np.array([int(s[0]) for s in slot.samples], dtype=np.int64)
    channels = channel_cache(slot.samples, cfg)
    y_series, _ = extract_y_detail(slot.samples, res.signal_source, cfg, channels=channels)
    raw = -np.where(np.isfinite(y_series), y_series, np.nan)
    keep = np.isfinite(raw)
    axis, dense, _ = sg.fill_gaps(frames[keep], raw[keep], cfg.max_gap)
    detrended = sg.detrend(dense, cfg.smooth_win, cfg.baseline_win)
    t = axis / fps
    amp_thr = float(res.stats.get("amplitude_threshold", 0.0))

    fig, ax = plt.subplots(figsize=(16, 4.5), dpi=100)
    ax.plot(t, detrended, color="#333", lw=0.9, label=f"detrended ({res.signal_source})")
    ax.axhline(amp_thr, color="tab:red", ls="--", lw=1.0, label=f"amp_thr={amp_thr:.1f}")
    ax.axhline(0, color="#bbb", lw=0.6)
    marks = {"jump": ("tab:green", "^"), "bump": ("tab:orange", "v"), "interpolated": ("tab:gray", "x")}
    for e in res.events:
        color, marker = marks.get(e.kind, ("tab:purple", "o"))
        if e.kind == "jump" and e.recovered:
            color, marker = "tab:blue", "*"
        ax.plot([e.frame / fps], [e.amplitude], marker, color=color, ms=9 if marker != "x" else 7,
                mec="k", mew=0.4, zorder=5)
    ax.set_xlim(start, end)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("up-positive displacement (px)")
    ax.set_title(f"{os.path.basename(video)} track#{tid}  total={res.stats.get('total_count')} "
                 f"period={res.period_sec}s rhythm={res.stats.get('rhythm_strength')} "
                 f"verdict={res.stats.get('verdict')} src={res.signal_source} | "
                 f"green^=measured  blue*=recovered  orange v=bump  gray x=filled")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    out = os.path.join(HERE, f"plot_{os.path.basename(video)[:12]}_{tid}_{int(start)}-{int(end)}s.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"信号图: {out}")
    print(f"窗口 {start}~{end}s: 窗口内事件 {sum(1 for e in res.events if start <= e.frame / fps <= end)} 个 "
          f"(实测 {sum(1 for e in res.events if e.kind == 'jump' and not e.recovered and start <= e.frame / fps <= end)}, "
          f"回捞 {sum(1 for e in res.events if e.kind == 'jump' and e.recovered and start <= e.frame / fps <= end)}, "
          f"bump {sum(1 for e in res.events if e.kind == 'bump' and start <= e.frame / fps <= end)}, "
          f"补齐 {sum(1 for e in res.events if e.kind == 'interpolated' and start <= e.frame / fps <= end)})")


if __name__ == "__main__":
    cfg = Config()
    # 尾部可跟 key=value 覆盖配置, 便于同一份素材对照两档参数(见 dump_tracks.py 同风格)
    for item in [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]:
        sys.argv.remove(item)
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
    vid_kw = sys.argv[1]
    video = next((v for v in VIDEOS if vid_kw in v), None)
    if not video:
        print(f"找不到视频 {vid_kw}, 可选: {[os.path.basename(v) for v in VIDEOS]}")
        raise SystemExit(1)
    if len(sys.argv) > 2 and sys.argv[2] == "--overview":
        overview(video, int(sys.argv[3]) if len(sys.argv) > 3 else 0, cfg)
    elif len(sys.argv) > 2 and sys.argv[2] == "--st":
        spacetime(video, int(sys.argv[3]), cfg,
                  float(sys.argv[4]) if len(sys.argv) > 4 else 0.0,
                  float(sys.argv[5]) if len(sys.argv) > 5 else 6.0)
    elif len(sys.argv) > 2 and sys.argv[2] == "--plot":
        plot_track(video, int(sys.argv[3]), cfg,
                   float(sys.argv[4]) if len(sys.argv) > 4 else 0.0,
                   float(sys.argv[5]) if len(sys.argv) > 5 else data_span(video))
    else:
        filmstrip(video, int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.0,
                  float(sys.argv[4]) if len(sys.argv) > 4 else 5.0,
                  float(sys.argv[5]) if len(sys.argv) > 5 else 10.0, cfg)


