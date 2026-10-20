"""把计数结果画回视频, 方便肉眼复核谁被数错了。"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

PALETTE = [
    (80, 220, 255),
    (120, 255, 120),
    (255, 160, 90),
    (220, 130, 255),
    (120, 200, 255),
    (90, 230, 200),
    (200, 200, 120),
    (255, 120, 160),
]


def color_for(track_id: int):
    return PALETTE[int(track_id) % len(PALETTE)]


def draw(
    frame: np.ndarray,
    items: Sequence[Tuple[int, Tuple[float, float, float, float]]],
    running: Dict[int, int],
    fps: float,
    time: float,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    for track_id, box in items:
        x1, y1, x2, y2 = (int(v) for v in box)
        color = color_for(track_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"#{track_id} x{running.get(track_id, 0)}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, top), (x1 + tw + 8, top + th + 6), color, -1)
        cv2.putText(out, label, (x1 + 4, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    strip_h = 34
    cv2.rectangle(out, (0, h - strip_h), (w, h), (18, 18, 18), -1)
    total = sum(running.values())
    text = f"t={time:6.2f}s  total={total}"
    cv2.putText(out, text, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    return out


STATE_COLORS = {
    "counting": (90, 220, 120),   # 正在计数
    "walking": (170, 170, 170),   # 走动路人, 不计数
    "weak": (90, 170, 255),       # 信号不足
    "warmup": (255, 210, 90),     # 预热中
}


def draw_live(frame, tracks, fps, time, show_signal: bool = True, coords_scale: float = 1.0):
    """实时页面用的叠加层: 正在计数的人画整框和数字, 不计数的只在脚底留个标记。

    coords_scale: 轨迹框的坐标是按原始分辨率量的, 而画面可能已经先缩小过,
    这里按比例换算, 保证框仍然贴在人身上。
    """
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.5, min(2.0, w / 1280.0))
    spark_h = max(12, int(34 * coords_scale))
    spark_w = max(1, int(round(2 * coords_scale)))
    strip = int(46 * scale)
    # 先画不计数的人, 再画正在计数的人: 脚底标记再怎么叠也压不住计数框和数字
    for item in sorted(tracks, key=lambda t: 0 if t.get("participant") else 1):
        box = item.get("box")
        if not box:
            continue
        x1, y1, x2, y2 = (int(v * coords_scale) for v in box)
        color = _state_color(item)
        if not item.get("participant"):
            _foot_mark(out, item, (x1, y1, x2, y2), color, scale, strip)
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(2, int(2 * scale)))
        label = f"#{item['id']}  {item['count']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, max(1, int(2 * scale)))
        top = max(0, y1 - th - 8)
        cv2.rectangle(out, (x1, top), (x1 + tw + 10, top + th + 8), color, -1)
        cv2.putText(out, label, (x1 + 5, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (25, 25, 25), max(1, int(2 * scale)), cv2.LINE_AA)
        if show_signal and item.get("signal"):
            _sparkline(out, item["signal"], (x1, y2), (x2, y2), color, spark_h, spark_w)
    cv2.rectangle(out, (0, h - strip), (w, h), (20, 20, 20), -1)
    counting = [t for t in tracks if t.get("participant")]
    total = sum(t.get("count", 0) for t in counting)
    text = f"t={time:5.1f}s   人数 {len(tracks)}   计数中 {len(counting)}   总计 {total}"
    cv2.putText(out, text, (int(12 * scale), h - int(14 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, (240, 240, 240), max(1, int(2 * scale)), cv2.LINE_AA)
    return out


def _state_color(item) -> tuple:
    if item.get("participant"):
        return STATE_COLORS["counting"]
    if "走动" in (item.get("reason") or "") or "走路" in (item.get("reason") or ""):
        return STATE_COLORS["walking"]
    if item.get("reason"):
        return STATE_COLORS["warmup"]
    return STATE_COLORS["weak"]


def _short_tag(item) -> str:
    """脚底标签只留一个词, 完整原因在右侧卡片里看。"""
    reason = item.get("reason") or ""
    if "走动" in reason or "走路" in reason:
        return "走动"
    if "只出现" in reason or "预热" in reason or "刚进入" in reason:
        return "预热"
    if "可见率" in reason or "遮挡" in reason:
        return "遮挡"
    if "数到" in reason or "幅度" in reason or "信噪比" in reason:
        return "信号弱"
    return "不计数"


def _halo_text(canvas, text, org, color, font_scale, thickness) -> None:
    """描边字, 不铺底色块, 尽量少挡画面。"""
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (12, 12, 12), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def _foot_mark(canvas, item, box, color, scale, strip) -> None:
    """不计数的人: 脚底一条短线 + 一个短标签, 不再画整框挡住别人。"""
    x1, _, x2, y2 = box
    h = canvas.shape[0]
    y = max(0, min(h - 2, y2))
    cv2.line(canvas, (x1, y), (x2, y), color, max(2, int(3 * scale)), cv2.LINE_AA)
    tag = f"#{item['id']} {_short_tag(item)} {item.get('count', 0)}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, max(1, int(scale)))
    ty = y + th + 7
    if ty + 3 > h - strip:  # 底下就是统计栏了, 挪到脚上方
        ty = y - int(9 * scale)
    _halo_text(canvas, tag, (x1, ty), color, 0.5 * scale, max(1, int(round(scale))))


def _sparkline(canvas, values, p0, p1, color, height: int = 34, thickness: int = 2) -> None:
    """把最近一段垂直位移信号画在框下方, 方便判断是不是真的在跳。"""
    pts = [float(v) for v in values][-60:]
    if len(pts) < 4:
        return
    x0, x1 = min(p0[0], p1[0]), max(p0[0], p1[0])
    y0 = min(canvas.shape[0] - height - 2, max(p0[1], p1[1]) + 4)
    span = max(1, x1 - x0)
    lo, hi = min(pts), max(pts)
    rng = max(1e-6, hi - lo)
    # 一个人 60 段就是 60 次绘制调用, 十几个人时这里比推理还贵, 合成一条折线一次画完
    steps = max(1, len(pts) - 1)
    poly = np.empty((len(pts), 1, 2), dtype=np.int32)
    for i, v in enumerate(pts):
        poly[i, 0, 0] = x0 + span * i / steps
        poly[i, 0, 1] = y0 + height - height * (v - lo) / rng
    cv2.polylines(canvas, [poly], False, color, thickness, cv2.LINE_AA)
