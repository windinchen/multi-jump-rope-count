"""跨帧关联: 给每个人一条稳定的轨迹。

多人跳绳计数的第一号误差来源是 ID 漂移(同一个人被拆成两条轨迹, 或两个人被
合并成一条)。这里用 IoU + 中心距离的混合得分做贪心关联, 并为断轨保留一个
重新认领窗口, 尽量让短暂遮挡后重新出现的人接回原来的 ID。
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .types import Detection


# 长跑(摄像头连续开几小时)时这三样必须封顶, 否则单条轨迹的框高列表会涨到几十万,
# 每帧的 np.median 和内存都会被拖垮。600 帧 = 25fps 下 24s, 比在线信号窗口还长,
# 中位数取"最近这段"反而更贴合当前距离, 短时复盘的结果与不封顶时一致。
HEIGHT_HISTORY = 600
TRACK_HISTORY = 4000   # 参与审计的历史轨迹条数上限
EVENT_HISTORY = 5000   # 断轨重认领事件条数上限


def _geom(box) -> Tuple[float, float, float]:
    x1, y1, x2, y2 = (float(v) for v in box)
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    return x1 + w / 2.0, y1 + h / 2.0, h


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(boxes: np.ndarray, others: np.ndarray) -> np.ndarray:
    """整批框两两 IoU: 关联时每帧算一次, 不再每对框回一次 Python。"""
    if boxes.size == 0 or others.size == 0:
        return np.zeros((boxes.shape[0], others.shape[0]), dtype=np.float64)
    x1 = np.maximum(boxes[:, None, 0], others[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], others[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], others[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], others[None, :, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[:, None]
    area_b = ((others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]))[None, :]
    union = area_a + area_b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


class _Track:
    def __init__(self, tid: int, det: Detection):
        self.id = tid
        self.box = np.asarray(det.box, dtype=np.float64)
        cx, cy, h = _geom(det.box)
        self.center = np.array([cx, cy], dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)
        self.created_frame = det.frame
        self.last_frame = det.frame
        self.hits = 1
        self.frames = 1
        self.missing = 0
        self.conf = float(det.conf)
        self.heights: Deque[float] = deque([h], maxlen=HEIGHT_HISTORY)
        self.gaps: Deque[int] = deque(maxlen=EVENT_HISTORY)

    def median_height(self) -> float:
        """框高中位数: 按轨迹算一次即可, 以前每个"轨迹×检测"对都重算(16 人素材每帧近千次)。"""
        return float(np.median(self.heights))

    def predict(self, frame: int) -> np.ndarray:
        return self.center + self.vel * (frame - self.last_frame)

    def update(self, det: Detection) -> None:
        cx, cy, h = _geom(det.box)
        target = np.array([cx, cy], dtype=np.float64)
        predicted = self.predict(det.frame)
        gain = 0.5 if self.missing == 0 else 0.85
        self.center = predicted + gain * (target - predicted)
        step = max(1, det.frame - self.last_frame)
        self.vel = 0.7 * self.vel + 0.3 * ((target - predicted) / step)
        if self.missing:
            self.gaps.append(self.missing)
        self.box = np.asarray(det.box, dtype=np.float64)
        self.last_frame = det.frame
        self.hits += 1
        self.frames += 1
        self.missing = 0
        self.conf = 0.85 * self.conf + 0.15 * float(det.conf)
        self.heights.append(h)


class Tracker:
    """轻量多目标跟踪器(零外部依赖, 便于在板端替换实现)。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.active: List[_Track] = []
        self.finished: List[_Track] = []
        self.all: Deque[_Track] = deque(maxlen=TRACK_HISTORY)
        self._next_id = 1
        self._known: set = set()
        self.events: Deque[dict] = deque(maxlen=EVENT_HISTORY)

    def _score(self, trk: _Track, det: Detection) -> float:
        cx, cy, h = _geom(det.box)
        pred = trk.predict(det.frame)
        ref = max(h, trk.median_height())
        proximity = max(0.0, 1.0 - float(np.hypot(pred[0] - cx, pred[1] - cy)) / (self.cfg.match_scale * ref))
        return self.cfg.match_iou_w * _iou(trk.box, det.box) + (1.0 - self.cfg.match_iou_w) * proximity

    def _revive(self, frame: int, pending: List[Detection]) -> List[Tuple[_Track, Detection]]:
        """在已断开的轨迹里找失而复得的人, 一个断轨只允许被认领一次。"""
        cfg = self.cfg
        live = {trk.id for trk in self.active}
        candidates: List[Tuple[float, int, int]] = []
        for fi, trk in enumerate(self.finished):
            if trk.id in live or trk.hits < cfg.min_hits:
                continue
            age = frame - trk.last_frame
            if age <= 0 or age > cfg.reid_window:
                continue
            ref = trk.median_height()
            for di, det in enumerate(pending):
                cx, cy, h = _geom(det.box)
                dist = float(np.hypot(trk.center[0] - cx, trk.center[1] - cy))
                if dist <= cfg.reid_dist * ref and 0.6 <= h / ref <= 1.7:
                    candidates.append((dist / (cfg.reid_dist * ref), fi, di))
        candidates.sort()
        used_f: set = set()
        used_d: set = set()
        revived: List[Tuple[_Track, Detection]] = []
        for _, fi, di in candidates:
            if fi in used_f or di in used_d:
                continue
            used_f.add(fi)
            used_d.add(di)
            trk = self.finished[fi]
            self.events.append(
                {"type": "reid", "track_id": trk.id, "frame": frame, "gap_frames": frame - trk.last_frame}
            )
            revived.append((trk, pending[di]))
        for di in sorted(used_d, reverse=True):
            pending.pop(di)
        # 认领就必须把这条轨迹从"已断开"里摘掉: 留在里面, 他下次再断开时同一个对象会被塞回来两次,
        # 之后一帧里两条重号能把同一个人认领两遍 -> active 里出现两个自己, hits 与框位一帧写两次。
        for fi in sorted(used_f, reverse=True):
            self.finished.pop(fi)
        return revived

    def update(self, frame: int, dets: List[Detection]) -> List[Tuple[int, Detection]]:
        cfg = self.cfg
        pending = list(dets)
        pairs: List[Tuple[_Track, Detection]] = []

        # 1) 断轨重认领
        revived: set = set()
        for trk, det in self._revive(frame, pending):
            trk.update(det)
            self.active.append(trk)
            pairs.append((trk, det))
            revived.add(id(trk))

        # 2) 常规贪心关联
        # 刚被认领回来的轨迹本轮已经配过一个框, 再让它参与关联会让同一个人同一帧配到两个框:
        # 框位/框高被写两次, 命中数翻倍, 下游还会把这一刻当成两次起跳。
        # 得分整批算: 16 人同跳时每帧近千个"轨迹×检测"对, 逐对回 Python 比推理还贵
        scored: List[Tuple[float, int, int]] = []
        if self.active and pending:
            det_boxes = np.array([d.box for d in pending], dtype=np.float64)
            trk_boxes = np.array([t.box for t in self.active], dtype=np.float64)
            ious = _iou_matrix(trk_boxes, det_boxes)
            pred = np.array([t.predict(frame) for t in self.active], dtype=np.float64)
            geom = np.array([_geom(b) for b in det_boxes], dtype=np.float64)
            medians = np.array([t.median_height() for t in self.active], dtype=np.float64)[:, None]
            refs = np.maximum(geom[None, :, 2], medians)
            distance = np.hypot(pred[:, None, 0] - geom[None, :, 0], pred[:, None, 1] - geom[None, :, 1])
            proximity = np.clip(1.0 - distance / (cfg.match_scale * np.maximum(refs, 1e-6)), 0.0, None)
            scores = cfg.match_iou_w * ious + (1.0 - cfg.match_iou_w) * proximity
            rows, cols = np.nonzero(scores >= cfg.match_thresh)
            order = np.argsort(-scores[rows, cols], kind="stable") if rows.size else []
            scored = [(float(scores[r, c]), int(r), int(c)) for r, c in zip(rows[order], cols[order])]
        used_t: set = set()
        used_d: set = set()
        for _, ti, di in scored:
            if ti in used_t or di in used_d or id(self.active[ti]) in revived:
                continue
            used_t.add(ti)
            used_d.add(di)
            trk = self.active[ti]
            trk.update(pending[di])
            pairs.append((trk, pending[di]))
        matched = {id(t) for t, _ in pairs}

        # 3) 未命中的活跃轨迹累计丢失, 超时则断开
        keep: List[_Track] = []
        for trk in self.active:
            if id(trk) in matched:
                keep.append(trk)
            else:
                trk.missing += 1
                (keep if trk.missing <= cfg.max_age else self.finished).append(trk)
        self.active = keep

        # 4) 剩下的框开新轨迹
        for di, det in enumerate(pending):
            if di in used_d:
                continue
            trk = _Track(self._next_id, det)
            self._next_id += 1
            self.active.append(trk)
            pairs.append((trk, det))

        self.finished = [t for t in self.finished if frame - t.last_frame <= cfg.reid_window + 5]

        for t, _ in pairs:
            if t.id not in self._known:
                self._known.add(t.id)
                self.all.append(t)
        return [(t.id, d) for t, d in pairs if t.hits >= cfg.min_hits]

    def track_meta(self) -> Dict[int, dict]:
        """每条轨迹的稳定性指标, 供审计与评估使用。"""
        meta: Dict[int, dict] = {}
        for trk in self.all:
            span = max(1, trk.last_frame - trk.created_frame + 1)
            meta[trk.id] = {
                "first_frame": trk.created_frame,
                "last_frame": trk.last_frame,
                "frames": trk.frames,
                "visibility": round(trk.frames / span, 4),
                "mean_conf": round(float(trk.conf), 4),
                "median_height": round(trk.median_height(), 2),
                "breaks": len(trk.gaps),
                "max_gap": int(max(trk.gaps)) if trk.gaps else 0,
            }
        return meta
