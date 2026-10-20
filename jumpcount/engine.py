"""实时引擎: 采集 -> YOLO 姿态 -> 跟踪 -> 在线计数, 供页面轮询与拉流。"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import Config, snr_k_to_sensitivity
from .detect import build_detector
from .online import OnlineCounter, box_iou
from .render import draw_live
from .tracking import Tracker


class LiveEngine:
    def __init__(self, source: str, weights: Optional[str], cfg: Optional[Config] = None, realtime: bool = True,
                 jpeg_width: int = 1280, stream_fps: float = 15.0):
        self.cfg = cfg or Config()
        self.source = source
        self.weights = weights
        self.realtime = realtime
        self.jpeg_width = jpeg_width
        self.stream_fps = float(stream_fps)  # 出图帧率上限: 计数用原始帧, 这里只影响页面看着跟不跟手
        self.detector = build_detector(weights, self.cfg)
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.paused = False
        self.frame_id = 0
        self.jpeg: Optional[bytes] = None
        self.started_at = time.time()
        self.state: dict = {
            "running": False,
            "paused": False,
            "source": str(source),
            "detector": self.detector.name,
            "fps": 0.0,
            "measured_fps": 0.0,
            "proc_fps": 0.0,
            "frame": 0,
            "time": 0.0,
            "people": 0,
            "participants": 0,
            "participants_confident": 0,
            "total": 0,
            "total_direct": 0,
            "total_filled": 0,
            "total_recovered": 0,
            "total_confident": 0,
            "rope_total": 0,
            "tracks": [],
            "events": [],
            "warnings": [],
            "stage_ms": {},
            "error": None,
        }

    # ---- 生命周期 ----
    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=3.0)

    def control(self, action: str, **kwargs) -> dict:
        if action == "reset":
            with self.lock:
                self._reset_needed = True
        elif action == "pause":
            self.paused = True
        elif action == "resume":
            self.paused = False
        elif action == "source" and kwargs.get("source") is not None:
            self.pending_source = str(kwargs["source"])
        elif action == "config":
            for key, value in kwargs.items():
                if hasattr(self.cfg, key) and value is not None:
                    setattr(self.cfg, key, value)
        return self.snapshot()

    _reset_needed = False

    # ---- 主循环 ----
    def _open(self, source):
        try:
            index = int(source)
        except (TypeError, ValueError):
            index = source
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"打不开画面源: {source}")
        return cap

    def _bind(self, source):
        """打开画面源, 帧率以画面源声明值为准(缺失时按 25fps)。"""
        cap = self._open(source)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1.0 or fps > 120:
            fps = 25.0
        return cap, fps, Tracker(self.cfg), OnlineCounter(self.cfg, fps)

    def _loop(self) -> None:
        cap = None
        frame_index = 0
        recent: List[float] = []
        proc: List[float] = []
        # 分段耗时(毫秒滑动平均): 页面卡顿排查用, 直接显示在页面底部
        stages: Dict[str, List[float]] = {k: [] for k in ("采集", "推理", "跟踪计数", "缩小", "叠加", "编码")}
        next_shot = 0.0
        try:
            cap, fps, tracker, counter = self._bind(self.source)
        except Exception as exc:  # 摄像头被占用或路径不存在
            with self.lock:
                self.state.update({"running": False, "error": str(exc)})
            return

        while not self.stop_flag.is_set():
            if getattr(self, "pending_source", None):
                pending, self.pending_source = self.pending_source, None
                try:
                    cap.release()
                    cap, fps, tracker, counter = self._bind(pending)
                    self.source = pending
                    frame_index = 0
                except Exception as exc:
                    with self.lock:
                        self.state["error"] = str(exc)
            if self._reset_needed:
                self._reset_needed = False
                counter.reset()
                tracker = Tracker(self.cfg)
                frame_index = 0
            if self.paused:
                time.sleep(0.1)
                with self.lock:
                    self.state["paused"] = True
                continue
            with self.lock:
                self.state["paused"] = False

            t0 = time.time()
            ok, frame = cap.read()
            t_read = time.time() - t0
            if not ok:
                cap.release()
                try:
                    cap, fps, tracker, counter = self._bind(self.source)
                    frame_index = 0
                except Exception as exc:
                    with self.lock:
                        self.state["error"] = str(exc)
                    break
                continue

            t1 = time.time()
            dets = self.detector.detect(frame, frame_index, frame_index / fps)
            t_det = time.time() - t1
            t1b = time.time()
            pairs = tracker.update(frame_index, dets)
            items = [(tid, det.box, det.keypoints, det.conf) for tid, det in pairs]
            emitted = counter.update(frame_index, frame_index / fps, items)
            tracks = counter.snapshot(frame_index)  # 框位已在 update 内按站位记录, 这里不再回填
            # 站位会替离开的人保管计数, 但画面上的框要及时消失, 否则会定格在空地上像多了几个路人
            ghost_limit = max(2, int(round(self.cfg.ghost_display_sec * fps)))
            live = [t for t in tracks if t.get("box") and t.get("absent_frames", 0) <= ghost_limit]
            t_count = time.time() - t1b

            t2 = time.time()
            encoded = (False, None)
            # 出图比计数慢半拍时, 浏览器会把看不过来的帧攒在手里越拖越久, 所以限个上限
            shot = self.stream_fps <= 0 or t2 >= next_shot
            if shot:
                if self.stream_fps > 0:
                    next_shot = t2 + 1.0 / self.stream_fps
                # 先缩小再叠加: 2.5K 原图上画框画折线比推理本身还费时间
                base = self._fit(frame)
                t_fit = time.time() - t2
                t3 = time.time()
                annotated = draw_live(
                    base,
                    # 去重只管画面, 计数列表原样保留: 砍掉的框连带砍掉了他的计数
                    _hide_duplicate_boxes(live, self.cfg),
                    fps,
                    frame_index / fps,
                    coords_scale=base.shape[1] / float(frame.shape[1]),
                )
                t_draw = time.time() - t3
                t4 = time.time()
                encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                t_enc = time.time() - t4
            else:
                t_fit = t_draw = t_enc = 0.0

            now = time.time()
            recent.append(now)
            proc.append(now - t0)
            if len(recent) > 60:
                recent.pop(0)
                proc.pop(0)
            timings = [("采集", t_read), ("推理", t_det), ("跟踪计数", t_count)]
            if shot:
                timings += [("缩小", t_fit), ("叠加", t_draw), ("编码", t_enc)]
            for key, value in timings:
                bucket = stages[key]
                bucket.append(value)
                if len(bucket) > 60:
                    del bucket[0]
            stage_ms = {k: round(1000.0 * sum(v) / len(v), 1) for k, v in stages.items() if v}
            stage_ms["合计"] = round(sum(stage_ms.values()), 1)
            measured = len(recent) / max(1e-6, (recent[-1] - recent[0])) if len(recent) > 5 else 0.0
            processed = len(proc) / max(1e-6, sum(proc)) if proc else 0.0
            frame_index += 1
            if encoded[0]:
                self.jpeg = encoded[1].tobytes()
                self.frame_id += 1

            participants = [t for t in tracks if t["participant"]]
            confident = [t for t in participants if t.get("verdict") == "in"]
            with self.lock:
                self.state.update(
                    {
                        "running": True,
                        "error": None,
                        "source": str(self.source),
                        "fps": round(float(fps), 3),
                        "measured_fps": round(measured, 2),
                        "proc_fps": round(processed, 2),
                        "frame": frame_index,
                        "time": round(frame_index / fps, 2),
                        "people": len(live),
                        "participants": len(participants),
                        "participants_confident": len(confident),
                        "total": int(sum(t["count"] + t.get("filled", 0) for t in participants)),
                        "total_direct": int(sum(t["count"] for t in participants)),
                        "total_filled": int(sum(t.get("filled", 0) for t in participants)),
                        "total_recovered": int(sum(t.get("recovered", 0) for t in participants)),
                        "total_confident": int(sum(t["count"] + t.get("filled", 0) for t in confident)),
                        "rope_total": int(sum(t["rope"] for t in participants)),
                        "tracks": tracks,
                        "stage_ms": stage_ms,
                        "events": list(counter.log)[-40:],
                        "warnings": _warnings(live, self.cfg),
                    }
                )

            if self.realtime and self._is_file(self.source):
                time.sleep(max(0.0, 1.0 / fps - (time.time() - t0)))
        if cap is not None:
            cap.release()
        with self.lock:
            self.state["running"] = False

    @staticmethod
    def _is_file(source) -> bool:
        return isinstance(source, str) and not source.isdigit() and os.path.exists(source)

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self.jpeg_width:
            return frame
        scale = self.jpeg_width / float(w)
        # 缩小到非整数倍(如 2560->1152)时 INTER_AREA 走的是通用慢路径, 单帧能吃掉 25ms,
        # 比推理还贵; 预览用双线性, 画质损失看不出来但快了百倍。
        return cv2.resize(
            frame,
            (self.jpeg_width, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_LINEAR,
        )

    def snapshot(self) -> dict:
        with self.lock:
            state = dict(self.state)
            state["tracks"] = [dict(t) for t in self.state["tracks"]]
            state["events"] = list(self.state["events"])
            state["policy"] = self.cfg.policy
            state["signal_source"] = self.cfg.signal_source
            state["imgsz"] = self.cfg.imgsz
            state["conf"] = self.cfg.conf
            state["sensitivity"] = snr_k_to_sensitivity(self.cfg.snr_k)  # 页面滑杆刻度
            state["snr_k"] = self.cfg.snr_k                              # 阈值原始倍数
        return state


def _hide_duplicate_boxes(tracks: List[dict], cfg: Config) -> List[dict]:
    """画面上把压在活人身上的旧框去掉, 当前帧命中的人一个都不许动。

    跟丢的人会把他最后一次框位留在原处, 叠在旁边还在动的人身上就像一个人套了好几个框。
    实测两个挨着站的孩子框重叠能到 0.91、框高只差 2%, 所以不能按重叠度砍活框:
    那样会把真跳者挤掉、总计数直接归零。这里只允许"更新鲜的框"盖掉"更旧的框"。

    只用于叠加层: 计数列表里每个人的计数都要留着, 否则替他保管次数的人一被去掉, 总计就少了。
    """
    ordered = sorted(tracks, key=lambda t: (t.get("absent_frames", 0), t["id"]))
    kept: List[dict] = []
    for t in ordered:
        absent = t.get("absent_frames", 0)
        if any(k.get("absent_frames", 0) < absent and box_iou(t["box"], k["box"]) >= cfg.dup_box_iou for k in kept):
            continue
        kept.append(t)
    kept.sort(key=lambda t: t["id"])
    return kept


def _warnings(tracks: List[dict], cfg: Config) -> List[dict]:
    out: List[dict] = []
    for t in tracks:
        if not t.get("box"):
            continue
        tid = t["id"]
        if not t["participant"]:
            out.append({"level": "muted", "track": tid, "text": f"#{tid} 不计数：{t['reason'] or '刚进入画面'}"})
            continue
        if t.get("verdict") == "provisional":
            out.append({"level": "warn", "track": tid, "text": f"#{tid} 低置信计入：{t.get('reason') or '证据偏弱'}"})
        if (t.get("snr") or 0) < cfg.snr_k:
            out.append({"level": "warn", "track": tid, "text": f"#{tid} 起跳幅度接近噪声水平（信噪比 {t.get('snr')}），计数可能偏少"})
        if (t.get("visibility") or 1) < 0.75:
            out.append({"level": "warn", "track": tid, "text": f"#{tid} 有 {round((1 - t['visibility']) * 100)}% 的时间被遮挡或跟丢，这期间可能漏计"})
        if t.get("rate_hz") and not (cfg.min_rate_hz * 0.8 <= t["rate_hz"] <= cfg.max_rate_hz):
            out.append({"level": "warn", "track": tid, "text": f"#{tid} 节奏 {t['rate_hz']}Hz 超出正常跳绳区间"})
    return out


def list_sources(extra: Optional[List[str]] = None, scan_dirs: Optional[List[str]] = None) -> List[dict]:
    """页面下拉框里的画面源: 本机摄像头 + 素材目录里的视频。"""
    out = [{"id": "0", "label": "本机摄像头", "kind": "camera"}]
    for directory in scan_dirs or []:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            if name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                path = os.path.join(directory, name)
                out.append({"id": path, "label": name, "kind": "file", "size_mb": round(os.path.getsize(path) / 1e6, 1)})
    for path in extra or []:
        if os.path.exists(path) and not any(o["id"] == path for o in out):
            out.append({"id": path, "label": os.path.basename(path), "kind": "file"})
    return out
