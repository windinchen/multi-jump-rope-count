"""实时引擎: 采集 -> YOLO 姿态 -> 跟踪 -> 在线计数, 供页面轮询与拉流。"""

from __future__ import annotations

import os
import re
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


# 页面只会用到这几个控制指令; 拼错的指令以前会被静默吞掉, 页面看着"成功"了其实什么都没发生
KNOWN_ACTIONS = ("reset", "pause", "resume", "source", "config", "status")


class FrameFeeder:
    """摄像头取帧线程: 只保留最新一帧, 处理不过来时把旧帧直接丢掉。

    原来主循环是单线程的 `read -> 推理 -> 跟踪计数 -> 出图`。摄像头的出帧速度由设备决定,
    一旦这一环比采集慢, 帧就在驱动队列里越积越多: 页面画面越走越晚, 延迟单调累积,
    计数算的还是"三十秒前的人", 而且页面上看不出落后了多少。

    把取帧单独拆出来以后队列长度恒为 1(覆盖式写入), 延迟就等于一帧。
    暂停时主循环不再取帧, 线程继续覆盖同一格, 恢复后拿到的是当下这一帧而不是积压的旧帧——
    原来暂停三十秒再恢复, 页面要先花三十秒把攒下的旧帧补播完才回到实时。

    视频文件**不能**这么做: 计数用的是原始帧, 丢帧等于丢计数, 文件源保持原来的单线程顺序读。
    """

    def __init__(self, cap, timeout: float = 3.0):
        self.cap = cap
        self.timeout = float(timeout)
        self.condition = threading.Condition()
        self.stop_flag = threading.Event()
        self.frame = None
        self.ready = False    # 有一帧还没被取走
        self.failed = False   # 读到结尾/读失败时才置位: 让主循环按"采集停了"去重开画面源
        self.released = False
        self.reads = 0        # 从驱动读到的帧数, 与取走的帧数对比就是丢帧数
        self.taken = 0        # 主循环取走的帧数
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_flag.is_set():
            try:
                ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None
            if self.stop_flag.is_set():
                break
            with self.condition:
                if self.stop_flag.is_set():
                    break
                if not ok:
                    self.failed = True
                    self.condition.notify_all()
                    break
                self.reads += 1
                self.frame = frame   # 上一帧没被取走就直接覆盖: 丢的是旧帧, 不是新帧
                self.ready = True
                self.condition.notify_all()

    def read(self):
        """取当下最新的一帧; 超过 timeout 没等到新帧就报"读不到", 交给主循环重开画面源。"""
        deadline = time.time() + self.timeout
        with self.condition:
            while not (self.ready or self.failed or self.stop_flag.is_set()):
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False, None
                self.condition.wait(remaining)
            if not self.ready:
                return False, None
            frame = self.frame
            self.frame = None
            self.ready = False
            self.taken += 1
            return True, frame

    def dropped(self) -> int:
        """被覆盖掉的旧帧数: 持续上涨说明推理跟不上摄像头的帧率。"""
        with self.condition:
            return self.reads - self.taken

    def release(self) -> None:
        self.stop_flag.set()
        with self.condition:
            self.condition.notify_all()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if not self.released:
            self.released = True
            self.cap.release()


class StageTimings:
    """分段耗时与帧率的滑动平均。

    单独成类是因为换画面源/视频循环重开时这些必须整体作废: 攒着上一个源的数字继续往下显示,
    看到的"每帧多少毫秒"其实是上一个源的平均值, 排查卡顿会被带到别的段上去。
    """

    KEYS = ("采集", "推理", "跟踪计数", "缩小", "叠加", "编码")

    def __init__(self, depth: int = 60):
        self.depth = max(2, int(depth))
        self.stages: Dict[str, List[float]] = {k: [] for k in self.KEYS}
        self.recent: List[float] = []   # 每帧完成时刻: 差值即画面源真实出帧率
        self.proc: List[float] = []     # 每帧总耗时: 倒数即本机处理帧率, 低于出帧率就是跟不上
        self.next_shot = 0.0            # 下一帧允许出图的最早时刻

    def reset(self) -> None:
        for bucket in self.stages.values():
            bucket.clear()
        self.recent.clear()
        self.proc.clear()
        self.next_shot = 0.0

    def shot_due(self, now: float, stream_fps: float) -> bool:
        """该不该出图: 浏览器吃不下来流会把帧攒在手里越拖越久, 所以限个上限。"""
        return stream_fps <= 0 or now >= self.next_shot

    def schedule_shot(self, now: float, stream_fps: float) -> None:
        if stream_fps > 0:
            self.next_shot = now + 1.0 / stream_fps

    def record(self, now: float, cost: float, timings: List[tuple]) -> None:
        self.recent.append(now)
        self.proc.append(cost)
        if len(self.recent) > self.depth:
            self.recent.pop(0)
            self.proc.pop(0)
        for key, value in timings:
            bucket = self.stages[key]
            bucket.append(value)
            if len(bucket) > self.depth:
                del bucket[0]

    def stage_ms(self) -> Dict[str, float]:
        out = {k: round(1000.0 * sum(v) / len(v), 1) for k, v in self.stages.items() if v}
        out["合计"] = round(sum(out.values()), 1)
        return out

    def measured_fps(self) -> float:
        if len(self.recent) <= 5:
            return 0.0
        return len(self.recent) / max(1e-6, self.recent[-1] - self.recent[0])

    def processed_fps(self) -> float:
        if not self.proc:
            return 0.0
        return len(self.proc) / max(1e-6, sum(self.proc))


class LiveEngine:
    """采集 -> YOLO 姿态 -> 跟踪 -> 在线计数, 供页面轮询与拉流。"""

    # 画面源抖动相关: 都是"最多等多久"的兜底值, 与计数精度无关, 所以不占配置项
    BIND_RETRY_SEC = 10.0     # 首次打开画面源的总重试时长(摄像头刚被占用/系统还没交出来)
    MAX_STUCK_BINDS = 3       # 连续这么多次"打开成功却一帧都读不出来"就放弃, 否则空转打满 CPU
    RETRY_DELAY = 0.5         # 重试之间的间隔
    CAMERA_READ_TIMEOUT = 3.0  # 等摄像头交出下一帧的上限: 超时按"采集停了"处理, 拔掉摄像头能自己重连
    MAX_DETECT_FAILURES = 30   # 连续这么多帧推理失败才认定链路真坏了: 偶发一帧跳过就行

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
        self.pending_source: Optional[str] = None
        # 换画面源失败要留话: error 会在下一帧被清掉, 页面根本来不及看到, 用户只会发现画面还是上一个源
        self.switch_error: Optional[str] = None
        self._reset_needed = False
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

    def feeding(self) -> bool:
        """采集线程还在不在: 供 /stream 判断这条连接还有没有必要继续占着。"""
        thread = self.thread
        return not self.stop_flag.is_set() and bool(thread and thread.is_alive())

    def control(self, action: str, **kwargs) -> dict:
        if action not in KNOWN_ACTIONS:
            raise ValueError(f"不认识的控制指令: {action or '(空)'}，可用: {', '.join(KNOWN_ACTIONS)}")
        if action == "reset":
            with self.lock:
                self._reset_needed = True
        elif action == "pause":
            self.paused = True
        elif action == "resume":
            self.paused = False
        elif action == "source" and kwargs.get("source") is not None:
            # 必须与主循环取走请求用同一把锁: 主循环是"读出 pending 再置空"两步,
            # 连着点两次切源时后一次会被前一次的置空抹掉, 页面显示选了 B、实际还在播 A。
            with self.lock:
                self.pending_source = str(kwargs["source"])
                # 换画面源是一次明确的操作: 停在暂停里切源会让页面永远等不到新帧
                self.paused = False
        elif action == "config":
            for key, value in kwargs.items():
                if value is None:
                    continue
                setattr(self.cfg, key, _checked_config_value(self.cfg, key, value))
        return self.snapshot()

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
        if self._is_camera(source):
            # 只有摄像头会有"驱动队列积压、画面越走越晚"的问题: 取帧拆到独立线程, 只留最新一帧。
            # 文件源不能拆: 计数用原始帧, 覆盖式取帧等于把中间的帧丢掉。
            cap = FrameFeeder(cap, timeout=self.CAMERA_READ_TIMEOUT)
        return cap, fps, Tracker(self.cfg), OnlineCounter(self.cfg, fps)

    @staticmethod
    def _is_camera(source) -> bool:
        """画面源写法是不是摄像头编号: 只有摄像头值得为"还没就绪/刚掉线"反复重试。"""
        try:
            return int(source) >= 0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _keeps_counts_on_restart(source) -> bool:
        """读不到帧、重开画面源之后, 跟踪与计数是否留着: 摄像头留, 文件不留。

        文件读到结尾是"同一素材重播", 计数必须归零才和画面对得上;
        摄像头停帧是设备抖动, 人还在原地, 归零等于把老师已经数出来的次数抹掉。
        """
        return LiveEngine._is_camera(source)

    def _retry_sec(self, source) -> float:
        return self.BIND_RETRY_SEC if self._is_camera(source) else 0.0

    def _bind_wait(self, source):
        """打开画面源, 摄像头在 BIND_RETRY_SEC 内有界重试。

        摄像头刚被别的程序占用、系统还没把它交出来, 或者跑到一半掉线, 都是一次 isOpened()=False;
        直接放弃等于让整场计数在这之后停摆。文件源不重试: 路径写错时多等十秒只会像程序卡住了。
        """
        deadline = time.time() + self._retry_sec(source)
        while True:
            try:
                return self._bind(source)
            except Exception:
                if self.stop_flag.is_set() or time.time() >= deadline:
                    raise
                time.sleep(self.RETRY_DELAY)

    def _loop(self) -> None:
        cap = None
        frame_index = 0
        timings = StageTimings()

        def restart(new_cap, new_fps, new_tracker, new_counter):
            """换画面源/循环重开: 帧号与耗时统计都要归零, 否则页面上是上一个源的秒表和上一个源的平均耗时。"""
            timings.reset()
            return new_cap, new_fps, new_tracker, new_counter

        try:
            cap, fps, tracker, counter = self._bind_wait(self.source)
        except Exception as exc:  # 摄像头被占用或路径不存在
            with self.lock:
                self.state.update({"running": False, "error": str(exc)})
            return
        read_since_bind = 0   # 本次打开后真正读出的帧数: 用来识别"能打开但一帧都解不出来"的坏源
        stuck_binds = 0
        detect_failures = 0   # 连续推理失败次数: 偶发一帧不致命, 一直失败就该停下来说话

        try:
            while not self.stop_flag.is_set():
                with self.lock:
                    pending, self.pending_source = self.pending_source, None
                if pending:
                    try:
                        cap.release()
                        cap, fps, tracker, counter = restart(*self._bind_wait(pending))
                        self.source = pending
                        frame_index = 0
                        read_since_bind = 0
                        stuck_binds = 0
                        with self.lock:
                            self.state["error"] = None
                            self.switch_error = None
                    except Exception as exc:
                        # 旧捕获器已经 release: 必须把原画面源开回来, 否则下一帧是在空捕获器上读,
                        # 会立刻"读到结尾"并把计数清零, 页面看着像换源把数换没了
                        with self.lock:
                            self.switch_error = f"切换到 {pending} 没成功, 画面仍是 {self.source}: {exc}"
                        try:
                            cap, fps, tracker, counter = restart(*self._bind_wait(self.source))
                            frame_index = 0
                            read_since_bind = 0
                            stuck_binds = 0
                        except Exception:
                            cap = None
                        if cap is None:
                            break
                        continue
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
                    # 重开之后一帧都没读出来 = 这个源解不出画面: 无限重开会满速空转, CPU 全烧在反复打开文件上
                    stuck_binds = 0 if read_since_bind else stuck_binds + 1
                    read_since_bind = 0
                    if stuck_binds >= self.MAX_STUCK_BINDS:
                        with self.lock:
                            self.state.update({"running": False,
                                               "error": f"画面源解不出内容, 已停止: {self.source}"})
                        break
                    if stuck_binds:
                        time.sleep(self.RETRY_DELAY)
                    # 重开之前先把手上的跟踪与计数留个引用: 摄像头重开之后要换回来
                    keep = (tracker, counter)
                    try:
                        cap, fps, tracker, counter = restart(*self._bind_wait(self.source))
                    except Exception as exc:
                        with self.lock:
                            self.state["error"] = str(exc)
                        break
                    if self._keeps_counts_on_restart(self.source):
                        # 摄像头停三秒(切到别的应用、驱动打了个嗝)之后人还在原地: 只换捕获器,
                        # 跟踪和计数用回原来那套, 帧号继续向前。否则这一趟重开把已经数出来的几十次抹平,
                        # 而且在线计数拿的是 frame/fps 当时间轴, 归零会让窗口里刚攒起的节律一起作废。
                        tracker, counter = keep
                    else:
                        # 文件读到结尾是同一素材重新开始: 帧号归零, 上一次的计数作废
                        frame_index = 0
                    continue

                read_since_bind += 1
                t1 = time.time()
                try:
                    dets = self.detector.detect(frame, frame_index, frame_index / fps)
                except Exception as exc:
                    # 偶发一帧推理报错(显存抖动、这一帧解出来的图不对劲)不该结束整场: 跳过这一帧,
                    # 把原因写进 error 让页面上看得见。一直失败才停下, 否则就是白跑一节课。
                    detect_failures += 1
                    if detect_failures >= self.MAX_DETECT_FAILURES:
                        raise RuntimeError(f"连续 {detect_failures} 帧推理失败: {exc}") from exc
                    with self.lock:
                        self.state["error"] = f"这一帧推理失败已跳过(连续第 {detect_failures} 次): {exc}"
                    frame_index += 1
                    continue
                detect_failures = 0
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
                shot = timings.shot_due(t2, self.stream_fps)
                if shot:
                    timings.schedule_shot(t2, self.stream_fps)
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
                spans = [("采集", t_read), ("推理", t_det), ("跟踪计数", t_count)]
                if shot:
                    spans += [("缩小", t_fit), ("叠加", t_draw), ("编码", t_enc)]
                timings.record(now, now - t0, spans)
                stage_ms = timings.stage_ms()
                measured = timings.measured_fps()
                processed = timings.processed_fps()
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
                            # 暂停超过一个信号窗口的人会被暂时判成"没在跳", 数到的次数还在他身上但没进总计。
                            # 单列出来, 页面上的总计掉下来时才有地方解释, 不必怀疑是计数丢了。
                            "total_excluded": int(sum(t["count"] + t.get("filled", 0) for t in tracks if not t["participant"])),
                            "rope_total": int(sum(t["rope"] for t in participants)),
                            "tracks": tracks,
                            "stage_ms": stage_ms,
                            "events": list(counter.log)[-40:],
                            "warnings": _warnings(live, self.cfg),
                        }
                    )

                if self.realtime and self._is_file(self.source):
                    time.sleep(max(0.0, 1.0 / fps - (time.time() - t0)))
        except Exception as exc:
            # 跑到一半抛异常(推理内部报错、显存不够、素材被写坏)时线程会直接死掉。
            # 不接住的话 running 还停在 True、error 还是空, 页面上是"实时计数中 + 数字悄悄不再涨",
            # 一节课数下来没人发现自己白跳了。计数原样留在 state 里, 页面上还能看到停在哪个数。
            with self.lock:
                self.state.update({"running": False, "error": f"实时链路异常停止: {exc}"})
        finally:
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
            # 页面上"入围门槛"滑杆的当前刻度: 不同步过去就永远显示默认值, 刷新一次就对不上一次
            state["min_track_visibility"] = self.cfg.min_track_visibility
            state["min_track_visibility_floor"] = self.cfg.min_track_visibility_floor
            state["known_actions"] = list(KNOWN_ACTIONS)
            state["switch_error"] = self.switch_error
        return state


def _checked_config_value(cfg: Config, key: str, value):
    """页面改参数时的取值校验: 越界的参数会让整段计数变成噪声或全部归零, 宁可直接报错。

    未知配置项同样报错: 以前 hasattr 一挡就静默跳过, 前端字段名写错时页面显示"已生效"而后台没动。
    """
    if not hasattr(cfg, key):
        raise ValueError(f"未知配置项: {key}")
    if key == "policy" and value not in ("jump", "rope"):
        raise ValueError(f"policy 只能是 jump 或 rope, 收到 {value}")
    if key == "signal_source" and value not in ("auto", "ankle", "hip", "head", "box_bottom", "box_center"):
        raise ValueError(f"signal_source 取值不合法: {value}")
    if key == "imgsz":
        value = int(value)
        if not 160 <= value <= 3840:
            raise ValueError(f"imgsz 需在 160~3840 之间, 收到 {value}")
    if key in ("conf", "min_track_visibility", "min_track_visibility_floor", "peak_frac"):
        value = float(value)
        low = 0.0
        high = 0.95 if key == "conf" else 1.0
        if not low <= value <= high:
            raise ValueError(f"{key} 需在 {low}~{high} 之间, 收到 {value}")
    if key == "device" and value not in (None, ""):
        # 这个值会被原样交给 ultralytics, 写错时表现为"每帧推理都抛异常": 不在这里挡住,
        # 页面上看到的就是画面突然不再计数, 和画面源坏了长得一模一样
        text = str(value).strip().lower()
        if not re.fullmatch(r"(cpu|mps|cuda(?::\d+)?|\d+(?:,\d+)*)", text):
            raise ValueError(f"device 只能是 cpu / mps / cuda / 0 / 0,1 这类取值, 收到 {value}")
        value = text
    if key in ("warmup_sec", "min_track_sec", "max_rate_hz", "min_rate_hz"):
        value = float(value)
        if value <= 0:
            raise ValueError(f"{key} 必须大于 0, 收到 {value}")
    return value


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
            if not name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                continue
            path = os.path.join(directory, name)
            # 素材正被人拷进拷出时列目录和取大小之间文件就没了: 这里不能让页面下拉框整个 500
            try:
                size_mb = round(os.path.getsize(path) / 1e6, 1)
            except OSError:
                continue
            out.append({"id": path, "label": name, "kind": "file", "size_mb": size_mb})
    for path in extra or []:
        if os.path.exists(path) and not any(o["id"] == path for o in out):
            out.append({"id": path, "label": os.path.basename(path), "kind": "file"})
    return out
