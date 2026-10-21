"""实时链路的工程约束: 控制接口校验、画面源列表、耗时统计与内存上限。"""

from __future__ import annotations

import http.client
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jumpcount.config import Config
from jumpcount.engine import (
    FrameFeeder,
    LiveEngine,
    StageTimings,
    _checked_config_value,
    list_sources,
)
from jumpcount.online import OnlineCounter
from jumpcount.server import Handler
from jumpcount.tracking import Tracker


class TestControlValidation(unittest.TestCase):
    """页面上的开关必须真的落到后台: 静默吞掉请求比报错更难查。"""

    def _engine(self) -> LiveEngine:
        return LiveEngine("0", None, Config())

    def test_unknown_action_is_rejected(self):
        engine = self._engine()
        with self.assertRaises(ValueError):
            engine.control("puase")

    def test_known_actions_are_accepted(self):
        engine = self._engine()
        for action in ("pause", "resume", "status", "reset"):
            self.assertIsInstance(engine.control(action), dict)

    def test_config_action_rejects_unknown_key(self):
        engine = self._engine()
        with self.assertRaises(ValueError):
            engine.control("config", min_visibility=0.4)   # 页面字段名, 落到后台必须是配置项名

    def test_config_action_applies_valid_key(self):
        engine = self._engine()
        state = engine.control("config", min_track_visibility=0.4)
        self.assertAlmostEqual(engine.cfg.min_track_visibility, 0.4)
        self.assertAlmostEqual(state["min_track_visibility"], 0.4)

    def test_out_of_range_values_are_rejected(self):
        engine = self._engine()
        for key, value in (("imgsz", 10), ("conf", 5.0), ("min_track_visibility", 1.8), ("warmup_sec", 0)):
            with self.assertRaises(ValueError, msg=f"{key}={value} 本该被拦下"):
                engine.control("config", **{key: value})

    def test_switching_source_resumes_a_paused_engine(self):
        """暂停中切画面源: 不解除暂停的话页面会永远停在"已暂停"上等不到新帧。"""
        engine = self._engine()
        engine.control("pause")
        self.assertTrue(engine.paused)
        engine.control("source", source="1")
        self.assertFalse(engine.paused)
        self.assertEqual(engine.pending_source, "1")

    def test_source_switch_cannot_lose_a_request(self):
        """连着点两次切源: 主循环"读出 pending 再置空"是两步, 写入必须走同一把锁才不会被抹掉。"""
        engine = LiveEngine("0", None, Config())
        held = threading.Event()
        done = threading.Event()

        def holder():
            with engine.lock:
                held.set()
                time.sleep(0.2)
            done.set()

        thread = threading.Thread(target=holder, daemon=True)
        thread.start()
        held.wait(1.0)
        started = time.time()
        engine.control("source", source="/tmp/素材B.mp4")
        waited = time.time() - started
        done.wait(1.0)
        thread.join(1.0)
        self.assertGreater(waited, 0.05, "写入没等锁 = 它绕过了主循环的读清两步")
        self.assertEqual(engine.pending_source, "/tmp/素材B.mp4")


    def test_snapshot_exposes_slider_bindings(self):
        """前端滑杆要靠这些字段对齐, 缺一项刷新一次就错位一次。"""
        state = self._engine().snapshot()
        for key in ("policy", "signal_source", "imgsz", "conf", "sensitivity", "snr_k",
                    "min_track_visibility", "min_track_visibility_floor", "known_actions"):
            self.assertIn(key, state, f"snapshot 缺字段 {key}")


class TestCheckedConfigValues(unittest.TestCase):
    def test_types_are_coerced(self):
        cfg = Config()
        self.assertEqual(_checked_config_value(cfg, "imgsz", "960"), 960)
        self.assertEqual(_checked_config_value(cfg, "conf", "0.4"), 0.4)

    def test_bad_enum_is_rejected(self):
        cfg = Config()
        with self.assertRaises(ValueError):
            _checked_config_value(cfg, "policy", "jumps")


class TestListSources(unittest.TestCase):
    def test_missing_size_does_not_break_the_dropdown(self):
        """素材正被拷进拷出时, 列目录之后取大小之前文件就没了: 不能让下拉框整个报错。"""
        import jumpcount.engine as engine_module

        directory = tempfile.mkdtemp()
        gone = os.path.join(directory, "已删除.mp4")
        open(gone, "wb").close()
        original = os.path.getsize

        def racing(path, *args, **kwargs):
            if path == gone:
                raise OSError("No such file or directory")
            return original(path, *args, **kwargs)

        engine_module.os.path.getsize = racing
        try:
            sources = list_sources(scan_dirs=[directory])
        finally:
            engine_module.os.path.getsize = original
        self.assertEqual([s["id"] for s in sources if s["kind"] == "file"], [])

    def test_present_file_reports_its_size(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "素材.mp4")
        with open(path, "wb") as fh:
            fh.write(b"x" * 2048)
        found = {s["id"]: s for s in list_sources(scan_dirs=[directory])}
        self.assertIn(path, found)
        self.assertEqual(found[path]["size_mb"], 0.0)


class TestStageTimings(unittest.TestCase):
    """页面底部的"每帧多少毫秒": 切源之后必须重新攒, 不能拿上一个源的平均值糊在这里。"""

    @staticmethod
    def _filled(depth: int = 60) -> StageTimings:
        timings = StageTimings(depth=depth)
        for i in range(depth + 20):
            now = 1000.0 + i * 0.04
            timings.record(now, 0.03, [("推理", 0.02), ("跟踪计数", 0.005)])
        return timings

    def test_smoothed_values_are_averages(self):
        timings = self._filled()
        stage_ms = timings.stage_ms()
        self.assertAlmostEqual(stage_ms["推理"], 20.0, places=1)
        self.assertAlmostEqual(stage_ms["合计"], 25.0, places=1)
        self.assertGreater(timings.measured_fps(), 20.0)
        self.assertGreater(timings.processed_fps(), 20.0)

    def test_reset_drops_the_previous_source(self):
        timings = self._filled()
        timings.schedule_shot(2000.0, 15.0)
        timings.reset()
        self.assertEqual(timings.stage_ms(), {"合计": 0.0})
        self.assertEqual(timings.measured_fps(), 0.0)
        self.assertEqual(timings.processed_fps(), 0.0)
        self.assertEqual(timings.next_shot, 0.0, "上一个源排的下一次出图时刻还留着")
        self.assertTrue(timings.shot_due(0.0, 15.0))

    def test_window_stays_bounded(self):
        timings = self._filled(depth=10)
        self.assertEqual(len(timings.recent), 10)
        self.assertEqual(len(timings.proc), 10)
        self.assertEqual(len(timings.stages["推理"]), 10)

    def test_shot_throttle_respects_the_cap(self):
        timings = StageTimings()
        self.assertTrue(timings.shot_due(0.0, 15.0))
        timings.schedule_shot(0.0, 15.0)
        self.assertFalse(timings.shot_due(0.03, 15.0), "15fps 上限内不该再出一张")
        self.assertTrue(timings.shot_due(0.08, 15.0))
        self.assertTrue(timings.shot_due(0.0, 0.0), "上限为 0 表示不限帧率")


class TestSourceRecovery(unittest.TestCase):
    """画面源抖动: 摄像头刚被别的程序占用、或跑到一半掉线, 不该让整场计数就此停摆。"""

    class Flaky(LiveEngine):
        """前 failures 次 _bind 必然失败, 之后返回一套假资源。"""

        def __init__(self, failures: int):
            super().__init__("0", None, Config())
            self.failures = failures
            self.attempts = 0

        def _bind(self, source):
            self.attempts += 1
            if self.attempts <= self.failures:
                raise RuntimeError(f"打不开画面源: {source}")
            return None, 25.0, Tracker(self.cfg), OnlineCounter(self.cfg, 25.0)

    def _engine(self, failures: int) -> "TestSourceRecovery.Flaky":
        engine = self.Flaky(failures)
        engine.RETRY_DELAY = 0.005     # 用例不等真实的退避间隔
        return engine

    def test_camera_is_retried_until_it_answers(self):
        engine = self._engine(failures=2)
        cap, fps, tracker, counter = engine._bind_wait("0")
        self.assertEqual(engine.attempts, 3, "摄像头第 3 次才就绪时不该提前放弃")
        self.assertEqual(fps, 25.0)
        self.assertIsNotNone(tracker)
        self.assertIsNotNone(counter)

    def test_file_source_fails_fast(self):
        """路径写错时多等十秒只会让人以为程序卡住了: 文件源一次试不出来就直接报错。"""
        engine = self._engine(failures=99)
        with self.assertRaises(RuntimeError):
            engine._bind_wait("/tmp/这段路径不存在.mp4")
        self.assertEqual(engine.attempts, 1)

    def test_retry_gives_up_at_the_deadline(self):
        engine = self._engine(failures=99)
        engine.BIND_RETRY_SEC = 0.03
        with self.assertRaises(RuntimeError):
            engine._bind_wait("0")
        self.assertGreater(engine.attempts, 1, "有界重试应该试多次")
        self.assertLess(engine.attempts, 20)

    def test_stop_flag_ends_the_retry_at_once(self):
        engine = self._engine(failures=99)
        engine.stop_flag.set()
        with self.assertRaises(RuntimeError):
            engine._bind_wait("0")
        self.assertEqual(engine.attempts, 1, "退出时不该再赖在重试里")

    def test_only_indices_count_as_a_camera(self):
        self.assertTrue(LiveEngine._is_camera("0"))
        self.assertTrue(LiveEngine._is_camera(1))
        self.assertFalse(LiveEngine._is_camera("/tmp/素材.mp4"))
        self.assertFalse(LiveEngine._is_camera(""))
        self.assertFalse(LiveEngine._is_camera(None))

    def test_feeding_is_false_before_start_and_after_stop(self):
        engine = self._engine(failures=0)
        self.assertFalse(engine.feeding(), "线程还没起来, /stream 不该认为还在出帧")
        engine.stop_flag.set()
        self.assertFalse(engine.feeding())


class TestFrameFeeder(unittest.TestCase):
    """摄像头取帧线程: 队列长度必须恒为 1, 且解不出帧时要能被主循环察觉。"""

    class FastCap:
        """按固定间隔吐帧, 帧内容就是帧号, 方便看"取到的是不是最新一帧"。"""

        def __init__(self, count: int, delay: float = 0.001):
            self.count = count
            self.delay = delay
            self.index = 0
            self.releases = 0

        def read(self):
            if self.index >= self.count:
                return False, None
            time.sleep(self.delay)
            self.index += 1
            return True, self.index

        def release(self):
            self.releases += 1

    class StalledCap(FastCap):
        """吐完 count 帧之后就再也不返回: 模拟摄像头被拔掉/驱动卡住。"""

        def __init__(self, count: int):
            super().__init__(count, delay=0.0)
            self.gate = threading.Event()

        def read(self):
            if self.index < self.count:
                self.index += 1
                return True, self.index
            self.gate.wait(5.0)
            return False, None

        def release(self):
            super().release()
            self.gate.set()

    def test_a_slow_consumer_gets_the_newest_frame_not_the_oldest(self):
        """消费端(推理+出图)比采集慢时, 攒下来的旧帧必须被丢掉, 否则延迟只涨不落。"""
        cap = self.FastCap(400, delay=0.001)
        feeder = FrameFeeder(cap, timeout=2.0)
        try:
            ok, first = feeder.read()
            self.assertTrue(ok)
            time.sleep(0.12)          # 采集线程这段时间里又出了上百帧
            ok, latest = feeder.read()
            self.assertTrue(ok)
            self.assertGreater(latest, first + 20, "取到的应该是当下最新一帧, 不是排队在最前面的旧帧")
            self.assertGreater(feeder.dropped(), 0, "没被取走的帧要能被统计出来, 否则看不出推理跟不上帧率")
        finally:
            feeder.release()

    def test_a_consumer_that_keeps_up_sees_every_frame(self):
        """消费端跟得上采集节奏时, 一帧都不能少: 计数用的是原始帧, 少一帧就是可能少一次起跳。"""
        cap = self.FastCap(30, delay=0.002)
        feeder = FrameFeeder(cap, timeout=2.0)
        try:
            taken = [feeder.read()[1] for _ in range(30)]
            self.assertEqual(taken, list(range(1, 31)), "跟得上节奏时必须逐帧不漏")
            self.assertEqual(feeder.dropped(), 0)
        finally:
            feeder.release()

    def test_a_burst_collapses_into_one_slot(self):
        """积压的帧只保留最新一帧: 两帧挤在一起到达时, 取帧端只会拿到后一帧。

        这是刻意的取舍: 摄像头场景下落后就永远追不回来, 补播旧帧只会让画面上的数字
        算的是三十秒前的人。文件源不走这条路径, 见 _bind。
        """
        cap = self.FastCap(2, delay=0.0)
        feeder = FrameFeeder(cap, timeout=2.0)
        try:
            self.assertEqual(feeder.read(), (True, 2), "拿到的应该是最新一帧")
            self.assertEqual(feeder.taken, 1, "两帧合并成一格, 取走一次就是取完了")
        finally:
            feeder.release()

    def test_a_stalled_camera_times_out_instead_of_hanging(self):
        """摄像头不再出帧时 read() 要按时返回 False, 主循环才会去重开画面源。"""
        cap = self.StalledCap(1)
        feeder = FrameFeeder(cap, timeout=0.05)
        try:
            self.assertEqual(feeder.read(), (True, 1))
            started = time.time()
            self.assertEqual(feeder.read(), (False, None), "卡住的采集线程要被判成停了")
            self.assertLess(time.time() - started, 0.5)
        finally:
            feeder.release()

    def test_the_end_of_a_feed_keeps_reporting_failure(self):
        """读到结尾之后每次 read 都要立刻返回 False: 让主循环的有界重开逻辑去数次数, 别每次都等满超时。"""
        cap = self.FastCap(1, delay=0.0)
        feeder = FrameFeeder(cap, timeout=2.0)
        try:
            self.assertTrue(feeder.read()[0])
            for _ in range(3):
                started = time.time()
                self.assertFalse(feeder.read()[0])
                self.assertLess(time.time() - started, 0.2, " EOF 之后不该再等满一个 timeout")
        finally:
            feeder.release()

    def test_release_stops_a_blocked_reader_and_is_idempotent(self):
        """release 时读帧线程可能正卡在 cap.read() 里: join 有上限, 且重复 release 不能重复关底层捕获器。"""
        cap = self.StalledCap(0)
        feeder = FrameFeeder(cap, timeout=0.05)
        feeder.release()
        feeder.release()
        self.assertFalse(feeder.thread.is_alive(), "取帧线程还活着就会一直占着摄像头")
        self.assertEqual(cap.releases, 1)

    def test_only_the_file_source_keeps_reading_in_the_main_loop(self):
        """文件源不能套取帧线程: 计数用原始帧, 覆盖式取帧会把中间的帧丢掉。"""
        engine = LiveEngine("/tmp/这段路径不存在.mp4", None, Config())
        self.assertFalse(engine._is_camera("/tmp/素材.mp4"))
        engine2 = LiveEngine("0", None, Config())
        self.assertTrue(engine2._is_camera(engine2.source))


class TestInferenceFailures(unittest.TestCase):
    """推理报错时实时链路怎么收场: 静默死掉最坏, 页面上会一直是"实时计数中"。"""

    class EndlessCap:
        """一直给帧、永远不到结尾: 把用例的注意力留在推理这一段。"""

        def __init__(self):
            self.releases = 0

        def get(self, prop):
            return 30.0

        def read(self):
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

        def release(self):
            self.releases += 1

    class Boom:
        """按 fail_every 的比例报错的假检测器; fail_every=1 表示每帧都失败。"""

        name = "boom"

        def __init__(self, fail_every: int, slow: float = 0.0):
            self.fail_every = fail_every
            self.slow = slow      # 每帧假装耗时: 让轮询能稳定采到"报错之后又恢复"的中间态
            self.calls = 0
            self.skips = 0

        def detect(self, frame, index, stamp):    # 第三个参数不要叫 time: 会挡住模块
            self.calls += 1
            if self.slow:
                time.sleep(self.slow)
            if self.calls % self.fail_every == 0:
                self.skips += 1
                raise RuntimeError("显存不够了")
            return []

    class FixedEngine(LiveEngine):
        def __init__(self, detector):
            super().__init__("0", None, Config())
            self.detector = detector

        def _open(self, source):
            return TestInferenceFailures.EndlessCap()

    def _run(self, engine, frames: int, limit: float = 15.0):
        engine.jpeg_width = 64
        engine.stream_fps = 0.001
        engine.start()
        started = time.time()
        while time.time() - started < limit:
            state = engine.snapshot()
            # 等的是"停下并写明原因"这句话本身: 每帧都失败时 running 从来就是 False,
            # 拿 running 当信号会在第一条跳过提示上就误判成已经停下
            if state["frame"] >= frames or "实时链路异常停止" in (state["error"] or ""):
                break
            time.sleep(0.02)
        result = engine.snapshot()
        engine.stop()
        return result

    def test_a_broken_detector_stops_with_a_message(self):
        """每帧都报错时必须停下并把原因写在 error 上, 不能留着 running 骗人。"""
        engine = self.FixedEngine(self.Boom(fail_every=1))
        engine.MAX_DETECT_FAILURES = 3
        state = self._run(engine, frames=50)
        self.assertFalse(state["running"], "推理再也起不来了还说自己在实时计数")
        self.assertIsNotNone(state["error"])
        self.assertIn("实时链路异常停止", state["error"])
        self.assertIn("显存不够了", state["error"], "要带上原始报错, 否则没法查")

    def test_a_single_bad_frame_is_only_skipped(self):
        """偶发一帧失败不该结束整场: 跳过的帧要让时间轴继续向前, 而不是当它没发生过。"""
        detector = self.Boom(fail_every=5, slow=0.01)
        engine = self.FixedEngine(detector)
        engine.jpeg_width = 64
        engine.stream_fps = 0.001
        engine.start()
        started = time.time()
        clean_after_skip = False
        state = engine.snapshot()
        while time.time() - started < 20.0:
            state = engine.snapshot()
            # 跳过之后的某一帧成功出了数,error 必须已经被清掉: 一直挂着旧报错等于页面永远在报警
            if state["frame"] >= 10 and state["error"] is None:
                clean_after_skip = True
            if state["frame"] >= 40 or "实时链路异常停止" in (state["error"] or ""):
                break
            time.sleep(0.002)
        engine.stop()
        self.assertTrue(clean_after_skip, "恢复成功之后还挂着上一帧的报错")
        self.assertTrue(state["running"], "只失败了几帧就结束整场, 太脆")
        self.assertGreaterEqual(detector.skips, 2, "没走到跳过分支等于没测到东西")
        # 跳过的帧也要把帧号推进: 帧号永远不小于"成功检测的次数", 且必须把失败的那些也算进去,
        # 否则在线计数的时间轴会退回重复的一帧, 窗口里的节律就被同一帧喂了两遍
        self.assertGreaterEqual(detector.calls, state["frame"])
        self.assertGreater(state["frame"], detector.calls - detector.skips)

    def test_device_values_are_validated(self):
        """device 会被原样交给 ultralytics: 写错时的表现是每帧推理都抛异常, 必须在这里就挡住。"""
        cfg = Config()
        self.assertEqual(_checked_config_value(cfg, "device", "MPS"), "mps")
        self.assertEqual(_checked_config_value(cfg, "device", "0"), "0")
        self.assertEqual(_checked_config_value(cfg, "device", "0,1"), "0,1")
        for bad in ("gpu", "cuda7", "mps2", "cuda:abc"):
            with self.assertRaises(ValueError, msg=f"{bad} 应该被拒绝"):
                _checked_config_value(cfg, "device", bad)


class TestCameraRestartKeepsCounts(unittest.TestCase):
    """摄像头停帧之后会重开画面源: 人还在原地, 不能把已经数出来的次数一起抹掉。"""

    class FakeCap:
        """每次"打开"只给 5 帧, 之后再也不出帧: 模拟驱动打嗝或被别的应用抢走摄像头。"""

        FRAMES_PER_BIND = 5

        def __init__(self):
            self.index = 0
            self.releases = 0

        def get(self, prop):
            return 30.0

        def read(self):
            if self.index >= self.FRAMES_PER_BIND:
                return False, None
            self.index += 1
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

        def release(self):
            self.releases += 1

    class CamEngine(LiveEngine):
        def __init__(self, source):
            super().__init__(source, None, Config())
            self.binds = 0

        def _open(self, source):
            self.binds += 1
            return TestCameraRestartKeepsCounts.FakeCap()

    def _run(self, source, target: int, limit: float):
        engine = self.CamEngine(source)
        engine.jpeg_width = 64
        engine.stream_fps = 0.001     # 用例只关心帧号与计数活着, 不需要出图
        engine.start()
        started = time.time()
        while time.time() - started < limit:
            if engine.snapshot()["frame"] >= target:
                break
            time.sleep(0.02)
        state = engine.snapshot()
        engine.stop()
        return state, engine

    def test_a_camera_reopen_continues_the_same_session(self):
        """跨过多次重开, 帧号仍要一路向前: 归零就是把这一局的计数作废。"""
        state, engine = self._run("0", target=12, limit=20.0)
        self.assertGreater(engine.binds, 1, "没有重开就说明这条用例没测到东西")
        self.assertGreaterEqual(state["frame"], 12, "摄像头重开不该把帧号打回本局开头")
        self.assertIsNone(state["error"])

    def test_a_file_reopen_restarts_the_session(self):
        """文件读到结尾是同一素材重播: 帧号必须归零, 否则页面秒表和画面越差越远。"""
        state, engine = self._run("/tmp/不会存在的素材.mp4", target=12, limit=3.0)
        self.assertGreater(engine.binds, 1)
        self.assertLess(state["frame"], self.FakeCap.FRAMES_PER_BIND + 1,
                        "文件循环重开必须从第 0 帧重新开始")

    def test_the_decision_only_depends_on_the_source_kind(self):
        self.assertTrue(LiveEngine._keeps_counts_on_restart("0"))
        self.assertTrue(LiveEngine._keeps_counts_on_restart(2))
        self.assertFalse(LiveEngine._keeps_counts_on_restart("/跳绳素材/a.mp4"))
        self.assertFalse(LiveEngine._keeps_counts_on_restart(""))


class TestStreamConnection(unittest.TestCase):
    """采集线程停了以后, 这条 /stream 必须自己收手: 浏览器对同一主机只有 6 条并发连接,
    攒下一堆再也等不到帧的死流, 整个页面会连状态都发不出去。"""

    class StubEngine:
        def __init__(self, feeding: bool):
            self.stop_flag = threading.Event()
            self.frame_id = 0
            self.jpeg = b""
            self._feeding = feeding

        def feeding(self) -> bool:
            return self._feeding

    def setUp(self):
        self.saved = (Handler.engine, Handler.STREAM_IDLE_SEC)
        Handler.STREAM_IDLE_SEC = 0.05

    def tearDown(self):
        Handler.engine, Handler.STREAM_IDLE_SEC = self.saved

    def _fetch_stream(self, engine, wait: float):
        Handler.engine = engine
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        port = server.server_address[1]
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=wait)
            conn.request("GET", "/stream")
            response = conn.getresponse()
            started = time.time()
            try:
                response.read(4096)      # 服务端关掉连接时返回空字节
                closed = True
            except OSError:            # 超时 = 连接还开着
                closed = False
            return closed, time.time() - started
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2.0)

    def test_a_dead_feed_closes_the_stream(self):
        closed, elapsed = self._fetch_stream(self.StubEngine(feeding=False), wait=3.0)
        self.assertTrue(closed, "采集线程已经停了, 这条流还挂着不放")
        self.assertLess(elapsed, 2.0, "关掉死流不该让人等上一整轮")

    def test_a_live_feed_keeps_the_stream_open(self):
        closed, _ = self._fetch_stream(self.StubEngine(feeding=True), wait=0.4)
        self.assertFalse(closed, "还在出帧的采集线程不该被当成死流关掉")


if __name__ == "__main__":
    unittest.main()
