"""核心算法单元测试: 信号/计数/跟踪/评估。"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jumpcount import signal as sg
from jumpcount.config import (
    SNR_K_DEFAULT,
    SNR_K_LOOSEST,
    SNR_K_STRICTEST,
    Config,
    sensitivity_to_gains,
    sensitivity_to_snr_k,
    snr_k_to_sensitivity,
)
from jumpcount.counter import (
    ankle_sync, channel_cache, classify, count_track, drift_spread, extract_y_detail, participation,
    pick_source, robust_spread, source_scores,
)
from jumpcount.detect import Detection, suppress_duplicates
from jumpcount.identity import build_slot, consolidate
from jumpcount.evaluate import _event_metrics, evaluate
from jumpcount.online import OnlineCounter
from jumpcount.tracking import Tracker


def _kp(x: float, y: float, lift: float, conf: float = 0.9) -> np.ndarray:
    """造一副 17 点姿态: 头/髋/脚踝随整体起跳等量上移。"""
    arr = np.zeros((17, 3))
    arr[:, :2] = (x, y)
    arr[:, 2] = conf
    arr[0] = (x, y - 270.0 - lift, conf)
    arr[11] = (x, y - 150.0 - lift, conf)
    arr[12] = (x + 10, y - 150.0 - lift, conf)
    arr[15] = (x - 12, y - 20.0 - lift, conf)
    arr[16] = (x + 12, y - 20.0 - lift, conf)
    return arr


def jump_signal(fps: float, duration: float, period: float, amp: float, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(fps * duration)
    out = np.zeros(n)
    for i in range(n):
        t = i / fps
        frac = (t % period) / period
        out[i] = 4 * amp * (frac / 0.55) * (1 - frac / 0.55) if frac <= 0.55 else 0.0
    if noise:
        out = out + rng.normal(0, noise, n)
    return out


class TestUndercountRecovery(unittest.TestCase):
    """少数(漏计)的几条主路径: 低幅起跳、遮挡空档、替身通道、像素下限。"""

    fps = 30.0

    def _samples(self, seconds, period=0.5, amp=28.0, flat=None, gap=None):
        sig = jump_signal(self.fps, seconds, period, amp)
        samples, frame = [], 0
        for i in range(sig.size):
            when = i / self.fps
            if gap and gap[0] <= when < gap[1]:
                frame += 1
                continue  # 人根本不在画面里: 这一帧没有观测
            value = 0.0 if (flat and flat[0] <= when < flat[1]) else sig[i]
            samples.append((frame, when, (100.0, 100.0 - value, 200.0, 400.0 - value), 300.0 - value))
            frame += 1
        return samples

    def test_flat_stretch_is_backfilled_by_rhythm(self):
        """人一直在画面里但有一段时间没数到峰: 按节奏补, 且与实测数分开记一笔。"""
        cfg = Config()
        res = count_track(1, self._samples(12, flat=(5.0, 6.0)), cfg, self.fps, "ankle")
        self.assertGreaterEqual(res.stats.get("filled", 0), 1, res.stats)
        self.assertEqual(res.stats["total_count"], res.count + res.stats["filled"])
        self.assertGreater(res.stats["rhythm_coverage"], 0.7, res.stats)

    def test_absent_person_is_not_backfilled(self):
        """人整段不在画面里: 不替他猜, 只记成"疑似漏计但未补"。"""
        cfg = Config()
        res = count_track(1, self._samples(12, gap=(5.0, 6.5)), cfg, self.fps, "ankle")
        self.assertEqual(res.stats.get("filled", 0), 0, res.stats)
        self.assertGreaterEqual(res.stats.get("misses_unobserved", 0), 1, res.stats)

    def test_backfill_is_capped_per_gap(self):
        cfg = Config()
        cfg.max_fill_per_gap = 1
        one = count_track(1, self._samples(12, flat=(4.0, 6.4)), cfg, self.fps, "ankle")
        cfg.max_fill_per_gap = 5
        many = count_track(1, self._samples(12, flat=(4.0, 6.4)), cfg, self.fps, "ankle")
        self.assertLessEqual(one.stats["filled"], many.stats["filled"])
        self.assertLessEqual(one.stats["filled"], 2, one.stats)

    def test_substitute_channel_does_not_leave_a_step(self):
        """脚踝整段缺失时换成髋补, 要按两者的差平移回来, 不能留一个几十像素的台阶。"""
        cfg = Config()
        truth = jump_signal(self.fps, 6, 0.5, 25.0)
        samples = []
        for i in range(truth.size):
            keypoints = _kp(330.0, 400.0, truth[i])
            if 2.0 <= i / self.fps < 3.5:  # 中段脚踝被绳和前面的人挡住: 坐标直接无效
                keypoints[15, 0] = -1.0
                keypoints[16, 0] = -1.0
            samples.append((i, i / self.fps, (300.0, 100.0, 360.0, 400.0), 0.0, keypoints))
        values, filled = extract_y_detail(samples, "ankle", cfg)
        self.assertGreater(filled, 20, filled)
        reference = -truth  # 向上为正的位移序列
        offset = float(np.nanmedian(values[:60] - reference[:60]))
        seam = slice(int(1.9 * self.fps), int(3.6 * self.fps))
        self.assertLess(float(np.nanmax(np.abs(values[seam] - (reference[seam] + offset)))), 6.0)

    def test_weak_ankle_channel_is_swapped_for_a_clearer_one(self):
        """远机位低抬腿: 脚踝抖的幅度比抬起来还大, 该换成髋/头数, 不能照着脚踝一路漏。"""
        cfg = Config()
        truth = jump_signal(self.fps, 9.0, 0.45, 26.0, noise=0.8, seed=5)
        rng = np.random.default_rng(3)
        samples = []
        for i in range(truth.size):
            keypoints = _kp(320.0, 470.0, truth[i])
            keypoints[15, 1] = 450.0 - truth[i] * 0.05 + rng.normal(0.0, 1.1)
            keypoints[16, 1] = 450.0 - truth[i] * 0.05 + rng.normal(0.0, 1.1)
            samples.append((i, i / self.fps, (290.0, 320.0, 350.0, 470.0), 0.0, keypoints))
        channels = channel_cache(samples)
        scores = {name: value[0] for name, value in source_scores(samples, cfg, self.fps, channels).items()}
        picked = pick_source(samples, cfg, self.fps, channels)
        self.assertNotEqual(picked, "ankle", scores)
        self.assertGreaterEqual(scores[picked], scores["ankle"] + cfg.channel_switch_margin, scores)
        improved = count_track(1, samples, cfg, self.fps)
        expected = 9.0 / 0.45  # 20 个起跳
        total = improved.count + improved.stats["filled"]
        self.assertGreaterEqual(total, 0.8 * expected, (improved.count, improved.stats["filled"]))


class TestParticipantVerdicts(unittest.TestCase):
    """参与判定从一票否决改成三态: 证据不足只降级, 确凿不是跳绳才剔除。"""

    def _stats(self, **over):
        base = {"duration_sec": 18.0, "visibility": 1.0, "excursion_ratio": 0.2, "lift_ratio": 0.06,
                "lift_ratio_best": 0.06, "ankle_sync": 0.99, "count": 30,
                "rhythm_strength": 0.55, "period_source": "峰间隔中位数"}
        base.update(over)
        return base

    def test_low_lift_but_clear_rhythm_is_provisional(self):
        verdict, reason = participation(self._stats(lift_ratio=0.005, lift_ratio_best=0.005, count=7), Config())
        self.assertEqual(verdict, "provisional", reason)
        self.assertTrue(classify(self._stats(lift_ratio=0.005, lift_ratio_best=0.005, count=7), Config())[0])

    def test_low_lift_without_rhythm_is_still_rejected(self):
        weak = self._stats(lift_ratio=0.002, lift_ratio_best=0.002, rhythm_strength=0.1)
        self.assertEqual(participation(weak, Config())[0], "out")

    def test_heavy_drift_is_rejected_even_with_synced_feet(self):
        """走得很远的人双脚也同步, 但 4 倍身高的位移不可能是原地跳。"""
        verdict, reason = participation(self._stats(excursion_ratio=4.1), Config())
        self.assertEqual(verdict, "out", reason)
        self.assertIn("走动", reason)

    def test_mild_drift_with_clear_rhythm_is_provisional(self):
        verdict, _ = participation(self._stats(excursion_ratio=1.45), Config())
        self.assertEqual(verdict, "provisional")

    def test_heavy_occlusion_is_provisional_only_within_floor(self):
        self.assertEqual(participation(self._stats(visibility=0.45), Config())[0], "provisional")
        self.assertEqual(participation(self._stats(visibility=0.18), Config())[0], "out")

    def test_box_size_jitter_means_track_switched_person(self):
        verdict, reason = participation(self._stats(height_cv=0.31), Config())
        self.assertEqual(verdict, "out", reason)

    def test_backfilled_count_also_proves_continued_jumping(self):
        """实测只数到 3 个但按节奏补到 12 个: 用总口径判断他确实在连跳。"""
        stats = self._stats(count=3, total_count=12)
        self.assertNotEqual(participation(stats, Config())[0], "out")


class TestDriftMeasure(unittest.TestCase):
    """横向活动范围要逐时间窗量: 一次 ID 漂移不该把原地跳的人判成走路。"""

    fps = 25.0
    height = 300.0

    def test_single_id_glitch_no_longer_looks_like_walking(self):
        frames = np.arange(int(self.fps * 30))
        cxs = np.full(frames.size, 300.0, dtype=np.float64)
        cxs[500:548] = 1500.0  # 有近两秒被认成了隔壁位置的人
        self.assertGreater(robust_spread(cxs) / self.height, 1.2, "整段分位数会被这一次漂移放大")
        self.assertLess(drift_spread(cxs, self.fps, 3.0, frames) / self.height, 1.2)

    def test_real_walker_still_looks_like_walking(self):
        frames = np.arange(int(self.fps * 30))
        cxs = 300.0 + 200.0 * (frames / self.fps)  # 每秒挪 200px, 每个 3s 窗口都走出去两个身位
        self.assertGreater(drift_spread(cxs, self.fps, 3.0, frames) / self.height, 1.2)


class TestDuplicateGuard(unittest.TestCase):
    """判重必须看框是否真的叠在一起, 否则并排站的真人会被并掉。"""

    fps = 25.0

    def _slot(self, track_id, cx, first=0, last=200, width=60.0):
        samples = [(f, f / self.fps, (cx - width / 2, 100.0, cx + width / 2, 400.0), 400.0)
                   for f in range(first, last)]
        centers = [(cx, 250.0) for _ in range(first, last)]
        return build_slot(track_id, samples, centers, 0.9)

    def test_side_by_side_neighbours_are_not_merged(self):
        cfg = Config()
        slots, actions = consolidate([self._slot(1, 300.0), self._slot(2, 348.0)], cfg)
        self.assertEqual(len(slots), 2, actions)
        self.assertEqual([a for a in actions if a["action"] == "drop_duplicate"], [])

    def test_overlapping_duplicate_ids_are_merged(self):
        cfg = Config()
        slots, actions = consolidate([self._slot(1, 300.0), self._slot(2, 304.0)], cfg)
        self.assertEqual(len(slots), 1)
        self.assertEqual(actions[0]["action"], "drop_duplicate")
        self.assertGreaterEqual(actions[0]["iou"], cfg.dup_iou_min)

    def test_duplicate_boxes_at_same_frame_need_overlap(self):
        cfg = Config()
        side_by_side = [Detection(0, 0.0, (270.0, 100.0, 330.0, 400.0), 0.9),
                        Detection(0, 0.0, (336.0, 100.0, 396.0, 400.0), 0.8)]
        self.assertEqual(len(suppress_duplicates(side_by_side, cfg)), 2)
        stacked = [Detection(0, 0.0, (270.0, 100.0, 330.0, 400.0), 0.9),
                   Detection(0, 0.0, (282.0, 104.0, 342.0, 404.0), 0.8)]
        self.assertEqual(len(suppress_duplicates(stacked, cfg)), 1)


class TestSignal(unittest.TestCase):
    def test_period_estimation(self):
        sig = jump_signal(30, 10, 0.5, 30.0) - 15.0
        period, strength = sg.autocorr_period(sg.detrend(sig, 3, 91), 30, 0.5, 4.5)
        self.assertIsNotNone(period)
        self.assertAlmostEqual(period, 0.5, delta=0.05)
        self.assertGreater(strength, 0.3)

    def test_fill_gaps_interpolates_short_holes(self):
        frames = np.array([0, 1, 2, 3, 10, 11, 12])
        values = np.array([0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
        axis, dense, holes = sg.fill_gaps(frames, values, max_gap=3)
        self.assertEqual(axis.size, 13)
        self.assertFalse(np.isnan(dense).any())
        self.assertEqual(holes, 1)

    def test_find_peaks_rejects_noise(self):
        clean = jump_signal(30, 6, 0.5, 20.0) - 10.0
        peaks = sg.find_peaks(sg.detrend(clean, 3, 91), 8, 3.0, 3.0)
        self.assertGreaterEqual(len(peaks), 10)
        self.assertLessEqual(len(peaks), 14)


class TestCounter(unittest.TestCase):
    def _samples(self, fps, duration, period, amp, noise=0.0, seed=0):
        sig = jump_signal(fps, duration, period, amp, noise, seed)
        # 整框随起跳上移: 框高恒为 300px, 底边与脚踝同步抬升
        return [(i, i / fps, (100.0, 100.0 - sig[i], 200.0, 400.0 - sig[i]), 300.0 - sig[i]) for i in range(sig.size)]

    def test_counts_steady_jumps(self):
        cfg = Config()
        res = count_track(1, self._samples(30, 10, 0.5, 28.0), cfg, 30.0, "ankle")
        self.assertAlmostEqual(res.count, 20, delta=2)
        self.assertAlmostEqual(res.rate_hz, 2.0, delta=0.25)

    def test_counts_under_noise(self):
        cfg = Config()
        res = count_track(1, self._samples(30, 12, 0.45, 24.0, noise=2.0, seed=3), cfg, 30.0, "ankle")
        self.assertAlmostEqual(res.count, round(12 / 0.45), delta=3)

    def test_short_track_rejected(self):
        cfg = Config()
        res = count_track(1, self._samples(30, 0.4, 0.5, 25.0), cfg, 30.0, "ankle")
        self.assertEqual(res.count, 0)
        self.assertEqual(res.stats.get("reject"), "too_short")

    def test_small_amplitude_still_counted(self):
        cfg = Config()
        res = count_track(1, self._samples(30, 10, 0.5, 6.0, seed=5), cfg, 30.0, "ankle")
        self.assertGreaterEqual(res.count, 17)


class TestParticipantGate(unittest.TestCase):
    """走动路人不能被当成跳绳的人。"""

    def _stats(self, **over):
        base = {"duration_sec": 30.0, "visibility": 0.9, "excursion_ratio": 0.25, "lift_ratio": 0.06, "ankle_sync": 0.71}
        base.update(over)
        return base

    def test_walker_rejected_by_excursion(self):
        ok, reason = classify(self._stats(excursion_ratio=6.4), Config())
        self.assertFalse(ok)
        self.assertIn("走动", reason)

    def test_alternating_feet_rejected_as_walking(self):
        ok, reason = classify(self._stats(ankle_sync=-0.55), Config())
        self.assertFalse(ok)
        self.assertIn("走路", reason)

    def test_real_jumper_passes(self):
        self.assertTrue(classify(self._stats(), Config())[0])

    def test_standing_person_rejected(self):
        ok, reason = classify(self._stats(lift_ratio=0.002), Config())
        self.assertFalse(ok)
        self.assertIn("起跳幅度", reason)

    def test_one_bad_box_is_not_walking(self):
        """个别错位框不能把原地跳的人算成"走了几倍身高"。"""
        standing = [300.0] * 40 + [1900.0]
        walking = [300.0 + 8.0 * i for i in range(40)]
        self.assertLess(robust_spread(standing), 20.0)
        self.assertGreater(robust_spread(walking), 250.0)

    def test_stray_peaks_are_not_jump_roping(self):
        """抖出来的一两个峰不算在跳绳, 连跳够数次才算。"""
        ok, reason = classify(self._stats(count=2), Config())
        self.assertFalse(ok)
        self.assertIn("只数到 2 个", reason)
        self.assertTrue(classify(self._stats(count=6), Config())[0])

    def test_counts_track_visibility_into_stats(self):
        cfg = Config()
        sig = jump_signal(30, 10, 0.5, 28.0)
        samples = [(i, i / 30.0, (100.0, 100.0 - sig[i], 200.0, 400.0 - sig[i]), 300.0 - sig[i]) for i in range(sig.size)]
        res = count_track(1, samples, cfg, 30.0, "ankle")
        self.assertEqual(res.stats["visibility"], res.visibility)
        self.assertTrue(res.stats["participant"], res.stats.get("excluded"))

    def test_ankle_sync_sign(self):
        cfg = Config()
        n = 90
        box = (0.0, 0.0, 60.0, 300.0)  # 关键点同步测试只用到脚踝, 框高取固定值


        def kp(lift_l, lift_r):
            arr = np.zeros((17, 3))
            arr[:, :2] = 1.0
            arr[:, 2] = 0.9
            arr[15] = (10.0, 280.0 - lift_l, 0.9)
            arr[16] = (30.0, 280.0 - lift_r, 0.9)
            return arr

        both, alt = [], []
        for i in range(n):
            both.append((i, i / 30.0, box, 0.0, kp(20 * abs(np.sin(i)), 20 * abs(np.sin(i)))))
            alt.append((i, i / 30.0, box, 0.0, kp(20 * abs(np.sin(i)), 20 * abs(np.cos(i)))))
        self.assertGreater(ankle_sync(both, cfg), 0.5)
        self.assertLess(ankle_sync(alt, cfg), 0.0)


class TestOnlineCounter(unittest.TestCase):
    """实时页面用的在线计数: 不能重复出数, 也不能把路人算进来。"""

    def _feed(self, hz, seconds, fps=25.0, amp=35.0, noise=0.0, walk=False, seed=1, cfg=None):
        rng = np.random.default_rng(seed)
        counter = OnlineCounter(cfg or Config(), fps)
        per = 1.0 / max(hz, 1e-6)
        for f in range(int(seconds * fps)):
            t = f / fps
            frac = (t % per) / per if hz > 0 else 0.0
            lift = (4 * amp * (frac / 0.55) * (1 - frac / 0.55) if frac <= 0.55 else 0.0) if hz > 0 else 0.0
            if noise:
                lift += float(rng.normal(0, noise))
            cx = 360.0 + (f * 6.0 if walk else 0.0)
            box = (cx - 60, 120.0 - lift, cx + 60, 420.0 - lift)
            kps = np.zeros((17, 3))
            kps[:, :2] = 1.0
            kps[:, 2] = 0.9
            kps[0] = (cx, 130.0 - lift, 0.9)
            kps[11] = (cx, 250.0 - lift, 0.9)
            if walk:
                kps[15] = (cx - 30, 400.0 - 30 * abs(np.sin(f * 0.6)), 0.9)
                kps[16] = (cx + 30, 400.0 - 30 * abs(np.cos(f * 0.6)), 0.9)
            else:
                kps[15] = (cx - 30, 400.0 - lift, 0.9)
                kps[16] = (cx + 30, 401.0 - lift, 0.9)
            counter.update(f, t, [(1, box, kps, 0.9)])
        return counter.snapshot()[0]

    def test_counts_steady_jumps(self):
        snap = self._feed(2.2, 20)
        self.assertAlmostEqual(snap["count"], 44, delta=4)
        self.assertAlmostEqual(snap["rate_hz"], 2.2, delta=0.35)
        self.assertTrue(snap["participant"], snap["reason"])

    def test_window_drift_does_not_double_count(self):
        """窗口滑动会让峰位漂移 1~2 帧, 曾经把 36 次数成 63 次。"""
        snap = self._feed(1.2, 30, noise=1.5)
        self.assertLess(snap["count"], 45)
        self.assertGreater(snap["count"], 28)

    def test_high_rate_still_counted(self):
        snap = self._feed(3.2, 15, amp=20, noise=1.0)
        self.assertAlmostEqual(snap["count"], 48, delta=6)

    def test_background_walker_not_counted(self):
        snap = self._feed(1.6, 15, walk=True)
        self.assertFalse(snap["participant"])
        # 先被"双脚交替迈步"拦下, 还是被"横向走了一大段"拦下都可以: 必须是走路类原因, 不能静默剔除
        self.assertTrue("走动" in snap["reason"] or "走路" in snap["reason"], snap["reason"])

    def test_idle_person_not_counted(self):
        snap = self._feed(0.0, 15)
        self.assertFalse(snap["participant"])
        self.assertEqual(snap["count"], 0)

    def test_reset_clears_counts(self):
        counter = OnlineCounter(Config(), 25.0)
        counter.update(0, 0.0, [(1, (0, 0, 60, 300), None, 0.9)])
        counter.reset()
        self.assertEqual(counter.snapshot(), [])


class TestTracker(unittest.TestCase):
    def test_one_person_one_track_across_dropout(self):
        cfg = Config(max_age=10, min_hits=2, reid_window=40)
        tracker = Tracker(cfg)
        ids = set()
        for f in range(60):
            if 25 <= f < 30:
                tracker.update(f, [])
                continue
            x = 100 + f
            tracker.update(f, [Detection(f, f / 30.0, (x, 100, x + 60, 300), 0.9)])
            for tid, _ in tracker.update(f, []):
                pass
        self.assertLessEqual(len(tracker.all), 2)

    def test_two_people_keep_distinct_ids(self):
        cfg = Config(min_hits=1)
        tracker = Tracker(cfg)
        seen = []
        for f in range(40):
            dets = [
                Detection(f, f / 30.0, (60, 100, 120, 320), 0.9),
                Detection(f, f / 30.0, (420, 120, 480, 340), 0.9),
            ]
            seen.append({tid for tid, _ in tracker.update(f, dets)})
        self.assertTrue(all(len(s) == 2 for s in seen[5:]))
        self.assertEqual(len(tracker.all), 2)

    def test_flapping_detection_does_not_spawn_tracks(self):
        cfg = Config(min_hits=3)
        tracker = Tracker(cfg)
        for f in range(20):
            dets = [Detection(f, f / 30.0, (100, 100, 160, 320), 0.9)] if f % 2 == 0 else []
            tracker.update(f, dets)
        self.assertLessEqual(len(tracker.all), 2)
        self.assertTrue(all(trk.hits >= 3 for trk in tracker.all))


class TestEvaluate(unittest.TestCase):
    def test_event_metrics_counts_misses_and_false(self):
        gt = [1.0, 2.0, 3.0, 4.0]
        pred = [1.02, 3.05, 5.5]
        tp, fp, fn, biases = _event_metrics(pred, gt, 0.15)
        self.assertEqual((tp, fp, fn), (2, 1, 2))
        self.assertTrue(all(abs(b) < 0.1 for b in biases))

    def test_end_to_end_accuracy(self):
        pred = {
            "duration": 20.0,
            "width": 1000.0,
            "tracks": [
                {"track_id": 1, "count": 20, "first_frame": 0, "last_frame": 600, "rate_hz": 2.0, "snr": 8.0,
                 "visibility": 1.0, "period_sec": 0.5, "stats": {"center_x_mean": 200.0}, "events": []},
                {"track_id": 2, "count": 15, "first_frame": 0, "last_frame": 600, "rate_hz": 1.5, "snr": 6.0,
                 "visibility": 1.0, "period_sec": 0.66, "stats": {"center_x_mean": 700.0}, "events": []},
            ],
        }
        gt = {"persons": [{"name": "P1", "center_x": 0.2, "count": 22}, {"name": "P2", "center_x": 0.7, "count": 15}]}
        out = evaluate(pred, gt, Config())
        self.assertEqual(out["paired"], 2)
        self.assertEqual(out["gt_total"], 37)
        self.assertEqual(out["pred_total"], 35)
        self.assertAlmostEqual(out["per_person"][0]["error"], -2)
        self.assertEqual(out["per_person"][1]["error"], 0)
        self.assertAlmostEqual(out["exact_match_rate"], 0.5)


class TestSeatIdentity(unittest.TestCase):
    """实时页面(固定机位): 一个人被拆成多条跟踪 ID 时, 计数要接在同一个站位上。"""

    fps = 25.0

    def _observe(self, counter: OnlineCounter, frame: int, cx: float = 360.0, track_id: int = 1,
                 hz: float = 2.2, amp: float = 35.0) -> None:
        t = frame / self.fps
        frac = (t % (1.0 / hz)) * hz
        lift = 4 * amp * (frac / 0.55) * (1 - frac / 0.55) if frac <= 0.55 else 0.0
        box = (cx - 60, 120.0 - lift, cx + 60, 420.0 - lift)
        kps = np.zeros((17, 3))
        kps[:, :2] = 1.0
        kps[:, 2] = 0.9
        kps[0] = (cx, 130.0 - lift, 0.9)
        kps[11] = (cx, 250.0 - lift, 0.9)
        kps[15] = (cx - 30, 400.0 - lift, 0.9)
        kps[16] = (cx + 30, 401.0 - lift, 0.9)
        counter.update(frame, t, [(track_id, box, kps, 0.9)])

    def test_fragmented_track_keeps_one_counter(self):
        """同一个位置先后出现三条断裂 ID: 应该是一个越来越大的数, 而不是三个人各数一段。"""
        counter = OnlineCounter(Config(), self.fps)
        for start, end, track_id in ((0, 120, 1), (140, 260, 2), (280, 400, 3)):
            for frame in range(start, end):
                self._observe(counter, frame, track_id=track_id)
        snaps = counter.snapshot(400)
        self.assertEqual(len(snaps), 1, [s["id"] for s in snaps])
        self.assertEqual(snaps[0]["fragments"], 3)
        self.assertGreater(snaps[0]["visibility"], 0.95, snaps[0])
        self.assertLessEqual(snaps[0]["visibility"], 1.0, "断档另起一段时可见率不能算超 100%")
        self.assertTrue(snaps[0]["participant"], snaps[0]["reason"])
        self.assertAlmostEqual(snaps[0]["count"], 2.2 * 3 * 3.5, delta=8)

    def test_returning_id_joins_the_live_seat(self):
        """断裂的旧号回来时并到当下正在被更新的站位上, 不能自己占一张卡各数一段。"""
        counter = OnlineCounter(Config(), self.fps)
        for frame in range(0, 60):
            self._observe(counter, frame, cx=360.0, track_id=1)
        for frame in range(60, 120):
            self._observe(counter, frame, cx=620.0, track_id=2)
        for frame in range(120, 300):
            self._observe(counter, frame, cx=620.0, track_id=1)
            self._observe(counter, frame, cx=620.0, track_id=2)
        snaps = counter.snapshot(300)
        self.assertEqual(len(snaps), 2, [s["id"] for s in snaps])
        live = [s for s in snaps if s["absent_frames"] <= 2]
        self.assertEqual(len(live), 1, [s["id"] for s in snaps])
        self.assertGreater(live[0]["count"], 8, live[0])
        self.assertLess(live[0]["count"], 26, "同一帧两个人头各数一遍会把这个数翻倍")

    def test_neighbours_keep_their_own_counts(self):
        """旁边站着另一个跳的人不能被并进来, 各数各的。"""
        counter = OnlineCounter(Config(), self.fps)
        for frame in range(int(12 * self.fps)):
            self._observe(counter, frame, cx=300.0, track_id=1)
            self._observe(counter, frame, cx=760.0, track_id=2)
        snaps = counter.snapshot(int(12 * self.fps))
        self.assertEqual(len(snaps), 2)
        for snap in snaps:
            self.assertEqual(snap["fragments"], 1)
            self.assertGreater(snap["count"], 15, snap)

    def test_seat_identity_can_be_turned_off(self):
        """同一段断裂数据: 关掉站位就还是两个各数一段的人。"""
        blocks = ((0, 120, 1), (130, 178, 2))
        off = Config()
        off.slot_identity = False
        merged, split = OnlineCounter(Config(), self.fps), OnlineCounter(off, self.fps)
        for counter in (merged, split):
            for start, end, track_id in blocks:
                for frame in range(start, end):
                    self._observe(counter, frame, track_id=track_id)
        self.assertEqual(len(merged.snapshot(177)), 1)
        self.assertEqual(len(split.snapshot(177)), 2)


class TestSensitivityScale(unittest.TestCase):
    """灵敏度滑杆 1~100 与振幅阈值倍数的换算。"""

    def test_endpoints_match_documented_scale(self):
        self.assertAlmostEqual(sensitivity_to_snr_k(1), SNR_K_STRICTEST, places=3)
        self.assertAlmostEqual(sensitivity_to_snr_k(50), SNR_K_DEFAULT, places=3)
        self.assertAlmostEqual(sensitivity_to_snr_k(100), SNR_K_LOOSEST, places=3)
        self.assertAlmostEqual(sensitivity_to_snr_k(50), Config().snr_k, places=3,
                               msg="滑杆中点必须等于流水线的默认阈值")

    def test_higher_slider_always_loosens_threshold(self):
        steps = [sensitivity_to_snr_k(v) for v in range(1, 101)]
        self.assertTrue(all(later < earlier for earlier, later in zip(steps, steps[1:])), steps)
        self.assertGreater(min(steps), 0.0, "阈值不能归零, 否则机身抖动也会被数成起跳")

    def test_out_of_range_is_clamped(self):
        self.assertEqual(sensitivity_to_snr_k(0), sensitivity_to_snr_k(1))
        self.assertEqual(sensitivity_to_snr_k(500), sensitivity_to_snr_k(100))

    def test_slider_reading_round_trips(self):
        for value in (1, 7, 25, 50, 63, 77, 100):
            self.assertEqual(snr_k_to_sensitivity(sensitivity_to_snr_k(value)), value)

    def test_all_three_gates_loosen_together(self):
        """只松噪声一项的话, 滑杆推过一半就没反应了: 峰值比例和像素下限必须一起动。"""
        loose, tight = sensitivity_to_gains(95), sensitivity_to_gains(50)
        self.assertLess(loose["snr_k"], tight["snr_k"])
        self.assertLess(loose["peak_frac"], tight["peak_frac"])
        self.assertLess(loose["floor_relief"], tight["floor_relief"])

    def test_default_gains_match_config(self):
        gains, cfg = sensitivity_to_gains(50), Config()
        self.assertAlmostEqual(gains["snr_k"], cfg.snr_k, places=3)
        self.assertAlmostEqual(gains["peak_frac"], cfg.peak_frac, places=4)
        self.assertAlmostEqual(gains["floor_relief"], cfg.floor_relief, places=3)


class TestSensitivityChangesCounts(unittest.TestCase):
    """机身抖动比起跳还高的信号: 滑杆推到任何一格都不能把抖动数成起跳。

    这根滑杆"有没有反应"改由 TestDistantLowLift 把关: 抖得厉害的信号里阈值封顶
    (峰值的 45%)本来就把两端都兜住了, 真正对滑杆敏感的是远景低抬腿那种一强一弱的起跳。
    """

    fps = 25.0

    def _count(self, slider: int) -> int:
        """2.2Hz 跳 16 秒(真值约 35 个), 抖动幅值和起跳同量级: 门限松紧应当直接体现在计数上。"""
        cfg = Config()
        for key, value in sensitivity_to_gains(slider).items():
            setattr(cfg, key, value)
        counter = OnlineCounter(cfg, self.fps)
        rng = np.random.default_rng(7)
        for frame in range(int(16 * self.fps)):
            t = frame / self.fps
            frac = (t % (1 / 2.2)) * 2.2
            lift = 4 * 5.0 * (frac / 0.55) * (1 - frac / 0.55) if frac <= 0.55 else 0.0
            lift += float(rng.normal(0, 2.6))  # 远景机位的机身抖动
            box = (300.0, 120.0 - lift, 420.0, 420.0 - lift)
            kps = np.zeros((17, 3))
            kps[:, :2] = 1.0
            kps[:, 2] = 0.9
            kps[0] = (360.0, 130.0 - lift, 0.9)
            kps[11] = (360.0, 250.0 - lift, 0.9)
            kps[15] = (330.0, 400.0 - lift, 0.9)
            kps[16] = (390.0, 401.0 - lift, 0.9)
            counter.update(frame, t, [(1, box, kps, 0.9)])
        return max(t["count"] for t in counter.snapshot(int(16 * self.fps)))

    def test_slider_stays_near_the_truth(self):
        """真值约 35 个: 最严到最松整根滑杆都要落在对得上真实个数的区间里。"""
        counts = {value: self._count(value) for value in (10, 50, 95)}
        for value, count in counts.items():
            self.assertLessEqual(count, 42, f"滑杆 {value} 把机身抖动数成了 {count} 个: {counts}")
            self.assertGreaterEqual(count, 25, f"滑杆 {value} 只数到 {count} 个, 真值约 35: {counts}")
        self.assertGreaterEqual(counts[95], counts[10], f"放松滑杆计数反而少了: {counts}")


class TestDistantLowLift(unittest.TestCase):
    """远景机位: 人只有 145px 高、左右脚一强一弱地跳, 弱的那一半起跳只有一像素多。

    16 人群像复盘: 有人实际跳了 106 个而页面只数到 75, 已计峰的间隔正好是节奏周期的两倍
    —— 固定 2px 的绝对下限把弱的那一半整片埋掉了, 而滑杆原来的松紧跨度也够不着。
    """

    fps = 25.0
    truth = 52  # 2.6Hz 跳 20 秒

    def _count(self, slider: int = 50, box_h: float = 145.0, ref_height: float = None) -> int:
        cfg = Config()
        if ref_height is not None:
            cfg.amplitude_ref_height = ref_height
        for key, value in sensitivity_to_gains(slider).items():
            setattr(cfg, key, value)
        counter = OnlineCounter(cfg, self.fps)
        rng = np.random.default_rng(11)
        foot = 1000.0
        for frame in range(int(self.truth / 2.6 * self.fps)):
            t = frame / self.fps
            frac = (t % (1 / 2.6)) * 2.6
            peak = 3.4 if int(t * 2.6) % 2 == 0 else 3.4 * 0.4  # 一强一弱交替抬腿
            lift = peak * 4 * (frac / 0.45) * (1 - frac / 0.45) if frac <= 0.45 else 0.0
            jitter = float(rng.normal(0, 0.1))
            box = (300.0, foot - box_h - lift, 300.0 + box_h * 0.42, foot - lift)
            kps = np.zeros((17, 3))
            kps[:, :2] = (300.0 + box_h * 0.2, foot - box_h * 0.5)
            kps[:, 2] = 0.9
            kps[0] = (300.0 + box_h * 0.2, foot - box_h - lift, 0.9)
            kps[15] = (300.0 + box_h * 0.1, foot - lift - jitter, 0.9)
            kps[16] = (300.0 + box_h * 0.3, foot - lift - jitter, 0.9)
            counter.update(frame, t, [(1, box, kps, 0.9)])
        snaps = counter.snapshot(int(self.truth / 2.6 * self.fps))
        return max((snap["count"] for snap in snaps), default=0)

    def test_absolute_floor_scales_with_distance(self):
        scaled, fixed = self._count(), self._count(ref_height=0.0)
        self.assertGreaterEqual(scaled, int(0.8 * self.truth), f"按框高缩小下限后只数到 {scaled}, 真值 {self.truth}")
        self.assertLessEqual(fixed, int(0.5 * self.truth),
                            f"固定 2px 下限本该把弱的那一半埋掉, 却数到了 {fixed}")

    def test_slider_actively_changes_these_counts(self):
        strict, loose = self._count(slider=10), self._count(slider=90)
        self.assertLessEqual(strict, int(0.6 * self.truth), f"最严一档就数到 {strict}, 门限没起作用")
        self.assertGreaterEqual(loose, int(0.85 * self.truth),
                               f"推到 90 只数到 {loose}(真值 {self.truth}), 等于滑杆没反应")


if __name__ == "__main__":
    unittest.main()
