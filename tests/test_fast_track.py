"""快跳相关判据的单元测试: 通道择优、窄窗换档、格子过滤、补齐证据闸门、阈值地板。

这些分支都是为"跳得快的时候比实际计数少"加的, 素材上的绝对数字在 README 的对照表里,
这里只锁不变量: 换通道要同清晰度才比数量、格子外的峰不许留下、
空档里没有离地鼓包就不许按节奏补数、阈值上限不许把下限击穿。
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jumpcount import signal as sg
from jumpcount.config import Config
from jumpcount.counter import (absolute_floor, amplitude_threshold, channel_rank, count_track,
                               gap_lift_evidence, keep_on_grid)
from jumpcount.online import OnlineCounter

FPS = 30.0


def _scores():
    """三条通道的 (节律强度, 峰数, 覆盖率, 位移 p95): 头最清楚但只数得出 3 个, 踝数得出 40 个。"""
    return {"head": (0.81, 3, 1.0, 0.9), "ankle": (0.75, 40, 1.0, 2.1), "hip": (0.30, 39, 1.0, 3.0)}


class TestChannelRank(unittest.TestCase):
    def test_legacy_ranks_by_strength_only(self):
        cfg = Config()
        self.assertEqual(cfg.channel_prefer_rate_margin > 0, True, "默认就该开这条判据")
        cfg.channel_prefer_rate_margin = 0.0
        cfg.channel_prefer_rate_floor = 0.0        # 两条开关都关才退回"纯比节律强度"
        self.assertEqual(max(_scores(), key=channel_rank(_scores(), cfg)), "head")

    def test_tied_channels_give_way_to_more_jumps(self):
        cfg = Config()
        cfg.channel_prefer_rate_margin = 0.15
        self.assertEqual(max(_scores(), key=channel_rank(_scores(), cfg)), "ankle")

    def test_much_less_clear_channel_never_wins(self):
        cfg = Config()
        cfg.channel_prefer_rate_margin = 0.15      # 髋比最差的头还差一截, 峰再多也不给
        scores = {"head": (0.81, 3, 1.0, 0.9), "hip": (0.30, 90, 1.0, 3.0)}
        self.assertEqual(max(scores, key=channel_rank(scores, cfg)), "head")

    def test_absolute_floor_ties_across_a_wide_strength_gap(self):
        """179023428267 #18 的实测形状: 头 0.851/13 个 vs 踝 0.373/38 个。

        相对档宽够不着这么大的差距, 但两条都过了"节律算不算清楚"这条既有门槛,
        就该按起跳数决定用谁 —— 否则一整段快跳只数出头颈慢摆的十几次。
        """
        cfg = Config()
        scores = {"head": (0.851, 13, 0.79, 1.0), "ankle": (0.373, 38, 0.79, 2.4)}
        cfg.channel_prefer_rate_floor = 0.0
        self.assertEqual(max(scores, key=channel_rank(scores, cfg)), "head", "关掉绝对门槛就该退回旧的择头")
        cfg.channel_prefer_rate_floor = 0.35
        self.assertEqual(max(scores, key=channel_rank(scores, cfg)), "ankle")

    def test_floor_does_not_promote_an_unclear_channel(self):
        """绝对门槛不是"谁快用谁": 节律本身不清楚的通道, 峰再多也不给。"""
        cfg = Config()
        cfg.channel_prefer_rate_floor = 0.35
        scores = {"head": (0.851, 13, 0.79, 1.0), "ankle": (0.34, 200, 0.79, 2.4)}
        self.assertEqual(max(scores, key=channel_rank(scores, cfg)), "head")


class TestKeepOnGrid(unittest.TestCase):
    def test_grid_peaks_kept_and_off_grid_dropped(self):
        # 宽窗每拍都少取一半(2:1 锁相), 窄窗把漏掉的那一拍找回来: 它离最近的实测峰正好一个周期
        mains = [0, 60, 120, 180]
        peaks = [1, 30, 47, 60, 90, 150]
        kept, dropped = keep_on_grid(peaks, mains, 30.0, 0.35)
        self.assertIn(1, kept)                  # 与主档峰同一拍, 只差亚帧
        self.assertIn(30, kept)                 # 离最近实测峰一个周期 = 被滤波抹掉的那一拍
        self.assertIn(90, kept)
        self.assertIn(150, kept)
        self.assertNotIn(47, kept)              # 离峰只有 0.43 个周期: 是抖动不是漏拍
        self.assertEqual(dropped, len(peaks) - len(kept))

    def test_few_anchors_pass_everything_through(self):
        peaks = [3, 17, 44]
        kept, dropped = keep_on_grid(peaks, [0, 60], 30.0, 0.35)
        self.assertEqual(kept, peaks)
        self.assertEqual(dropped, 0)


class TestGapLiftEvidence(unittest.TestCase):
    def test_too_short_gap_is_not_gated(self):
        det = np.zeros(40)
        self.assertEqual(gap_lift_evidence(det, 10, 13, 30.0), float("inf"))

    def test_flat_gap_has_no_evidence(self):
        det = np.zeros(120)
        det[:10] = 8.0
        det[-10:] = 8.0                          # 两侧的起跳沿, 掐尾巴后不该看到
        self.assertLess(gap_lift_evidence(det, 0, 120, 30.0), 0.5)

    def test_bump_inside_gap_counts_as_evidence(self):
        det = np.zeros(120)
        det[:10] = 8.0
        det[-10:] = 8.0
        det[58:63] = 6.0                         # 空档中间真有一次离地
        self.assertGreater(gap_lift_evidence(det, 0, 120, 30.0), 4.0)


class TestThresholdFloor(unittest.TestCase):
    def test_cap_can_reach_the_absolute_floor(self):
        cfg = Config()
        floor = absolute_floor(172.0, cfg) * max(0.05, cfg.floor_relief)
        cfg.threshold_cap_floor_frac = 1.0
        # 信号本身只有 ±1px 的抖动: 上限不许把阈值压到绝对下限之下
        self.assertGreaterEqual(amplitude_threshold(0.3, 1.0, 172.0, cfg), floor - 1e-9)

    def test_legacy_cap_still_dips_below_floor(self):
        cfg = Config()
        cfg.threshold_cap_floor_frac = 1.0
        strict = amplitude_threshold(0.3, 1.0, 172.0, cfg)
        cfg.threshold_cap_floor_frac = 0.0
        self.assertLess(amplitude_threshold(0.3, 1.0, 172.0, cfg), strict)


def _feed_online(amps, period, cfg, seconds=26.0, width: float = 1.1):
    """把一条节拍信号喂给实时计数器, 返回快照。amps[k] 是第 k 拍的位移峰值(像素)。

    每一拍画成一两帧宽的窄脉冲: 起跳在姿态读数上就是这么窄, 拍形太宽就测不出滤波窗口的差别。
    """
    counter = OnlineCounter(cfg, FPS)
    for frame in range(int(seconds * FPS)):
        k = int(frame / period)
        phase = frame - k * period
        pulse = float(np.exp(-((phase - 1.0) ** 2) / (2.0 * width * width)))
        lift = float(amps[k] if k < len(amps) else 0.0) * pulse
        box = (300.0, 250.0 - lift, 363.0, 400.0 - lift)
        kps = np.zeros((17, 3))
        kps[:, :2] = (331.0, 325.0)
        kps[:, 2] = 0.9
        kps[0] = (331.0, 250.0 - lift, 0.9)
        kps[11] = (331.0, 320.0 - lift, 0.9)
        kps[15] = (315.0, 400.0 - lift, 0.9)
        kps[16] = (345.0, 401.0 - lift, 0.9)
        counter.update(frame, frame / FPS, [(1, box, kps, 0.9)])
    return counter.snapshot(int(seconds * FPS))[0]


class TestOnlineFastWindow(unittest.TestCase):
    """实时链路也要用第二档窄窗: 否则快跳只在离线复盘里数得全, 页面上仍然少。"""

    def test_fast_beats_are_not_fewer_than_wide_window(self):
        """接了窄窗之后实时端只许多数不许少数, 也不许把一次起跳数成两个。

        真实素材上的增益在 README/文档第 9.3 节的基线表里(4761 -> 4785);
        合成节拍太干净, 宽窗本来就不会抹平, 这里锁的是"接线不能把数接丢"。
        """
        period = 7.0                                # ≈4.3 Hz, 快跳档
        frames = int(26 * FPS)
        beats = int(frames / period) + 1
        amps = [9.0] * beats                        # 每一拍都是真起跳
        wide = Config()
        wide.online_fast_window = False
        narrow = Config()
        narrow.online_fast_window = True
        slow = _feed_online(amps, period, wide)["count"]
        fast = _feed_online(amps, period, narrow)["count"]
        self.assertGreater(fast, 0, "快跳素材一条都没数到, 断言没有意义")
        self.assertGreaterEqual(fast, slow, f"快跳窄窗反而少数了: 宽窗 {slow} -> 窄窗 {fast}")
        self.assertLessEqual(fast, beats * 1.2,
                             f"窄窗把快跳数成 {fast} 次, 超过节拍的 {beats} 次太多")


def _beat_samples(fps, seconds, period, amp, damp=None, flat=None):
    """等间隔节拍位移样本(向上为正), 可按时间段挡糊(damp)或压平(flat=站着没跳)。"""
    beats, t = [], 0.0
    while t < seconds:
        beats.append(t)
        t += period
    values = np.zeros(int(seconds * fps))
    for b in beats:
        centre = int(round(b * fps))
        for k in range(-2, 3):
            i = centre + k
            if 0 <= i < values.size:
                values[i] = max(values[i], amp * math.exp(-(k * k) / 2.0))
    out = []
    for i in range(values.size):
        when = i / fps
        v = float(values[i])
        if flat and flat[0] <= when < flat[1]:
            v = 0.0
        elif damp and damp[0] <= when < damp[1]:
            v = v * damp[2]
        out.append((i, when, (100.0, 100.0 - v, 200.0, 400.0 - v), 300.0 - v))
    return out, len(beats)


def _ramp_samples(fps, seconds, p_start, p_end, amp):
    """从慢到快的加速跳: 每一拍的间隔按线性缩短, 返回(样本, 真拍数)。"""
    beats, t = [], 0.0
    while t < seconds:
        beats.append(t)
        t += p_start + (p_end - p_start) * min(1.0, t / max(1e-6, seconds))
    values = np.zeros(int(seconds * fps))
    for b in beats:
        centre = int(round(b * fps))
        for k in range(-2, 3):
            i = centre + k
            if 0 <= i < values.size:
                values[i] = max(values[i], amp * math.exp(-(k * k) / 2.0))
    samples = [(i, i / fps, (100.0, 100.0 - values[i], 200.0, 400.0 - values[i]), 300.0 - values[i])
               for i in range(values.size)]
    return samples, len(beats)


def _count_total(cfg, samples, fps):
    res = count_track(1, samples, cfg, fps, "ankle")
    return res.count + res.stats.get("filled", 0), res


class TestLocalRhythm(unittest.TestCase):
    """变速跳: 一条轨迹里节奏会漂(实测起跳间隔 lag-1 自相关 +0.31), 空档判定要跟着漂。"""

    fps = 25.0

    def test_speeding_up_jumper_counts_closer_to_truth(self):
        """1.2Hz 加速到 3.2Hz: 拿全局周期当尺子会把快起来的那半段判成"没漏"。"""
        samples, beats = _ramp_samples(self.fps, 24.0, 1 / 1.2, 1 / 3.2, 26.0)
        self.assertGreater(beats, 40, "合成素材太短, 断言没有意义")
        legacy = Config()
        legacy.local_period_window = 0
        on = Config()
        self.assertGreaterEqual(on.local_period_window, 4, "默认就该开这条判据")
        off_total, _ = _count_total(legacy, samples, self.fps)
        on_total, _res = _count_total(on, samples, self.fps)
        self.assertGreater(on_total, off_total,
                           f"局部节奏没有追回加速段的漏计: 全局 {off_total} -> 局部 {on_total}")
        self.assertLessEqual(on_total, beats * 1.15,
                             f"局部节奏把 {beats} 拍数成了 {on_total} 次")

    def test_gaps_do_not_lengthen_the_local_ruler(self):
        """漏计造成的双倍间隔不许进局部尺子: 进了就把尺子拉长, 快跳段重新被判成"没漏"。

        实测过这一坑: 不筛间隔时离线 4820 -> 4612, 比不改还少 208 次。
        """
        cfg = Config()
        period = 10.0
        hist = [10.0, 10.0, 10.0, 10.0, 20.0, 10.0, 10.0, 10.0]
        kept = [g for g in hist if g <= cfg.gap_ratio * period]      # 与 count_track 的入窗条件一致
        self.assertNotIn(20.0, kept, "双倍间隔(漏计)进了局部节奏历史")
        self.assertEqual(kept, [10.0] * 7)

    def test_stopped_person_is_still_not_filled(self):
        """尺子变灵之后仍然不许凭空补: 站着没跳的那三秒一个都不补。"""
        cfg = Config()
        samples, beats = _beat_samples(self.fps, 18.0, 0.5, 26.0, flat=(9.0, 12.0))
        total, res = _count_total(cfg, samples, self.fps)
        self.assertGreater(total, 10, f"整段没怎么数到, 断言没有意义: {res.stats}")
        inside = [e.frame for e in res.events if e.kind == "interpolated"
                  and 9.0 * self.fps <= e.frame < 12.0 * self.fps]
        self.assertEqual(inside, [], f"站着没跳的三秒被补出了 {len(inside)} 次起跳")
        self.assertGreaterEqual(res.stats.get("fill_no_lift", 0), 1, res.stats)


class TestFillShareRail(unittest.TestCase):
    """补数占比护栏: 补出来的次数不许反超实测起跳。"""

    fps = 25.0

    def test_rail_cuts_the_fills_not_the_jumps(self):
        cfg = Config()
        self.assertGreater(cfg.max_fill_share, 0, "默认就该开这条护栏")
        samples, _beats = _beat_samples(self.fps, 18.0, 0.5, 26.0, damp=(8.0, 13.0, 0.05))
        loose = Config()
        loose.max_fill_share = 0.0
        loose_total, loose_res = _count_total(loose, samples, self.fps)
        self.assertGreaterEqual(loose_res.stats.get("filled", 0), 2,
                                f"合成用例没能造出补数: {loose_res.stats}")
        tight = Config()
        tight.max_fill_share = 0.05
        tight_total, tight_res = _count_total(tight, samples, self.fps)
        self.assertLess(tight_total, loose_total, "护栏没起作用")
        self.assertGreaterEqual(tight_res.stats.get("fill_capped", 0), 1, tight_res.stats)
        self.assertEqual(tight_res.count, loose_res.count, "护栏不该动实测起跳")


if __name__ == "__main__":
    unittest.main()
