"""能量包络回捞的单元测试: 弱起跳要捞回来, 没跳的时候不能凭空数。

判据取自心电 R 峰检测(Pan-Tompkins / Hamilton): 带通 → 差分平方 → 滑动积分成包络,
再在"按节奏本该有起跳"的空档里逐档压低阈值 search-back, 最后由多通道投票 + 节拍间距签字。
这里只断言不变量(不许丢实测峰、回捞峰必须拉开一个节拍、人不在场一律不捞),
真实素材上的绝对数字在 README 的对照表里。
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jumpcount import signal as sg
from jumpcount.config import Config
from jumpcount.counter import channel_on_axis, count_track, recover_peaks, refine_peak
from jumpcount.online import OnlineCounter

FPS = 25.0
PERIOD = 11.0          # ≈2.27 Hz, 跳绳常用节奏
N = 660
MIN_DISTANCE = 7       # ≈0.6 × 周期, 与 Config.refractory_ratio 一致
AMP_THR = 4.0


def _train(amps, n: int = N, period: float = PERIOD, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """按节拍生成起跳波形: amps[k] 是第 k 拍的幅度, 形状与 counter 里的实测波形一致。"""
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    duty = 0.55
    for i in range(n):
        k = int(i / period)
        frac = (i % period) / period
        amp = amps[k] if k < len(amps) else 0.0
        if frac <= duty:
            out[i] = 4 * amp * (frac / duty) * (1 - frac / duty)
    return out + (rng.normal(0, noise, n) if noise else 0.0)


def _beats(n: int = N, period: float = PERIOD) -> int:
    return int(n / period)


def _cases(weak=(), weak_amp: float = 3.4, noise: float = 0.0, seed: int = 0):
    """造一套踝部弱、髋头强的三通道素材: 弱拍在踝部只留零头, 髋和头照常抬。"""
    strong = [8.0] * _beats()
    weak_amps = [weak_amp if k in weak else 8.0 for k in range(_beats())]
    return (_train(weak_amps, noise=noise, seed=seed),
            _train(strong, noise=noise, seed=seed + 1),
            _train(strong, noise=noise, seed=seed + 2))


def _zero_beats(amps, beats) -> list:
    """把指定几拍的幅度抹平: 模拟某个部位被绳/前面的人挡住, 坐标糊成一条直线。"""
    out = list(amps)
    for k in beats:
        if 0 <= k < len(out):
            out[k] = 0.0
    return out


def _recover(det, channels, cfg, observed=None, amp_thr: float = AMP_THR):
    base = sg.find_peaks(det, MIN_DISTANCE, amp_thr, 0.8 * amp_thr)
    frames = np.arange(det.size, dtype=np.int64)
    seen = np.ones(det.size, dtype=bool) if observed is None else observed
    kept, recovered = recover_peaks(det, frames, frames, base, seen, MIN_DISTANCE, PERIOD, amp_thr,
                                    channels, "ankle", cfg, FPS)
    return base, kept, recovered


class TestEnvelopeSignal(unittest.TestCase):
    """包络本身: 峰位不许漂, 噪声不许聚成鼓包。"""

    def test_bandpass_does_not_move_the_peak(self):
        lift = _train([0.0] * (_beats() - 1) + [8.0] + [0.0] * _beats())
        band = sg.bandpass(lift, FPS, 0.7, 4.0)
        self.assertAlmostEqual(int(np.argmax(band)), int(np.argmax(lift)), delta=1)

    def test_envelope_bumps_about_once_per_jump(self):
        lift = _train([8.0] * _beats())
        env = sg.energy_envelope(lift, FPS, 0.7, 4.0, max(3, int(round(0.4 * PERIOD))))
        peaks = sg.find_peaks(env, int(0.6 * PERIOD), 0.1 * float(np.percentile(env, 95)), 0.0)
        self.assertGreaterEqual(len(peaks), _beats() - 2)
        self.assertLessEqual(len(peaks), _beats() + 2)

    def test_flat_noise_stays_below_jump_energy(self):
        jumps = sg.energy_envelope(_train([8.0] * _beats()), FPS, 0.7, 4.0, 4)
        noise = sg.energy_envelope(_train([0.0] * _beats(), noise=1.0, seed=7), FPS, 0.7, 4.0, 4)
        self.assertLess(float(np.percentile(noise, 95)), float(np.percentile(jumps, 50)))

    def test_gap_candidates_report_their_level(self):
        det = sg.detrend(_train([8.0] * _beats()), 3, 91)
        env = sg.energy_envelope(det, FPS, 0.7, 4.0, 4)
        thr = 0.1 * float(np.percentile(env, 95))
        base = sg.find_peaks(env, MIN_DISTANCE, thr, 0.8 * thr)
        cands = sg.gap_candidates(env, base, MIN_DISTANCE, thr, 1.66 * PERIOD, (0.5, 0.35))
        for pos, level in cands:
            self.assertIn(level, (0.5, 0.35))
            self.assertTrue(all(abs(pos - p) >= MIN_DISTANCE for p in base),
                            "候选必须落在空档里, 不能贴着实测峰")

    def test_refine_peak_snaps_to_displacement_maximum(self):
        lift = _train([8.0] * _beats())
        center = int(3 * PERIOD + 0.275 * PERIOD)
        pos, value = refine_peak(lift, center + 2, int(round(0.4 * PERIOD)))
        self.assertAlmostEqual(pos, center, delta=1)
        self.assertGreater(value, 0.8 * float(lift.max()))

    def test_channel_on_axis_aligns_channels_to_frames(self):
        frames_all = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
        axis = np.arange(7, dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.assertEqual(list(channel_on_axis(values, frames_all, axis)), [float(v) for v in range(1, 8)])
        gap_frames = np.array([0, 2, 4, 6, 8, 10, 12], dtype=np.int64)
        dense = channel_on_axis(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), gap_frames,
                                np.arange(13, dtype=np.int64))
        self.assertEqual(list(dense), [1.0 + 0.5 * i for i in range(13)])


class TestRecoverGates(unittest.TestCase):
    """回捞的四道闸: 该捞的捞回来, 不该数的一次数不上。"""

    def setUp(self):
        self.cfg = Config()

    def _channels(self, weak=(), noise=0.0, seed=0):
        ankle, hip, head = _cases(weak=weak, noise=noise, seed=seed)
        return {"hip": -hip, "head": -head}, sg.detrend(ankle, 3, 91)

    def test_measured_peaks_are_never_dropped(self):
        channels, det = self._channels(weak=(5, 9, 14), noise=0.4, seed=3)
        base, kept, _ = _recover(det, channels, self.cfg)
        self.assertLessEqual(set(base), set(kept),
                             f"回捞把实测峰挤掉了: {sorted(set(base) - set(kept))}")

    def test_weak_jump_is_recovered(self):
        channels, det = self._channels(weak=(5, 9, 14, 20, 26), noise=0.4, seed=3)
        base, kept, recovered = _recover(det, channels, self.cfg)
        self.assertGreater(len(recovered), 0, "整段弱拍一个都没捞回来")
        self.assertGreater(len(kept), len(base))

    def test_recovered_peaks_keep_one_beap_distance(self):
        """回捞峰与左右邻居必须拉开 recover_gap_ratio 个周期: 挤在旁边就是双计。"""
        channels, det = self._channels(weak=(5, 9, 14, 20, 26, 33, 40), noise=0.5, seed=11)
        _, kept, recovered = _recover(det, channels, self.cfg)
        floor = self.cfg.recover_gap_ratio * PERIOD
        for pos in recovered:
            others = [q for q in kept if q != pos]
            near = min((abs(pos - q) for q in others), default=1e9)
            self.assertGreaterEqual(near, floor - 1e-6,
                                    f"回捞峰 {pos} 距邻居只有 {near:.1f} 帧 (< {floor:.1f})")

    def test_strict_gap_gate_lets_nothing_through(self):
        channels, det = self._channels(weak=(5, 9, 14, 20, 26), noise=0.4, seed=3)
        strict = Config()
        strict.recover_gap_ratio = 6.0     # 要拉开六个周期才承认: 一个都塞不进去
        _, _, recovered = _recover(det, channels, strict)
        self.assertEqual(recovered, set(), f"节拍间距闸没生效: {sorted(recovered)}")

    def test_unobserved_span_is_never_recovered(self):
        channels, det = self._channels(weak=(5, 9, 14, 20, 26), noise=0.4, seed=3)
        base, kept, recovered = _recover(det, channels, self.cfg, observed=np.zeros(det.size, bool))
        self.assertEqual(recovered, set())
        self.assertEqual(sorted(kept), sorted(base), "人不在画面里时峰集应当原样返回")

    def test_pure_noise_adds_almost_nothing(self):
        """没人跳: 包络全是噪声, 回捞不能凭空造出一串起跳。"""
        channels, det = self._channels(noise=1.2, seed=21)
        base, kept, recovered = _recover(det, channels, self.cfg, amp_thr=1.5)
        self.assertLessEqual(len(recovered), max(2, int(0.1 * max(1, len(base)) + 2)),
                             f"噪声里捞出 {sorted(recovered)}")

    def test_single_lead_cannot_pass_two_votes(self):
        """只剩一条投票通道: 要求两票的闸门就该拦住低幅度回捞。"""
        ankle, hip, _head = _cases(weak=(5, 9, 14, 20, 26), noise=0.4, seed=3)
        det = sg.detrend(ankle, 3, 91)
        strict = Config()
        strict.envelope_lead_votes = 3     # 主通道 + 两条替身才够, 只给一条
        _, _, strict_rec = _recover(det, {"hip": -hip}, strict)
        loose = Config()
        loose.envelope_lead_votes = 1
        _, _, loose_rec = _recover(det, {"hip": -hip}, loose)
        self.assertLessEqual(len(strict_rec), len(loose_rec),
                             f"投票门槛提高反而捞得更多: {sorted(strict_rec)} vs {sorted(loose_rec)}")

    def test_count_track_never_counts_less_with_envelope(self):
        ankle, hip, head = _cases(weak=(5, 9, 14, 20, 26), noise=0.4, seed=3)
        samples = []
        for i in range(N):
            box = (100.0, 100.0 - ankle[i], 200.0, 400.0 - ankle[i])
            kps = np.zeros((17, 3))
            kps[:, :2] = (150.0, 300.0)
            kps[:, 2] = 0.9
            kps[0] = (150.0, 300.0 - head[i], 0.9)
            kps[11] = (150.0, 250.0 - hip[i], 0.9)
            kps[12] = (160.0, 250.0 - hip[i], 0.9)
            kps[15] = (138.0, 380.0 - ankle[i], 0.9)
            kps[16] = (162.0, 380.0 - ankle[i], 0.9)
            samples.append((i, i / FPS, box, 300.0 - ankle[i], kps))
        on = count_track(1, samples, Config(), FPS)
        off_cfg = Config()
        off_cfg.envelope_detect = False
        # 回捞现在有两条独立路径(能量包络 + 节拍格认领亚阈值鼓包), 这条测的是包络那条,
        # 要一起关掉节拍格认领, 否则"关掉回捞却仍有回捞数"是假警报
        off_cfg.grid_fill_frac = 0.0
        off = count_track(1, samples, off_cfg, FPS)
        self.assertGreaterEqual(on.count, off.count, f"开回捞反而少数: {on.count} < {off.count}")
        flagged = sum(1 for e in on.events if getattr(e, "recovered", False))
        self.assertEqual(on.stats.get("recovered", 0), flagged,
                         "stats 里的回捞数要与事件标记一致")
        self.assertEqual(off.stats.get("recovered", 0), 0)


class TestGapFillEvidence(unittest.TestCase):
    """按节奏补数的取证与口径: 补的每一笔要有别的部位作证, 报表还要说清"应有"按哪段时长算。"""

    OCCLUDED = range(24, 28)     # 脚踝被挡住的几拍(约 1.8s), 起跳还在但位移被抹平

    def _samples(self, ank_flat=(), hip_flat=(), head_flat=(), skip=None):
        strong = [8.0] * _beats()
        ank = _train(_zero_beats(strong, ank_flat), noise=0.4, seed=5)
        hip = _train(_zero_beats(strong, hip_flat), noise=0.4, seed=6)
        head = _train(_zero_beats(strong, head_flat), noise=0.4, seed=7)
        samples = []
        for i in range(N):
            if skip is not None and skip[0] <= i < skip[1]:
                continue     # 人整个离开画面: 这一帧连框都没有
            box = (100.0, 100.0 - ank[i], 200.0, 400.0 - ank[i])
            kps = np.zeros((17, 3))
            kps[:, :2] = (150.0, 300.0)
            kps[:, 2] = 0.9
            kps[0] = (150.0, 300.0 - head[i], 0.9)
            kps[11] = (150.0, 250.0 - hip[i], 0.9)
            kps[12] = (160.0, 250.0 - hip[i], 0.9)
            kps[15] = (138.0, 380.0 - ank[i], 0.9)
            kps[16] = (162.0, 380.0 - ank[i], 0.9)
            samples.append((i, i / FPS, box, 300.0 - ank[i], kps))
        return samples

    def test_occluded_feet_are_witnessed_by_other_channels(self):
        """脚踝被挡平的这段: 髋和头还在按拍抬, 就许补(config.fill_evidence_votes)。

        旧口径只看主通道, 挡平的那段取证恒为 0, 一笔都不敢补 —— 这是快跳漏算里最大的一块。
        """
        data = self._samples(ank_flat=self.OCCLUDED)
        loose = count_track(1, data, Config(), FPS, "ankle")
        self.assertGreaterEqual(loose.stats["filled"], 1, loose.stats)
        self.assertEqual(loose.stats["fill_no_lift"], 0, loose.stats)
        self.assertGreaterEqual(loose.stats["evidence_channels"], 2,
                                f"取证通道没接上: {loose.stats['evidence_channels']}")
        # 退回只看主通道: 同一段素材一笔都不补, 证明放行靠的是别的部位的旁证
        only_main = Config()
        only_main.fill_evidence_votes = 0
        off = count_track(1, data, only_main, FPS, "ankle")
        self.assertEqual(off.stats["filled"], 0, off.stats)
        self.assertGreaterEqual(off.stats["fill_no_lift"], 1, off.stats)

    def test_second_vote_needs_two_body_channels(self):
        """votes=2 要求两个部位同时达标: 只剩髋在动时不许补, 单通道抖动才算不数。"""
        data = self._samples(ank_flat=self.OCCLUDED, head_flat=self.OCCLUDED)
        one = count_track(1, data, Config(), FPS, "ankle")        # 默认一票: 髋达标即放行
        self.assertGreaterEqual(one.stats["filled"], 1, one.stats)
        two_cfg = Config()
        two_cfg.fill_evidence_votes = 2
        two = count_track(1, data, two_cfg, FPS, "ankle")
        self.assertEqual(two.stats["filled"], 0, two.stats)
        self.assertGreaterEqual(two.stats["fill_no_lift"], 1, two.stats)

    def test_expected_count_also_reports_the_visible_span(self):
        """人中途离场 8 秒: "应有次数"要有一条只按真在画面里的秒数算的口径。

        整段口径会把离场那 8 秒也算成"该跳没跳", 完整率永远追不平, 还会凭空招来补数。
        """
        samples = self._samples(skip=(220, 430))
        res = count_track(1, samples, Config(), FPS, "ankle")
        self.assertLess(res.stats["observed_sec"], res.stats["duration_sec"] - 5.0, res.stats)
        self.assertAlmostEqual(res.stats["expected_visible"],
                               round(res.rate_hz * res.stats["observed_sec"], 1), delta=0.2)
        self.assertLess(res.stats["expected_visible"], res.stats["expected_count"] * 0.8, res.stats)
        self.assertGreater(res.stats["rhythm_coverage_visible"], res.stats["rhythm_coverage"] + 0.2,
                           res.stats)
        # 按可见段算其实数到了九成五: 缺口集中在人不在画面的那一段, 不是计数器失灵
        self.assertGreaterEqual(res.stats["rhythm_coverage_visible"], 0.9, res.stats)


class TestOnlineRecovery(unittest.TestCase):
    """实时链路的接线: 计数器要能在页面上单独交代回捞了多少个。"""

    def _feed(self, lift_of, seconds: float = 26.0) -> dict:
        counter = OnlineCounter(Config(), FPS)
        for frame in range(int(seconds * FPS)):
            lift = lift_of(frame)
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

    def test_standing_person_gets_nothing(self):
        """站着抖一下: 参与判定必须把它剔掉, 回捞也不能把幻影放大成一片。"""
        rng = np.random.default_rng(3)
        snap = self._feed(lambda _f: float(rng.normal(0, 0.3)))
        self.assertFalse(snap["participant"], f"站着不动仍被判成在跳绳: {snap['reason']}")
        # 上限按"每秒不到一个"定: 这条轨迹已经被剔出总量, 残余幻影只说明噪声里总能凑出几个鼓包;
        # 一旦节拍闸失效, 26s 会被回捞成几十个, 量级上分得很开。
        self.assertLessEqual(snap.get("recovered"), 18,
                             f"静止抖动被回捞成 {snap['recovered']} 个起跳, 节拍闸失效")

    def test_jumper_snapshot_reports_recovered_field(self):
        snap = self._feed(lambda f: 12.0 * abs(np.sin(np.pi * (f % PERIOD) / PERIOD)))
        self.assertIn("recovered", snap)
        self.assertLessEqual(snap["recovered"], snap["count"])
        self.assertGreater(snap["count"], 20, f"稳定起跳只数到 {snap['count']}")


if __name__ == "__main__":
    unittest.main()
