"""在线计数: 边看画面边出数, 供实时页面使用。

与离线复盘共用同一套阈值逻辑, 区别只在于:
  * 信号放在滑动窗口里, 每帧重算一次去基线与峰值;
  * 只承认"窗口尾部留足去基线上下文"的峰, 避免边界抖动造成重复计数;
  * 参与判定(是否在跳绳)用累计量在线算, 走动的路人当场标出来。
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from . import signal as sg
from .config import Config
from .counter import (OUT, amplitude_threshold, candidate_values, channel_quality, channel_rank,
                      LIFT_EVIDENCE_CHANNELS, drift_spread, gap_evidence_from_denses,
                      gap_evidence_votes, participation, pick_smooth_pass, recover_grid_peaks,
                      recover_peaks)


class TrackCounter:
    def __init__(self, track_id: int, cfg: Config, fps: float):
        self.track_id = track_id
        self.cfg = cfg
        self.fps = fps
        self.window = max(cfg.baseline_win * 3, int(round(fps * cfg.online_window_sec)))
        self.guard = cfg.smooth_win + 2  # 动态确认延迟, 峰值确认后立即出数
        self.poll_every = max(1, int(round(fps / max(1.0, cfg.poll_hz))))
        self._polled_frame = -(10 ** 9)
        self.buffer: Deque[Optional[float]] = Deque(maxlen=self.window)
        self.buffer_body: Deque[Optional[float]] = Deque(maxlen=self.window)  # 头/髋通道, 用来量真实离地幅度
        self.channel_buffers: Dict[str, Deque[Optional[float]]] = {
            name: Deque(maxlen=self.window) for name in ("ankle", "hip", "head")
        }
        self.channel_used: Optional[str] = None
        self.channel_scores: Dict[str, Tuple[float, int, float, float]] = {}
        self._scored_at: Optional[float] = None  # 上次重选通道的时刻: 得分逐帧几乎不变, 不必每次出数都重算三条
        self.frames: Deque[int] = Deque(maxlen=self.window)
        self.lifts: List[float] = []
        self.first_frame: Optional[int] = None
        self.last_frame: Optional[int] = None
        self.emitted_frame = -1
        self.count = 0
        self.rope = 0
        self.filled = 0  # 按节奏补齐的起跳: 单独记一笔, 与实测数分开
        self.fill_no_lift = 0  # 空档里毫无离地证据 = 人停下了, 不替他凭空补数
        self.fill_phase = 0  # 遮挡空档里靠"出洞没脱拍"补上的次数, 见 config.fill_hole_phase_tol
        self.recovered = 0  # 能量包络 + 多通道投票找回的起跳: 有信号依据, 与凭空补齐分开
        self.events: List[dict] = []
        self.height = 0.0
        self.height_n = 0
        self.height_sum = 0.0
        self.height_sum2 = 0.0
        self.centers: Deque[float] = Deque(maxlen=600)
        self.center_frames: Deque[int] = Deque(maxlen=600)  # 与 centers 对齐的帧号, 分窗口量横向活动范围要用
        self.conf_sum = 0.0
        self.conf_n = 0
        self.missing = 0
        self.span_total = 0      # 已结束片段的总帧数(可见率分母)
        self.observed_total = 0  # 已结束片段里真正看到的帧数
        self.fragments = 1
        self.raw_ids: Set[int] = set()
        self.period_frames: Optional[float] = None
        self.last_jump_time: Optional[float] = None
        self.stats: dict = {}
        self.box: Optional[list] = None
        self.source_seen: Dict[str, int] = {}
        self.channel_delta: Dict[str, Tuple[float, int]] = {}  # 替身通道 -> 主通道的平移量(滑动平均)
        self.rhythm_strength: Optional[float] = None
        self.rhythm_clarity: Optional[float] = None
        self.last_peak_frame: Optional[float] = None  # 上一个评估过的峰(含热身期跳过的), 空档判定的参考
        self.recovered_frames: Set[int] = set()  # 本次窗口里属于回捞的峰下标(窗口每次重算)
        self.carry_count = 0  # 上一次离场时寄存下来、这次接回来的次数: 与本轮实测分开显示
        # 局部节奏历史: 只存"当作正常拍收下"的起跳间隔, 供空档判定当尺子(见 config.local_period_window)
        self.gap_hist: Deque[float] = Deque(maxlen=32)
        self.sync_n = 0
        self.sync_a = 0.0
        self.sync_b = 0.0
        self.sync_aa = 0.0
        self.sync_bb = 0.0
        self.sync_ab = 0.0

    # ---- 特征累积 ----
    def observe(self, frame: int, box, keypoints, conf: float) -> None:
        if self.last_frame is not None and frame > self.last_frame + 1:
            gap = frame - self.last_frame - 1
            if gap > self.cfg.max_gap:
                # 断档太久(人离场或被长时间遮挡): 另起一段, 不把缺席时间算进可见率
                self._close_segment()
                self.buffer.clear()
                self.buffer_body.clear()
                for buf in self.channel_buffers.values():
                    buf.clear()
                self.frames.clear()
            else:
                for missed in range(self.last_frame + 1, frame):
                    self.buffer.append(None)
                    self.buffer_body.append(None)
                    for buf in self.channel_buffers.values():
                        buf.append(None)
                    self.frames.append(missed)
                self.missing += gap
        if self.first_frame is None:
            self.first_frame = frame
        cands = candidate_values(keypoints, box, self.cfg)
        y, source = feature_y_from(box, keypoints, self.cfg)
        self.source_seen[source] = self.source_seen.get(source, 0) + 1
        self.buffer.append(self._to_dominant(y, source, cands))
        self.buffer_body.append(cands.get("head", cands.get("hip")))
        for name, buf in self.channel_buffers.items():
            buf.append(cands.get(name))
        self.frames.append(frame)
        self.last_frame = frame
        h = max(1.0, float(box[3] - box[1]))
        self.height = h if self.height == 0 else 0.8 * self.height + 0.2 * h
        self.height_n += 1
        self.height_sum += h
        self.height_sum2 += h * h
        if self.height_n > 600:  # 滑动衰减, 与双脚同步性一致
            for attr in ("height_n", "height_sum", "height_sum2"):
                setattr(self, attr, getattr(self, attr) * 0.5)
        cx = (float(box[0]) + float(box[2])) / 2.0
        self.centers.append(cx)
        self.center_frames.append(int(frame))
        self.conf_sum += float(conf)
        self.conf_n += 1

    def _to_dominant(self, value: Optional[float], source: str, cands: Dict[str, float]) -> Optional[float]:
        """把替身通道的读数平移回主通道。

        脚踝被绳挡一下就换成髋, 读数会整体跳一个台阶(髋本来就比脚踝高几十像素),
        台阶本身会被数成一次起跳, 也会把真正的起跳埋掉。两个通道同时可见时
        它们的差是中位稳定的, 攒够样本后按这个差把替身平移回来。
        """
        if value is None or source not in cands or not self.source_seen:
            return value
        dominant = max(self.source_seen, key=self.source_seen.get)
        if source == dominant or dominant not in cands:
            return value
        delta, seen = self.channel_delta.get(source, (cands[dominant] - cands[source], 0))
        delta = delta if seen == 0 else 0.8 * delta + 0.2 * (cands[dominant] - cands[source])
        self.channel_delta[source] = (delta, seen + 1)
        if seen < 3:
            return value
        return value + delta

    # ---- 选信号通道 ----
    def _selected_values(self, frames: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        """当前窗口里哪条身体部位的节律最清楚就用哪条计数。

        远机位或低抬腿时脚踝位移只有一两个像素, 一半真起跳压在阈值以下
        (实测同一个人脚踝只数到 29 个, 换成髋是 45 个); 髋和头又会被走路摆动骗到,
        所以要它明确胜出才换、换回去也要胜出同样多, 免得两条通道打平时来回跳。

        三条通道各要跑一遍去基线+自相关, 每秒六次全算会把计数环节的成本翻倍,
        所以按 channel_reselect_sec 隔一秒才重选一次, 期间沿用刚选定的通道取数。
        """
        fallback = np.array([np.nan if v is None else v for v in self.buffer], dtype=np.float64)
        if self.cfg.channel_select != "rhythm" or self.height <= 0:
            return fallback
        reselect = (self.channel_used is None or self._scored_at is None
                    or now is None or now - self._scored_at >= self.cfg.channel_reselect_sec)
        if reselect:
            scores: Dict[str, Tuple[float, int, float, float]] = {}
            # 通道择优每秒只重算一次, 分数里顺带记下每条通道自己的位移 p95:
            # "头部零位移"闸门要用, 单独再跑一遍去基线会把这条链路的成本翻倍。
            for name in self.channel_buffers:
                values = np.array([np.nan if v is None else v for v in self.channel_buffers[name]], dtype=np.float64)
                if np.isfinite(values).sum() < 20:
                    continue
                scores[name] = channel_quality(frames[:values.size], -values[:frames.size], self.height, self.cfg, self.fps)
            self.channel_scores = scores
            self._scored_at = now
            if not scores:
                return fallback
            best = max(scores, key=channel_rank(scores, self.cfg))
            current = self.channel_used if self.channel_used in scores else best
            if best != current and scores[best][0] < scores[current][0] + self.cfg.channel_switch_margin:
                best = current
            self.channel_used = best
        best = self.channel_used
        values = np.array([np.nan if v is None else v for v in self.channel_buffers[best]], dtype=np.float64)
        return values[:frames.size] if values.size > frames.size else values

    # ---- 在线出数 ----
    def poll(self, now: float, force: bool = False) -> List[dict]:
        if self.first_frame is None or len(self.buffer) < max(12, int(self.fps * 1.2)):
            return []
        # 峰值搜索比取信号贵得多: 每人每秒扫几次就够, 窗口仍然覆盖所有帧, 不会漏峰
        if not force and self.last_frame - self._polled_frame < self.poll_every:
            return []
        self._polled_frame = self.last_frame
        frames = np.array(self.frames, dtype=np.int64)
        values = self._selected_values(frames, now)
        valid = ~np.isnan(values)
        if valid.sum() < 10:
            return []
        if frames.size != values.size:
            frames = frames[:values.size]
        idx = np.arange(values.size)
        dense = values.copy()
        dense[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
        lift = -dense
        baseline_win = self.cfg.baseline_win if lift.size >= self.cfg.baseline_win else _odd(lift.size // 2)
        # 与离线同一口径(向下取整): 实时链路是用户实际看到的那个数,
        # 这里向上取整会让快跳在实时端先被锁成两倍周期, 离线再准也补不回来
        hard_min = max(2, int(self.fps / self.cfg.max_rate_hz))
        # 与离线同一档快跳窄窗: 3 帧中值会把快跳的一拍整个抹平, 实时端先少数,
        # 离线再怎么补也只是修历史。关掉即退回"只用 smooth_win 一档"。
        used = pick_smooth_pass(lift, valid, baseline_win, self.height, self.fps, self.cfg, hard_min) \
            if self.cfg.online_fast_window else None
        if used is None:
            detrended = sg.detrend(lift, self.cfg.smooth_win, baseline_win)
            # 与离线复盘同一条噪声口径: 缺帧处插过值会被拉平, 混进估计里噪声假性变小, 阈值跟着变松
            noise = sg.hf_sigma_masked(detrended, valid)
            peak_level = float(np.percentile(detrended, 95))
            amp_thr = amplitude_threshold(noise, peak_level, self.height, self.cfg)
            coarse = sg.find_peaks(detrended, hard_min, amp_thr, 0.8 * amp_thr)
        else:
            detrended, noise = used["detrended"], used["noise"]
            peak_level, amp_thr, coarse = used["peak_level"], used["amp_thr"], used["coarse"]
        self.lifts = [round(float(v), 2) for v in detrended[-120:]]
        lift_ratio_best = peak_level / self.height if self.height else 0.0
        body = np.array([np.nan if v is None else v for v in self.buffer_body], dtype=np.float64)
        if body.size == detrended.size and np.isfinite(body).sum() >= 12:
            bidx = np.arange(body.size)
            bvalid = np.isfinite(body)
            body_filled = body.copy()
            body_filled[~bvalid] = np.interp(bidx[~bvalid], bidx[bvalid], body[bvalid])
            body_detrended = sg.detrend(-body_filled, self.cfg.smooth_win, baseline_win)
            lift_ratio_best = max(lift_ratio_best, float(np.percentile(body_detrended, 95)) / max(self.height, 1e-6))
        spacing = float(np.median(np.diff(coarse))) if len(coarse) >= 3 else 0.0
        if hard_min <= spacing <= self.fps / self.cfg.min_rate_hz:
            self.period_frames = spacing
        elif len(coarse) < 3 and int(valid.sum()) >= int(0.9 * self.window):
            # 窗口已经装满(约 12s 信号)却连三个峰都数不出来: 节奏不再成立, 要把周期忘掉。
            # 以前周期一旦估出来就永远留着, 人早就停下不跳了,"节奏清楚"的资格还在往下发,
            # 低置信计入、按节奏补、包络回捞都据此继续出数, 停跳的人会被继续加数。
            self.period_frames = None
        period = self.period_frames or hard_min / self.cfg.refractory_ratio
        min_distance = max(hard_min, int(round(self.cfg.refractory_ratio * period)))
        # 与离线同一判据: 不应期到不到的地方看有没有落回地面, 快跳时不把真起跳当重复峰丢
        landing: set = set()
        if self.cfg.landing_valley_frac > 0:
            peaks, landing = sg.merge_close_peaks(detrended, coarse, min_distance,
                                                  self.cfg.landing_valley_frac, amp_thr)
        else:
            peaks = sg.find_peaks(detrended, min_distance, amp_thr, 0.8 * amp_thr)
        self.recovered_frames = set()
        if self.cfg.envelope_detect and self.cfg.search_back_levels:
            channels = {name: np.array([np.nan if v is None else v for v in buf], dtype=np.float64)
                        for name, buf in self.channel_buffers.items()}
            size = detrended.size
            axis = frames[:size] if frames.size >= size else np.arange(size, dtype=np.int64)
            peaks, self.recovered_frames = recover_peaks(
                detrended, axis, axis, peaks, valid[:size], min_distance, period, amp_thr,
                {name: values[:size] for name, values in channels.items()},
                self.channel_used or "", self.cfg, self.fps)
        if len(peaks) >= 3:
            intervals = np.diff(peaks)
            self.rhythm_strength = round(float(np.clip(
                1.0 - np.median(np.abs(intervals - period)) / max(period, 1e-6), 0.0, 1.0)), 3)
        else:
            # 与周期同理: 这一窗口的峰不够, 上一个窗口量出来的节律强度不能继续作数
            self.rhythm_strength = None
        # 另一个口径(自相关取在实测周期那一格), 与离线同一函数: 见 config.rhythm_measure 的注释
        self.rhythm_clarity = round(sg.autocorr_clarity(detrended, int(round(period))), 3)
        # 节拍格认领亚阈值鼓包(与离线同一条, 见 config.grid_fill_frac):
        # 远景快跳的人抬腿只有半个像素, 整段卡在阈值下面, 但节拍格是准的 —— 实时端不接这条
        # 就会比离线复盘少数一截(实测离线 5656 而实时停在 5247)。
        # 放在节律强度算完之后: 认领来的峰不该反过来给"节奏清不清楚"作证(自我循环)。
        if self.cfg.grid_fill_frac > 0:
            peaks, grid_extra = recover_grid_peaks(detrended, valid[:detrended.size], list(peaks),
                                                   period, amp_thr, float(self.rhythm_clarity), self.cfg)
            self.recovered_frames |= grid_extra

        # 确认延迟取一个跳跃周期: 峰值必须已经完整下落才承认, 兼顾实时性与稳定性
        guard = max(self.guard, int(round(period)))
        limit = int(frames[-1]) - guard
        # 出数最小间隔: 窗口滑动会让同一个峰的帧位置漂移 1~2 帧, 没有这道闸会重复出数
        min_emit_gap = max(2, int(round(0.5 * period)))
        emitted: List[dict] = []

        def coverage_between(lo: float, hi: float) -> float:
            """空档区间里真正观测到的比例: 缓冲区被清空过(人离场/长时间遮挡)时分子会掉下来。"""
            span = int(hi) - int(lo) + 1
            if span <= 1:
                return 0.0
            selected = (frames >= int(lo)) & (frames <= int(hi))
            return float(valid[selected].sum()) / span

        rhythm_trusted = (self.rhythm_strength is not None
                          and self.rhythm_strength >= self.cfg.keep_rhythm_strength)
        # 与离线同构的局部节奏: 变速跳时拿全局周期当尺子, 快起来的那半段会被判成"没漏"
        local_n = max(0, int(getattr(self.cfg, "local_period_window", 0) or 0))
        period_low = float(max(2.0, hard_min))
        period_high = float(self.fps) / max(0.05, self.cfg.min_rate_hz)
        # 取证通道只在真出现"该补"的空档时才建: 建一次要三条通道各去一次基线, 逐帧建太贵
        extras: Dict[str, Tuple[np.ndarray, float]] = {}
        extras_ready = False
        for p in peaks:
            at = int(frames[p])
            if at <= self.emitted_frame or at > limit or at - self.emitted_frame < min_emit_gap:
                continue
            if self.cfg.warmup_sec and at - self.first_frame < self.cfg.warmup_sec * self.fps:
                self.emitted_frame = at
                # 热身期跳过的峰也是节奏参考: 不记下来, 下一个峰会把整段热身当成漏计补回来
                self.last_peak_frame = float(at)
                continue
            sub = sg.parabolic_peak(detrended, p)
            frame_pos = float(frames[0]) + sub
            when = round(frame_pos / self.fps, 3)
            kind = "jump"
            scale = period
            if local_n >= 4 and len(self.gap_hist) >= local_n:
                scale = min(max(float(np.percentile(np.asarray(list(self.gap_hist)[-local_n:],
                                                                     dtype=np.float64),
                                                        float(self.cfg.local_period_quantile))),
                                period_low), period_high)
            last_jump = next((e for e in reversed(self.events) if e["kind"] == "jump"), None)
            if last_jump is not None:
                gap = frame_pos - last_jump["frame"]
                if gap < self.cfg.double_under_ratio * scale and p not in landing:
                    kind = "bump"
                elif gap > self.cfg.gap_ratio * scale:
                    self.events.append(
                        {"kind": "gap", "frame": round(frame_pos, 2), "time": when, "track": self.track_id,
                         "gap": round(gap / max(period, 1e-6), 2)}
                    )
            reference = frame_pos
            if last_jump is not None:
                reference = max(last_jump["frame"], self.last_peak_frame or float("-inf"))
            elif self.last_peak_frame is not None:
                reference = self.last_peak_frame
            gap_back = reference if reference < frame_pos else None
            if gap_back is not None:
                missing = int(round((frame_pos - gap_back) / max(scale, 1e-6))) - 1
                cov = coverage_between(gap_back, frame_pos)
                beats_gap = (frame_pos - gap_back) / max(scale, 1e-6)
                # 遮挡空档旁路(与离线同一条): 覆盖率不够时改由"出洞后的第一拍有没有脱拍"签字。
                # 遮挡期间的位移是插值填出来的, 离地取证闸门在那种数据上量不到东西,
                # 见 config.fill_hole_phase_tol
                phase_ok = (self.cfg.fill_hole_phase_tol > 0 and missing >= 1
                            and cov < self.cfg.min_fill_coverage
                            and abs(beats_gap - round(beats_gap)) <= self.cfg.fill_hole_phase_tol)
                # 空档是少数的主因: 遮挡十几帧就跳不到峰, 人一直都在跳。
                # 但只有在"人确实在画面里、节奏也是实测出来的"时才补, 补的个数还封顶,
                # 否则节奏一估歪就整段虚高, 宁少不虚。
                if (missing >= 1 and self.cfg.interpolate_gaps and rhythm_trusted
                        and (cov >= self.cfg.min_fill_coverage or phase_ok)):
                    lo = max(0, int(round(gap_back - frames[0])))
                    hi = min(detrended.size, int(round(frame_pos - frames[0])) + 1)
                    if self.cfg.fill_evidence_votes > 0 and not extras_ready:
                        extras_ready = True
                        extras = self.evidence_signals(frames, valid, detrended.size,
                                                       baseline_win, int(used["smooth_win"]) if used else self.cfg.smooth_win)
                    if (self.cfg.fill_min_lift_frac > 0 and not phase_ok
                            and gap_evidence_votes(detrended, amp_thr, extras, lo, hi, scale,
                                                   int(self.cfg.fill_evidence_votes)) < self.cfg.fill_min_lift_frac):
                        self.fill_no_lift += missing
                    else:
                        steps = min(missing, self.cfg.max_fill_per_gap)
                        if self.cfg.max_fill_share > 0 and self.count >= 4:
                            # 与离线同一条护栏: 补数只能是实测起跳的零头
                            quota = int(self.count * self.cfg.max_fill_share)
                            steps = max(0, min(steps, quota - self.filled))
                        if self.cfg.fill_cap_expected and period > 0 and self.first_frame is not None:
                            # 与离线同一条护栏: 补完不许超过"这段画面按节奏本该有的次数"。
                            # 寄存接回来的次数发生在画面之外, 不在预算里, 要先减掉再比。
                            # 尺子取"全局周期"和"局部快尺"里更短的一条: 全局周期被锁成两倍时,
                            # 按它算的预算只有真实拍数的一半, 快跳的人会被这条上限整个卡死不补 ——
                            # 那恰好是最需要补数的场景(实测实时 5086 -> 5054)。
                            ruler = min(period, scale)
                            warm = int(round(self.cfg.warmup_sec * self.fps))
                            beats = (limit - self.first_frame - warm) / max(ruler, 1e-6)
                            room = int(math.ceil(beats)) - (self.count + self.filled - self.carry_count)
                            steps = max(0, min(steps, room))
                        for step in range(1, steps + 1):
                            fill_pos = gap_back + (frame_pos - gap_back) * step / (steps + 1)
                            record = {"kind": "interpolated", "frame": round(fill_pos, 2),
                                      "time": round(fill_pos / self.fps, 3),
                                      "amplitude": round(float(amp_thr) * 0.6, 2), "track": self.track_id}
                            if phase_ok:
                                record["phase_filled"] = True
                                self.fill_phase += 1
                            self.events.append(record)
                            self.filled += 1
                            self.rope += 1
                            self.last_jump_time = record["time"]
                            emitted.append(record)
            if kind == "jump" and local_n >= 4 and last_jump is not None:
                back = max(last_jump["frame"], self.last_peak_frame or float("-inf"))
                if frame_pos - back <= self.cfg.gap_ratio * period:
                    # 只有被当作正常拍收下的间隔才进窗口: 漏计的双倍间隔会把尺子拉长
                    self.gap_hist.append(frame_pos - back)
            self.emitted_frame = at
            self.last_peak_frame = frame_pos
            record = {"kind": kind, "frame": round(frame_pos, 2), "time": when,
                      "amplitude": round(float(detrended[p]), 2), "track": self.track_id}
            if p in self.recovered_frames:
                # 幅度不足但包络+多通道投票通过的起跳: 打上标记, 页面上与实测峰区分开
                record["recovered"] = True
                self.recovered += 1
            self.events.append(record)
            self.events = self.events[-200:]
            if kind == "bump":
                self.rope += 1
                if self.cfg.policy == "rope":
                    self.count += 1
                    self.last_jump_time = when
                continue
            self.count += 1
            self.rope += 1
            self.last_jump_time = when
            emitted.append(record)
        span, observed = self._coverage()
        self.stats = {
            "count": self.count,
            "filled": self.filled,
            "fill_no_lift": self.fill_no_lift,
            "fill_phase": self.fill_phase,
            "recovered": self.recovered,
            "total_count": self.count + self.filled,
            "snr": round(peak_level / noise, 2) if noise > 1e-6 else 0.0,
            "lift_ratio": round(peak_level / self.height, 4) if self.height else 0.0,
            "lift_ratio_best": round(lift_ratio_best, 4),
            # 与离线同一口径的"头部零位移"证据: 复用通道择优那一趟算出来的各通道位移 p95
            "head_p95_px": self._head_p95(),
            "body_p95_px": self._body_p95(),
            "excursion_ratio": round(drift_spread(self.centers, self.fps, self.cfg.drift_window_sec,
                                                  list(self.center_frames)) / self.height, 3) if self.height else 0.0,
            "visibility": round(observed / max(1, span), 3),
            "mean_conf": round(self.conf_sum / max(1, self.conf_n), 3),
            "duration_sec": round(observed / self.fps, 2),
            "rate_hz": round(self.fps / self.period_frames, 2) if self.period_frames else None,
            "rhythm_strength": self.rhythm_strength,
            "rhythm_clarity": self.rhythm_clarity,
            "period_source": "峰间隔中位数" if self.period_frames else "fallback",
            "height_cv": self.height_cv(),
            # 双脚同步性是"漂移的真人 vs 路人"唯一的反证, 不给判定层就等于实时链路只能剔除不能降级
            "ankle_sync": self.sync_value(),
            # 这里留空而不是写 "none": snapshot 里要在没做过通道择优时退回"实际用过最多的那条通道"。
            # 写成非空的 "none" 会把那道退回逻辑挡掉, 页面上"信号"一栏永远是 none。
            "signal_source": self.channel_used or "",
            "channel_scores": {name: {"rhythm": v[0], "peaks": v[1], "coverage": v[2]}
                               for name, v in self.channel_scores.items()},
        }
        return emitted

    def evidence_signals(self, frames: np.ndarray, observed: np.ndarray, size: int,
                         baseline_win: int, smooth_win: int) -> Dict[str, Tuple[np.ndarray, float]]:
        """空档取证用的各部位去基线与阈值 —— 与离线 counter.gap_evidence_signals 同一套口径。

        实时端的部位通道缓冲区本来就按帧对齐(与 recover_peaks 的投票同一份数据),
        所以不需要再搬一次帧轴, 剩下的去基线/噪声/阈值全部走同一个函数。
        """
        if self.cfg.fill_evidence_votes <= 0:
            return {}
        denses: Dict[str, np.ndarray] = {}
        for name in LIFT_EVIDENCE_CHANNELS:
            if name == (self.channel_used or ""):
                continue
            buf = self.channel_buffers.get(name)
            if not buf:
                continue
            values = np.array([np.nan if v is None else v for v in buf], dtype=np.float64)
            values = values[:size]
            if values.size != size:
                continue
            good = np.isfinite(values)
            if good.sum() < 12:
                continue
            idx = np.arange(size)
            filled = values.copy()
            filled[~good] = np.interp(idx[~good], idx[good], values[good])
            denses[name] = -filled
        return gap_evidence_from_denses(denses, observed[:size], self.height, self.cfg,
                                       smooth_win, baseline_win, size)

    def height_cv(self) -> Optional[float]:
        """框高的变异系数: 一个人不该忽大忽小, 突然变大变小多半是 ID 换到了别人身上。"""
        if self.height_n < 20:
            return None
        mean = self.height_sum / self.height_n
        var = max(0.0, self.height_sum2 / self.height_n - mean * mean)
        return round(float(np.sqrt(var) / max(mean, 1e-6)), 3)

    def carry_forward(self, carry: dict) -> None:
        """把上次离场时寄存的次数接回来: 计数器被淘汰不等于他跳过的次数作废。

        只接累计量, 信号缓冲/覆盖率/相关性都保持新轨迹的观测: 接回来的必须是"这个人自己跳的",
        不能把上一个人的可见率、双脚同步性一起继承过来替新证据背书。
        """
        self.count += int(carry.get("count", 0))
        self.filled += int(carry.get("filled", 0))
        self.recovered += int(carry.get("recovered", 0))
        self.rope += int(carry.get("rope", 0))
        self.carry_count = int(carry.get("count", 0)) + int(carry.get("filled", 0))
        if self.last_jump_time is None and carry.get("last_jump_time") is not None:
            self.last_jump_time = carry["last_jump_time"]

    # ---- 覆盖率: 只统计"人在画面里"的那些段 ----
    def _head_p95(self) -> Optional[float]:
        """头部位移 p95(像素): 头部通道观测不足时给出 None, 闸门自动不启用。"""
        head = self.channel_scores.get("head")
        if head is None or len(head) < 4 or head[2] < self.cfg.min_head_coverage:
            return None
        return round(float(head[3]), 3)

    def _body_p95(self) -> Optional[float]:
        """踝/髋/头里动得最大的那条通道位移 p95(像素)。"""
        values = [float(score[3]) for score in self.channel_scores.values() if len(score) >= 4]
        return round(max(values), 3) if values else None

    def _coverage(self) -> Tuple[int, int]:
        current = 0 if self.first_frame is None or self.last_frame is None else self.last_frame - self.first_frame + 1
        return self.span_total + current, self.observed_total + max(0, current - self.missing)

    def _close_segment(self) -> None:
        current = 0 if self.first_frame is None or self.last_frame is None else max(0, self.last_frame - self.first_frame + 1)
        self.span_total += current
        self.observed_total += max(0, current - self.missing)
        self.first_frame = None
        self.missing = 0

    def snapshot(self, frame: Optional[int] = None) -> dict:
        stats = getattr(self, "stats", {})
        verdict, reason = judge_detail(stats, self.cfg)
        return {
            "id": self.track_id,
            "verdict": verdict,
            "fragments": self.fragments,
            "absent_frames": int(frame - self.last_frame) if frame is not None and self.last_frame is not None else 0,
            "count": self.count,
            "filled": self.filled,
            "recovered": self.recovered,
            "total_count": self.count + self.filled,
            "carried": self.carry_count,
            "rope": self.rope,
            "participant": verdict != OUT,
            "reason": reason,
            "rate_hz": stats.get("rate_hz"),
            "snr": stats.get("snr"),
            "visibility": stats.get("visibility"),
            "lift_ratio": stats.get("lift_ratio"),
            "lift_ratio_best": stats.get("lift_ratio_best"),
            "head_p95_px": stats.get("head_p95_px"),
            "body_p95_px": stats.get("body_p95_px"),
            "excursion_ratio": stats.get("excursion_ratio"),
            "ankle_sync": self.sync_value(),
            "duration_sec": stats.get("duration_sec"),
            "rhythm_strength": stats.get("rhythm_strength"),
            "rhythm_clarity": stats.get("rhythm_clarity"),
            "height_cv": stats.get("height_cv"),
            "period_source": stats.get("period_source"),
            "last_jump": self.last_jump_time,
            "signal_source": stats.get("signal_source")
                             or (max(self.source_seen, key=self.source_seen.get) if self.source_seen else "none"),
            "signal": getattr(self, "lifts", []),
            "box": self.box if hasattr(self, "box") else None,
        }

    # ---- 双脚同步性(走路 vs 跳绳) ----
    def sync_value(self) -> Optional[float]:
        if self.sync_n < 25:
            return None
        n = float(self.sync_n)
        ma, mb = self.sync_a / n, self.sync_b / n
        va = self.sync_aa / n - ma * ma
        vb = self.sync_bb / n - mb * mb
        if va <= 1e-6 or vb <= 1e-6:
            return None
        cov = self.sync_ab / n - ma * mb
        return round(float(cov / np.sqrt(va * vb)), 3)


def _odd(value: int, floor: int = 5) -> int:
    value = max(floor, int(value))
    return value if value % 2 else value - 1


PREFERRED = ("ankle", "hip", "head", "box_bottom")


def feature_y_from(box, keypoints, cfg: Config) -> Tuple[Optional[float], str]:
    """按脚踝->髋->头的优先级取垂直位移; 框底边只在完全没有关键点时才用(它常被阴影粘在地面)。"""
    cands = candidate_values(keypoints, box, cfg)
    order = ((cfg.signal_source,) if cfg.signal_source != "auto" else ()) + PREFERRED
    for source in order:
        if source in cands:
            return cands[source], source
    return None, "none"


def judge_detail(stats: dict, cfg: Config) -> Tuple[str, str]:
    """三态参与判定: in / provisional / out。"""
    merged = dict(stats)
    merged.setdefault("ankle_sync", None)
    return participation(merged, cfg)


def judge(stats: dict, cfg: Config) -> Tuple[bool, str]:
    verdict, reason = judge_detail(stats, cfg)
    return verdict != OUT, reason


def box_iou(a, b) -> float:
    if not a or not b:
        return 0.0
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


class SeatBook:
    """固定机位下把「站位」当身份: 检测闪烁把一个人拆成多条轨迹时接回同一个人。

    跳绳时人基本守在一个圈里, 站位比短时运动预测可靠得多; 而跟踪 ID 一旦断裂,
    同一个人的可见率会被摊薄到一半以下, 直接被"可见率过低"踢出计数。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.seats: Dict[int, dict] = {}
        self.by_track: Dict[int, int] = {}
        self._next = 0

    def resolve(self, track_id: int, frame: int, box) -> int:
        cx = (float(box[0]) + float(box[2])) / 2.0
        bottom = float(box[3])
        height = max(1.0, float(box[3] - box[1]))
        seat_id = self.by_track.get(track_id)
        if seat_id is None:
            seat_id = self._claim(frame, box, cx, bottom, height)
            self.by_track[track_id] = seat_id
        else:
            fresher = self._fresher_twin(frame, box, seat_id)
            if fresher is not None:
                # 同一个人被拆成两个跟踪号: 让后来的号替他把数接着数, 画面上只留一个框, 不再各数一段
                self.by_track[track_id] = fresher
                seat_id = fresher
        seat = self.seats[seat_id]
        if seat["observed"]:
            blend = 0.15
            for key, value in (("cx", cx), ("bottom", bottom), ("height", height)):
                seat[key] += blend * (value - seat[key])
        else:
            seat["cx"], seat["bottom"], seat["height"] = cx, bottom, height
        seat["box"] = [float(v) for v in box]
        seat["observed"] += 1
        seat["last_frame"] = frame
        seat["tracks"].add(track_id)
        return seat_id

    def _fresher_twin(self, frame: int, box, mine: int) -> Optional[int]:
        """自己的框压在别人站位上、而那个站位这一帧更新鲜时, 说明自己只是同一个人另立的号。

        只在"自己这一帧没被更新到"时才并过去: 两个真人同时被跟踪时各自的站位一样新,
        不会被并成一个, 挨着站的孩子不会被误并。
        """
        own = self.seats.get(mine)
        if not own or not own["box"]:
            return None
        picked, best_gap = None, frame - own["last_frame"]
        for seat_id, seat in self.seats.items():
            if seat_id == mine or not seat["box"]:
                continue
            if box_iou(box, seat["box"]) < self.cfg.slot_overlap_iou:
                continue
            gap = frame - seat["last_frame"]
            if gap < best_gap:
                picked, best_gap = seat_id, gap
        return picked

    def _claim(self, frame: int, box, cx: float, bottom: float, height: float) -> int:
        picked, best_score = None, float("inf")
        for seat_id, seat in self.seats.items():
            ref = max(1.0, min(height, seat["height"]))
            if max(height, seat["height"]) / ref > self.cfg.slot_height_ratio:
                continue
            score = math.hypot(cx - seat["cx"], 0.7 * (bottom - seat["bottom"])) / ref
            if score > self.cfg.slot_dist_ratio:
                continue
            # 位置被别的轨迹占着时, 只有框几乎重合才允许直接认领: 那说明是同一个人被拆成了两条轨迹,
            # 而不是旁边圈里的另一个人。以前一律拒绝, 结果同一个人被拆出好几个框各数各的。
            if frame - seat["last_frame"] < self.cfg.slot_claim_gap and box_iou(box, seat.get("box")) < self.cfg.slot_overlap_iou:
                continue
            if score < best_score:
                picked, best_score = seat_id, score
        if picked is not None:
            return picked
        self._next += 1
        self.seats[self._next] = {
            "cx": cx, "bottom": bottom, "height": height,
            "box": [float(v) for v in box],
            "first_frame": frame, "last_frame": frame, "observed": 0, "tracks": set(),
        }
        return self._next

    def fragments(self, seat_id: int) -> int:
        return len(self.seats.get(seat_id, {}).get("tracks", ()))

    def raw_ids(self, seat_id: int) -> Set[int]:
        return set(self.seats.get(seat_id, {}).get("tracks", ()))

    def prune(self, frame: int) -> None:
        for seat_id in [sid for sid, seat in self.seats.items() if frame - seat["last_frame"] > self.cfg.slot_keep_frames]:
            self.seats.pop(seat_id, None)
            for track_id, mapped in list(self.by_track.items()):
                if mapped == seat_id:
                    self.by_track.pop(track_id, None)


class OnlineCounter:
    """管理画面里所有人的在线计数器。"""

    def __init__(self, cfg: Config, fps: float):
        self.cfg = cfg
        self.fps = fps
        self.tracks: Dict[int, TrackCounter] = {}
        self.log: Deque[dict] = Deque(maxlen=120)
        self.seats = SeatBook(cfg) if cfg.slot_identity else None
        # 站位 -> 离场时寄存的累计量。计数器被淘汰后总次数会倒退, 出去喘口气回来的人还得从 0 数起,
        # 这里把"他确实跳过的次数"留在站位上, 下一个人站在同一个圈里才接得走。
        self.bank: "OrderedDict[int, dict]" = OrderedDict()

    def get(self, track_id: int) -> TrackCounter:
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackCounter(track_id, self.cfg, self.fps)
        return self.tracks[track_id]

    def update(self, frame: int, now: float, items) -> List[dict]:
        """items: [(track_id, box, keypoints, conf)]"""
        emitted: List[dict] = []
        claimed: Dict[int, Tuple] = {}
        for track_id, box, keypoints, conf in items:
            key = self.seats.resolve(track_id, frame, box) if self.seats else track_id
            previous = claimed.get(key)
            if previous is None or conf > previous[3]:
                claimed[key] = (track_id, box, keypoints, conf)
        for key, (track_id, box, keypoints, conf) in claimed.items():
            fresh = key not in self.tracks
            counter = self.get(key)
            if fresh and self.cfg.bank_counts:
                carry = self._take_bank(frame, box)
                if carry:
                    counter.carry_forward(carry)
            counter.observe(frame, box, keypoints, conf)
            counter.box = [round(float(v), 1) for v in box]
            if self.seats:
                counter.fragments = self.seats.fragments(key)
                counter.raw_ids = self.seats.raw_ids(key)
            self._accumulate_sync(counter, keypoints)
            emitted.extend(counter.poll(now))
        for record in emitted:
            self.log.append(record)
        expire = max(self.cfg.max_age * 4, self.cfg.slot_keep_frames if self.seats else 0)
        stale = [tid for tid, c in self.tracks.items() if c.last_frame is not None and frame - c.last_frame > expire]
        for tid in stale:
            counter = self.tracks.pop(tid)
            if self.cfg.bank_counts:
                self._retire_to_bank(tid, counter, frame)
        self._drop_stale_bank(frame)
        if self.seats:
            self.seats.prune(frame)
        return emitted

    # ---- 离场寄存 ----
    def _retire_to_bank(self, key: int, counter: TrackCounter, frame: int) -> None:
        """被淘汰的计数器里, 只有"判定确实在跳绳"的人才值得把次数留在站位上。"""
        if not counter.box:
            return
        if counter.count + counter.filled < self.cfg.min_count_provisional:
            return
        verdict, _ = judge_detail(counter.stats or {}, self.cfg)
        if verdict == OUT:
            return
        box = counter.box
        self.bank.pop(key, None)
        self.bank[key] = {
            "cx": (float(box[0]) + float(box[2])) / 2.0,
            "bottom": float(box[3]),
            "height": max(1.0, float(box[3]) - float(box[1])),
            "count": counter.count, "filled": counter.filled,
            "recovered": counter.recovered, "rope": counter.rope,
            "last_jump_time": counter.last_jump_time, "frame": frame,
        }
        while len(self.bank) > max(1, int(self.cfg.bank_max)):
            self.bank.popitem(last=False)

    def _take_bank(self, frame: int, box) -> Optional[dict]:
        """新开的计数器正站在一个寄存过的圈里: 把上一次留下的次数接走。

        认领条件与站位本身一致(位置距离/框高倍数), 并且要求那个位置当下没有别的人在跳,
        否则会把寄存的次数塞给一个正站在旁边的人。
        """
        keep_frames = int(self.cfg.bank_keep_sec * self.fps)
        cx = (float(box[0]) + float(box[2])) / 2.0
        bottom = float(box[3])
        height = max(1.0, float(box[3]) - float(box[1]))
        picked, best = None, float("inf")
        busy = self._position_busy(frame, box)
        for key, entry in self.bank.items():
            if frame - entry["frame"] > keep_frames:
                continue
            ref = max(1.0, min(height, entry["height"]))
            if max(height, entry["height"]) / ref > self.cfg.slot_height_ratio:
                continue
            if busy:
                continue
            score = math.hypot(cx - entry["cx"], 0.7 * (bottom - entry["bottom"])) / ref
            if score > self.cfg.slot_dist_ratio or score >= best:
                continue
            picked, best = key, score
        return self.bank.pop(picked) if picked is not None else None

    def _position_busy(self, frame: int, box) -> bool:
        for other in self.tracks.values():
            if other.last_frame is None or frame - other.last_frame > max(2, int(self.fps)):
                continue
            if other.box and box_iou(other.box, box) >= self.cfg.slot_overlap_iou:
                return True
        return False

    def _drop_stale_bank(self, frame: int) -> None:
        keep_frames = int(self.cfg.bank_keep_sec * self.fps)
        for key in [k for k, entry in self.bank.items() if frame - entry["frame"] > keep_frames]:
            self.bank.pop(key, None)

    def _accumulate_sync(self, counter: TrackCounter, keypoints) -> None:
        if keypoints is None:
            return
        arr = np.asarray(keypoints, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 17:
            return
        left, right = arr[15], arr[16]
        if left[2] < self.cfg.kp_conf or right[2] < self.cfg.kp_conf:
            return
        counter.sync_n += 1
        counter.sync_a += float(left[1])
        counter.sync_b += float(right[1])
        counter.sync_aa += float(left[1]) ** 2
        counter.sync_bb += float(right[1]) ** 2
        counter.sync_ab += float(left[1]) * float(right[1])
        if counter.sync_n > 600:  # 滑动衰减, 让判定跟上最新状态
            for attr in ("sync_n", "sync_a", "sync_b", "sync_aa", "sync_bb", "sync_ab"):
                setattr(counter, attr, getattr(counter, attr) * 0.5)

    def snapshot(self, frame: Optional[int] = None) -> List[dict]:
        return [self.tracks[tid].snapshot(frame) for tid in sorted(self.tracks)]

    def reset(self) -> None:
        self.tracks.clear()
        self.log.clear()
        self.bank.clear()
        if self.seats:
            self.seats = SeatBook(self.cfg)
