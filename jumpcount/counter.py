"""把一条轨迹的垂直位移序列变成"跳了几个"以及"为什么可能数错"。"""

from __future__ import annotations

import bisect
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import signal as sg
from .config import AUTO_ORDER, COCO_KEYPOINTS, Config
from .types import Detection, JumpEvent, TrackResult


def feature_y(det: Detection, source: str, cfg: Optional[Config] = None) -> Optional[float]:
    """取出该帧用于计数的 y 坐标(图像坐标, 向下为正)。"""
    if source.startswith("box"):
        x1, y1, x2, y2 = det.box
        return (y1 + y2) / 2.0 if source == "box_center" else float(y2)
    if det.keypoints is None:
        return None
    idx = COCO_KEYPOINTS.get(source)
    if not idx:
        return None
    kps = np.asarray(det.keypoints, dtype=np.float64)
    if kps.size == 0 or kps.shape[0] < max(idx) + 1:
        return None
    picked = [kps[i] for i in idx if kps.shape[0] > i and kps[i, 2] >= 0 and kps[i, 0] >= 0]
    if not picked:
        return None
    confs = np.array([min(1.0, max(0.0, float(k[2]))) for k in picked])
    good = confs >= 0.0
    if not good.any():
        return None
    pts = np.array([[float(k[0]), float(k[1])] for k, g in zip(picked, good) if g])
    if source == "ankle" and pts.shape[0] > 1 and \
            (cfg.ankle_combine if cfg is not None else "high") == "high":
        # 取抬得高的一只脚: 左右脚交替抬的跳法里, 两只脚取均值会把一半起跳压到阈值以下
        # (实测人群素材里有人只数到实际的一半, 已计峰间隔正好是节奏周期的两倍)
        return float(pts[:, 1].min())
    return float(pts[:, 1].mean())


def candidate_values(kps, box, cfg: Config) -> Dict[str, float]:
    """一帧里各候选信号取到的 y 值(取不到的不写, 之后按覆盖率择优)。"""
    out: Dict[str, float] = {}
    det = Detection(frame=0, time=0.0, box=tuple(float(v) for v in box), conf=1.0, keypoints=kps)
    for source in ("ankle", "hip", "head", "box_center", "box_bottom"):
        value = feature_y(det, source, cfg)
        if value is not None:
            out[source] = value
    return out


def absolute_floor(median_h: float, cfg: Config) -> float:
    """像素级振幅下限: 远处的人按固定像素卡下限会整片埋掉真起跳。

    16 人群像里中位框高只有 172px, 真实离地在踝部只留下 1~2px, 而 min_amplitude=2px
    是给近景定的, 直接套到远景上就砍掉了半数起跳。所以按框高等比缩小这条下限,
    缩到 amplitude_floor_min 封底, 再与"占身高比例"那条取大(它本来就随远近自适应)。
    """
    floor = cfg.min_amplitude
    if cfg.amplitude_ref_height > 0 and median_h > 0:
        scale = min(1.0, max(cfg.amplitude_floor_min, median_h / cfg.amplitude_ref_height))
        floor *= scale
    return max(floor, cfg.amplitude_ratio * median_h)


def amplitude_threshold(noise: float, peak_level: float, median_h: float, cfg: Config) -> float:
    """计数用的振幅下限: 绝对像素/框高比例/噪声/峰值比例取大, 但不超过信号自身幅值的 45%。"""
    relief = max(0.05, cfg.floor_relief)
    floor = absolute_floor(median_h, cfg) * relief
    relative_floor = cfg.amplitude_ratio * median_h * relief
    thr = max(floor, cfg.snr_k * noise, cfg.peak_frac * peak_level)
    return min(thr, max(relative_floor, 0.45 * peak_level))


def channel_quality(frames: np.ndarray, up: np.ndarray, median_h: float,
                    cfg: Config, fps: float) -> Tuple[float, int, float]:
    """这条通道数得清不清楚: (节律强度, 过阈值的峰数, 观测覆盖率)。判据与计数同一套, 免得选完源又数不出东西。"""
    keep = np.isfinite(up)
    if keep.sum() < 20:
        return 0.0, 0, 0.0
    f, v = frames[keep], up[keep]
    axis, dense, _ = sg.fill_gaps(f, v, cfg.max_gap)
    observed = np.zeros(axis.size, dtype=bool)
    observed[f - int(axis[0])] = True
    baseline_win = cfg.baseline_win if dense.size >= cfg.baseline_win else _odd(dense.size // 2)
    det = sg.detrend(dense, cfg.smooth_win, baseline_win)
    noise = sg.hf_sigma_masked(det, observed)
    peak_level = float(np.percentile(det, 95)) if det.size else 0.0
    thr = amplitude_threshold(noise, peak_level, median_h, cfg)
    hard_min = max(2, int(round(fps / cfg.max_rate_hz)))
    coarse = sg.find_peaks(det, hard_min, thr, 0.8 * thr)
    ac_period, strength = sg.autocorr_period(det, fps, cfg.min_rate_hz, cfg.max_rate_hz)
    _, period_src = _choose_period(coarse, ac_period, fps, cfg)
    if period_src == "fallback":
        strength = 0.0
    return round(float(strength), 3), int(len(coarse)), round(float(observed.mean()), 3)


def source_scores(samples: List[Tuple], cfg: Config, fps: float,
                  channels: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Tuple[float, int, float]]:
    """踝/髋/头三条通道各自的"数得清"程度。"""
    usable: Dict[str, int] = {}
    for sample in samples:
        kps = sample[4] if len(sample) > 4 else None
        for source in candidate_values(kps, sample[2], cfg):
            usable[source] = usable.get(source, 0) + 1
    frames = np.array([int(s[0]) for s in samples], dtype=np.int64)
    heights = np.array([float(s[2][3]) - float(s[2][1]) for s in samples], dtype=np.float64)
    median_h = float(np.median(heights)) if heights.size else 0.0
    ch = channels if channels is not None else channel_cache(samples, cfg)
    out: Dict[str, Tuple[float, int, float]] = {}
    for source in ("ankle", "hip", "head"):
        if usable.get(source, 0) < max(1, int(0.5 * len(samples))) or source not in ch:
            continue
        values = ch[source]
        if not np.isfinite(values).any():
            continue
        out[source] = channel_quality(frames, -np.where(np.isfinite(values), values, np.nan), median_h, cfg, fps)
    return out


def pick_source(samples: List[Tuple], cfg: Config, fps: Optional[float] = None,
                channels: Optional[Dict[str, np.ndarray]] = None,
                scores_out: Optional[Dict[str, Tuple[float, int, float]]] = None) -> str:
    """先按覆盖率筛掉取不到点的部位, 再在剩下的里面挑节律最清楚的。

    一优先用脚踝是错的: 人离得远或小腿抬得低时脚踝位移只有一两个像素, 一半真起跳落在阈值以下
    (实测同一个人脚踝只数到 29 个, 换成髋能数到 45 个)。但髋/头会被走路摆动干扰,
    所以只在节律强度明确胜出时才换, 覆盖率门槛保持不变。
    """
    if cfg.signal_source != "auto":
        return cfg.signal_source
    usable: Dict[str, int] = {}
    for sample in samples:
        kps = sample[4] if len(sample) > 4 else None
        box = sample[2]
        for source in candidate_values(kps, box, cfg):
            usable[source] = usable.get(source, 0) + 1
    fallback = "box_bottom"
    for source in AUTO_ORDER:
        if usable.get(source, 0) >= max(1, int(0.5 * len(samples))):
            fallback = source
            break
    if cfg.channel_select != "rhythm" or fps is None:
        return fallback
    scores = source_scores(samples, cfg, fps, channels)
    if scores_out is not None:
        scores_out.update(scores)
    if not scores:
        return fallback
    best = max(scores, key=lambda s: (scores[s][0], scores[s][1]))
    base = scores.get(fallback, (0.0, 0, 0.0))
    return best if scores[best][0] >= base[0] + cfg.channel_switch_margin else fallback


# 某个身体部位这一帧取不到时按这条链找替身, 并按"两者同时可见时的中位差"把替身平移回原通道。
# 直接换成框底边会在信号上留下一个台阶, 而台阶本身会被当成一次起跳, 反过来也会把真实起跳埋掉。
SUBSTITUTE_CHAIN = {
    "ankle": ("hip", "head", "box_bottom"),
    "hip": ("head", "ankle", "box_bottom"),
    "head": ("hip", "ankle", "box_bottom"),
    "box_center": ("head", "hip", "box_bottom"),
    "box_bottom": ("head", "hip", "box_center"),
}


def channel_y(samples: List[Tuple], source: str, cfg: Optional[Config] = None) -> np.ndarray:
    """单一身体部位的逐帧垂直位移, 取不到的帧留 NaN。"""
    values = np.full(len(samples), np.nan)
    for i, sample in enumerate(samples):
        kps = sample[4] if len(sample) > 4 else None
        det = Detection(frame=0, time=0.0, box=tuple(float(v) for v in sample[2]), conf=1.0, keypoints=kps)
        value = feature_y(det, source, cfg)
        if value is not None:
            values[i] = value
    return values


def extract_y_detail(samples: List[Tuple], source: str, cfg: Config,
                     channels: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, int]:
    """计数用的垂直位移序列: 缺帧用其它部位平移补齐, 返回 (序列, 补齐帧数)。"""
    primary = _channel_of(samples, source, channels, cfg)
    values = primary.copy()
    substituted = 0
    for alternate in SUBSTITUTE_CHAIN.get(source, ("box_bottom",)):
        missing = ~np.isfinite(values)
        if not missing.any():
            break
        other = _channel_of(samples, alternate, channels, cfg)
        both = np.isfinite(primary) & np.isfinite(other)
        if both.sum() < 5:
            continue
        shift = float(np.median(primary[both] - other[both]))
        fill = missing & np.isfinite(other)
        values[fill] = other[fill] + shift
        substituted += int(fill.sum())
    return values, substituted


def extract_y(samples: List[Tuple], source: str, cfg: Config) -> np.ndarray:
    return extract_y_detail(samples, source, cfg)[0]


def _channel_of(samples: List[Tuple], source: str, channels: Optional[Dict[str, np.ndarray]],
                cfg: Optional[Config] = None) -> np.ndarray:
    if channels is not None and source in channels:
        return channels[source]
    return channel_y(samples, source, cfg)


ALL_CHANNELS = ("ankle", "hip", "head", "box_bottom", "box_center")


def channel_cache(samples: List[Tuple], cfg: Optional[Config] = None) -> Dict[str, np.ndarray]:
    """一条轨迹的各位部位通道一次算完: 计数、幅度对照都要用, 反复重算最费时间。"""
    return {source: channel_y(samples, source, cfg) for source in ALL_CHANNELS}


def select_source(dets: List[Detection], cfg: Config) -> str:
    """为一条轨迹固定一种信号来源, 避免中途换特征造成阶跃。"""
    if cfg.signal_source != "auto":
        return cfg.signal_source
    usable: Dict[str, int] = {}
    for det in dets:
        for src in AUTO_ORDER:
            if feature_y(det, src, cfg) is not None:
                usable[src] = usable.get(src, 0) + 1
    for src in AUTO_ORDER:
        if usable.get(src, 0) >= max(1, int(0.4 * len(dets))):
            return src
    return "box_bottom"


def ankle_sync(samples: List[Tuple], cfg: Config) -> Optional[float]:
    """左右脚踝垂直位移的相关性: 跳绳时双脚一起起跳(正相关), 走路时交替落地(负相关)。"""
    left: List[float] = []
    right: List[float] = []
    for sample in samples:
        kps = sample[4] if len(sample) > 4 else None
        if kps is None:
            continue
        arr = np.asarray(kps, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 17:
            continue
        lft, rgt = arr[15], arr[16]
        if lft[2] >= cfg.kp_conf and rgt[2] >= cfg.kp_conf and lft[0] >= 0 and rgt[0] >= 0:
            left.append(float(lft[1]))
            right.append(float(rgt[1]))
    if len(left) < 20:
        return None
    a = np.array(left)
    b = np.array(right)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return None
    return float(np.corrcoef(a, b)[0, 1])


IN, PROVISIONAL, OUT = "in", "provisional", "out"


def rhythm_is_clear(stats: dict, cfg: Config) -> bool:
    """节奏是否清楚: 节律强度达标且周期是真测出来的(不是兜底值)。

    这是"能不能原谅幅度小/遮挡多"的唯一凭据: 真跳绳的人节奏一定清楚,
    路人抖出来的峰对不上任何固定周期。
    """
    if stats.get("period_source") == "fallback":
        return False
    strength = stats.get("rhythm_strength")
    return strength is not None and float(strength) >= cfg.keep_rhythm_strength


def participation(stats: dict, cfg: Config) -> Tuple[str, str]:
    """三态参与判定: in=计入, provisional=低置信仍计入, out=剔除。

    以前的判定是一票否决, 幅度小或遮挡多的人整条轨迹直接归零, 人还在跳但次数全丢。
    现在把"确凿不是跳绳"(走动/双脚交替/根本没那么高)和"证据不足"分开:
    后者按低置信计入总量, 并在审计和页面上标出来。
    """
    if stats.get("reject"):
        return OUT, "观测点太少"
    if stats.get("duration_sec", 0) < cfg.min_track_sec:
        return OUT, f"只出现 {stats.get('duration_sec', 0):.1f}s"
    clear = rhythm_is_clear(stats, cfg)
    excursion = stats.get("excursion_ratio", 0.0)
    if excursion > cfg.max_excursion_ratio:
        sync = stats.get("ankle_sync")
        together = sync is not None and sync >= cfg.min_ankle_strong
        if excursion > cfg.max_excursion_hard or not (clear and together):
            return OUT, f"水平移动了 {excursion:.1f} 倍身高, 是在走动不是原地跳"
        return PROVISIONAL, f"水平移动 {excursion:.1f} 倍身高但双脚同步起跳, 低置信计入"
    lift = max(float(stats.get("lift_ratio", 0.0)), float(stats.get("lift_ratio_best", 0.0)))
    if lift < cfg.min_lift_ratio:
        if not clear:
            return OUT, f"起跳幅度仅为身高的 {lift * 100:.1f}%, 没有明显离地"
        return PROVISIONAL, f"起跳幅度仅为身高的 {lift * 100:.1f}%, 节奏清楚, 低置信计入"
    sync = stats.get("ankle_sync")
    if sync is not None and cfg.require_pose_sync and sync < cfg.min_ankle_sync:
        return OUT, f"左右脚位移相关性 {sync:.2f} 为负, 是交替迈步的走路动作"
    height_cv = stats.get("height_cv")
    if height_cv is not None and height_cv > cfg.max_height_cv:
        return OUT, f"框高波动 {height_cv:.2f} 过大, 这条轨迹在不同人之间跳"
    visibility = stats.get("visibility", 0)
    if visibility < cfg.min_track_visibility:
        if visibility < cfg.min_track_visibility_floor or not clear:
            return OUT, f"可见率 {visibility:.2f} 过低"
        return PROVISIONAL, f"可见率 {visibility:.2f} 偏低但节奏清楚, 低置信计入"
    # 判断"是否在连续跳绳"看总证据(实测 + 按节奏补齐), 没有计数的场景(如逐条复盘)不启用这条判据
    count = stats.get("total_count", stats.get("count"))
    if count is not None and count < cfg.min_count_active:
        if count < cfg.min_count_provisional or not clear:
            return OUT, f"只数到 {count} 个, 还不够确认在连续跳绳"
        return PROVISIONAL, f"只数到 {count} 个但节奏清楚, 低置信计入"
    return IN, ""


def classify(stats: dict, cfg: Config) -> Tuple[bool, str]:
    """判断这条轨迹是不是真的在跳绳, 而不是路过的人。"""
    verdict, reason = participation(stats, cfg)
    return verdict != OUT, reason


def best_lift_ratio(samples: List[Tuple], median_h: float, cfg: Config,
                    channels: Optional[Dict[str, np.ndarray]] = None) -> float:
    """各身体部位里最大的离地幅度/身高: 脚踝抬得少, 但头髋能反映真实离地高度。"""
    if median_h <= 0:
        return 0.0
    frames = np.array([int(s[0]) for s in samples], dtype=np.int64)
    best = 0.0
    for source in ("ankle", "hip", "head", "box_bottom"):
        y = extract_y_detail(samples, source, cfg, channels)[0]
        keep = np.isfinite(y)
        if keep.sum() < 12:
            continue
        axis, dense, _ = sg.fill_gaps(frames[keep], -y[keep], cfg.max_gap)
        if dense.size < 12:
            continue
        window = cfg.baseline_win if dense.size >= cfg.baseline_win else _odd(dense.size // 2)
        detrended = sg.detrend(dense, cfg.smooth_win, window)
        best = max(best, float(np.percentile(detrended, 95)) / median_h)
    return round(best, 4)


def _odd(value: int, floor: int = 5) -> int:
    value = max(floor, int(value))
    return value if value % 2 else value - 1


def robust_spread(values, low: float = 5.0, high: float = 95.0) -> float:
    """横向活动范围: 用 5~95 分位而不是极差, 免得个别错位框就把原地跳的人判成走路。"""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 6:
        return 0.0
    return float(np.percentile(arr, high) - np.percentile(arr, low))


def drift_spread(values, fps: float, window_sec: float = 3.0, frames=None,
                 low: float = 10.0, high: float = 90.0) -> float:
    """逐时间窗口量横向活动范围, 取各窗中位数。

    整段分位数会把一次 ID 漂移放大成"走了两三个身位": 只有一帧被认成隔壁的人,
    全程范围就炸了, 原地跳的真人被当路人剔除(少数的主因之一)。
    真在走动的人每个窗口都在移动, 窗口中位数照样大; 抖动只影响个别窗口。
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    axis = np.asarray(frames, dtype=np.float64) if frames is not None else np.arange(arr.size, dtype=np.float64)
    if axis.shape != arr.shape:
        axis = np.arange(arr.size, dtype=np.float64)
    keep = np.isfinite(arr) & np.isfinite(axis)
    if keep.sum() < 6:
        return 0.0
    arr, axis = arr[keep], axis[keep]
    order = np.argsort(axis, kind="stable")
    arr, axis = arr[order], axis[order]
    win = max(3.0, window_sec * max(fps, 1e-6))
    bins = np.floor((axis - axis[0]) / win).astype(np.int64)
    spreads = [float(np.percentile(arr[bins == b], high) - np.percentile(arr[bins == b], low))
               for b in np.unique(bins) if int((bins == b).sum()) >= 6]
    if not spreads:
        return float(np.percentile(arr, high) - np.percentile(arr, low))
    return float(np.median(spreads))


def _choose_period(coarse: List[int], ac_period: Optional[float], fps: float, cfg: Config) -> Tuple[float, str]:
    lo = fps / cfg.max_rate_hz
    hi = fps / cfg.min_rate_hz
    if len(coarse) >= 3:
        spacing = float(np.median(np.diff(coarse)))
        if lo <= spacing <= hi:
            return spacing, "峰间隔中位数"
    if ac_period:
        spacing = ac_period * fps
        if lo <= spacing <= hi:
            return spacing, "自相关"
    return fps / ((cfg.min_rate_hz + cfg.max_rate_hz) / 2.0), "fallback"


def channel_on_axis(values_by_sample: np.ndarray, frames_all: np.ndarray, axis: np.ndarray) -> Optional[np.ndarray]:
    """把按"样本序号"存的身体部位通道搬到稠密帧轴上(缺帧插值)。

    投票要用别的身体部位通道, 但它们的缺帧模式和主通道不一样,
    不搬到同一条帧轴上就没法比同一时刻的鼓包。
    """
    keep = np.isfinite(values_by_sample)
    idx = frames_all - int(axis[0])
    ok = (idx >= 0) & (idx < axis.size) & keep
    if int(ok.sum()) < 6:
        return None
    dense = np.full(axis.size, np.nan)
    dense[idx[ok]] = values_by_sample[ok]
    valid = ~np.isnan(dense)
    dense[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), dense[valid])
    return dense


def refine_peak(values: np.ndarray, center: int, radius: int) -> Tuple[int, float]:
    """把包络峰对回原始位移上的高点: 出数时刻与幅度都按原始位移报。"""
    lo, hi = max(0, center - radius), min(values.size, center + radius + 1)
    if hi - lo <= 0:
        return int(center), 0.0
    offset = int(np.argmax(values[lo:hi]))
    return lo + offset, float(values[lo + offset])


def recover_peaks(detrended: np.ndarray, axis: np.ndarray, frames_all: np.ndarray, base: List[int],
                  observed: np.ndarray, min_distance: int, period_frames: float, amp_thr: float,
                  channels: Dict[str, np.ndarray], primary: str, cfg: Config, fps: float
                  ) -> Tuple[List[int], set]:
    """能量包络回捞: 候选来自包络, 是否算起跳由原始位移和其它通道同刻鼓包共同签字。

    位移阈值这边已经没有余量了(远景人真实起跳在踝部只留 1~2px, 阈值再松先进的是噪声),
    所以换一条判据: ① 只在"按节奏本该有起跳"的空档与首尾段里找;
    ② 那段时间人确实在画面里; ③ 原始位移上要能看出一个鼓包;
    ④ 至少 envelope_lead_votes 条身体部位通道在同一时刻
    也各有自己的包络鼓包 —— 真起跳时踝/髋/头一起动, 单个通道的抖动不会。
    """
    win = max(3, int(round(cfg.envelope_win_ratio * period_frames)))
    low_hz, high_hz = cfg.envelope_band_hz
    env = sg.energy_envelope(detrended, fps, low_hz, high_hz, win)
    env_p95 = float(np.percentile(env, 95)) if env.size else 0.0
    if env_p95 <= 0:
        return base, set()
    env_thr = cfg.envelope_frac * env_p95

    leads: List[Tuple[np.ndarray, float]] = []
    window = cfg.baseline_win if detrended.size >= cfg.baseline_win else _odd(detrended.size // 2)
    for name in ("ankle", "hip", "head"):
        if name == primary or name not in channels:
            continue
        series = -np.where(np.isfinite(channels[name]), channels[name], np.nan)
        moved = channel_on_axis(series, frames_all, axis)
        if moved is None:
            continue
        other = sg.energy_envelope(sg.detrend(moved, cfg.smooth_win, window), fps, low_hz, high_hz, win)
        p95 = float(np.percentile(other, 95)) if other.size else 0.0
        if p95 > 0:
            leads.append((other, cfg.envelope_lead_frac * p95))

    tol = max(1, int(round(cfg.envelope_lead_tol * period_frames)))

    def votes(pos: int, level: float) -> int:
        lo, hi = max(0, pos - tol), min(detrended.size, pos + tol + 1)
        # 候选是在哪一档捞出来的, 投票就按同一档算, 否则弱起跳刚出空档又被票回阈值以下
        total = 1 if float(env[lo:hi].max()) >= env_thr * level else 0
        for other, other_thr in leads:
            if hi <= other.size and float(other[lo:hi].max()) >= other_thr * level:
                total += 1
        return total

    cands = sg.find_peaks(env, min_distance, env_thr, 0.8 * env_thr)
    scored = [(int(p), 1.0) for p in cands]
    scored += sg.gap_candidates(env, cands, min_distance, env_thr,
                                cfg.search_back_ratio * period_frames, tuple(cfg.search_back_levels))
    radius = max(1, int(round(0.4 * period_frames)))
    accepted = list(base)
    recovered: set = set()
    gap_floor = cfg.recover_gap_ratio * period_frames
    # 空档中点即"本该有起跳"的时刻: 拿实测峰间距的中位数当节拍格, 单侧空档按一格算
    spacing = np.diff(sorted(int(p) for p in base))
    grid = float(np.median(spacing)) if spacing.size >= 2 else float(period_frames)
    tol_center = max(0.0, float(cfg.recover_center_tol))
    for cand, level in scored:
        if any(abs(cand - known) < min_distance for known in accepted):
            continue
        pos, value = refine_peak(detrended, int(cand), radius)
        if pos != int(cand) and any(abs(pos - known) < min_distance for known in accepted):
            continue     # 对回原位移后落到了实测峰身上: 同一次起跳, 不是新的一次
        if observed.size == detrended.size:
            lo, hi = max(0, pos - min_distance), min(observed.size, pos + min_distance + 1)
            if float(observed[lo:hi].mean()) < cfg.recover_min_coverage:
                continue      # 人那段时间根本不在画面里, 不替他猜
        if value >= amp_thr:
            accepted.append(pos)                       # 包络和原始位移都同意, 按正常起跳算
        elif value >= cfg.recover_raw_ratio * amp_thr and votes(pos, level) >= cfg.envelope_lead_votes:
            at = bisect.bisect_left(accepted, pos)
            prev = accepted[at - 1] if at > 0 else None
            nxt = accepted[at] if at < len(accepted) else None
            # 落点还得像个节拍: 与左右邻居都拉开一个周期, 且踩在格子上
            if prev is not None and pos - prev < gap_floor:
                continue
            if nxt is not None and nxt - pos < gap_floor:
                continue
            if prev is not None and nxt is not None:
                if abs(pos - (prev + nxt) / 2.0) > tol_center * (nxt - prev):
                    continue      # 不在正中: 是抖动不是被埋的那一拍
            else:
                anchor = prev if prev is not None else nxt
                if anchor is None or abs(abs(pos - anchor) - grid) > tol_center * grid:
                    continue      # 单侧空档要正好差一个节拍
            bisect.insort(accepted, pos)
            recovered.add(pos)                         # 幅度不够但多通道投票通过, 单独记一笔
    kept = sg.nms_by_height(detrended, accepted, min_distance)
    return kept, recovered & set(kept)


def count_track(
    track_id: int,
    samples: List[Tuple],
    cfg: Config,
    fps: float,
    source: Optional[str] = None,
) -> TrackResult:
    """samples: [(frame, time, box, feature_y, keypoints)]，feature_y 可为 None 交由选源逻辑补齐。"""
    frames = np.array([int(s[0]) for s in samples], dtype=np.int64)
    times = np.array([float(s[1]) for s in samples], dtype=np.float64)
    frames_all = frames.copy()   # 投票通道按样本序号取数, 搬到帧轴时要用它对齐
    heights = np.array([float(s[2][3]) - float(s[2][1]) for s in samples], dtype=np.float64)
    channels = channel_cache(samples, cfg)
    ch_scores: Dict[str, Tuple[float, int, float]] = {}
    resolved = source if source and source != "auto" else pick_source(samples, cfg, fps, channels, ch_scores)
    y_series, substituted = extract_y_detail(samples, resolved, cfg, channels=channels)
    usable = int(np.isfinite(channels[resolved]).sum()) if resolved in channels else 0
    if usable < max(6, int(0.4 * len(samples))) and resolved != "box_bottom":
        resolved = "box_bottom"
        y_series, substituted = extract_y_detail(samples, resolved, cfg, channels=channels)
        usable = int(np.isfinite(channels[resolved]).sum())
    raw = -np.where(np.isfinite(y_series), y_series, np.nan)  # 向上为正
    median_h = float(np.median(heights)) if heights.size else 0.0
    cxs = np.array([float(s[2][0] + s[2][2]) / 2.0 for s in samples], dtype=np.float64)
    min_frames = max(6, int(round(0.6 * fps)))

    result = TrackResult(
        track_id=track_id,
        first_frame=int(frames[0]),
        last_frame=int(frames[-1]),
        count=0,
        count_rope=0,
        signal_source=resolved,
        median_height=round(median_h, 2),
    )
    result.stats["sample_count"] = int(frames.size)
    result.stats["signal_coverage"] = round(usable / max(1, len(samples)), 3)
    result.stats["substituted_frames"] = int(substituted)
    if ch_scores:
        result.stats["channel_scores"] = {k: {"rhythm": v[0], "peaks": v[1], "coverage": v[2]}
                                          for k, v in ch_scores.items()}
    result.stats["height_cv"] = round(float(np.std(heights) / max(1.0, median_h)), 3)
    result.stats["bottom_spread_ratio"] = round(
        robust_spread([float(s[2][3]) for s in samples]) / max(1.0, median_h), 3)
    result.stats["duration_sec"] = round((int(frames[-1]) - int(frames[0]) + 1) / fps, 2)
    result.stats["visibility"] = result.visibility
    result.stats["excursion_px"] = round(drift_spread(cxs, fps, cfg.drift_window_sec, frames=frames), 1)
    if frames.size < min_frames or times[-1] - times[0] < 1.0:
        result.stats["reject"] = "too_short"
        return result

    keep = np.isfinite(raw)
    frames, times, raw = frames[keep], times[keep], raw[keep]
    if frames.size < min_frames:
        result.stats["reject"] = "too_short"
        return result
    axis, dense, long_holes = sg.fill_gaps(frames, raw, cfg.max_gap)
    times_dense = np.interp(axis, frames, times)  # 插值后帧轴比观测点多, 时间轴需要同步补齐
    observed = np.zeros(axis.size, dtype=bool)
    observed[frames - int(axis[0])] = True
    span = max(1, axis[-1] - axis[0] + 1)
    result.visibility = round(frames.size / span, 4)
    baseline_win = cfg.baseline_win if dense.size >= cfg.baseline_win else _odd(dense.size // 2)
    detrended = sg.detrend(dense, cfg.smooth_win, baseline_win)

    noise = sg.hf_sigma_masked(detrended, observed)
    mad = sg.mad_sigma(detrended)
    peak_level = float(np.percentile(detrended, 95)) if detrended.size else 0.0
    result.snr = round(min(99.0, peak_level / noise), 2) if noise > 1e-6 else float(peak_level > 0) * 99.0

    # 阈值不能超过信号本身, 否则密集人群里的小幅起跳会被整段抹掉;
    # 下压时只压到"框高比例下限", 不压回绝对像素下限(细节见 amplitude_threshold)。
    amp_thr = amplitude_threshold(noise, peak_level, median_h, cfg)

    # 两遍节奏估计: 先按物理上限粗取峰, 再用实测峰间隔的中位数当作这个人自己的节奏,
    # 避免自相关把噪声尖峰当成周期(会把一次起跳数成两三次)
    hard_min = max(2, int(round(fps / cfg.max_rate_hz)))
    coarse = sg.find_peaks(detrended, hard_min, amp_thr, 0.8 * amp_thr)
    ac_period, strength = sg.autocorr_period(detrended, fps, cfg.min_rate_hz, cfg.max_rate_hz)
    period_frames, period_src = _choose_period(coarse, ac_period, fps, cfg)
    result.stats["rhythm_strength"] = round(float(strength), 3)
    result.stats["period_source"] = period_src
    result.stats["period_estimated"] = period_src != "fallback"
    result.stats["coarse_peaks"] = len(coarse)
    period = period_frames / fps
    result.period_sec = round(float(period), 4)
    result.rate_hz = round(fps / period_frames, 3) if period_frames > 0 else 0.0

    min_distance = max(hard_min, int(round(cfg.refractory_ratio * period_frames)))
    idx = sg.find_peaks(detrended, min_distance, amp_thr, 0.8 * amp_thr)

    # 位移阈值已经贴到信号下限, 剩下的漏计靠"节奏该有而没数到"的空档 + 多通道同刻鼓包找回
    recovered: set = set()
    if cfg.envelope_detect and period_src != "fallback":
        idx, recovered = recover_peaks(detrended, axis, frames_all, idx, observed, min_distance,
                                       period_frames, amp_thr, channels, resolved, cfg, fps)
    result.stats["recovered"] = len(recovered)

    warmup = int(round(cfg.warmup_sec * fps))
    idx = [i for i in idx if axis[i] - axis[0] >= warmup]
    recovered &= set(idx)

    events: List[JumpEvent] = []
    bumps = 0
    prev: Optional[float] = None
    estimated_misses = 0
    filled = 0
    misses_unobserved = 0
    # 只有节奏是真测出来的、且人一直在画面里却没数到峰, 才按节奏补; 否则宁可少报也不虚报。
    rhythm_trusted = period_src != "fallback" and strength >= cfg.keep_rhythm_strength
    for i in idx:
        sub = sg.parabolic_peak(detrended, i)
        frame_pos = float(axis[0]) + sub
        t = float(np.interp(frame_pos, axis, times_dense))
        amp = float(detrended[i])
        kind = "jump"
        if prev is not None:
            gap = frame_pos - prev
            if gap < cfg.double_under_ratio * period_frames:
                kind = "bump"
                bumps += 1
            elif gap > cfg.gap_ratio * period_frames:
                expected = int(round(gap / period_frames))
                estimated_misses += max(0, expected - 1)
                lo = max(0, int(round(prev - axis[0])))
                coverage = float(observed[lo:i].mean()) if i > lo else 0.0
                if coverage < cfg.min_fill_coverage:
                    misses_unobserved += max(0, expected - 1)
                elif cfg.interpolate_gaps and expected >= 2 and rhythm_trusted:
                    for k in range(1, min(expected - 1, cfg.max_fill_per_gap) + 1):
                        fill_pos = prev + gap * k / expected
                        events.append(
                            JumpEvent(
                                frame=fill_pos,
                                time=float(np.interp(fill_pos, axis, times_dense)),
                                amplitude=round(amp_thr * 0.6, 3),
                                confidence=0.3,
                                kind="interpolated",
                            )
                        )
                        filled += 1
        events.append(JumpEvent(frame=round(frame_pos, 3), time=round(t, 4), amplitude=round(amp, 3),
                               confidence=round(min(1.0, amp / max(amp_thr, 1e-6)), 3), kind=kind,
                               recovered=bool(i in recovered)))
        prev = frame_pos

    # 实测与按节奏补齐分两笔记: 混在一起会让灵敏度滑杆失去意义(门限再严也能靠补数追平),
    # 复盘时也看不出哪些是真抓到的起跳。总量 = count + filled。
    direct = [e for e in events if e.kind == "jump"]
    filled_events = [e for e in events if e.kind == "interpolated"]
    result.events = events
    result.count = len(direct) if cfg.policy == "jump" else len(direct) + bumps
    result.count_rope = len(direct) + bumps + len(filled_events)
    result.stats["noise"] = round(noise, 3)
    result.stats["noise_mad"] = round(mad, 3)
    result.stats["amplitude_threshold"] = round(float(amp_thr), 3)
    result.stats["peak_level_p95"] = round(peak_level, 3)
    result.stats["bumps"] = bumps
    result.stats["estimated_misses"] = estimated_misses
    result.stats["filled"] = filled
    result.stats["misses_unobserved"] = misses_unobserved
    result.stats["long_holes"] = long_holes
    result.stats["visibility"] = result.visibility
    result.stats["lift_ratio"] = round(peak_level / median_h, 4) if median_h > 0 else 0.0
    result.stats["lift_ratio_best"] = best_lift_ratio(samples, median_h, cfg, channels)
    result.stats["excursion_ratio"] = round(result.stats["excursion_px"] / median_h, 3) if median_h > 0 else 0.0
    sync = ankle_sync(samples, cfg)
    result.stats["ankle_sync"] = round(sync, 3) if sync is not None else None
    result.stats["count"] = result.count
    expected = round(result.rate_hz * result.stats["duration_sec"], 1) if result.rate_hz else 0.0
    result.stats["expected_count"] = expected
    result.stats["total_count"] = result.count + int(result.stats.get("filled", 0))
    result.stats["rhythm_coverage"] = (round(result.stats["total_count"] / expected, 3)
                                       if expected > 0 else 0.0)
    verdict, reason = participation(result.stats, cfg)
    result.stats["verdict"] = verdict
    result.stats["participant"] = verdict != OUT
    if verdict == OUT:
        result.stats["excluded"] = reason
    return result
