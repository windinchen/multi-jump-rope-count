"""把一条轨迹的垂直位移序列变成"跳了几个"以及"为什么可能数错"。"""

from __future__ import annotations

import bisect
import math
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
    """计数用的振幅下限: 绝对像素/框高比例/噪声/峰值比例取大, 但不超过信号自身幅值的 45%。

    那条 45% 上限是为了让"整段都在跳"的人不被自己的峰值卡死, 但它会把阈值一路压到
    绝对像素下限之下: 站着不动的人信号只有 ±1px, 上限跟着落到 0.8px, 于是抖动全被当成起跳
    (复盘 179023471330 #11: 全身最大位移 5.7px 却数到 29 次)。
    threshold_cap_floor_frac 给上限再垫一层地板 —— 0 是历史行为, 1 表示上限不许低于绝对下限。
    """
    relief = max(0.05, cfg.floor_relief)
    floor = absolute_floor(median_h, cfg) * relief
    relative_floor = cfg.amplitude_ratio * median_h * relief
    thr = max(floor, cfg.snr_k * noise, cfg.peak_frac * peak_level)
    cap_floor = max(relative_floor, cfg.threshold_cap_floor_frac * floor)
    return min(thr, max(cap_floor, 0.45 * peak_level))


def gap_lift_evidence(detrended: np.ndarray, lo: int, hi: int, period_frames: float) -> float:
    """空档内部的离地证据有多高: 掐掉两侧起跳的尾巴, 只看中间有没有鼓包。

    不掐尾巴的话, 区间最大值永远是上一拍起跳自己身上沿, 判据恒真。
    中间空得太短(掐完不足三帧)就不拦: 那种空档最多补一个, 拦下来省的比亏的少。
    """
    skip = max(1, int(round(0.4 * max(1.0, period_frames))))
    inner = detrended[lo + skip: hi - skip]
    if inner.size < 3:
        return float("inf")
    return float(np.max(sg.median_filter(inner, 3)))


# 空档取证用的身体部位通道: 踝/髋/头/框底边。不含框中心 —— 它与框底边同源,
# 一起投票等于同一个观测算两票。
LIFT_EVIDENCE_CHANNELS = ("ankle", "hip", "head", "box_bottom")


def gap_evidence_from_denses(denses: Dict[str, np.ndarray], observed: np.ndarray, median_h: float,
                             cfg: Config, smooth_win: int, baseline_win: int,
                             size: int) -> Dict[str, Tuple[np.ndarray, float]]:
    """给定已对齐到帧轴的身体部位位移, 各自去基线并算出**各通道自己的**计数阈值。

    空档取证要横向比不同部位, 阈值必须按各通道自己的噪声与幅度算(与 channel_quality 同一口径),
    直接套主通道阈值会把幅度天然更小的通道一律判成"没离地"。
    滤波窗口跟住实际选用的那一档: 快跳用窄窗时取证也按窄窗, 否则被抹平的起跳在取证里同样找不到。
    """
    out: Dict[str, Tuple[np.ndarray, float]] = {}
    for name, dense in denses.items():
        if dense.size != size or dense.size < 12:
            continue
        det = sg.detrend(dense, smooth_win, baseline_win)
        noise = sg.hf_sigma_masked(det, observed)
        p95 = float(np.percentile(det, 95)) if det.size else 0.0
        thr = float(amplitude_threshold(noise, p95, median_h, cfg))
        if thr > 1e-6:
            out[name] = (det, thr)
    return out


def gap_evidence_signals(channels: Dict[str, np.ndarray], frames_all: np.ndarray, axis: np.ndarray,
                         observed: np.ndarray, median_h: float, cfg: Config,
                         smooth_win: int, baseline_win: int, primary: str) -> Dict[str, Tuple[np.ndarray, float]]:
    """把各身体部位通道搬到主通道那条稠密帧轴上, 再算取证用的去基线与阈值。"""
    denses: Dict[str, np.ndarray] = {}
    for name in LIFT_EVIDENCE_CHANNELS:
        if name == primary or name not in channels:
            continue
        series = -np.where(np.isfinite(channels[name]), channels[name], np.nan)
        dense = channel_on_axis(series, frames_all, axis)
        if dense is not None:
            denses[name] = dense
    return gap_evidence_from_denses(denses, observed, median_h, cfg, smooth_win, baseline_win, axis.size)


def gap_evidence_votes(detrended: np.ndarray, amp_thr: float, extras: Dict[str, Tuple[np.ndarray, float]],
                       lo: int, hi: int, period_frames: float, votes: int) -> float:
    """空档的离地证据 = 各通道证据比里第 votes 高的那一个; votes<=0 时退回只看主通道。

    取"第 N 高"而不是最大值, 是为了用同一个参数表达 1 票(任一通道达标)与 2 票(两条通道同时达标)。
    """
    values = [gap_lift_evidence(detrended, lo, hi, period_frames) / max(float(amp_thr), 1e-6)]
    if votes >= 1:
        for name in sorted(extras):
            det, thr = extras[name]
            values.append(gap_lift_evidence(det, lo, hi, period_frames) / max(thr, 1e-6))
    if votes <= 0 or len(values) < votes:
        return values[0]
    values.sort(reverse=True)
    return float(values[votes - 1])


def channel_quality(frames: np.ndarray, up: np.ndarray, median_h: float,
                    cfg: Config, fps: float) -> Tuple[float, int, float, float]:
    """这条通道数得清不清楚: (节律强度, 过阈值的峰数, 观测覆盖率, 位移 p95 像素)。

    判据与计数同一套, 免得选完源又数不出东西。最后那个 p95 顺手带出来给
    "头部零位移"闸门用: 通道择优每秒只重算一次, 闸门复用它就不额外花钱。
    """
    keep = np.isfinite(up)
    if keep.sum() < 20:
        return 0.0, 0, 0.0, 0.0
    f, v = frames[keep], up[keep]
    axis, dense, _ = sg.fill_gaps(f, v, cfg.max_gap)
    observed = np.zeros(axis.size, dtype=bool)
    observed[f - int(axis[0])] = True
    baseline_win = cfg.baseline_win if dense.size >= cfg.baseline_win else _odd(dense.size // 2)
    det = sg.detrend(dense, cfg.smooth_win, baseline_win)
    noise = sg.hf_sigma_masked(det, observed)
    peak_level = float(np.percentile(det, 95)) if det.size else 0.0
    thr = amplitude_threshold(noise, peak_level, median_h, cfg)
    # 向下取整: 向上取整会把"最短可取峰间隔"悄悄拉长, 让 max_rate_hz 的名义上限够不到
    hard_min = max(2, int(fps / cfg.max_rate_hz))
    coarse = sg.find_peaks(det, hard_min, thr, 0.8 * thr)
    ac_period, strength = sg.autocorr_period(det, fps, cfg.min_rate_hz, cfg.max_rate_hz)
    _, period_src = _choose_period(coarse, ac_period, fps, cfg)
    if period_src == "fallback":
        strength = 0.0
    return round(float(strength), 3), int(len(coarse)), round(float(observed.mean()), 3), round(peak_level, 3)


def source_scores(samples: List[Tuple], cfg: Config, fps: float,
                  channels: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Tuple[float, int, float, float]]:
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
    out: Dict[str, Tuple[float, int, float, float]] = {}
    for source in ("ankle", "hip", "head"):
        if usable.get(source, 0) < max(1, int(0.5 * len(samples))) or source not in ch:
            continue
        values = ch[source]
        if not np.isfinite(values).any():
            continue
        out[source] = channel_quality(frames, -np.where(np.isfinite(values), values, np.nan), median_h, cfg, fps)
    return out


def channel_rank(scores: Dict[str, Tuple[float, int, float, float]], cfg: Config):
    """通道排序键。默认纯比节律强度; channel_prefer_rate_margin>0 时先按强度整档分层, 同档内改比谁数得出的起跳多。

    快跳的人头颈几乎不动但信号平滑, 自相关强度反而压过脚踝, 纯比强度会选到"慢摆"那条,
    一整段快跳就被数成十几次(实测 179023428267 #18: 头 11 次 vs 踝 69 次)。
    """
    top = max((float(value[0]) for value in scores.values()), default=0.0)
    margin = cfg.channel_prefer_rate_margin
    floor = float(getattr(cfg, "channel_prefer_rate_floor", 0.0) or 0.0)

    def rank(name: str):
        strength, peaks = scores[name][0], scores[name][1]
        if margin > 0 or floor > 0:
            # 只分两层: "数得清" 与 "明显差一截"。数得清的里面比谁数得出的起跳多。
            # 门槛取相对档宽与绝对门槛里更宽松的一条: 头 0.851 对踝 0.373 时档宽够不着,
            # 但两条都过了"节律算清楚"这条既有门槛, 就该按起跳数决定用谁。
            good = (margin > 0 and strength >= top - margin) or (floor > 0 and strength >= floor)
            return 1 if good else 0, peaks, strength
        return strength, peaks

    return rank


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
    best = max(scores, key=channel_rank(scores, cfg))
    base = scores.get(fallback, (0.0, 0, 0.0, 0.0))
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
    key = "rhythm_clarity" if cfg.rhythm_measure == "period" else "rhythm_strength"
    strength = stats.get(key)
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
    # 头部零位移闸门: 站着拿绳的观众也会被前景行人带出"踝部波纹", 节律还能过关, 但身体没离地。
    # 放在幅度判定之前: 这类假节律的幅度指标通常是"合格"的, 走不到需要低置信兜底的分支。
    head_p95 = stats.get("head_p95_px")
    body_p95 = stats.get("body_p95_px")
    if (cfg.min_head_move_px > 0 and head_p95 is not None and body_p95
            and head_p95 < cfg.min_head_move_px and head_p95 < cfg.min_head_share * body_p95):
        return OUT, (f"头部只动了 {head_p95:.2f}px(全身最大 {body_p95:.1f}px), "
                     f"身体没有整体离地, 是站着的人被前景带动")
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


def channel_lift_p95(samples: List[Tuple], cfg: Config,
                     channels: Optional[Dict[str, np.ndarray]] = None
                     ) -> Dict[str, Tuple[float, float]]:
    """每条身体部位通道去基线后的 (位移 p95 像素, 观测覆盖率)。

    择优只看"哪条抬得最高", 这里把每条通道自己动了多少都留下来:
    真跳绳时全身一起离地, 头部必然跟着抬; 站着拿绳的观众只有框底边/踝被前景行人带动,
    头部读数停在亚像素。两者用同一个口径分开, 才分得清真起跳和伪造的波纹。
    """
    frames = np.array([int(s[0]) for s in samples], dtype=np.int64)
    total = max(1, len(samples))
    out: Dict[str, Tuple[float, float]] = {}
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
        out[source] = (float(np.percentile(detrended, 95)), round(float(keep.sum()) / total, 3))
    return out


def best_lift_ratio(samples: List[Tuple], median_h: float, cfg: Config,
                    channels: Optional[Dict[str, np.ndarray]] = None) -> float:
    """各身体部位里最大的离地幅度/身高: 脚踝抬得少, 但头髋能反映真实离地高度。"""
    if median_h <= 0:
        return 0.0
    p95 = channel_lift_p95(samples, cfg, channels)
    if not p95:
        return 0.0
    return round(max(value[0] for value in p95.values()) / median_h, 4)


def body_motion_stats(lift_p95: Dict[str, Tuple[float, float]],
                      cfg: Config) -> Tuple[Optional[float], Optional[float]]:
    """(头部位移 p95, 全身最大位移 p95): 只在头部通道观测够多时给出头部读数。

    只比踝/髋/头三条身体通道, 不带框底边 —— 框底边正是最容易被阴影和前景带动的通道,
    让它参与"全身最大"会替假节律作证。头部被挡住时返回 None, 闸门自动不启用。
    """
    body = [value[0] for name, value in lift_p95.items() if name in ("ankle", "hip", "head")]
    if not body:
        return None, None
    head = lift_p95.get("head")
    if head is None or head[1] < cfg.min_head_coverage:
        return None, round(max(body), 3)
    return round(head[0], 3), round(max(body), 3)


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


def recover_grid_peaks(detrended: np.ndarray, observed: np.ndarray, base: List[int],
                       period_frames: float, amp_thr: float, clarity: float,
                       cfg: Config) -> Tuple[List[int], set]:
    """把"踩在节拍格上但幅度不足阈值"的鼓包认领成起跳, 见 config.grid_fill_frac。

    阈值本身已经贴到信号下限, 再往下压就是放进噪声(扫描过, 见文档第 10 节)。
    这里换一条判据: 用"什么时候该有起跳"反过来决定"多小的鼓包算起跳"。
    格子由已认峰的公共相位定出, 只有空格子上的鼓包才被认领,
    并且必须与相邻已认峰隔开 recover_gap_ratio 个周期以上 —— 与包络回捞同一条防双计距离。
    """
    if (cfg.grid_fill_frac <= 0 or period_frames <= 1.0 or len(base) < 4
            or clarity < cfg.grid_fill_clarity or amp_thr <= 0):
        return list(base), set()
    period = float(period_frames)
    tol = max(1, int(round(cfg.grid_fill_tol * period)))
    floor = cfg.grid_fill_frac * float(amp_thr)
    gap_floor = cfg.recover_gap_ratio * period
    phase = float(np.median([float(float(p) % period) for p in base]))
    accepted = sorted(int(p) for p in base)
    extra: set = set()
    for k in range(int(detrended.size / period) + 1):
        center = int(round(phase + k * period))
        if center < 1 or center >= detrended.size - 1:
            continue
        if any(abs(center - known) < gap_floor for known in accepted):
            continue
        lo, hi = max(1, center - tol), min(detrended.size - 1, center + tol + 1)
        if hi <= lo:
            continue
        pos = lo + int(np.argmax(detrended[lo:hi]))
        # 插值填出来的帧不算证据: 遮挡期间"看着有个鼓包"多半是插值把两端连起来的线
        if not bool(observed[pos]) or float(detrended[pos]) < floor:
            continue
        if any(abs(pos - known) < gap_floor for known in accepted):
            continue
        accepted.append(pos)
        extra.add(pos)
    accepted.sort()
    return accepted, extra


def keep_on_grid(peaks: List[int], mains: List[int], period_frames: float,
                 tol: float) -> Tuple[List[int], int]:
    """窄窗的峰只留两类: 与宽窗实测峰是同一拍的(只差亚帧), 或踩在整数倍周期格子上的(被滤波抹掉的那一拍)。

    宽窗实测峰太少时不作过滤: 锚点不够, 格子本身就画不准, 宁可原样放行。
    """
    if period_frames <= 0 or len(mains) < 4:
        return list(peaks), 0
    kept: List[int] = []
    for p in peaks:
        dist = min(abs(m - p) for m in mains)
        if dist <= tol * period_frames:
            kept.append(p)                                   # 同一拍, 只是亚帧位置有差
            continue
        if dist > 0.6 * period_frames:
            k = round(dist / period_frames)
            if k >= 1 and abs(dist - k * period_frames) <= tol * period_frames:
                kept.append(p)                               # 踩在格子上 = 被抹掉的那一拍
    return kept, len(peaks) - len(kept)


def analyze_signal(
    dense: np.ndarray,
    observed: np.ndarray,
    baseline_win: int,
    median_h: float,
    fps: float,
    cfg: Config,
    hard_min: int,
    smooth_win: int,
) -> dict:
    """给定一档滤波窗口, 走完"去基线 -> 噪声 -> 阈值 -> 粗取峰 -> 节奏估计"整条前段。

    抽出来是为了能用不同窗口重跑同一份位移: 窗口越宽越抗抖, 但窄于一两帧的真起跳会被抹平,
    到底该用哪一档取决于这个人跳得多快, 只能各算一遍再比。
    """
    detrended = sg.detrend(dense, smooth_win, baseline_win)
    noise = sg.hf_sigma_masked(detrended, observed)
    mad = sg.mad_sigma(detrended)
    peak_level = float(np.percentile(detrended, 95)) if detrended.size else 0.0
    snr = round(min(99.0, peak_level / noise), 2) if noise > 1e-6 else float(peak_level > 0) * 99.0
    # 阈值不能超过信号本身, 否则密集人群里的小幅起跳会被整段抹掉;
    # 下压时只压到"框高比例下限", 不压回绝对像素下限(细节见 amplitude_threshold)。
    amp_thr = amplitude_threshold(noise, peak_level, median_h, cfg)
    # 两遍节奏估计: 先按物理上限粗取峰, 再用实测峰间隔的中位数当作这个人自己的节奏,
    # 避免自相关把噪声尖峰当成周期(会把一次起跳数成两三次)
    # 同上: 这里决定了"一次起跳最短隔几帧", 向上取整会把快跳压成两倍周期
    coarse = sg.find_peaks(detrended, hard_min, amp_thr, 0.8 * amp_thr)
    ac_period, strength = sg.autocorr_period(detrended, fps, cfg.min_rate_hz, cfg.max_rate_hz)
    period_frames, period_src = _choose_period(coarse, ac_period, fps, cfg)
    clarity = float(sg.autocorr_clarity(detrended, int(round(period_frames)))) if period_frames > 0 else 0.0
    return {
        "smooth_win": int(smooth_win),
        "detrended": detrended,
        "noise": noise,
        "mad": mad,
        "peak_level": peak_level,
        "snr": snr,
        "amp_thr": amp_thr,
        "coarse": coarse,
        "strength": float(strength),
        "period_frames": period_frames,
        "period_src": period_src,
        "clarity": clarity,
        "n_peaks": len(coarse),
    }


def pick_smooth_pass(
    dense: np.ndarray,
    observed: np.ndarray,
    baseline_win: int,
    median_h: float,
    fps: float,
    cfg: Config,
    hard_min: int,
    stats: Optional[dict] = None,
) -> dict:
    """挑一档滤波窗口做后续计数, 并把两档的对照数字写进 stats 供复盘。

    只在主档已经把节奏估到"快跳"档时才试窄窗口: 慢跳时窄窗口只会把关键点抖动放进来。
    替换条件要同时满足"明显多取到峰"和"节律没变差", 否则宁可留在抗抖的那一档。
    """
    main = analyze_signal(dense, observed, baseline_win, median_h, fps, cfg, hard_min, cfg.smooth_win)
    used = main
    alt = None
    fast = cfg.smooth_win_fast
    if (fast >= cfg.smooth_win or fast < 1 or cfg.smooth_fast_min_rate <= 0
            or main["period_src"] == "fallback" or main["period_frames"] <= 0
            or fps / main["period_frames"] < cfg.smooth_fast_min_rate):
        tried = False
    else:
        tried = True
        alt = analyze_signal(dense, observed, baseline_win, median_h, fps, cfg, hard_min, fast)
        if (alt["n_peaks"] >= cfg.smooth_fast_gain * main["n_peaks"]
                and alt["clarity"] >= main["clarity"] - cfg.smooth_fast_margin):
            kept, dropped = keep_on_grid(alt["coarse"], main["coarse"], main["period_frames"],
                                         cfg.recover_center_tol)
            # 换窗只是为了找回被滤波抹掉的那几拍: 留在节拍格子外的新峰是抖动, 不要
            if len(kept) >= max(cfg.smooth_fast_min_extra, main["n_peaks"]):
                used = dict(alt, coarse=kept, n_peaks=len(kept))
                used["detrended"] = alt["detrended"]
                if stats is not None:
                    stats["smooth_fast_dropped"] = dropped
    if stats is not None:
        stats["smooth_win_used"] = used["smooth_win"]
        stats["smooth_fast_tried"] = int(tried)
        stats["smooth_peaks_used"] = int(used["n_peaks"])
        stats["smooth_peaks_other"] = int((alt if used is main else main)["n_peaks"]) if tried else -1
    return used


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
    hard_min = max(2, int(fps / cfg.max_rate_hz))
    # 快跳时窄于一两帧的真起跳会被 3 帧中值整个抹平, 所以按节奏挑一档窗口再往下走
    pass_used = pick_smooth_pass(dense, observed, baseline_win, median_h, fps, cfg, hard_min,
                                 stats=result.stats)
    detrended = pass_used["detrended"]
    noise, mad, peak_level = pass_used["noise"], pass_used["mad"], pass_used["peak_level"]
    amp_thr, coarse = pass_used["amp_thr"], pass_used["coarse"]
    strength, period_frames, period_src = pass_used["strength"], pass_used["period_frames"], pass_used["period_src"]
    result.snr = pass_used["snr"]
    result.stats["rhythm_strength"] = round(float(strength), 3)
    # 另一个口径: 自相关取在"这个人自己的周期"那一格上。band 口径几乎人人合格, 见 config 注释。
    result.stats["rhythm_clarity"] = round(pass_used["clarity"], 3)
    result.stats["period_source"] = period_src
    result.stats["period_estimated"] = period_src != "fallback"
    result.stats["coarse_peaks"] = len(coarse)
    period = period_frames / fps
    result.period_sec = round(float(period), 4)
    result.rate_hz = round(fps / period_frames, 3) if period_frames > 0 else 0.0

    min_distance = max(hard_min, int(round(cfg.refractory_ratio * period_frames)))
    # 不应期到不到的地方, 用"有没有落回地面"接着判: 阈值和粗取峰完全同一批, 不引入新的门限。
    if cfg.landing_valley_frac > 0:
        idx, landing = sg.merge_close_peaks(detrended, coarse, min_distance,
                                            cfg.landing_valley_frac, amp_thr)
    else:
        idx, landing = sg.find_peaks(detrended, min_distance, amp_thr, 0.8 * amp_thr), set()
    result.stats["landing_kept"] = len(landing)

    # 位移阈值已经贴到信号下限, 剩下的漏计靠"节奏该有而没数到"的空档 + 多通道同刻鼓包找回
    recovered: set = set()
    if cfg.envelope_detect and period_src != "fallback":
        idx, recovered = recover_peaks(detrended, axis, frames_all, idx, observed, min_distance,
                                       period_frames, amp_thr, channels, resolved, cfg, fps)
    grid_recovered: set = set()
    if period_src != "fallback":
        idx, grid_recovered = recover_grid_peaks(detrended, observed, idx, period_frames, amp_thr,
                                                 float(pass_used["clarity"]), cfg)
    # 回捞总数在合并节拍格认领之后写(见下面 warmup 过滤处), 保持"stats 数 == 事件标记数"的不变量
    result.stats["recovered_grid"] = len(grid_recovered & set(idx))

    warmup = int(round(cfg.warmup_sec * fps))
    idx = [i for i in idx if axis[i] - axis[0] >= warmup]
    recovered &= set(idx)
    # 节拍格认领的峰与包络回捞一样打上 recovered 标记(它们不是干净的实测峰), 但计数另记一笔
    grid_recovered &= set(idx)
    recovered |= grid_recovered
    result.stats["recovered"] = len(recovered)

    events: List[JumpEvent] = []
    bumps = 0
    prev: Optional[float] = None
    estimated_misses = 0
    filled = 0
    misses_unobserved = 0
    fill_no_lift = 0
    fill_phase = 0
    extras: Optional[Dict[str, Tuple[np.ndarray, float]]] = None
    # 只有节奏是真测出来的、且人一直在画面里却没数到峰, 才按节奏补; 否则宁可少报也不虚报。
    rhythm_trusted = period_src != "fallback" and strength >= cfg.keep_rhythm_strength
    # 局部节奏: 只改"该不该有起跳、该补几个"的尺子, 不改取峰间距(见 config.local_period_window)
    local_n = max(0, int(getattr(cfg, "local_period_window", 0) or 0))
    gaps_hist: List[float] = []
    period_low = float(max(2.0, hard_min))
    period_high = float(fps) / max(0.05, cfg.min_rate_hz)
    fastest_scale = float(period_frames) if period_frames > 0 else 0.0
    for i in idx:
        sub = sg.parabolic_peak(detrended, i)
        frame_pos = float(axis[0]) + sub
        t = float(np.interp(frame_pos, axis, times_dense))
        amp = float(detrended[i])
        kind = "jump"
        if prev is not None:
            gap = frame_pos - prev
            scale = period_frames
            if local_n >= 4 and len(gaps_hist) >= local_n:
                scale = min(max(float(np.percentile(np.asarray(gaps_hist[-local_n:], dtype=np.float64),
                                                float(cfg.local_period_quantile))),
                                period_low), period_high)
            fastest_scale = min(fastest_scale, scale) if fastest_scale > 0 else scale
            if gap < cfg.double_under_ratio * scale and i not in landing:
                kind = "bump"
                bumps += 1
            elif gap > cfg.gap_ratio * scale:
                expected = int(round(gap / scale))
                estimated_misses += max(0, expected - 1)
                lo = max(0, int(round(prev - axis[0])))
                coverage = float(observed[lo:i].mean()) if i > lo else 0.0
                # 空档内部连一点离地都没有 = 人停下没跳, 不能按节奏凭空补数
                if cfg.fill_evidence_votes > 0 and extras is None:
                    extras = gap_evidence_signals(channels, frames_all, axis, observed, median_h, cfg,
                                                  int(pass_used["smooth_win"]), baseline_win, resolved)
                evidence = gap_evidence_votes(detrended, amp_thr, extras or {}, lo, i, scale,
                                              int(cfg.fill_evidence_votes))
                no_lift = (cfg.fill_min_lift_frac > 0 and evidence < cfg.fill_min_lift_frac)
                if no_lift:
                    fill_no_lift += max(0, expected - 1)
                # 遮挡空档旁路: 只在"覆盖率不够(人不在画面里)"时启用, 见 config.fill_hole_phase_tol。
                # 不加这个限定会出大事: 人站着没跳的那段位移是平的但一直在观测, 两端峰的距离
                # 常常正好是整数个周期, 相位残差≈0 会被判成"节奏没断"而凭空补好几个 (test_core 抓过)。
                residual = abs(gap / max(scale, 1e-6) - expected)
                phase_ok = (cfg.fill_hole_phase_tol > 0 and expected >= 2
                            and coverage < cfg.min_fill_coverage
                            and residual <= cfg.fill_hole_phase_tol)
                if (coverage < cfg.min_fill_coverage or no_lift) and not phase_ok:
                    misses_unobserved += max(0, expected - 1)
                elif cfg.interpolate_gaps and expected >= 2 and rhythm_trusted:
                    conf = min(0.9, evidence)
                    if phase_ok:
                        conf = max(conf, round(1.0 - residual, 3))
                        fill_phase += max(0, expected - 1)
                    for k in range(1, min(expected - 1, cfg.max_fill_per_gap) + 1):
                        fill_pos = prev + gap * k / expected
                        events.append(
                            JumpEvent(
                                frame=fill_pos,
                                time=float(np.interp(fill_pos, axis, times_dense)),
                                amplitude=round(amp_thr * 0.6, 3),
                                confidence=round(conf, 3),
                                kind="interpolated",
                            )
                        )
                        filled += 1
        events.append(JumpEvent(frame=round(frame_pos, 3), time=round(t, 4), amplitude=round(amp, 3),
                               confidence=round(min(1.0, amp / max(amp_thr, 1e-6)), 3), kind=kind,
                               recovered=bool(i in recovered)))
        if local_n >= 4 and prev is not None and gap <= cfg.gap_ratio * period_frames:
            # 只有"当作正常拍收下"的间隔才进局部节奏历史: 漏计造成的双倍间隔一旦进桶,
            # 中位数被拉长, 尺子反过来变钝, 实测比不改还少 208 次。
            gaps_hist.append(gap)
        prev = frame_pos

    # 实测与按节奏补齐分两笔记: 混在一起会让灵敏度滑杆失去意义(门限再严也能靠补数追平),
    # 复盘时也看不出哪些是真抓到的起跳。总量 = count + filled。
    direct = [e for e in events if e.kind == "jump"]
    filled_events = [e for e in events if e.kind == "interpolated"]
    # 补数两条护栏取更严的一条: ①占比护栏(补数不许超过实数的这个比例);
    # ②节奏应有上限(补完不许超过"整段时长 ÷ 实测周期"), 见 config.fill_cap_expected。
    # 超配额时按证据强度留强去弱, 剩下的仍按时间顺序留在事件流里。
    if filled_events:
        quota = len(filled_events)
        if cfg.max_fill_share > 0 and len(direct) >= 4:
            quota = min(quota, int(math.floor(cfg.max_fill_share * len(direct))))
        scored = len(direct) + (bumps if cfg.policy != "jump" else 0)
        if cfg.fill_cap_expected and period_frames > 0:
            # 尺子与实时端同一条: 取全局周期与局部快尺里更短的一条, 别用被锁长的周期卡死快跳
            ruler = min(float(period_frames), fastest_scale) if fastest_scale > 0 else float(period_frames)
            beats = (float(axis[-1] - axis[0] + 1) - warmup) / max(ruler, 1e-6)
            quota = min(quota, max(0, int(math.ceil(beats)) - scored))
        if len(filled_events) > quota:
            ranked = sorted(filled_events, key=lambda e: -e.confidence)[:quota]
            keep = {id(e) for e in ranked}
            dropped = len(filled_events) - quota
            events = [e for e in events if e.kind != "interpolated" or id(e) in keep]
            filled_events = [e for e in events if e.kind == "interpolated"]
            filled = len(filled_events)
            result.stats["fill_capped"] = dropped
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
    result.stats["fill_no_lift"] = fill_no_lift
    result.stats["fill_phase"] = fill_phase
    result.stats["evidence_channels"] = len(extras) if extras is not None else 0
    result.stats["long_holes"] = long_holes
    result.stats["visibility"] = result.visibility
    # 一次去基线同时供两件事用: 各通道最大离地幅度(择优) + 头部是否真的跟着离地(闸门)
    lift_p95 = channel_lift_p95(samples, cfg, channels)
    result.stats["lift_ratio"] = round(peak_level / median_h, 4) if median_h > 0 else 0.0
    result.stats["lift_ratio_best"] = (round(max(value[0] for value in lift_p95.values()) / median_h, 4)
                                       if lift_p95 and median_h > 0 else 0.0)
    head_p95, body_p95 = body_motion_stats(lift_p95, cfg)
    result.stats["head_p95_px"] = head_p95
    result.stats["body_p95_px"] = body_p95
    result.stats["channel_p95_px"] = {name: round(value[0], 3) for name, value in lift_p95.items()}
    result.stats["excursion_ratio"] = round(result.stats["excursion_px"] / median_h, 3) if median_h > 0 else 0.0
    sync = ankle_sync(samples, cfg)
    result.stats["ankle_sync"] = round(sync, 3) if sync is not None else None
    result.stats["count"] = result.count
    observed_sec = float(frames.size) / float(fps) if fps > 0 else 0.0
    # 应有次数有两个口径, 都报出来, 各自防不同的偏差:
    # expected_count(墙钟口径) = 频率 × 轨迹首尾时长。这是历史基线一直在用的口径,
    #   也是虚高判定(total > 1.05 × expected)的依据。
    # expected_visible(可见段口径) = 频率 × 人真在画面里的秒数。缺口归因(out/tune/probe_hole.py)
    #   显示墙钟口径的"应有"里有 967 次(45%)落在人被挡住/检测丢帧的时间段 —— 那些时间没有观测,
    #   记成"应有"等于凭空造缺口。但可见段口径太脆: 频率只要低估 10%,
    #   实测就有三成轨迹被判成"数过头"(虚高 2 条 -> 33 条), 所以它只当"看得见的部分数到几成"的参考。
    expected = round(result.rate_hz * result.stats["duration_sec"], 1) if result.rate_hz else 0.0
    expected_visible = round(result.rate_hz * observed_sec, 1) if result.rate_hz else 0.0
    result.stats["observed_sec"] = round(observed_sec, 2)
    result.stats["expected_count"] = expected
    result.stats["expected_visible"] = expected_visible
    result.stats["total_count"] = result.count + int(result.stats.get("filled", 0))
    result.stats["rhythm_coverage"] = (round(result.stats["total_count"] / expected, 3)
                                       if expected > 0 else 0.0)
    result.stats["rhythm_coverage_visible"] = (round(result.stats["total_count"] / expected_visible, 3)
                                               if expected_visible > 0 else 0.0)
    verdict, reason = participation(result.stats, cfg)
    result.stats["verdict"] = verdict
    result.stats["participant"] = verdict != OUT
    if verdict == OUT:
        result.stats["excluded"] = reason
    return result
