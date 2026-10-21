"""计数流水线的可调参数。"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Optional


# ---- 灵敏度滑杆(实时页面 1~100) → 起跳判据的三档松紧 ----
SENS_MIN = 1
SENS_MAX = 100
SENS_DEFAULT = 50
# 上端不敢放到接近零: 实测滑杆推到 95 时总计数会比实际多两三成(一次起跳被数成两个),
# 所以最松一档仍保留足够的门限, 让整根滑杆都落在"能对上真实个数"的区间里。
# 但 16 人群像复盘发现上端收得太紧: 人只有 170px 高时真实起跳在踝部只留下 1~2px,
# 滑杆推到 100 也只能把下限松到 1.4px, 半数真起跳仍在阈值以下(整段只数到实际的一半),
# 所以把最松一档整体再往下放一档, 滑杆上半段才有可感知的变化。
SNR_K_STRICTEST, SNR_K_DEFAULT, SNR_K_LOOSEST = 5.4, 3.0, 0.4
PEAK_FRAC_STRICTEST, PEAK_FRAC_DEFAULT, PEAK_FRAC_LOOSEST = 0.15, 0.08, 0.025
RELIEF_STRICTEST, RELIEF_DEFAULT, RELIEF_LOOSEST = 1.6, 1.0, 0.4


def _ramp(value, at_min, at_mid, at_max) -> float:
    """滑杆 1→50→100 上一段指数插值: 每滑一格的感觉变化大致相同。"""
    v = min(float(SENS_MAX), max(float(SENS_MIN), float(value)))
    if v <= SENS_DEFAULT:
        span, low, high = (v - SENS_MIN) / (SENS_DEFAULT - SENS_MIN), at_min, at_mid
    else:
        span, low, high = (v - SENS_DEFAULT) / (SENS_MAX - SENS_DEFAULT), at_mid, at_max
    return low * (high / low) ** span


def sensitivity_to_snr_k(value) -> float:
    """滑杆对应的噪声倍数(振幅阈值 = snr_k * 噪声)。"""
    return round(_ramp(value, SNR_K_STRICTEST, SNR_K_DEFAULT, SNR_K_LOOSEST), 3)


def sensitivity_to_gains(value) -> dict:
    """把灵敏度滑杆摊到三道门限上, 只松噪声一项的话滑到一半就没反应了。

    起跳门限取三者最大: 噪声倍数、绝对像素/框高比例下限、信号峰值的一个比例。
    以前只有噪声倍数跟着滑杆走, 滑到 50 以下峰值比例(0.08)和像素下限就接管了一切,
    所以用户把滑杆推到最高也看不到计数变多。现在三者一起松: 滑杆 1 最严,
    50 与默认配置一致, 100 最松(但像素下限不会归零, 机身抖动仍兜得住)。
    """
    return {
        "snr_k": sensitivity_to_snr_k(value),
        "peak_frac": round(_ramp(value, PEAK_FRAC_STRICTEST, PEAK_FRAC_DEFAULT, PEAK_FRAC_LOOSEST), 4),
        "floor_relief": round(_ramp(value, RELIEF_STRICTEST, RELIEF_DEFAULT, RELIEF_LOOSEST), 3),
    }


def snr_k_to_sensitivity(value) -> int:
    """反推滑杆读数: 让状态/日志把阈值显示回页面上那根 1~100 的刻度。"""
    k = max(1e-6, float(value))
    if k >= SNR_K_DEFAULT:
        span = math.log(k / SNR_K_STRICTEST) / math.log(SNR_K_DEFAULT / SNR_K_STRICTEST)
        raw = SENS_MIN + span * (SENS_DEFAULT - SENS_MIN)
    else:
        span = math.log(k / SNR_K_DEFAULT) / math.log(SNR_K_LOOSEST / SNR_K_DEFAULT)
        raw = SENS_DEFAULT + span * (SENS_MAX - SENS_DEFAULT)
    return int(round(min(float(SENS_MAX), max(float(SENS_MIN), raw))))


@dataclass
class Config:
    # ---- 检测 / 跟踪 ----
    classes: tuple = (0,)          # COCO: 0 = person
    imgsz: int = 1280              # 2560x1440 监控画面下 640 会把人压到 ~80px, 检测闪烁严重
    conf: float = 0.35             # 检测置信度下限
    device: Optional[str] = None   # None=自动, 可填 0 / cpu / mps
    match_thresh: float = 0.35     # 关联得分下限
    match_iou_w: float = 0.6       # 得分中 IoU 的权重, 其余给中心距离
    match_scale: float = 0.6       # 中心距离得分的衰减尺度(乘以框高)
    nms_iou: float = 0.45          # 同一人重复出框的 IoU 抑制阈值
    dup_center_ratio: float = 0.3  # 中心距离小于框高该比例视为同一人的重复框
    dup_min_iou: float = 0.25      # 中心距离判重还要求框真的重叠: 紧挨着站的两个人中心距离可以很近但框不重叠
    max_age: int = 25              # 连续丢失多少帧后判定轨迹断开: 16 人群像里人被人挡 1s 是常态,
                                   # 断开就等于这段起跳全丢(实测 15 帧时有 470 次数到却被丢, 放宽到 25 剩 289)
    min_hits: int = 3              # 新建轨迹需要连续命中的帧数(抑制闪烁误检)
    reid_window: int = 45          # 断开后允许重新认领的时间窗口(帧)
    reid_dist: float = 0.9         # 重新认领时的最大中心距离/框高比例
    merge_gap_frames: int = 60     # 碎帧缝合允许的最大时间间隔(帧)
    merge_dist_ratio: float = 0.9  # 碎帧缝合允许的水平距离/框高比例
    dup_overlap: float = 0.5       # 判定重复轨迹的最小时间重叠比例
    dup_dist_ratio: float = 0.45   # 判定重复轨迹的水平距离/框高比例
    dup_iou_min: float = 0.35      # 重复轨迹在重叠帧上的框重叠下限: 并排站的人水平距离也近, 只按距离判重会把真人并掉
    min_track_sec: float = 2.0     # 短于此长度的轨迹不参与总计数
    min_track_visibility: float = 0.5
    # ---- 离场寄存: 计数器被淘汰时把已经数到的次数留在站位上 ----
    bank_counts: bool = True       # 关掉即退回"计数器淘汰就把这段计数丢掉"
    bank_keep_sec: float = 120.0   # 站位无人超过这么久才真的忘记他寄存的次数: 出去喘口气回来的人不再从 0 数起
    bank_max: int = 32             # 同时寄存的站位条数(按最久未用淘汰), 防止摄像头长跑堆积

    # ---- 站位身份(实时页面: 固定机位下用站位当身份) ----
    slot_identity: bool = True     # 检测闪烁把一个人拆成多条轨迹时, 按站位接回同一个人
    slot_dist_ratio: float = 0.5   # 认领站位允许的最大位置距离 / 框高
    slot_claim_gap: int = 8        # 站位空置多少帧后才允许新轨迹认领(防止把旁边的人并进来)
    slot_height_ratio: float = 1.6 # 认领站位允许的框高变化倍数
    slot_keep_frames: int = 250    # 站位消失多久后彻底忘记(帧), 约 10s
    ghost_display_sec: float = 0.35 # 失联超过这么久就不再画他的框/脚标; 计数仍由站位继续保管
    slot_overlap_iou: float = 0.55 # 框重叠超过此值判定就是同一个人, 即使站位刚被占用也直接认领(防止同一人另立新框)
    dup_box_iou: float = 0.5       # 两个框重叠超过此值视为同一个人, 只把更旧的框从画面上去掉(旧框定格会让人看着像套了好几个框); 不影响计数

    # ---- 信号 ----
    signal_source: str = "auto"    # auto|ankle|hip|head|box_bottom|box_center
    channel_select: str = "rhythm"  # auto 时的选源口径: coverage=只按覆盖率, rhythm=在踝/髋/头里挑节律最清楚的
    channel_switch_margin: float = 0.08  # 换通道要在节律强度上明确胜出这么多才换, 免得两个通道打平时来回跳
    kp_conf: float = 0.30          # 关键点可见性下限
    ankle_combine: str = "high"    # 双踝合成: high=取抬得高的一只(左右脚交替抬时不再漏掉一半), mean=取两者均值
    max_gap: int = 12              # 可插值的最大缺帧数
    smooth_win: int = 3            # 中值滤波窗口(奇数)
    baseline_win: int = 91         # 去基线的滑动中值窗口(帧), 约 3s

    # ---- 计数 ----
    min_rate_hz: float = 0.5       # 合理跳绳频率下限
    max_rate_hz: float = 3.5       # 合理跳绳频率上限
    min_amplitude: float = 2.0     # 绝对振幅下限(像素)
    amplitude_ratio: float = 0.004 # 相对振幅下限(占框高比例)
    amplitude_ref_height: float = 260.0  # 绝对像素下限对应的框高: 更矮的人按比例缩小这条下限, 0=不缩放
    amplitude_floor_min: float = 0.35    # 按比例缩小后, 绝对下限最多缩到原值的这个比例
    snr_k: float = 3.0             # 振幅阈值 = snr_k * 噪声
    peak_frac: float = 0.08        # 振幅阈值不低于信号峰值的该比例(灵敏度调高时一起放松)
    floor_relief: float = 1.0      # 绝对像素/框高比例下限的缩放, <1 表示放松
    refractory_ratio: float = 0.6  # 最小间隔 = refractory_ratio * 周期; 太小时一次起跳会被数成两个
    double_under_ratio: float = 0.62  # 间隔小于该比例*周期 -> 双摇过绳
    gap_ratio: float = 1.8         # 间隔大于该比例*周期 -> 疑似漏计
    interpolate_gaps: bool = True  # 是否按周期补齐疑似漏计的空档: 遮挡与错位造成的空档是少数的主因, 默认开
    policy: str = "jump"           # jump=按起跳计数, rope=按过绳计数
    poll_hz: float = 6.0           # 每人每秒做几次峰值搜索(取信号仍逐帧, 不影响漏峰)
    channel_reselect_sec: float = 1.0  # 在线重选信号通道的间隔(秒): 三条通道的得分逐帧几乎不变, 每次出数都重算太贵

    # ---- 能量包络回捞: 把"真起跳但幅度压在阈值下"的找回来 ----
    # 位移阈值本身已经压到信号下限(复盘: 远景人真实起跳在踝部只有 1~2px), 继续松就先把噪声放进来。
    # 改成带通→差分→平方→滑动积分得到能量包络: 小幅但整段同向的运动相对零散抖动占便宜,
    # 再只在"按节奏本该有起跳"的空档里回捞, 并用别的身体部位通道做同刻投票, 误计不会跟着涨。
    envelope_detect: bool = True       # 关掉即退回"只在原始位移上按阈值找峰"
    envelope_band_hz: tuple = (0.7, 4.0)   # 带通上下限(Hz), 超过采样率一半自动收
    envelope_win_ratio: float = 0.4    # 滑动积分窗 = 该比例 × 节奏周期
    envelope_frac: float = 0.05        # 包络检出阈值 = 该比例 × 包络 p95
    envelope_lead_frac: float = 0.06   # 投票通道算"同一时刻也有鼓包"的阈值(占它自己包络 p95)
    envelope_lead_votes: int = 2       # 低幅度回捞峰需要几条通道同时刻有鼓包(1=不投票)
    envelope_lead_tol: float = 0.35    # 投票允许的时刻偏差 = 该比例 × 周期
    recover_raw_ratio: float = 0.30    # 回捞峰在原始位移上至少要达到正常阈值的这个比例
    recover_min_coverage: float = 0.7  # 回捞时段至少观测到这么多帧: 与盲补分开收, 回捞本身还要信号签字
    recover_gap_ratio: float = 0.75    # 回捞峰与左右邻居的最小间隔 = 该比例 × 周期: 挤进 0.7 倍以内就是双计
    recover_center_tol: float = 0.35   # 回捞峰必须落在左右实测峰的正中(允许偏差 = 该比例 × 空档长): 站着抖的人也凑得出"空档"
    search_back_ratio: float = 1.66    # 空档超过该倍周期才回捞(沿用 R 峰检测的经验值)
    search_back_levels: tuple = (0.5, 0.35)  # 空档内逐档压低的幅度

    # ---- 参与判定(区分原地跳绳与走动路人) ----
    max_excursion_ratio: float = 1.2   # 水平活动范围超过框高该比例视为在走动
    max_excursion_hard: float = 2.5    # 超过该比例一律剔除: 再同步也不可能是原地跳, 大概率是 ID 在不同人之间来回跳
    drift_window_sec: float = 3.0      # 量水平活动范围的窗口(秒): 逐窗取中位数, 单次 ID 漂移不再放大成"走了几倍身高"
    min_lift_ratio: float = 0.012      # 起跳幅度/框高 下限
    min_ankle_sync: float = -0.10      # 左右脚踝位移相关性下限(走路为负相关)
    min_ankle_strong: float = 0.85     # 双脚同步起跳的下限: 位置漂移大但双脚同步时只降级不剔除
    require_pose_sync: bool = True     # 有姿态时启用双脚同步性判据
    warmup_sec: float = 1.0        # 每个轨迹开始阶段不计数
    online_window_sec: float = 12.0  # 在线计数的滑动信号窗口(秒)
    min_count_active: int = 5      # 至少连跳这么多次才算"在跳绳", 挡掉偶然抖出来的一两个峰
    # ---- 三态参与判定: 硬指标不过就剔除, 只是幅度/可见率偏弱而节奏清楚就低置信计入 ----
    keep_rhythm_strength: float = 0.35      # 节律强度下限: 达到这条, 幅度/可见率偏弱也只降级为低置信
    min_track_visibility_floor: float = 0.25  # 可见率低到这条以下彻底剔除, 不再低置信计入
    min_count_provisional: int = 2          # 数到的次数不足 min_count_active 时, 至少要有这么多次才允许低置信计入
    max_height_cv: float = 0.18             # 框高变异系数上限: 超过说明这条轨迹的观测在不同人之间跳, 计数不可信

    # ---- 按节奏补齐漏计的空档 ----
    min_fill_coverage: float = 0.7   # 空档区间里至少观测到这么多帧才补: 人本来就不在画面里时不该替他猜
    max_fill_per_gap: int = 2        # 单个空档最多补几个: 节奏估歪时不至于整段虚高

    # ---- 评估 ----
    tolerance_sec: float = 0.15    # 事件级匹配容差
    tolerance_ratio: float = 0.4   # 或用周期比例作为容差, 取较大者

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        names = {f.name for f in fields(cls)}
        unknown = set(raw) - names
        if unknown:
            raise ValueError(f"配置项无法识别: {sorted(unknown)}")
        if "classes" in raw:
            raw["classes"] = tuple(raw["classes"])
        return cls(**raw)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)


COCO_KEYPOINTS = {
    "head": [0],
    "shoulder": [5, 6],
    "hip": [11, 12],
    "ankle": [15, 16],
}

AUTO_ORDER = ["ankle", "hip", "head", "box_bottom"]
