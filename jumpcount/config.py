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
    # 同一清晰度档内改比"谁数得出的起跳多": 快跳者的头颈几乎不动但信号平滑,
    # 自相关强度反而压过脚踝, 选到慢摆通道后一整段快跳只剩十几次。0 = 历史行为(纯比强度)。
    # 档宽按"与最清楚那条差多少"分层: 头 0.851 / 踝 0.373 这种差距太大, 永远不同档,
    # 所以另有一条按绝对门槛同档的口径(channel_prefer_rate_floor), 两者取更宽的那一条。
    channel_prefer_rate_margin: float = 0.15
    # 绝对清晰度门槛: 只要通道节律强度 >= 这条就算"都数得清", 一律同档按起跳数比较。
    # 与 keep_rhythm_strength 同值: 那是本仓库已有的"节律算不算清楚"的唯一口径, 不另造标准。
    # 取 0 = 关闭(退回只看相对档宽)。
    channel_prefer_rate_floor: float = 0.0
    kp_conf: float = 0.30          # 关键点可见性下限
    ankle_combine: str = "high"    # 双踝合成: high=取抬得高的一只(左右脚交替抬时不再漏掉一半), mean=取两者均值
    max_gap: int = 12              # 可插值的最大缺帧数
    smooth_win: int = 3            # 中值滤波窗口(奇数)
    # 快跳自适应: 中值滤波窗口和"一拍有多长"此消彼长。25fps 下快跳一次起跳只占 1~2 帧,
    # 3 帧中值会把这种窄峰整个抹平: 峰取不到 -> 周期被估成两倍 -> 计数腰斩, 这正是"跳得快就少很多"。
    # 做法是先按 smooth_win 估节奏, 节奏不低于 smooth_fast_min_rate 时再按窄窗口重做一遍,
    # 只有窄档明显多取到峰(smooth_fast_gain)且节律没变差(smooth_fast_margin)才替换。
    smooth_win_fast: int = 1       # 第二档(窄)窗口, >= smooth_win 表示关闭
    # 实时链路要不要也用这第二档窄窗: 离线复盘早就两档都跑, 实时端一直只用宽窗,
    # 于是"实时比离线少数"。开启后实时与离线走同一个 pick_smooth_pass。
    online_fast_window: bool = True
    # 触发第二档的频率下限(Hz), 0=关闭。2.4 -> 2.0 实测 +23 次且双计特征反而降到 2.40%;
    # 再放到 1.5 能再多 122 次, 但被换的几条轨迹他道作证率从 ≥64% 掉到 27.9%
    # (out/tune/probe_corate.py), 买来的峰没有旁证, 与「没跳不计数」冲突, 停在 2.0。
    smooth_fast_min_rate: float = 2.0
    smooth_fast_gain: float = 1.15     # 窄档要比宽档多取到这么多倍的粗取峰才替换
    smooth_fast_margin: float = 0.05   # 允许窄档节律清晰度比宽档低这么多
    smooth_fast_min_extra: int = 8     # 过滤后窄窗至少要有这么多峰才换窗: 短轨迹锚点太少, 换窗等于瞎猜
    baseline_win: int = 91         # 去基线的滑动中值窗口(帧), 约 3s

    # ---- 计数 ----
    min_rate_hz: float = 0.5       # 合理跳绳频率下限
    # 合理跳绳频率上限: 这条同时决定"粗取峰的最小间隔"和"自相关能看的最短周期"。
    # 定在 3.5 时快跳会被 2:1 锁相腰斩: 真实周期短于 fps/3.5(25fps 下 7 帧) 就取不到峰,
    # 峰间隔中位数与自相关只能锁到两倍周期, 计数直接少一半(合成扫频 3.6Hz 起达成率 0.95->0.43)。
    # 4.5 覆盖到 25fps+中值滤波还能分辨的范围, 再往上(5.0)回捞峰开始出现自身双计。
    max_rate_hz: float = 4.5
    min_amplitude: float = 2.0     # 绝对振幅下限(像素)
    amplitude_ratio: float = 0.004 # 相对振幅下限(占框高比例)
    amplitude_ref_height: float = 260.0  # 绝对像素下限对应的框高: 更矮的人按比例缩小这条下限, 0=不缩放
    amplitude_floor_min: float = 0.35    # 按比例缩小后, 绝对下限最多缩到原值的这个比例
    snr_k: float = 3.0             # 振幅阈值 = snr_k * 噪声
    peak_frac: float = 0.08        # 振幅阈值不低于信号峰值的该比例(灵敏度调高时一起放松)
    floor_relief: float = 1.0      # 绝对像素/框高比例下限的缩放, <1 表示放松
    # 阈值上限(45% 峰值)允许把下限压到多低: 0 = 历史行为(可以压到只剩框高比例那条),
    # 1 = 上限不许低于绝对像素下限, 站着不动的人不会再被 ±1px 抖动数出一串起跳。
    threshold_cap_floor_frac: float = 0.0
    refractory_ratio: float = 0.6  # 最小间隔 = refractory_ratio * 周期; 太小时一次起跳会被数成两个
    # 落回地面判据: 两个峰靠得比不应期还近时, 看它们之间有没有真的落回地面
    # (谷底/较小峰高 <= 该比例 = 落回过 = 两次真起跳, 两个峰都保留; 0=关闭, 退回纯间隔一刀切)。
    # 快跳时周期只有六七帧, 纯按间隔扣峰会把真起跳一起扣掉, 这正是"跳得快就少很多"的主因之一。
    # 实测 0.35 档: 计入不变(4818), 双计特征 2.48%->2.30%, 间隔离散度 0.698->0.694 ——
    # 它救回被不应期误杀的快跳峰, 同时把"一次起跳的假肩"合成一个, 两个方向各抵一半, 精度净赚。
    # 再松(0.5)就开始丢计数(-3), 更紧(0.2)与 0.35 同分, 取中间值。
    landing_valley_frac: float = 0.35
    double_under_ratio: float = 0.62  # 间隔小于该比例*周期 -> 双摇过绳
    # 间隔大于该比例*周期 -> 疑似漏计。1.8 时快跳的短空档根本进不了补齐/回捞的视野:
    # 扫到 1.5 计入 +70 次且双计不变(2.51%), 再往前 1.3 已饱和(+70 不再涨), 收到 2.0 反而少 153 次。
    gap_ratio: float = 1.5
    interpolate_gaps: bool = True  # 是否按周期补齐疑似漏计的空档: 遮挡与错位造成的空档是少数的主因, 默认开
    # 局部节奏窗口: 一条轨迹里节奏会漂(实测起跳间隔 lag-1 自相关 +0.31, 115 条里 61 条 >0.25,
    # 50 条前后三段差 >=50%)。拿全局周期当尺子, 快起来的那半段会被判成"没漏", 补不上也回捞不到。
    # 这里取最近 N 个起跳间隔的低分位数(local_period_quantile)当局部周期, 只用于"这一格该不该有起跳/该补几个"的判断;
    # 取峰与不应期仍按全局周期, 免得局部周期一短就把一次起跳的余振数成两次。0 = 关闭。
    # 进窗口的只有"当作正常拍收下"的间隔(<= gap_ratio 倍全局周期), 否则漏计的双倍间隔会把
    # 局部中位数拉长, 尺子反而变钝(实测比不改还少 208 次)。
    local_period_window: int = 8
    # 局部节奏取哪个分位数: 50=中位数, 10=取偏快的那一侧。间隔序列只会因漏计变长、
    # 不会因漏计变短, 所以低分位数才是"这一拍有多长"的无偏估计。
    # 扫档: 50 -> 4853, 25 -> 4905, 10 -> 4961(离线), 且补数的他道作证率不降
    # (45.9% -> 45.6%, out/tune/probe_fillproof.py), 取 10。
    local_period_quantile: float = 10.0
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
    # 回捞峰在原始位移上至少要达到正常阈值的这个比例。0.30 把远景真跳(踝部只有 1~2px)挡在外面:
    # 放到 0.20 计入 +15, 双计与回捞自身双计同向下降(多通道投票仍然把关)。
    recover_raw_ratio: float = 0.20
    recover_min_coverage: float = 0.7  # 回捞时段至少观测到这么多帧: 与盲补分开收, 回捞本身还要信号签字
    recover_gap_ratio: float = 0.75    # 回捞峰与左右邻居的最小间隔 = 该比例 × 周期: 挤进 0.7 倍以内就是双计
    recover_center_tol: float = 0.35   # 回捞峰必须落在左右实测峰的正中(允许偏差 = 该比例 × 空档长): 站着抖的人也凑得出"空档"
    search_back_ratio: float = 1.66    # 空档超过该倍周期才回捞(沿用 R 峰检测的经验值)
    # 空档内逐档压低的幅度。多一档 0.25 让"幅度被压得很低"的空档也有最后一次机会, +10 次且精度不变差。
    search_back_levels: tuple = (0.5, 0.35, 0.25)

    # ---- 参与判定(区分原地跳绳与走动路人) ----
    max_excursion_ratio: float = 1.2   # 水平活动范围超过框高该比例视为在走动
    max_excursion_hard: float = 2.5    # 超过该比例一律剔除: 再同步也不可能是原地跳, 大概率是 ID 在不同人之间来回跳
    drift_window_sec: float = 3.0      # 量水平活动范围的窗口(秒): 逐窗取中位数, 单次 ID 漂移不再放大成"走了几倍身高"
    min_lift_ratio: float = 0.012      # 起跳幅度/框高 下限
    # ---- 头部零位移闸门: 挡住"站着拿绳的观众被造出假节律" ----
    # 复盘 179023471330 #12(93 次): 踝部位移 4.7px 看着像在跳, 头部位移只有 0.85px,
    # 踝-头相关性 -0.14 —— 那是前景行人经过时把框底边/踝一起带出来的波纹, 人根本没离地。
    # 真跳绳者全身一起离地, 头部必然跟着抬(真跳样本头部位移普遍 ≥2px)。
    # 单看"踝-头相关性"会误杀真跳(179023428267 #2 真跳 117 次但相关性只有 0.09),
    # 所以用更窄的合取条件: 头部位移本身在亚像素级 *且* 远小于全身最大位移。
    min_head_move_px: float = 1.0      # 头部去基线位移 p95 低于这条 = 亚像素, 身体没离地(0=关闭闸门)
    min_head_share: float = 0.30       # 同时还要头部位移小于全身最大位移的这个比例, 才判定"只有局部在动"
    min_head_coverage: float = 0.6     # 头部通道至少观测到这么多帧才启用闸门: 头被挡住时不该替他猜没离地
    min_ankle_sync: float = -0.10      # 左右脚踝位移相关性下限(走路为负相关)
    min_ankle_strong: float = 0.85     # 双脚同步起跳的下限: 位置漂移大但双脚同步时只降级不剔除
    require_pose_sync: bool = True     # 有姿态时启用双脚同步性判据
    warmup_sec: float = 1.0        # 每个轨迹开始阶段不计数
    online_window_sec: float = 12.0  # 在线计数的滑动信号窗口(秒)
    min_count_active: int = 5      # 至少连跳这么多次才算"在跳绳", 挡掉偶然抖出来的一两个峰
    # ---- 三态参与判定: 硬指标不过就剔除, 只是幅度/可见率偏弱而节奏清楚就低置信计入 ----
    keep_rhythm_strength: float = 0.35      # 节律强度下限: 达到这条, 幅度/可见率偏弱也只降级为低置信
    # "节律强度"到底量哪一格: band = 频率带内自相关最大值(历史口径), period = 实测周期那一格的自相关。
    # band 的最大值几乎总被最短滞后钉住, 实测 129 条轨迹里 121 条都 ≥0.35 —— 这道闸其实不把关;
    # period 口径只有 80 条合格, 且与"实测起跳间隔的离散度"正相关(0.52), band 口径是 -0.06。
    # 默认仍按 band: 换成 period 会把 41 条轨迹 / 627 次的低置信与补齐资格一并削掉(精度换召回),
    # 与"先把快跳数全"的目标相反。两个数都会写进 stats, 换口径只需改这一项。
    rhythm_measure: str = "band"     # band | period
    min_track_visibility_floor: float = 0.25  # 可见率低到这条以下彻底剔除, 不再低置信计入
    min_count_provisional: int = 2          # 数到的次数不足 min_count_active 时, 至少要有这么多次才允许低置信计入
    max_height_cv: float = 0.18             # 框高变异系数上限: 超过说明这条轨迹的观测在不同人之间跳, 计数不可信

    # ---- 按节奏补齐漏计的空档 ----
    min_fill_coverage: float = 0.7   # 空档区间里至少观测到这么多帧才补: 人本来就不在画面里时不该替他猜
    max_fill_per_gap: int = 3        # 单个空档最多补几个: 节奏估歪时不至于整段虚高
    # (2 -> 3 是在下面这条证据闸门开了之后才放的: 补的每一个都要在空档内部有离地鼓包作证据,
    #  实测 +66 次(4634 -> 4700)而双计特征与间隔离散度都不变。)
    # 补齐的证据闸门: 空档里连一点离地都没有(局部最大值都够不到计数阈值的这个比例), 就是人停下没跳,
    # 不该按节奏凭空补数。目视复盘过一条真跳轨迹: 12.6s 后他停下来弯腰, 位移掉到阈值以下,
    # 补齐却照样在 12.80s / 13.05s 各造了一次起跳 —— 这就是"没跳也被计数"的来源。
    # 0 = 关闭(退回只看观测覆盖率的旧行为)。
    fill_min_lift_frac: float = 0.35
    # 空档取证用几个身体通道的证据: 上面那条闸门只看"选中的那一条通道"有没有离地鼓包,
    # 但远景的人被前排挡住时, 踝部只剩 1~2px 位移而头部/框底边还在明显起落 ——
    # 真起跳时踝/髋/头/框底边一起动, 单通道噪声不会(out/tune/probe_ev.py 实测:
    # 被这条闸门拦下的 450 次里"最大通道证据比"中位数 1.49, 也就是别的通道其实早就不止阈值;
    # 且这些被拦空档的他道旁证率 93%, 与已经放行的那批 94% 持平 —— 拦掉的不是猜, 是漏)。
    # 判据取"第 N 高的通道证据比", 所以 1 = 任一通道达标就补, 2 = 至少两条通道同时达标才补。
    # 0 = 关闭(退回只看主通道, 即 fill_min_lift_frac 的原始口径)。
    # 扫档(离线总数, 配下面 fill_cap_expected=True): 0 -> 4951, 1 -> 5170, 2 -> 5110, 3 -> 4981。
    # 取 1: 双计特征 2.40%、回捞自身双计 0.39%、节奏离散度 0.669 三条精度证据一格没变,
    #       代价是离线计数多 0.3 ms/帧。要更保守就调 2(少 60 次, 取证要两条通道同时点头)。
    fill_evidence_votes: int = 1
    # (取证通道固定为踝/髋/头/框底边四条, 不含框中心: 框中心与框底边同源,
    #  一起投票会让同一个观测算两票, 2 票闸门形同虚设。)
    # 补数占比护栏: 一条轨迹里"按节奏补出来的次数"不许超过实测起跳数的这个比例。
    # 局部节奏尺子越灵(quantile 越低), 找出来的空档越多, 补数就可能反客为主;
    # 这条护栏保证"补"不会反客为主。0 = 关闭。
    # 只对实测起跳 >= 4 次的轨迹生效: 远景被遮挡的人本来就数不出几个峰, 全靠补数达标,
    # 对他们套比例等于把整条抹掉(实测在跳绳 70 -> 69)。
    # 扫档: 0.25 -> 4693, 0.5 -> 4825, 0.75 -> 4928, 1.0 -> 4961(离线)。
    # 取 1.0: 只拦"补的比实数还多"的病态轨迹, 正常轨迹碰不到。
    max_fill_share: float = 1.0
    # 遮挡空档的旁路闸门: 空档里观测覆盖率不够(人被前排挡住/检测丢了几帧)时, 上面那条
    # min_fill_coverage 会直接拒绝补数 —— 但这种情况下空档内部的位移是插值填出来的,
    # 离地取证闸门在这种数据上也没有意义(插值把鼓包抹平了)。改由**相位连续性**签字:
    # 只对**覆盖率不足**的空档生效: 人明明在画面里却一片平静的段落, 两端峰的距离常常正好是
    # 整数个周期(相位残差≈0), 不加这个限定会把"站着没跳"也判成"节奏没断"而凭空补数。
    # 若"空档跨度 ÷ 局部周期"与整数的偏差不超过本参数, 说明出洞后的第一拍仍然落在起跳前的
    # 节拍格上 —— 人被挡住的这段时间里节奏没有断, 他确实还在按同一节奏跳, 该补。
    # 反过来, 如果他中途停下来没跳, 出洞后的第一拍必然落在节拍格外面, 偏差会明显大于本参数。
    # 0 = 关闭(遮挡空档一律不补)。
    fill_hole_phase_tol: float = 0.25
    # 节拍格上认领亚阈值鼓包: 远景的人抬腿在踝部只有 0.3~0.7px, 而计数阈值由"框高比例下限"托底在
    # 0.8px 左右, 于是整段快跳整片卡在阈值下面(out/tune/trace_one.py 里 179023471330 的 9 号:
    # 57 个候选鼓包, 节奏却清楚得很, 节律 0.887)。
    # 判据不是"把阈值再降一点"(那会在别的人身上放进噪声, 已扫过), 而是**踩没踩节拍格**:
    # 噪声尖峰恰好落在格子附近的概率只有约 2×grid_fill_tol, 而真起跳一定在格子上。
    # 所以只认领"落在格子上 + 幅度 >= grid_fill_frac × 阈值 + 与已认峰隔 0.75 个周期以上"的鼓包。
    # 0 = 关闭。
    # 扫档(离线总数 / 双计特征 / 节奏离散度): 0 -> 5536 / 2.40% / 0.669, 0.2 -> 5685 / 2.32% / 0.656,
    # 0.3 -> 5656 / 2.33% / 0.663, 0.5 -> 5604 / 2.35% / 0.678, 0.65 -> 5579 / 2.36% / 0.664。
    # 取 0.3: 再往下的认领幅度已低于噪声本身(0.2 档 = 0.16px 对噪声 0.24px),
    # 而按"新增认领峰的他道作证率"算只有 38.7%, 与按节奏补数(39%)同级、低于包络回捞(44%),
    # 已经接近"信节奏不信信号"的边界, 不该再松。要更激进就调 0.2(多 29 次)。
    grid_fill_frac: float = 0.3
    # 认领的资格门槛: 节奏清晰度(rhythm_clarity, 自相关取在实测周期那一格)要达到这条才允许认领。
    # 比 keep_rhythm_strength=0.35 高一档 —— 这条是"拿幅度换确定性", 节奏不够硬就不该换。
    grid_fill_clarity: float = 0.6
    # 节拍格容差(占周期比例): 候选鼓包与格子中心的偏差不超过这个比例才算"踩中格子"。
    # 概率意义: 随机噪声踩中格子的概率约 2×本参数, 取 0.25 即 50%, 越小越严但漏得越多。
    grid_fill_tol: float = 0.25
    # 补数第二条护栏(与上面那条取更严的一条): 补完之后, 一条轨迹的总次数不许超过
    # "整段可见时长 ÷ 实测周期"这个节奏应有上限。取证闸门放开后补数会变多变密,
    # 这条从总量上封顶, 保证"补"永远不可能把次数抬到比"一直不停跳"还多 —— 
    # 也就是用户口径里的"没跳的时候不计数": 一个人哪怕站着不动, 补数也只能补到节奏上限为止。
    # 实测: 多通道取证取 1 票时离线 5193 但出现 2 条虚高(总数 > 1.05 应有), 加这条后虚高归零。
    fill_cap_expected: bool = True

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
