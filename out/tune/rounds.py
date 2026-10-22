"""一轮一个假设的单变量扫描: 每个候选只动一个配置项, 串行跑, 输出一张对照表。

用法: .venv/bin/python out/tune/rounds.py            # 跑内置候选清单
     .venv/bin/python out/tune/rounds.py "名字=键=值,键=值" ...   # 只跑指定候选
判读: 计入要涨, 而"两条数重了的直接证据"(双计特征 / 回捞峰自身双计)与虚高条数不许变差;
     在跳绳人数不应靠放宽闸门变化。数值涨了但双计同涨 = 拆人, 不算收益。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from jumpcount.config import Config  # noqa: E402
from quick import evaluate  # noqa: E402

# 名字 -> (假设, 覆盖项)。每项都必须还在文档第 10 节禁区之外。
CANDIDATES = [
    ("band_hi", "带通上限 4.0 低于新的频率上限 4.5, 快跳的包络被削掉", {"envelope_band_hz": (0.7, 5.0)}),
    ("win_ratio", "积分窗 = 0.4×周期, 快跳时只有 2.8 帧, 再短一档", {"envelope_win_ratio": 0.3}),
    ("lead_votes1", "多通道投票从 2 票放到 1 票(旧基线不涨双计, 新速率下复核)", {"envelope_lead_votes": 1}),
    ("searchback3", "空档 search-back 再加一档更低的幅度", {"search_back_levels": (0.5, 0.35, 0.25)}),
    ("refractory", "不应期 0.6 → 0.5, 快跳时不至于吞峰", {"refractory_ratio": 0.5}),
    ("peak_frac", "峰值比例下限 0.08 → 0.05", {"peak_frac": 0.05}),
    ("recover_raw", "回捞峰在原始位移上的下限 0.30 → 0.20", {"recover_raw_ratio": 0.20}),
    ("gap_ratio", "疑似漏计的间隔阈值 1.8 → 1.5, 更早开始找", {"gap_ratio": 1.5}),
    ("smooth1", "中值滤波窗口 3 → 1, 保住快跳的单拍波形", {"smooth_win": 1}),
    ("baseline51", "去基线窗口 91 → 51 帧, 快跳时基线跟得更紧", {"baseline_win": 51}),
    ("env_frac", "包络阈值 0.05 → 0.03 × p95", {"envelope_frac": 0.03}),
    ("rec_gap", "回捞峰与邻居最小间隔 0.75 → 0.8 周期(更严, 试精度的上限)", {"recover_gap_ratio": 0.80}),
    ("combo_recall", "把三项各自都为正的召回项叠起来: 更早开始找空档 + 更低的回捞原始位移下限 + search-back 多一档",
     {"gap_ratio": 1.5, "recover_raw_ratio": 0.20, "search_back_levels": (0.5, 0.35, 0.25)}),
    ("combo_loose", "再叠上多通道投票放到 1 票", {"gap_ratio": 1.5, "recover_raw_ratio": 0.20,
                                                  "search_back_levels": (0.5, 0.35, 0.25), "envelope_lead_votes": 1}),
    ("gap_only", "只动 gap_ratio(单变量已测, 这里作组合的对照锚点)", {"gap_ratio": 1.5}),
    ("gap13", "gap_ratio 再往前一档 1.3, 找拐点", {"gap_ratio": 1.3}),
    ("gap20", "gap_ratio 收到 2.0, 确认 1.8 不是最优", {"gap_ratio": 2.0}),
    # ---- 2026-10-21 补充轮次: 落回地面判据 / 倍频假设 / 一直没扫过的参数 ----
    ("land_off", "关掉落回地面判据(现行默认 0.35 的对照组)", {"landing_valley_frac": 0.0}),
    ("land20", "落回判据更紧: 谷底要低于 0.2 倍计数阈值", {"landing_valley_frac": 0.2}),
    ("land50", "落回判据更松: 谷底放到 0.5 倍计数阈值", {"landing_valley_frac": 0.5}),
    ("land70", "落回判据放到 0.7 倍阈值, 找拐点", {"landing_valley_frac": 0.7}),
    ("octave", "倍频假设: 快跳的真起跳藏在每个周期的中点, 空档触发+不应期+回捞间距一起放松",
     {"search_back_ratio": 1.05, "refractory_ratio": 0.5, "recover_gap_ratio": 0.45}),
    ("octave_sb", "只把空档触发从 1.66 收到 1.05 倍周期(让中点峰进得了候选)", {"search_back_ratio": 1.05}),
    ("octave_rg", "只把回捞最小间距从 0.75 收到 0.45 倍周期", {"recover_gap_ratio": 0.45}),
    ("mincount3", "确认在跳绳只要 3 次(现 5 次): 短轨迹里真跳的人不该被人数门槛剔掉",
     {"min_count_active": 3}),
    ("vis40", "低置信可见率下限 0.5->0.4, 量一下这条边界", {"min_track_visibility": 0.4}),
    ("lead_tol", "多通道投票的时刻容差 0.35->0.5 倍周期", {"envelope_lead_tol": 0.5}),
    ("lead_frac", "投票通道的鼓包阈值 0.06->0.04 倍它自己的 p95", {"envelope_lead_frac": 0.04}),
    ("fill3", "单个空档最多补 2->3 个", {"max_fill_per_gap": 3}),
    ("head_off", "关掉头部零位移闸门: 确认被剔的 122 次是假计数而不是真跳", {"min_head_move_px": 0.0}),
    # ---- 中值滤波 × 落回判据: 上一轮 smooth_win=1 换来 +162 次但双计 3.41%, 那时还没有落回判据 ----
    ("sm1_land35", "中值滤波窗口 3->1 保住快跳的单拍波形, 用落回判据当双计闸", {"smooth_win": 1, "landing_valley_frac": 0.35}),
    ("sm1_land70", "同上, 落回收紧到 0.7 倍阈值", {"smooth_win": 1, "landing_valley_frac": 0.7}),
    ("sm1_pfrac12", "滤波窗口 1 + 峰值比例下限 0.08->0.12, 用阈值把噪声峰压回去",
     {"smooth_win": 1, "peak_frac": 0.12, "landing_valley_frac": 0.35}),
    # ---- 节律度量口径 ----
    ("rhy_period", "节律强度改在「实测周期」那一格上取(带内最大值那条几乎不把关)", {"rhythm_measure": "period"}),
    ("rhy_period20", "同上, 但下限从 0.35 放到 0.20, 试着把召回换回来",
     {"rhythm_measure": "period", "keep_rhythm_strength": 0.20}),
    # ---- 2026-10-21 阈值上限地板轮: 45% 峰值上限会把阈值压到绝对像素下限之下 ----
    ("capfloor50", "阈值上限允许压到绝对下限的一半", {"threshold_cap_floor_frac": 0.5}),
    ("capfloor100", "阈值上限不许低于绝对像素下限(站着不动的人不再被 ±1px 抖动数出起跳)",
     {"threshold_cap_floor_frac": 1.0}),
    ("head20", "头部零位移闸门收紧: 头部位移下限 1.0 -> 2.0px", {"min_head_move_px": 2.0}),
    ("headshare35", "头部零位移闸门收紧: 占比下限 0.30 -> 0.35", {"min_head_share": 0.35}),
    ("capfloor100_head15", "阈值地板 + 头部下限 1.5px 一起上", {"threshold_cap_floor_frac": 1.0, "min_head_move_px": 1.5}),
    # ---- 通道择优: 快跳者被平滑的头部慢摆通道抢走, 一整段快跳只剩十几次 ----
    ("chrate05", "同清晰度档内取计数更多的通道, 档宽 0.05", {"channel_prefer_rate_margin": 0.05}),
    ("chrate10", "同上, 档宽 0.10", {"channel_prefer_rate_margin": 0.10}),
    ("chrate15", "同上, 档宽 0.15", {"channel_prefer_rate_margin": 0.15}),
    ("chrate25", "档宽放到 0.25, 找拐点", {"channel_prefer_rate_margin": 0.25}),
    ("chrate10_cf100", "档宽 0.10 + 阈值上限地板 1.0", {"channel_prefer_rate_margin": 0.10,
                                                        "threshold_cap_floor_frac": 1.0}),
]


def apply(cfg: Config, over: dict) -> Config:
    for key, value in over.items():
        setattr(cfg, key, value)
    return cfg


def main() -> None:
    wanted = set(sys.argv[1:])
    picks = []
    for name, why, over in CANDIDATES:
        if wanted and name not in wanted:
            continue
        picks.append((name, why, over))
    if wanted:
        for item in wanted:
            if "=" not in item:
                continue
            name = item.split("=", 1)[0]
            if name not in {p[0] for p in picks}:
                rest = item.split("=", 1)[1]
                over = {}
                for pair in rest.split(","):
                    k, v = pair.split("=", 1)
                    if v.lower() in ("true", "false"):
                        over[k] = v.lower() == "true"    # 布尔开关, 不能留字符串(非空即真)
                    else:
                        try:
                            over[k] = int(v) if v.lstrip("-").isdigit() else float(v)
                        except ValueError:
                            over[k] = v      # 字符串取值(例如 rhythm_measure=period)
                picks.append((name, "命令行临时候选", over))

    base = evaluate(Config())
    rows = [("基线", base)]
    for name, why, over in picks:
        m = evaluate(apply(Config(), over))
        rows.append((name, m))
        print(f"[done] {name}: {m['total']:.0f} (基线 {base['total']:.0f})  双计 {m['double_rate']:.2%}", flush=True)

    print()
    print(f"{'候选':<14}{'计入':>7}{'Δ':>7}{'Δ%':>7}{'完整率':>8}{'补齐':>6}{'实时':>7}{'双计%':>8}"
          f"{'回捞双计%':>10}{'CV':>7}{'虚高':>5}{'在跳绳':>7}{'回捞':>6}{'快跳条':>7}{'窄档条':>7}")
    for name, m in rows:
        d = m['total'] - base['total']
        print(f"{name:<14}{m['total']:>7.0f}{d:>+7.0f}{100 * d / max(1, base['total']):>+6.1f}%"
              f"{m['coverage']:>8.3f}{m['filled']:>6}{m['online_total']:>7}{100 * m['double_rate']:>8.2f}"
              f"{100 * m['rec_double_rate']:>10.2f}{m['cv']:>7.3f}{m['over']:>5}{m['online_people']:>7}"
              f"{m['recovered']:>6}{m['fast_tracks']:>7}{m['narrow_tracks']:>7}")
    print()
    for name, why, over in picks:
        print(f"  {name}: {why} -> {over}")


if __name__ == "__main__":
    main()
