"""无需人工标注的自检: 指出每一条计数为什么可信或不可信。"""

from __future__ import annotations

from typing import Dict, List

from .config import Config

LABELS = {
    "TOO_SHORT": "轨迹太短",
    "LOW_SNR": "信号信噪比低",
    "WEAK_RHYTHM": "节律不清晰",
    "NO_PERIOD": "周期估计失败",
    "RATE_OUT_OF_RANGE": "计数频率超出合理区间",
    "TRACK_FRAGMENTED": "轨迹频繁断开",
    "LOW_VISIBILITY": "目标可见率过低",
    "LOW_CONF": "检测置信度偏低",
    "SMALL_TARGET": "目标像素过小",
    "ESTIMATED_MISSES": "存在疑似漏计空档",
    "SPLIT_TRACK": "同一人被拆成多条轨迹",
    "MERGED_TRACK": "两个人被并成一条轨迹",
    "DEGRADED_SOURCE": "关键点缺失, 退化为框底边",
    "CROWD_OVERLAP": "人群重叠严重",
    "AMPLITUDE_CLAMPED": "振幅阈值被压低",
    "TRACK_EXCLUDED": "未参与总计数",
    "WALKER": "走动路人",
    "NO_LIFT": "无明显起跳",
    "PROVISIONAL": "低置信计入",
    "FILLED_GAPS": "按节奏补齐漏计",
    "SIGNAL_SUBSTITUTED": "信号靠替身部位补齐",
}


def _issue(level: str, code: str, track_id, message: str, **extra) -> dict:
    return {
        "level": level,
        "code": code,
        "label": LABELS.get(code, code),
        "track_id": track_id,
        "message": message,
        "extra": extra,
    }


def audit(summary: dict, cfg: Config) -> dict:
    issues: List[dict] = []
    fps = summary.get("fps") or 30.0
    duration = summary.get("duration") or 0.0
    tracks = summary.get("tracks", [])

    if not tracks:
        issues.append(_issue("error", "NO_TRACK", None, "整段视频没有得到任何有效轨迹, 无法计数"))

    for trk in tracks:
        tid = trk["track_id"]
        stats = trk.get("stats", {})
        if stats.get("reject"):
            issues.append(_issue("warn", "TOO_SHORT", tid, f"轨迹仅 {stats.get('sample_count')} 个观测点, 直接放弃计数"))
            continue
        if stats.get("excluded"):
            reason = stats["excluded"]
            code = "WALKER" if "走动" in reason or "走路" in reason else ("NO_LIFT" if "起跳幅度" in reason else "TRACK_EXCLUDED")
            issues.append(_issue("info", code, tid, f"不计入总量({trk['count']} 次已剔除): {reason}"))
            continue
        if stats.get("filled", 0):
            issues.append(_issue(
                "info", "FILLED_GAPS", tid,
                f"另有 {stats['filled']} 个空档按实测节奏补齐(已计入总量, 单列出来便于核对)",
                filled=stats["filled"],
            ))
        if stats.get("verdict") == "provisional":
            issues.append(_issue(
                "warn", "PROVISIONAL", tid,
                f"低置信计入 {trk['count']} 次: {stats.get('provisional_reason') or ''}".rstrip(": "),
                reason=stats.get("provisional_reason"),
            ))
        if stats.get("sample_count", 0) and stats.get("substituted_frames", 0) > 0.3 * stats["sample_count"]:
            issues.append(_issue(
                "info", "SIGNAL_SUBSTITUTED", tid,
                f"{stats['substituted_frames']}/{stats['sample_count']} 帧关键点缺失, 已用其它身体部位平移补齐(台阶已校正)",
                substituted=stats["substituted_frames"],
            ))
        if trk.get("signal_source") == "box_bottom" and cfg.signal_source == "auto":
            issues.append(_issue("info", "DEGRADED_SOURCE", tid, "未取得关键点, 使用检测框底边作为跳跃信号(幅度更小, 易受遮挡影响)"))
        if trk.get("snr", 0) < cfg.snr_k:
            issues.append(_issue("warn", "LOW_SNR", tid, f"信噪比 {trk.get('snr')} < {cfg.snr_k}, 计数量程内噪声可能压过真实起跳", snr=trk.get("snr")))
        if stats.get("period_estimated") is False:
            issues.append(_issue("warn", "NO_PERIOD", tid, "无法用自相关估计跳跃周期, 峰间隔阈值不可靠"))
        elif stats.get("rhythm_strength", 1.0) < 0.25:
            issues.append(_issue("warn", "WEAK_RHYTHM", tid, f"节律强度仅 {stats.get('rhythm_strength')}, 计数阈值退化为人工作业水平"))
        if trk.get("rate_hz") and not (cfg.min_rate_hz * 0.8 <= trk["rate_hz"] <= cfg.max_rate_hz * 1.2):
            issues.append(_issue("warn", "RATE_OUT_OF_RANGE", tid, f"估算节奏 {trk['rate_hz']}Hz 超出 {cfg.min_rate_hz}-{cfg.max_rate_hz}Hz 合理区间"))
        if trk.get("visibility", 1.0) < 0.8:
            issues.append(_issue("warn", "LOW_VISIBILITY", tid, f"可见率 {trk['visibility']} 偏低, 断档期靠插值撑信号"))
        if stats.get("track_mean_conf", 1.0) < 0.5:
            issues.append(_issue("warn", "LOW_CONF", tid, f"平均检测置信度 {stats.get('track_mean_conf')} 偏低"))
        if trk.get("median_height") and trk["median_height"] < 48:
            issues.append(_issue("warn", "SMALL_TARGET", tid, f"框高仅 {trk['median_height']}px, 起跳位移可能不足 2px"))
        if stats.get("long_holes", 0) >= 2:
            issues.append(_issue("warn", "TRACK_FRAGMENTED", tid, f"信号出现 {stats['long_holes']} 段超过 {cfg.max_gap} 帧的空白(遮挡或跟丢), 这些区间的跳跃不可靠"))
        elif stats.get("fragments", 1) >= 2:
            issues.append(_issue("info", "TRACK_FRAGMENTED", tid, f"由 {stats['fragments']} 段碎帧缝合而成, 接缝处的跳跃可能被合并或重复"))
        if stats.get("estimated_misses", 0) >= 2:
            filled = int(stats.get("filled", 0))
            unobserved = int(stats.get("misses_unobserved", 0))
            if cfg.interpolate_gaps and filled:
                # 已经补过了还提示"可用 --interpolate-gaps 补齐"会让人以为没生效
                tail = f"其中 {filled} 个已按实测节奏补齐"
                if unobserved:
                    tail += f", {unobserved} 个因当时不在画面里无法补"
                issues.append(_issue(
                    "info", "ESTIMATED_MISSES", tid,
                    f"检测到 {stats['estimated_misses']} 次超过 {cfg.gap_ratio} 倍周期的空档, {tail}",
                    estimated_misses=stats["estimated_misses"], filled=filled, misses_unobserved=unobserved,
                ))
            else:
                issues.append(_issue(
                    "warn", "ESTIMATED_MISSES", tid,
                    f"检测到 {stats['estimated_misses']} 次超过 {cfg.gap_ratio} 倍周期的空档, 疑似漏计(可用 --interpolate-gaps 补齐)",
                    estimated_misses=stats["estimated_misses"],
                ))
        if stats.get("peak_level_p95", 0) <= stats.get("amplitude_threshold", 0) * 1.02 and stats.get("sample_count", 0) > 30:
            issues.append(_issue("info", "AMPLITUDE_CLAMPED", tid, "振幅阈值贴近信号幅值, 小幅起跳可能被整段丢弃"))

    issues.extend(_check_split_merge(tracks))

    crowd = summary.get("crowding", {})
    if crowd.get("p95_max_iou", 0) > 0.45:
        issues.append(_issue("warn", "CROWD_OVERLAP", None, f"框重叠 95 分位达 {crowd['p95_max_iou']}, 跟踪 ID 极易互串, 建议降低机位遮挡或换 ReID 关联"))

    severity = {"error": 0, "warn": 1, "info": 2}
    issues.sort(key=lambda x: severity.get(x["level"], 3))
    score = _reliability(issues, tracks)
    return {
        "score": score,
        "verdict": "可靠" if score >= 80 else ("基本可用" if score >= 60 else "存疑"),
        "issues": issues,
        "counts_by_code": _by_code(issues),
    }


def _by_code(issues: List[dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in issues:
        out[it["code"]] = out.get(it["code"], 0) + 1
    return out


def _reliability(issues: List[dict], tracks: List[dict]) -> float:
    if not tracks:
        return 0.0
    penalty = 0.0
    for it in issues:
        penalty += {"error": 35.0, "warn": 9.0, "info": 2.0}.get(it["level"], 0.0)
    return round(max(0.0, min(100.0, 100.0 - penalty / max(1, len(tracks)) * 1.6)), 1)


def _alternates(a: dict, b: dict) -> bool:
    """一条轨迹刚结束另一条就在同一位置冒出来, 是 ID 来回切换的特征。"""
    gap_ab = abs(b["first_frame"] - a["last_frame"])
    gap_ba = abs(a["first_frame"] - b["last_frame"])
    return min(gap_ab, gap_ba) <= 30


def _check_split_merge(tracks: List[dict]) -> List[dict]:
    """同一人被拆成两条轨迹 -> 总体多计; 两人并一条 -> 总体少计。"""
    issues: List[dict] = []
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            a, b = tracks[i], tracks[j]
            overlap = min(a["last_frame"], b["last_frame"]) - max(a["first_frame"], b["first_frame"])
            span_a = max(1, a["last_frame"] - a["first_frame"])
            span_b = max(1, b["last_frame"] - b["first_frame"])
            if overlap <= 0:
                continue
            frac = overlap / max(span_a, span_b)
            dx = abs(a["stats"].get("center_x_mean", 0) - b["stats"].get("center_x_mean", 0))
            ref = max(1.0, min(a.get("median_height", 1.0), b.get("median_height", 1.0)))
            if frac > 0.5 and dx < 0.45 * ref:
                issues.append(_issue(
                    "error", "SPLIT_TRACK", [a["track_id"], b["track_id"]],
                    f"轨迹 {a['track_id']} 与 {b['track_id']} 时间高度重叠且水平位置仅差 {dx:.1f}px, 很可能是同一个人被数了两次",
                    overlap_frac=round(frac, 3), dx=round(dx, 2),
                ))
            elif frac > 0.75 and dx > 3.0 * ref and _alternates(a, b):
                issues.append(_issue(
                    "warn", "MERGED_TRACK", [a["track_id"], b["track_id"]],
                    f"轨迹 {a['track_id']} 与 {b['track_id']} 首尾相接反复交替出现, 同一个人的 ID 在来回切换, 交接处会漏计或重计",
                    overlap_frac=round(frac, 3), dx=round(dx, 1),
                ))
    return issues
