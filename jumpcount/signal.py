"""一维垂直位移信号的处理: 平滑、去基线、周期估计、亚帧峰值。"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def median_filter(values: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or values.size < 3:
        return values.astype(np.float64, copy=True)
    win = win if win % 2 else win + 1
    pad = win // 2
    padded = np.pad(values.astype(np.float64), pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, win)
    return np.median(windows, axis=-1)


def moving_average(values: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return values.astype(np.float64, copy=True)
    kernel = np.ones(win, dtype=np.float64) / win
    padded = np.pad(values.astype(np.float64), win // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def running_baseline(values: np.ndarray, win: int) -> np.ndarray:
    """缓慢变化的基线(用于剔除相机抖动、人员身高/透视差异造成的漂移)。"""
    return median_filter(values, win if win % 2 else win + 1)


def detrend(values: np.ndarray, smooth_win: int = 3, baseline_win: int = 91) -> np.ndarray:
    smoothed = median_filter(values, smooth_win)
    if baseline_win > 1:
        smoothed = smoothed - running_baseline(smoothed, baseline_win)
    return moving_average(smoothed, 1)


def mad_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - med)))


def hf_sigma(values: np.ndarray) -> float:
    """帧间抖动级别的噪声: 用一阶差分的稳健标准差, 避免把起跳本身当成噪声。"""
    if values.size < 4:
        return 0.0
    diff = np.diff(values)
    return float(1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2.0))


def autocorr_period(
    values: np.ndarray, fps: float, min_hz: float, max_hz: float
) -> Tuple[Optional[float], float]:
    """用自相关估计主周期, 返回 (周期秒, 归一化强度)。"""
    x = values - values.mean()
    n = x.size
    if n < 8:
        return None, 0.0
    spec = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    if ac[0] <= 0:
        return None, 0.0
    ac = ac / ac[0]
    lo = max(2, int(round(fps / max_hz)))
    hi = min(n - 1, int(round(fps / min_hz)))
    if hi <= lo:
        return None, 0.0
    band = ac[lo : hi + 1]
    idx = int(np.argmax(band))
    lag = lo + idx
    # 抛物线插值细化峰值位置
    if 0 < idx < band.size - 1:
        y0, y1, y2 = band[idx - 1], band[idx], band[idx + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        lag = lag + float(np.clip(shift, -1, 1))
    return lag / fps, float(band[idx])


def parabolic_peak(values: np.ndarray, i: int) -> float:
    if i <= 0 or i >= values.size - 1:
        return float(i)
    y0, y1, y2 = values[i - 1], values[i], values[i + 1]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return float(i)
    return float(i + np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))


def fill_gaps(frames: np.ndarray, values: np.ndarray, max_gap: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """在缺失帧上插值, 返回逐帧对齐的时间轴与信号(不可补处为 NaN)。"""
    if frames.size == 0:
        return np.arange(0), np.zeros(0), 0
    start, stop = int(frames[0]), int(frames[-1])
    axis = np.arange(start, stop + 1)
    dense = np.full(axis.size, np.nan)
    dense[frames - start] = values
    valid = ~np.isnan(dense)
    holes = 0
    if not valid.all():
        idx = np.arange(axis.size)
        dense[~valid] = np.interp(idx[~valid], idx[valid], dense[valid])
        run = 0
        for flag in valid:
            if not flag:
                run += 1
            else:
                if run > max_gap:
                    holes += 1
                run = 0
        if run > max_gap:
            holes += 1
    return axis, dense, holes


def find_peaks(
    values: np.ndarray, min_distance: int, amplitude: float, min_prominence: float
) -> List[int]:
    """极简峰值检测(不依赖 scipy): 局部极大 + 间隔 + 幅度 + 突出度。"""
    n = values.size
    if n < 3:
        return []
    peaks: List[int] = []
    for i in range(1, n - 1):
        if values[i] >= values[i - 1] and values[i] > values[i + 1]:
            peaks.append(i)
    if not peaks:
        return []
    # 幅度过滤
    peaks = [p for p in peaks if values[p] >= amplitude]
    # 突出度过滤: 峰值到两侧最低点的较高者(简化鞍点公式)
    filtered = []
    for p in peaks:
        left_min = values[:p].min() if p > 0 else values[p]
        right_min = values[p + 1 :].min() if p < n - 1 else values[p]
        saddle = max(left_min, right_min)
        if values[p] - saddle >= min_prominence:
            filtered.append(p)
    peaks = filtered
    # 非极大值抑制, 保留更高的峰
    peaks.sort(key=lambda p: -values[p])
    kept: List[int] = []
    for p in peaks:
        if all(abs(p - q) >= min_distance for q in kept):
            kept.append(p)
    return sorted(kept)


def bandpass(values: np.ndarray, fps: float, low_hz: float, high_hz: float) -> np.ndarray:
    """零相位带通: 只留跳绳节奏所在的频段。

    关键点抖动在高频, 走动与透视漂移在低频, 两端一起削掉比再套一层中值平滑干净得多
    —— 中值窗口一长, 先没的总是小幅起跳。用 FFT 做零相位, 不引入 scipy 也不产生相位延迟
    (峰的位置就是起跳时刻, 相位一漂计数就会整体错位)。
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n < 8 or fps <= 0:
        return x.copy()
    nyq = fps / 2.0
    lo, hi = float(low_hz), min(float(high_hz), nyq * 0.9)
    if hi <= lo:
        return x.copy()
    spec = np.fft.rfft(x - x.mean())
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    ramp = max(0.05, 0.3 * lo)
    gain = np.zeros_like(freqs)
    gain[(freqs >= lo) & (freqs <= hi)] = 1.0
    low_ramp = (freqs >= lo - ramp) & (freqs < lo)
    gain[low_ramp] = 0.5 - 0.5 * np.cos(np.pi * (freqs[low_ramp] - (lo - ramp)) / ramp)
    high_ramp = (freqs > hi) & (freqs <= hi + ramp)
    gain[high_ramp] = 0.5 + 0.5 * np.cos(np.pi * (freqs[high_ramp] - hi) / ramp)
    return np.fft.irfft(spec * gain, n=n)


def energy_envelope(values: np.ndarray, fps: float, low_hz: float, high_hz: float,
                    win: int) -> np.ndarray:
    """带通 → 差分 → 平方 → 滑动积分: 把一次起跳聚成一个鼓包。

    平方让"幅度小但整段同向"的相干运动相对噪声占便宜(噪声是零散差分, 起跳是一整段速度),
    滑动积分再把起跳的上升沿与落地沿并成一个包 —— 双计从源头上少一半。
    """
    band = bandpass(values, fps, low_hz, high_hz)
    diff = np.diff(band, prepend=band[0])
    return moving_average(diff * diff, max(1, int(win)))


def nms_by_height(values: np.ndarray, positions, min_distance: int) -> List[int]:
    kept: List[int] = []
    for p in sorted(set(int(v) for v in positions), key=lambda q: -float(values[q])):
        if all(abs(p - q) >= min_distance for q in kept):
            kept.append(p)
    return sorted(kept)


def gap_candidates(values: np.ndarray, base: List[int], min_distance: int, amplitude: float,
                   gap_frames: float, levels=(0.5, 0.35)) -> List[Tuple[int, float]]:
    """在"按节奏本该有起跳却没数到"的空档里, 用逐档压低的阈值取回捞候选。

    把全局阈值一次压到底会把整段噪声都放进来(实测灵敏度推到顶就多两三成);
    只在本该有起跳的那段里压低, 误计风险被关在空档内部。首尾两段一样参与:
    信号整段塌陷时(人还在跳、幅度一路压在阈值下)空档只有一侧有实测峰,
    以前一个都补不出来。候选是否真是起跳由调用方拿原始信号和别的通道复核。

    返回 (位置, 取到它的那一档): 复核必须沿用同一档, 取候选时压到 0.35、
    投票却按原阈值算, 弱起跳会刚被捞出来又被票掉。
    """
    peaks = sorted(int(p) for p in base)
    if not peaks or min_distance <= 0:
        return []
    out: List[Tuple[int, float]] = []
    seen = set(peaks)
    for level in levels:
        amp = amplitude * float(level)
        spans = [(left + min_distance, right - min_distance)
                 for left, right in zip(peaks, peaks[1:]) if right - left > gap_frames]
        spans.append((0, peaks[0] - min_distance))
        spans.append((peaks[-1] + min_distance, values.size))
        for lo, hi in spans:
            if hi - lo < 3:
                continue
            for p in find_peaks(values[lo:hi], min_distance, amp, 0.8 * amp):
                pos = lo + int(p)
                if all(abs(pos - q) >= min_distance for q in seen):
                    out.append((pos, float(level)))
                    seen.add(pos)
    return out


def hf_sigma_masked(values: np.ndarray, observed: np.ndarray) -> float:
    """只用真正观测到的连续片段估计噪声。

    插值补齐的段落被拉平过, 混进来会让噪声假性变小(阈值跟着变松, 抖动被数成起跳);
    缺口两侧那一帧的差分跨越了整段缺席, 混进来又会让噪声虚高(阈值变严, 小幅起跳被切掉)。
    """
    if values.size == 0 or observed.size != values.size:
        return hf_sigma(values)
    runs: List[float] = []
    start = None
    for i in range(values.size + 1):
        inside = i < values.size and bool(observed[i])
        if inside and start is None:
            start = i
        elif not inside and start is not None:
            if i - start >= 6:
                runs.append(hf_sigma(values[start:i]))
            start = None
    if not runs:
        return hf_sigma(values)
    return float(np.median(runs))
