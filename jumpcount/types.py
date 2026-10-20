"""流水线中流动的轻量数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

Box = Tuple[float, float, float, float]


@dataclass
class Detection:
    """单帧内的一个目标（检测框 + 可选关键点）。"""

    frame: int
    time: float
    box: Box
    conf: float
    keypoints: Optional[np.ndarray] = None


@dataclass
class Observation:
    """某个轨迹在某一帧的观测值。"""

    frame: int
    time: float
    box: Box
    signal: float
    conf: float


@dataclass
class JumpEvent:
    frame: float
    time: float
    amplitude: float
    confidence: float
    kind: str = "jump"
    recovered: bool = False   # 幅度不够, 靠能量包络 + 多通道投票找回的起跳


@dataclass
class TrackResult:
    track_id: int
    first_frame: int
    last_frame: int
    count: int
    count_rope: int
    events: List[JumpEvent] = field(default_factory=list)
    period_sec: float = 0.0
    rate_hz: float = 0.0
    signal_source: str = ""
    visibility: float = 0.0
    median_height: float = 0.0
    snr: float = 0.0
    breaks: int = 0
    stats: dict = field(default_factory=dict)
