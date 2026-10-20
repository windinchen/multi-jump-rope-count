"""合成一段多人跳绳视频, 用于在没有真机素材时端到端验证计数与评估链路。"""

from __future__ import annotations

import json
import math
import os
from typing import List

import cv2
import numpy as np


class Jumper:
    def __init__(self, cx_frac: float, ground_y: float, height: int, period: float, amp: float, phase: float, start: float):
        self.cx_frac = cx_frac
        self.ground_y = ground_y
        self.height = height
        self.period = period
        self.amp = amp
        self.phase = phase
        self.start = start
        self.hidden: tuple = (float("inf"), float("inf"))
        self.sway = 0.0

    def lift(self, t: float) -> float:
        if t < self.start:
            return 0.0
        local = (t - self.start) / self.period
        frac = local - math.floor(local)
        air = 0.55
        if frac > air:
            return 0.0
        u = frac / air
        return 4 * self.amp * u * (1 - u)

    def cycles(self, duration: float) -> List[float]:
        out = []
        k = 0
        while True:
            t = self.start + k * self.period + self.phase
            if t > duration:
                break
            if t >= self.start:
                out.append(round(t, 4))
            k += 1
        return out

    def draw(self, canvas: np.ndarray, t: float, width: int, dx: float, dy: float) -> None:
        if self.hidden[0] <= t <= self.hidden[1]:
            return
        lift = self.lift(t)
        cx = int(self.cx_frac * width + dx + self.sway * math.sin(2 * math.pi * 0.3 * t))
        feet = int(self.ground_y - lift + dy)
        head_r = max(6, int(self.height * 0.11))
        torso_h = int(self.height * 0.42)
        torso_w = int(self.height * 0.26)
        torso_top = feet - int(self.height * 0.62)
        cv2.ellipse(canvas, (cx, feet - int(self.height * 0.8) + torso_h // 2), (torso_w, torso_h), 0, 0, 360, (225, 225, 225), -1)
        cv2.circle(canvas, (cx, torso_top - head_r), head_r, (235, 235, 235), -1)
        spread = int(torso_w * (0.4 + 0.6 * min(1.0, lift / max(self.amp, 1e-6))))
        for sgn in (-1, 1):
            cv2.line(canvas, (cx, feet - int(self.height * 0.42)), (cx + sgn * spread, feet), (215, 215, 215), max(3, int(self.height * 0.05)))
            cv2.line(canvas, (cx + sgn * spread, feet), (cx + sgn * (spread + 6), feet + int(self.height * 0.06)), (200, 200, 200), max(2, int(self.height * 0.04)))


def generate(
    path: str,
    gt_path: str,
    persons: int = 3,
    duration: float = 10.0,
    fps: float = 30.0,
    width: int = 854,
    height: int = 480,
    seed: int = 7,
    shake: bool = True,
    occlude: bool = True,
) -> dict:
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit("无法写出视频, 请检查编码支持")
    jumpers: List[Jumper] = []
    for i in range(persons):
        cx = (i + 0.5) / persons
        jumpers.append(
            Jumper(
                cx_frac=float(np.clip(cx + rng.uniform(-0.03, 0.03), 0.08, 0.92)),
                ground_y=float(height * (0.78 + 0.04 * (i % 2))),
                height=int(height * rng.uniform(0.42, 0.55)),
                period=float(rng.uniform(0.42, 0.62)),
                amp=float(height * rng.uniform(0.055, 0.11)) if i != persons - 1 else float(height * 0.03),
                phase=float(rng.uniform(0, 0.2)),
                start=float(rng.uniform(0.6, 1.4)),
            )
        )
    if occlude and len(jumpers) > 1:
        jumpers[0].hidden = (duration * 0.55, duration * 0.55 + 1.2)

    n_frames = int(round(duration * fps))
    for f in range(n_frames):
        t = f / fps
        canvas = np.full((height, width, 3), 28, dtype=np.uint8)
        cv2.rectangle(canvas, (0, int(height * 0.86)), (width, height), (46, 46, 46), -1)
        dx = 4 * math.sin(2 * math.pi * 0.7 * t) if shake else 0.0
        dy = 3 * math.sin(2 * math.pi * 0.5 * t + 1.0) if shake else 0.0
        for j in jumpers:
            j.draw(canvas, t, width, dx, dy)
        noise = rng.normal(0, 3, canvas.shape)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        writer.write(canvas)
    writer.release()

    gt = {
        "video": os.path.abspath(path),
        "fps": fps,
        "persons": [
            {
                "name": f"P{i + 1}",
                "center_x": round(j.cx_frac, 4),
                "count": len(j.cycles(duration)),
                "events": j.cycles(duration),
            }
            for i, j in enumerate(jumpers)
        ],
    }
    with open(gt_path, "w", encoding="utf-8") as fh:
        json.dump(gt, fh, ensure_ascii=False, indent=2)
    return gt
