"""频率上限扫描: 跳得越快是否越数不出来, 以及是哪一道闸门先卡住。

用法: .venv/bin/python out/tune/sweep_rate.py [key=value ...]
合成信号从 0.8Hz 扫到 6Hz, 对照"应有次数"; 再单独拆出 max_rate_hz / refractory / policy 的影响。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

from jumpcount.config import Config  # noqa: E402
from jumpcount.counter import count_track, participation  # noqa: E402


def kp(x, y, lift, conf=0.9):
    arr = np.zeros((17, 3))
    arr[:, :2] = (x, y)
    arr[:, 2] = conf
    arr[0] = (x, y - 270.0 - lift, conf)
    arr[11] = (x, y - 150.0 - lift, conf)
    arr[12] = (x + 10, y - 150.0 - lift, conf)
    arr[15] = (x - 12, y - 20.0 - lift, conf)
    arr[16] = (x + 12, y - 20.0 - lift, conf)
    return arr


def samples(fps, duration, rate, amp, noise=0.6, seed=1, box_h=260.0):
    rng = np.random.default_rng(seed)
    period = 1.0 / rate
    n = int(fps * duration)
    out = []
    for i in range(n):
        t = i / fps
        frac = (t % period) / period
        lift = 4 * amp * (frac / 0.55) * (1 - frac / 0.55) if frac <= 0.55 else 0.0
        lift += rng.normal(0, noise)
        y_bottom = 800.0 - lift
        out.append((i, t, (600.0, y_bottom - box_h, 700.0, y_bottom), None, kp(650.0, y_bottom, lift)))
    return out


def main() -> None:
    cfg = Config()
    over = {}
    for item in sys.argv[1:]:
        key, value = item.split("=", 1)
        low = value.lower()
        cast = low == "true" if low in ("true", "false") else None
        if cast is None:
            try:
                cast = int(value)
            except ValueError:
                try:
                    cast = float(value)
                except ValueError:
                    cast = value
        setattr(cfg, key, cast)
        over[key] = cast
    fps = 25.0
    print(f"fps={fps} 时长=20s 幅度=框高的3% 配置: {over or '默认'}  "
          f"(max_rate_hz={cfg.max_rate_hz} refractory={cfg.refractory_ratio} policy={cfg.policy} smooth={cfg.smooth_win})")
    print(f"{'真实Hz':>7}{'应有':>7}{'实测':>7}{'补齐':>6}{'总计':>7}{'bump':>6}{'达成率':>8}{'估Hz':>7}{'周期源':>12}{'判定':>6}")
    for rate in (0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.5, 5.0, 6.0):
        s = samples(fps, 20.0, rate, amp=0.03 * 260.0)
        res = count_track(1, s, cfg, fps)
        expected = rate * 20.0
        total = res.count + res.stats.get("filled", 0)
        verdict = participation(res.stats, cfg)[0]
        print(f"{rate:>7.1f}{expected:>7.0f}{res.count:>7}{res.stats.get('filled', 0):>6}{total:>7}"
              f"{res.stats.get('bumps', 0):>6}{total / expected:>8.2f}{res.rate_hz:>7.2f}"
              f"{str(res.stats.get('period_source'))[:11]:>12}{verdict:>6}")


if __name__ == "__main__":
    main()
