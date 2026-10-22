"""逐条差分两份 tracks_*.tsv, 按抬脚比分组统计损失落在哪一类人身上。

用法: .venv/bin/python out/tune/diff_tracks.py A.tsv B.tsv
  A=参照(计数多的那份) B=对照; 第3列计数, 第9列抬脚比, 第10列阈值, 第6列verdict
"""
from __future__ import annotations

import sys
from collections import defaultdict


def load(path: str) -> dict:
    rows = {}
    for line in open(path, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 8 or not f[0].strip():
            continue
        rows[(f[0], f[1])] = {
            "count": float(f[2]),
            "rate": float(f[3]),
            "verdict": f[5],
            "rhythm": float(f[6]),
            "lift": float(f[8]) if len(f) > 8 else 0.0,
            "thr": float(f[9]) if len(f) > 9 else 0.0,
            "vis": float(f[10]) if len(f) > 10 else 0.0,
            "dur": float(f[11]) if len(f) > 11 else 0.0,
        }
    return rows


def main() -> None:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(a) | set(b))
    groups = defaultdict(lambda: [0, 0.0, 0.0, 0])  # 人数, A计数, B计数, 损失
    buckets = ["<1%", "1~2%", "2~3%", ">=3%"]

    def bucket(lift: float) -> str:
        return buckets[0] if lift < 0.01 else buckets[1] if lift < 0.02 else buckets[2] if lift < 0.03 else buckets[3]

    detail = []
    for k in keys:
        ra, rb = a.get(k), b.get(k)
        ca = ra["count"] if ra else 0.0
        cb = rb["count"] if rb else 0.0
        lift = max(ra["lift"] if ra else 0.0, rb["lift"] if rb else 0.0)
        g = groups[bucket(lift)]
        g[0] += 1
        g[1] += ca
        g[2] += cb
        g[3] += ca - cb
        if abs(ca - cb) >= 1 or (ra is None) != (rb is None):
            detail.append((k, ca, cb, lift, ra["thr"] if ra else -1, rb["thr"] if rb else -1,
                           ra["verdict"] if ra else "-", rb["verdict"] if rb else "-",
                           rb["rate"] if rb else (ra["rate"] if ra else 0)))

    print(f"{'抬脚比':<6}{'人数':>5}{'A计入':>8}{'B计入':>8}{'损失':>7}{'损失%':>8}")
    ta = tb = 0.0
    for bk in buckets:
        g = groups[bk]
        ta += g[1]
        tb += g[2]
        pct = 100 * g[3] / g[1] if g[1] else 0.0
        print(f"{bk:<6}{g[0]:>5}{g[1]:>8.0f}{g[2]:>8.0f}{g[3]:>7.0f}{pct:>7.1f}%")
    print(f"{'合计':<6}{sum(g[0] for g in groups.values()):>5}{ta:>8.0f}{tb:>8.0f}{ta - tb:>7.0f}"
          f"{100 * (ta - tb) / ta if ta else 0:>7.1f}%")
    print("\n逐条差异 (A计数 B计数 抬脚比 A阈值 B阈值 A判定 B判定 频率)")
    for k, ca, cb, lift, ta_, tb_, va, vb, rate in sorted(detail, key=lambda x: -(x[1] - x[2])):
        print(f"  {k[0]} #{k[1]:<4} {ca:>5.0f} {cb:>5.0f} {lift:>7.4f} {ta_:>7.2f} {tb_:>7.2f} {va:<4} {vb:<4} {rate:>5.2f}")


if __name__ == "__main__":
    main()
