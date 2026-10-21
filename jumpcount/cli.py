"""命令行入口: 生成素材 / 跑计数 / 对答案 / 自检。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .config import Config
from .evaluate import evaluate, load_ground_truth, report_md
from .pipeline import run, write_outputs
from .synthetic import generate


def _coerce(text: str):
    low = text.strip()
    if low.lower() in ("true", "false"):
        return low.lower() == "true"
    for cast in (int, float):
        try:
            return cast(low)
        except ValueError:
            pass
    if low.startswith("["):
        return json.loads(low)
    return low


def _apply_overrides(cfg: Config, overrides: Optional[List[str]]) -> Config:
    for item in overrides or []:
        if "=" not in item:
            raise SystemExit(f"--set 需要 key=value 形式, 收到 {item}")
        key, value = item.split("=", 1)
        if not hasattr(cfg, key):
            raise SystemExit(f"未知配置项 {key}")
        setattr(cfg, key, _coerce(value))
    return cfg


def cmd_gen(args) -> int:
    gt = generate(args.out, args.gt, persons=args.persons, duration=args.duration, seed=args.seed)
    total = sum(p["count"] for p in gt["persons"])
    print(f"已生成 {args.out} ({args.duration:.1f}s), 真值总计 {total} 个: "
          + ", ".join(f"{p['name']}={p['count']}" for p in gt["persons"]))
    print(f"真值标注写入 {args.gt}")
    return 0


def cmd_template(args) -> int:
    from .evaluate import blank_template

    blank_template(args.out, names=args.persons)
    print(f"人工计数模板已写入 {args.out}: 填 count(必填), center_x(建议, 0~1 的水平位置), events(可选, 分号分隔秒)")
    return 0


def cmd_run(args) -> int:
    cfg = Config()
    if args.config:
        cfg = Config.from_json(args.config)
    _apply_overrides(cfg, args.set)
    if args.policy:
        cfg.policy = args.policy
    if args.signal:
        cfg.signal_source = args.signal
    if args.conf is not None:
        cfg.conf = args.conf
    if args.max_rate:
        cfg.max_rate_hz = args.max_rate
    if getattr(args, "interpolate_gaps", None) is not None:
        cfg.interpolate_gaps = bool(args.interpolate_gaps)
    from .detect import build_detector

    detector = build_detector(args.weights, cfg, kind=args.detector)
    summary = run(args.video, cfg, detector, max_frames=args.max_frames, progress=not args.quiet)
    paths = write_outputs(summary, args.out, save_video=not args.no_video)
    print(f"检测器: {detector.name}")
    print(f"视频: {os.path.basename(summary['video'])}  {summary['duration']:.1f}s @ {summary['fps']:.1f}fps  "
          f"({summary['fps_processed']:.1f} fps 处理速度)")
    provisional = summary.get("provisional_tracks", [])
    print(f"参与计数人数: {summary['participants']} (其中低置信 {len(provisional)} 条, 剔除 {len(summary['excluded_tracks'])} 条)  "
          f"总次数: {summary['total_jumps']} = 实测 {summary.get('total_jumps_direct', summary['total_jumps'])}"
          f" + 按节奏补 {summary.get('total_jumps_filled', 0)}  过绳次数: {summary['total_rope_passes']}")
    if summary.get("expected_by_rhythm"):
        print(f"按各自实测节奏应有约 {summary['expected_by_rhythm']:.0f} 次 -> 数到 {summary['rhythm_coverage'] * 100:.0f}%")
    if summary["identity_actions"]:
        dup = sum(1 for a in summary["identity_actions"] if a["action"] == "drop_duplicate")
        stitch = sum(1 for a in summary["identity_actions"] if a["action"] == "stitch_fragment")
        print(f"身份归并: 剔除重复轨迹 {dup} 条, 缝合碎帧 {stitch} 处")
    tags = {"in": "计入  ", "provisional": "低置信", "out": "剔除  "}
    for trk in summary["tracks"]:
        stats = trk.get("stats", {})
        tag = tags.get(stats.get("verdict", "in"), "计入  ")
        print(f"  #{trk['track_id']:>2} {tag} 次数 {trk['count']:>4}(+补 {stats.get('filled', 0):>2}) "
              f"节奏应有 {stats.get('expected_count', 0):>6} "
              f"完整率 {stats.get('rhythm_coverage', 0):>4.2f}  节奏 {trk['rate_hz']:.2f}Hz  信号 {trk['signal_source']:<10} "
              f"信噪比 {trk['snr']:<6} 可见率 {trk['visibility']:.2f}")
    audit = summary["audit"]
    print(f"自检可信度 {audit['score']} ({audit['verdict']}), 问题 {len(audit['issues'])} 条")
    for issue in audit["issues"][:8]:
        tid = issue["track_id"]
        tag = f"#{tid}" if isinstance(tid, int) else ("#" + "/#".join(map(str, tid)) if isinstance(tid, list) else "-")
        print(f"  [{issue['level']:5}] {issue['label']:<12} {tag} {issue['message']}")
    print("输出: " + ", ".join(f"{k}={v}" for k, v in paths.items()))
    return 0


def cmd_eval(args) -> int:
    with open(args.pred, "r", encoding="utf-8") as fh:
        pred = json.load(fh)
    gt = load_ground_truth(args.gt)
    cfg = Config()
    for item in args.set or []:
        key, value = item.split("=", 1)
        setattr(cfg, key, _coerce(value))
    result = evaluate(pred, gt, cfg)
    text = report_md(result)
    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, "accuracy.json")
    md_path = os.path.join(args.out, "accuracy.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"报告: {md_path}\n数据: {json_path}")
    return 0


def cmd_selftest(args) -> int:
    import tempfile

    work = tempfile.mkdtemp(prefix="jumpcount_")
    video = os.path.join(work, "synth.mp4")
    gt_path = os.path.join(work, "gt.json")
    out_dir = os.path.join(work, "out")
    generate(video, gt_path, persons=3, duration=10.0, seed=7)
    args.detector = "blob"
    args.weights = None
    args.video = video
    args.out = out_dir
    args.no_video = True
    args.quiet = True
    args.max_frames = None
    args.config = None
    args.policy = None
    args.signal = None
    args.conf = None
    args.max_rate = None
    args.interpolate_gaps = False
    args.set = None
    cmd_run(args)
    args.pred = os.path.join(out_dir, "counts.json")
    args.gt = gt_path
    result = evaluate(json.load(open(os.path.join(out_dir, "counts.json"), encoding="utf-8")), load_ground_truth(gt_path), Config())
    acc = result.get("count_accuracy") or 0.0
    print(f"\n自检准确率 {acc * 100:.1f}% (阈值 80%)")
    if acc < 0.8:
        print("自检未通过", file=sys.stderr)
        return 1
    print("自检通过")
    return 0


def cmd_inspect(args) -> int:
    """对一个人用多种信号分别计数并画出对照图。"""
    from .detect import build_detector
    from .inspect import compare, plot

    cfg = Config()
    if args.config:
        cfg = Config.from_json(args.config)
    _apply_overrides(cfg, args.set)
    detector = build_detector(args.weights, cfg, kind=args.detector)
    summary = run(args.video, cfg, detector, max_frames=args.max_frames, progress=not args.quiet)
    slots = summary.get("_slot_samples", {})
    targets = [t for t in summary["tracks"] if (args.track is None or t["track_id"] == args.track)]
    os.makedirs(args.out, exist_ok=True)
    report = {}
    for trk in targets[: args.limit]:
        samples = slots.get(trk["track_id"])
        if not samples:
            continue
        one = compare(samples, cfg, summary["fps"], trk["track_id"])
        report[str(trk["track_id"])] = one
        png = os.path.join(args.out, f"signal_check_track{trk['track_id']}.png")
        plot(one, png, fps=summary["fps"], title=f"{os.path.basename(args.video)} 轨迹 #{trk['track_id']} 多信号对照")
        print(f"\n轨迹 #{trk['track_id']} 对照（图: {png}）")
        print(f"  {'信号来源':<12}{'计数':>6}{'过绳':>6}{'节奏Hz':>9}{'信噪比':>9}{'幅度/身高':>10}  周期来源")
        for source, item in one.items():
            print(f"  {source:<12}{item['count']:>6}{item['count_rope']:>6}{item['rate_hz']:>9}{item['snr']:>9}{item['lift_ratio']:>10}  {item['period_source']}")
        pose = {k: v for k, v in one.items() if k in ("ankle", "hip", "head")}
        rates = [i["rate_hz"] for i in pose.values() if i["rate_hz"]]
        if len(rates) >= 2 and max(rates) / max(min(rates), 1e-6) > 1.35:
            print("  ⚠ 各身体部位的节奏差了很多: 高节奏那条大概率把一次起跳数成了两次")
        box = one.get("box_bottom")
        if box and pose:
            best = max((i["count"] for i in pose.values()), default=0)
            if best and box["count"] < 0.6 * best:
                print(f"  ⚠ 框底边只数到 {box['count']} 次(身体部位约 {best} 次): 检测框底边被阴影/地面粘住, 不要用 --signal box_bottom")
    with open(os.path.join(args.out, "signal_check.json"), "w", encoding="utf-8") as fh:
        slim = {k: {s: {key: v for key, v in item.items() if key not in ("frames", "lift")} for s, item in per.items()} for k, per in report.items()}
        json.dump(slim, fh, ensure_ascii=False, indent=2)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jumpcount", description="多人跳绳计数与计数准确性分析")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gen", help="生成合成多人跳绳视频与真值")
    g.add_argument("--out", default="data/synth.mp4")
    g.add_argument("--gt", default="data/synth_gt.json")
    g.add_argument("--persons", type=int, default=3)
    g.add_argument("--duration", type=float, default=10.0)
    g.add_argument("--seed", type=int, default=7)
    g.set_defaults(func=cmd_gen)

    t = sub.add_parser("gt-template", help="生成人工计数标注模板")
    t.add_argument("--out", default="data/gt_template.json")
    t.add_argument("--persons", type=int, default=4)
    t.set_defaults(func=cmd_template)

    r = sub.add_parser("run", help="对一段视频做逐人计数并自检")
    r.add_argument("--video", required=True)
    r.add_argument("--weights", default=None, help="YOLO 权重(.pt), 不填则使用像素回退检测器")
    r.add_argument("--detector", choices=["auto", "yolo", "blob"], default="auto")
    r.add_argument("--out", default="out/run")
    r.add_argument("--config", default=None, help="JSON 配置文件")
    r.add_argument("--set", nargs="*", help="覆盖配置项, 例如 --set max_rate_hz=3.5 policy=rope")
    r.add_argument("--policy", choices=["jump", "rope"], default=None)
    r.add_argument("--signal", choices=["auto", "ankle", "hip", "head", "box_bottom", "box_center"], default=None)
    r.add_argument("--conf", type=float, default=None)
    r.add_argument("--max-rate", type=float, default=None)
    # 默认就是开的, 以前只有 store_true 等于"只能开不能关", 想关只能去改配置文件
    r.add_argument("--interpolate-gaps", dest="interpolate_gaps", action=argparse.BooleanOptionalAction,
                   default=None, help="按节奏补齐疑似漏计空档(默认开, 用 --no-interpolate-gaps 关掉)")
    r.add_argument("--max-frames", type=int, default=None)
    r.add_argument("--no-video", action="store_true", help="不输出标注视频")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("eval", help="对比预测计数与人工计数")
    e.add_argument("--pred", required=True, help="run 输出的 counts.json")
    e.add_argument("--gt", required=True, help="人工标注 JSON 或 CSV")
    e.add_argument("--out", default=None, help="报告输出目录, 默认与 counts.json 同目录")
    e.add_argument("--set", nargs="*", help="覆盖评估配置, 例如 --set tolerance_sec=0.2")
    e.set_defaults(func=cmd_eval)

    s = sub.add_parser("selftest", help="合成视频端到端自检(不需要 YOLO)")
    s.set_defaults(func=cmd_selftest)

    i = sub.add_parser("inspect", help="同一人用脚踝/髋/头/框底边分别计数的对照图")
    i.add_argument("--video", required=True)
    i.add_argument("--weights", default=None)
    i.add_argument("--detector", choices=["auto", "yolo", "blob"], default="auto")
    i.add_argument("--track", type=int, default=None, help="只看某条轨迹, 默认全部")
    i.add_argument("--limit", type=int, default=2, help="最多画几条轨迹")
    i.add_argument("--out", default="out/inspect")
    i.add_argument("--config", default=None)
    i.add_argument("--set", nargs="*")
    i.add_argument("--max-frames", type=int, default=None)
    i.add_argument("--quiet", action="store_true")
    i.set_defaults(func=cmd_inspect)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "eval" and not args.out:
        args.out = os.path.dirname(os.path.abspath(args.pred))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
