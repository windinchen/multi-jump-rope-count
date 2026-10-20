"""内置 HTTP 服务: 提供实时计数页面、状态接口与画面流。"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs, urlparse

from .config import Config, sensitivity_to_gains
from .engine import LiveEngine, list_sources

WEB_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")
NUMERIC = {"conf": float, "imgsz": int, "snr_k": float, "peak_frac": float, "floor_relief": float,
           "max_rate_hz": float, "warmup_sec": float, "min_track_sec": float}


class Handler(BaseHTTPRequestHandler):
    engine: Optional[LiveEngine] = None
    sources = []
    scan_dirs: list = []
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 关掉逐请求日志, 保持终端干净
        return

    # ---- 工具 ----
    def _send(self, code: int, body: bytes, ctype: str, extra=None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _file(self, path: str) -> None:
        if not os.path.isfile(path):
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        with open(path, "rb") as fh:
            self._send(200, fh.read(), ctype)

    # ---- 路由 ----
    def do_GET(self) -> None:
        parts = urlparse(self.path)
        route = parts.path
        if route in ("/", "/index.html"):
            self._file(os.path.join(WEB_ROOT, "index.html"))
        elif route in ("/app.js", "/style.css", "/favicon.svg"):
            self._file(os.path.join(WEB_ROOT, route.lstrip("/")))
        elif route == "/api/state":
            self._json(self.engine.snapshot())
        elif route == "/api/sources":
            current = str(self.engine.state.get("source"))
            # 每次请求现扫目录: 素材往文件夹里一丢, 页面刷新就能看到, 不用重启服务
            extra = [current] if current and not current.isdigit() and os.path.exists(current) else None
            self._json({"sources": list_sources(scan_dirs=self.scan_dirs, extra=extra), "current": current})
        elif route == "/stream":
            self._stream()
        elif route == "/snapshot":
            self._snapshot()
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json({"error": "请求体不是合法 JSON"}, 400)
            return
        if route == "/api/control":
            action = payload.get("action", "")
            kwargs = {}
            for key, cast in NUMERIC.items():
                if payload.get(key) is not None:
                    kwargs[key] = cast(payload[key])
            if payload.get("policy"):
                kwargs["policy"] = str(payload["policy"])
            if payload.get("signal"):
                kwargs["signal_source"] = str(payload["signal"])
            if payload.get("sensitivity") is not None:  # 灵敏度滑杆 1~100 → 三道起跳门限一起松(1 最严, 100 最松)
                kwargs.update(sensitivity_to_gains(payload["sensitivity"]))
            if payload.get("min_visibility") is not None:  # 入围门槛: 轨迹可见率下限
                kwargs["min_track_visibility"] = float(payload["min_visibility"])
            if payload.get("device"):
                kwargs["device"] = str(payload["device"])
            self.engine.control(action, source=payload.get("source"), **kwargs)
            self._json(self.engine.snapshot())
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def _snapshot(self) -> None:
        """单张最新画面: 浏览器吃不下来流时的退路, 由页面按帧率轮询。"""
        blob = self.engine.jpeg
        if not blob:
            self._send(404, b"no frame yet", "text/plain; charset=utf-8")
            return
        self._send(200, blob, "image/jpeg")

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        # 流没有正文长度: 声明连接即边界, Safari 才会边到边画而不是等连接结束
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        last = -1
        try:
            while not self.engine.stop_flag.is_set():
                if self.engine.frame_id != last and self.engine.jpeg:
                    last = self.engine.frame_id
                    blob = self.engine.jpeg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(blob))
                    self.wfile.write(blob)
                    self.wfile.write(b"\r\n")
                else:
                    time.sleep(0.02)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            try:
                self.close_connection = True
            except Exception:
                pass


def serve(args) -> int:
    cfg = Config()
    for item in args.set or []:
        key, value = item.split("=", 1)
        if not hasattr(cfg, key):
            raise SystemExit(f"未知配置项 {key}")
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                value = {"true": True, "false": False}.get(value.lower(), value)
        setattr(cfg, key, value)
    if args.policy:
        cfg.policy = args.policy
    if args.signal:
        cfg.signal_source = args.signal
    if args.imgsz:
        cfg.imgsz = args.imgsz
    if args.conf is not None:
        cfg.conf = args.conf
    if args.device:
        cfg.device = args.device

    engine = LiveEngine(args.source, args.weights, cfg, realtime=not args.no_realtime,
                        jpeg_width=args.stream_width, stream_fps=args.stream_fps)
    engine.start()
    Handler.engine = engine
    Handler.scan_dirs = [str(d) for d in (args.scan or [])]
    Handler.sources = list_sources(scan_dirs=Handler.scan_dirs,
                                   extra=[args.source] if args.source and not str(args.source).isdigit() else None)

    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as exc:
        raise SystemExit(f"端口 {args.port} 已被占用，换一个：--port {args.port + 1}  ({exc})")
    server.daemon_threads = True
    host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    print("=" * 62)
    print(f"  多人跳绳计数页面已启动: http://{host}:{args.port}")
    print(f"  画面源: {args.source}    模型: {args.weights}")
    print(f"  模型加载: {engine.detector.name}    Ctrl+C 退出")
    print("=" * 62)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在退出…")
    finally:
        server.shutdown()
        engine.stop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jumpcount-server", description="多人跳绳实时计数页面")
    p.add_argument("--source", default="0", help="摄像头编号或视频文件路径, 默认本机摄像头")
    p.add_argument("--weights", default=None, help="YOLO 权重路径(.pt), 例如 跳绳素材/yolo26s-pose.pt")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--scan", action="append", default=["跳绳素材"], help="扫描视频素材的目录, 可重复")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--device", default=None, help="cpu / mps / 0")
    p.add_argument("--policy", choices=["jump", "rope"], default=None)
    p.add_argument("--signal", choices=["auto", "ankle", "hip", "head", "box_bottom"], default=None)
    p.add_argument("--stream-width", type=int, default=1152)
    p.add_argument("--stream-fps", type=float, default=15.0,
                   help="页面出图帧率上限, 0=不限; 计数始终用全部原始帧")
    p.add_argument("--no-realtime", action="store_true", help="视频文件不按帧率节流, 尽快跑完")
    p.add_argument("--set", nargs="*", help="覆盖配置项, 例如 --set min_track_sec=3")
    return p


def main(argv=None) -> int:
    return serve(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
