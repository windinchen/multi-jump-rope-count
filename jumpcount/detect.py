"""检测层适配: YOLO(检测/姿态) 与一个不依赖 torch 的像素回退检测器。"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .config import Config
from .types import Detection


def _iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def suppress_duplicates(dets: List[Detection], cfg: Config) -> List[Detection]:
    """同一个人被模型出两个框时, 后处理只能留一个, 否则同一次起跳会被数两遍。"""
    if len(dets) < 2:
        return dets
    ordered = sorted(dets, key=lambda d: -d.conf)
    kept: List[Detection] = []
    for det in ordered:
        cx = (det.box[0] + det.box[2]) / 2.0
        h = max(1.0, det.box[3] - det.box[1])
        dup = False
        for other in kept:
            ocx = (other.box[0] + other.box[2]) / 2.0
            oh = max(1.0, other.box[3] - other.box[1])
            if _iou(det.box, other.box) >= cfg.nms_iou:
                dup = True
                break
            # 中心距离近还得框真的压在一起才算重复: 紧挨着站的两个人中心距离可以只有半个身位,
            # 只按距离判重会把旁边那位的框整帧吃掉, 他后面的跳跃就再也没有观测了。
            if (
                abs(cx - ocx) <= cfg.dup_center_ratio * min(h, oh)
                and abs(h - oh) < 0.35 * max(h, oh)
                and _iou(det.box, other.box) >= cfg.dup_min_iou
            ):
                dup = True
                break
        if not dup:
            kept.append(det)
    return kept


class BaseDetector:
    name = "base"
    provides_keypoints = False

    def detect(self, frame: np.ndarray, index: int, time: float) -> List[Detection]:
        raise NotImplementedError


class UltralyticsDetector(BaseDetector):
    """跑自定义 YOLO 权重(检测或 pose 模型皆可)。"""

    def __init__(self, weights: str, cfg: Config):
        from ultralytics import YOLO  # 延迟导入, 未装 torch 时也能用其他检测器

        self.model = YOLO(weights)
        self.cfg = cfg
        self.name = weights
        names = getattr(self.model, "names", {}) or {}
        lowered = {str(v).lower(): k for k, v in names.items()}
        # 自定义模型常常只有一两个类别, 若给了 COCO 的 classes 会全部过滤掉
        self.classes = cfg.classes
        if names and cfg.classes and not any(c in names for c in cfg.classes):
            if "person" in lowered:
                self.classes = (lowered["person"],)
            else:
                self.classes = tuple(sorted(names))

    def detect(self, frame: np.ndarray, index: int, time: float) -> List[Detection]:
        results = self.model.predict(
            frame,
            conf=self.cfg.conf,
            imgsz=self.cfg.imgsz,
            classes=list(self.classes) if self.classes else None,
            device=self.cfg.device,
            verbose=False,
        )
        if not results:
            return []
        res = results[0]
        out: List[Detection] = []
        boxes = getattr(res, "boxes", None)
        kps = getattr(res, "keypoints", None)
        kdata = None
        if kps is not None and getattr(kps, "data", None) is not None:
            kdata = kps.data.cpu().numpy()
        if boxes is None:
            return out
        for i in range(boxes.xyxy.shape[0]):
            box = tuple(float(v) for v in boxes.xyxy[i].tolist())
            conf = float(boxes.conf[i].item())
            k = kdata[i] if kdata is not None and i < kdata.shape[0] else None
            if k is not None and k.shape[0] >= 17 and k[:, 2].max() > 1.0:
                # 部分训练框架把 vis 写成 0/1 索引或概率, 统一压到 [0,1]
                k = k.copy()
                k[:, 2] = np.clip(k[:, 2], 0.0, 1.0)
            out.append(Detection(frame=index, time=time, box=box, conf=conf, keypoints=k))
        return suppress_duplicates(out, self.cfg)


class BlobDetector(BaseDetector):
    """暗背景亮前景的像素级检测器: 用于在没有 torch 时端到端验证计数逻辑。"""

    def __init__(self, cfg: Config, thresh: int = 90, min_area: int = 250):
        self.cfg = cfg
        self.thresh = thresh
        self.min_area = min_area
        self.name = f"blob(thresh={thresh})"

    def detect(self, frame: np.ndarray, index: int, time: float) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(gray, self.thresh, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h < 20 or w > 0.8 * frame.shape[1]:
                continue
            out.append(
                Detection(
                    frame=index,
                    time=time,
                    box=(float(x), float(y), float(x + w), float(y + h)),
                    conf=min(1.0, area / 4000.0),
                )
            )
        return suppress_duplicates(out, self.cfg)


def build_detector(weights: Optional[str], cfg: Config, kind: str = "auto") -> BaseDetector:
    if kind == "blob":
        return BlobDetector(cfg)
    if not weights:
        if kind == "yolo":
            raise SystemExit("需要指定 --weights 才能使用 YOLO 检测器")
        return BlobDetector(cfg)
    return UltralyticsDetector(weights, cfg)
