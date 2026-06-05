"""
RKNN (NPU) object detector for the mapping drone — combines the organizer's
getDepthAndDetect.py + rknndecoder.py into a reusable class.

rknnlite is imported lazily so this module imports fine on a laptop without an
NPU; only RKNNDetector.__init__ needs the runtime. The decoder (rknn_decoder)
and the deprojection helper are testable without hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from detection.rknn_decoder import decode_yolov11_rknn


@dataclass
class Detection3D:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    center_u: int
    center_v: int
    distance_m: float
    x_m: float
    y_m: float
    z_m: float


def deproject_pixel(fx, fy, cx, cy, u: int, v: int, depth_m: float) -> tuple[float, float, float]:
    """Pinhole back-projection (matches generateTopDown.py / organizer math)."""
    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    return x, y, depth_m


class RKNNDetector:
    def __init__(
        self,
        model_path: str,
        class_names: list[str],
        model_input_size: tuple[int, int] = (640, 640),
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> None:
        from rknnlite.api import RKNNLite

        self.class_names = class_names
        self.model_input_size = model_input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.rknn = RKNNLite()
        if self.rknn.load_rknn(model_path) != 0:
            raise RuntimeError(f"Failed to load RKNN model: {model_path}")
        if self.rknn.init_runtime() != 0:
            raise RuntimeError("Failed to init NPU runtime")

    def infer(self, color_bgr: np.ndarray):
        import cv2

        resized = cv2.resize(color_bgr, self.model_input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.uint8)
        img_input = np.expand_dims(rgb, axis=0)
        outputs = self.rknn.inference(inputs=[img_input])
        return decode_yolov11_rknn(
            outputs,
            img_shape=resized.shape,
            model_input_size=self.model_input_size,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
        )

    def detect_with_depth(self, color_bgr, depth_frame, intrinsics) -> list[Detection3D]:
        """
        depth_frame: pyrealsense2 depth frame (for get_distance)
        intrinsics: pyrealsense2 intrinsics (for rs2_deproject_pixel_to_point)
        """
        import pyrealsense2 as rs

        boxes, scores, classes = self.infer(color_bgr)
        results: list[Detection3D] = []
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.astype(int)
            cu = int((x1 + x2) / 2)
            cv = int((y1 + y2) / 2)
            if not (0 <= cu < intrinsics.width and 0 <= cv < intrinsics.height):
                continue
            distance = depth_frame.get_distance(cu, cv)
            if distance <= 0:
                continue
            point = rs.rs2_deproject_pixel_to_point(intrinsics, [cu, cv], distance)
            cid = int(cls)
            name = self.class_names[cid] if 0 <= cid < len(self.class_names) else f"id_{cid}"
            results.append(
                Detection3D(
                    class_id=cid,
                    class_name=name,
                    confidence=float(score),
                    bbox_xyxy=(x1, y1, x2, y2),
                    center_u=cu,
                    center_v=cv,
                    distance_m=float(distance),
                    x_m=float(point[0]),
                    y_m=float(point[1]),
                    z_m=float(point[2]),
                )
            )
        return results

    def release(self) -> None:
        try:
            self.rknn.release()
        except Exception:
            pass
