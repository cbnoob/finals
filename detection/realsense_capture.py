"""Intel RealSense aligned image + depth frames for mapping drone.

D430 modules may expose depth + infrared only, with no RGB color stream. The
capture class therefore tries color first, then falls back to Infrared 1 as a
grayscale image converted to BGR for the ArUco detector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class FramePair:
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    intrinsics: Intrinsics


class RealSenseCapture:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        image_source: str = "auto",
    ) -> None:
        if rs is None:
            raise ImportError("pyrealsense2 not installed")
        self.pipeline = rs.pipeline()
        self.image_source = image_source
        self._image_stream = rs.stream.color
        self._image_format = rs.format.bgr8
        self.align = None

        profile = self._start_pipeline(width, height, fps, image_source)
        image_profile = profile.get_stream(self._image_stream)
        intr = image_profile.as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(
            fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy
        )

    def _start_pipeline(self, width: int, height: int, fps: int, image_source: str):
        sources = [image_source] if image_source != "auto" else ["color", "infrared"]
        last_error: Exception | None = None
        for source in sources:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            try:
                if source == "color":
                    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                    self._image_stream = rs.stream.color
                    self._image_format = rs.format.bgr8
                    self.align = rs.align(rs.stream.color)
                elif source == "infrared":
                    cfg.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
                    self._image_stream = rs.stream.infrared
                    self._image_format = rs.format.y8
                    self.align = None
                else:
                    raise ValueError("image_source must be auto, color, or infrared")
                profile = self.pipeline.start(cfg)
                self.image_source = source
                print(f"RealSense image source: {source}")
                return profile
            except Exception as exc:
                last_error = exc
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
        raise RuntimeError(f"Could not start RealSense streams: {last_error}")

    def get_frames(self) -> FramePair:
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames) if self.align is not None else frames
        depth = aligned.get_depth_frame()
        if self._image_stream == rs.stream.color:
            image = aligned.get_color_frame()
        else:
            image = aligned.get_infrared_frame(1)
        if not depth or not image:
            raise RuntimeError("RealSense frame timeout")
        depth_mm = np.asanyarray(depth.get_data())
        raw_image = np.asanyarray(image.get_data())
        if self._image_stream == rs.stream.color:
            color_bgr = raw_image
        else:
            color_bgr = np.dstack([raw_image, raw_image, raw_image])
        return FramePair(color_bgr=color_bgr, depth_mm=depth_mm, intrinsics=self.intrinsics)

    def stop(self) -> None:
        self.pipeline.stop()
