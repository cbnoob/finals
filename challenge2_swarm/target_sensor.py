"""
Target sensing abstraction for the swarm SEARCH phase.

The mission loop doesn't care whether targets come from a real YOLO model on a
HULA camera feed or from a simulated convoy — it just calls sense() each tick
and save_snapshot() when something new is found.

  YoloTargetSensor  -> real: runs TargetDetector on stream.latest_frame
  SimTargetSensor   -> dry-run: "sees" ground robots within camera footprint
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class SensedTarget:
    confidence: float
    bbox_xyxy: tuple[int, int, int, int] | None = None
    world_n: float | None = None
    world_e: float | None = None
    target_id: int | None = None  # stable id (sim); None for real YOLO


class TargetSensor(Protocol):
    def sense(self, ctx) -> list[SensedTarget]: ...

    def save_snapshot(self, ctx, targets: list[SensedTarget], path: Path) -> int: ...


class YoloTargetSensor:
    """Real sensor: YOLO on the HULA camera frame."""

    def __init__(self, detector, conf: float = 0.4) -> None:
        self.detector = detector
        self.conf = conf
        self._last_frame = None

    def sense(self, ctx) -> list[SensedTarget]:
        stream = ctx.stream
        frame = stream.latest_frame if stream else None
        if frame is None:
            return []
        bgr = frame.to_rgb()
        self._last_frame = bgr
        dets = self.detector.detect(bgr, conf=self.conf)
        return [
            SensedTarget(confidence=d.confidence, bbox_xyxy=d.bbox_xyxy)
            for d in dets
        ]

    def save_snapshot(self, ctx, targets: list[SensedTarget], path: Path) -> int:
        if self._last_frame is None:
            return 0
        from detection.target_detector import Detection

        dets = [
            Detection("robomaster", t.confidence, t.bbox_xyxy or (0, 0, 0, 0))
            for t in targets
        ]
        self.detector.save_snapshot(self._last_frame, path, dets)
        return len(targets)


class SimTargetSensor:
    """
    Dry-run sensor: a ground robot is 'seen' when the drone's UWB position is
    within camera_footprint_m of it. Produces a synthetic snapshot image.
    """

    def __init__(self, uwb, robots, camera_footprint_m: float = 0.35) -> None:
        self.uwb = uwb
        self.robots = robots  # list of GroundRobot
        self.footprint = camera_footprint_m

    def sense(self, ctx) -> list[SensedTarget]:
        n, e, ready = self.uwb.get_tag_ne(ctx.tag_id)
        if not ready:
            return []
        seen: list[SensedTarget] = []
        for robot in self.robots:
            rn, re = robot.position()
            dist = math.hypot(rn - n, re - e)
            if dist <= self.footprint:
                conf = max(0.5, 1.0 - dist / max(self.footprint, 1e-6))
                seen.append(
                    SensedTarget(
                        confidence=round(conf, 2),
                        world_n=rn,
                        world_e=re,
                        target_id=robot.robot_id,
                    )
                )
        return seen

    def save_snapshot(self, ctx, targets: list[SensedTarget], path: Path) -> int:
        import cv2
        import numpy as np

        n, e, _ = self.uwb.get_tag_ne(ctx.tag_id)
        img = np.full((300, 400, 3), 70, dtype=np.uint8)
        cv2.putText(
            img, f"Drone {ctx.tag_id} @ N={n:.2f} E={e:.2f}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )
        for i, t in enumerate(targets):
            cx = 200 + int((t.world_e or 0 - e) * 300)
            cy = 150 - int((t.world_n or 0 - n) * 300)
            cv2.rectangle(img, (cx - 30, cy - 20), (cx + 30, cy + 20), (0, 200, 0), 2)
            cv2.putText(
                img, f"robot{t.target_id} {t.confidence:.2f}", (cx - 35, cy - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1,
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        return len(targets)
