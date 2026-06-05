"""
ROS2 UWB subscriber — extracted from organizer kolomee.py.

Runs rclpy.spin in a daemon thread so MAVSDK asyncio loop is not blocked.

ROS2 (rclpy / geometry_msgs) is imported lazily inside start_uwb_thread so this
module can be imported and unit-tested on machines without ROS2 installed.
"""

from __future__ import annotations

import threading
from typing import Tuple

# Shared state updated by the ROS callback (set in start_uwb_thread).
_current_n = 0.0
_current_e = 0.0
_ready = False

_uwb_node = None
_ros_thread: threading.Thread | None = None


def _update_position(n: float, e: float) -> None:
    """Called from the ROS callback."""
    global _current_n, _current_e, _ready
    _current_n = n
    _current_e = e
    _ready = True


def get_uwb_position() -> Tuple[float, float, bool]:
    return (_current_n, _current_e, _ready)


def set_simulated_position(n: float, e: float) -> None:
    """Dry-run only: set UWB position without ROS2."""
    _update_position(n, e)


def start_uwb_thread(topic: str = "uwb_tag"):
    """Initialize ROS2 and start the UWB subscriber in a daemon thread."""
    global _uwb_node, _ros_thread

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    class UwbNode(Node):
        def __init__(self) -> None:
            super().__init__("uwb_listener_node")
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
            self.subscription = self.create_subscription(
                PoseStamped, topic, self._callback, qos
            )

        def _callback(self, msg: "PoseStamped") -> None:
            # Organizer mapping: x -> East, y -> North
            _update_position(float(msg.pose.position.y), float(msg.pose.position.x))

    if not rclpy.ok():
        rclpy.init(args=None)
    _uwb_node = UwbNode()
    _ros_thread = threading.Thread(target=rclpy.spin, args=(_uwb_node,), daemon=True)
    _ros_thread.start()
    print("ROS2 UWB subscriber thread started.")
    return _uwb_node


def shutdown_uwb() -> None:
    global _uwb_node
    try:
        import rclpy

        if _uwb_node is not None:
            _uwb_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    except Exception as exc:
        print(f"ROS2 shutdown: {exc}")
    finally:
        _uwb_node = None


async def wait_for_uwb(timeout_s: float = 30.0) -> Tuple[float, float]:
    import asyncio
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        n, e, ready = get_uwb_position()
        if ready:
            return n, e
        await asyncio.sleep(0.2)
    raise TimeoutError("UWB data not ready")
