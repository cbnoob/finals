#!/usr/bin/env bash
# Idempotent dependency installer for the mapping drone / Ubuntu 22.04.
#
# Run on the drone computer from the repo root:
#   bash scripts/install_drone_deps.sh
#
# The script checks each import first. If a package is already available, it
# leaves it alone instead of forcing upgrades.

set -u

PYTHON_BIN="${PYTHON_BIN:-python3}"

have_import() {
  "$PYTHON_BIN" - "$1" <<'PY'
import importlib.util
import sys

name = sys.argv[1]
sys.exit(0 if importlib.util.find_spec(name) else 1)
PY
}

install_pip_if_missing() {
  import_name="$1"
  package_name="$2"
  if have_import "$import_name"; then
    echo "OK: $import_name already installed"
    return 0
  fi
  echo "Installing Python package: $package_name"
  "$PYTHON_BIN" -m pip install --user "$package_name"
}

echo "=== Drone dependency installer ==="
echo "Python: $("$PYTHON_BIN" --version)"

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  echo "pip not found; installing python3-pip via apt..."
  sudo apt-get update
  sudo apt-get install -y python3-pip
fi

install_pip_if_missing numpy numpy
install_pip_if_missing yaml PyYAML
install_pip_if_missing cv2 opencv-python
install_pip_if_missing mavsdk mavsdk

if have_import rclpy; then
  echo "OK: rclpy already installed"
else
  if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
  fi

  if have_import rclpy; then
    echo "OK: rclpy available after sourcing ROS Humble"
  else
    echo "Installing ROS2 Python packages for rclpy..."
    sudo apt-get update
    sudo apt-get install -y ros-humble-rclpy ros-humble-geometry-msgs
  fi
fi

if have_import pyrealsense2; then
  echo "OK: pyrealsense2 already installed"
else
  echo "pyrealsense2 missing; trying pip wheel first..."
  if "$PYTHON_BIN" -m pip install --user pyrealsense2; then
    echo "OK: pyrealsense2 installed by pip"
  else
    echo "pip pyrealsense2 failed; trying apt package names if available..."
    sudo apt-get update
    sudo apt-get install -y python3-pyrealsense2 librealsense2-utils || true
  fi
fi

echo
echo "=== Final import check ==="
"$PYTHON_BIN" scripts/check_env.py

echo
echo "If rclpy is still missing, run this before the mission:"
echo "  source /opt/ros/humble/setup.bash"
echo
echo "If pyrealsense2 is still missing but rs-enumerate-devices works, ask the"
echo "organiser which Python environment has librealsense Python bindings enabled."
