# RoboVerse Drone Challenge (University)

Competition code for **Challenge 1** (mapping drone + ArUco landing pads) and **Challenge 2** (3× HULA swarm + convoy detection), built from organizer references:

- `kolomee.py` — UWB + velocity offboard navigation
- `huladola.py` + `dola.py` — swarm discovery and video
- ArUco + depth sample — fiducial + RealSense deprojection

## Repository layout

```
config/challenge.yaml      # IDs, waypoints, gains — edit before competition
common/                    # UWB ROS listener, velocity navigator
detection/                 # ArUco+depth, RealSense, occupancy grid, RKNN/YOLO
challenge1_mapping/        # Mapping mission → landing_pad_report.json
challenge2_swarm/          # Swarm FSM + snapshots
reference/organizer_samples/  # Unmodified organizer sample codes (RealSense, RKNN)
run_challenge1.py
run_challenge2.py
scripts/train_yolo.py      # Train detector on laptop
scripts/aruco_demo.py      # Visual ArUco check (no hardware)
scripts/occupancy_demo.py  # Visual occupancy grid check (no hardware)
```

## Mission handoff: mapping drone → swarm

The mapping drone (Challenge 1) produces the map; the swarm (Challenge 2) consumes it.
`output/challenge1/landing_pad_report.json` carries:
- `valid_landing_zones` — world N/E of valid pads → swarm flies to / lands on these.
- `arena_bounds` — the surveyed N/E extent → swarm's lawnmower search covers exactly
  this area (`swarm.use_map_bounds: true`). Falls back to `swarm.search_area` if no map.

Markers are **20 cm × 20 cm**. With the size known, the ArUco detector falls back to
**pose estimation (`solvePnP`)** when depth has holes — common on flat markers at 3.5 m —
instead of dropping the detection. At 3.5 m a 20 cm marker is only ~50 px wide (1280 px,
D430), so keep resolution high and verify detection in the arena.

## Cameras & flight height (confirmed)

- **Mounting:** down-facing (matches organizer `generateTopDown.py`).
- **Model:** Intel RealSense **D430 or D450**; **resolution is configurable** — set it
  high (config `camera_width/height`, default 1280×720) since targets are far at altitude.
- **Minimum flight height: 3.5 m** — survey/search altitude must stay at or above this
  (`mapping_drone.takeoff_height_m` is 3.7 m).

Footprint = how much ground one frame covers, from `common/camera_model.py`. At 3.5 m:

| Module | Color footprint | Depth footprint | Lawnmower leg spacing | Ground res @1280px |
|---|---|---|---|---|
| D430 | 4.8 × 2.7 m | 6.6 × 3.9 m | ~2.1 m | ~3.8 mm/px |
| D450 | 7.0 × 4.5 m | 6.6 × 3.9 m | ~3.6 m | ~5.5 mm/px |

Consequences baked into the code/config:
- **Search legs are meters apart, not centimeters** — set `swarm.search_spacing_m` from the
  table (e.g. ~2 m for D430), *not* the dry-run placeholder value.
- **Small objects = few pixels.** A 0.2 m marker at 3.5 m is only ~50 px wide at 1280 — use
  high resolution and large ArUco markers, and verify YOLO still fires at that scale.
- A down-facing camera sees the floor at ~3.5 m everywhere; obstacle extraction from the
  occupancy grid needs a height threshold below the floor plane.

## Detection backends

Two different machines run two different detectors — keep them straight:

| | Mapping drone (Challenge 1) | Swarm C2 (Challenge 2) |
|---|---|---|
| Hardware | Rockchip NPU | Windows/Ubuntu laptop |
| Detector | `detection/rknn_detector.py` (`rknnlite` + `rknn_decoder.py`) | `detection/target_detector.py` (`ultralytics`) |
| Fiducials | `detection/aruco_depth.py` (ArUco/QR/AprilTag, no training) | — |
| 3D position | `rs.rs2_deproject_pixel_to_point` (distortion-aware) | no depth on HULA |
| Mapping | `detection/occupancy_grid.py` (top-down grid) | — |

`detection/rknn_decoder.py` is the **YOLOv11** decoder (applies sigmoid to class
scores — required for RKNN exports). The organizer's single-image
`testrknn_with_display.py` is **YOLOv8** style (no sigmoid); match the decoder to
whatever model you export. Originals kept in `reference/organizer_samples/`.

## Before competition day

1. **Confirm ArUco dictionary** with organizers (`DICT_6X6_250` in config).
2. Fill **`valid_marker_ids`** / **`invalid_marker_ids`** in `config/challenge.yaml`.
3. Set **`survey_waypoints`** to cover all landing pads in UWB N/E coordinates (measure in arena).
4. Train YOLO on RoboMaster targets → save to `models/robomaster_best.pt`.
5. Copy this repo to **C2 Terminal** (Ubuntu VM for mapping drone; Windows for swarm).

## Challenge 1 — Mapping drone

**Where:** Mapping drone onboard computer (NoMachine from C2).

**Dependencies (Ubuntu 22.04):** ROS2 Humble, `mavsdk`, `pyrealsense2`, `opencv-python`, `rclpy`.

```bash
cd roboverse-drone-challenge
python3 run_challenge1.py
```

**What it does** (stitched from the organizer samples):

1. Subscribes to UWB (`uwb_tag` topic) — *kolomee.py*
2. Arms, starts offboard, flies survey waypoints using **velocity control** (not position goto) — *kolomee.py*
3. Hovers at each point, grabs aligned RealSense color+depth — *getSyncDepthColor.py*
4. Builds a **top-down occupancy grid** per waypoint — *generateTopDown.py*
5. Detects ArUco, classifies valid/invalid, converts each pad to **world N/E** coordinates — *ArUco sample*

**Outputs** in `output/challenge1/`:
- `landing_pad_report.json` — observations + `valid_landing_zones` (world N/E) for Challenge 2
- `arena_map.png` — top-down map: survey path + valid (green) / invalid (red) pads
- `occupancy_wpNN.png` — per-waypoint occupancy grids

## Challenge 2 — Swarm

**Where:** C2 laptop on the **same WiFi** as HULA drones.

**Dependencies:** `pyhulax`, `opencv-python`, optional `ultralytics` for YOLO.

```bash
python run_challenge2.py
```

**What it does:**

1. `Dola` discovers drone IPs
2. Connects 3 drones, starts video streams
3. Per-drone **state machine**: `TAKEOFF → GO_TO_REGION → SEARCH → SNAPSHOT → RETURN → DONE`
4. **Primary goal = find the ground-robot convoy and snapshot each robot** (see below)

**SEARCH — coverage + multi-target (the scoring objective):**
- The arena (`swarm.search_area`) is split into one **vertical strip per drone** so they don't overlap.
- Each drone flies a **lawnmower (boustrophedon) path** (`challenge2_swarm/search_pattern.py`) so its
  camera passes over every cell of its strip.
- On **every tick** it runs the detector (`challenge2_swarm/target_sensor.py`). When a *new* robot is
  seen it transitions to `SNAPSHOT`, saves an annotated image to `output/snapshots/`, then **resumes
  searching** — so one drone can find several robots.
- Robots already photographed are de-duplicated (`target_dedup_m` / `snapshot_cooldown_s`).
- When a drone's strip is fully covered it `RETURN`s to its landing zone.

The detector is swappable behind one interface: `YoloTargetSensor` (real YOLO on the HULA camera)
vs `SimTargetSensor` (dry-run convoy). The state machine is identical for both.

**SDK docs:** https://pyhulax.xenops.ae

**UWB on C2** (organizer `UWBParserThread.py` → `common/uwb_c2.py`):
- USB serial parser gives each drone tag's `(N, E)` position
- Navigation uses the same P-controller as the mapping drone, mapped to `pyhulax` `move(Direction, speed)`
- Reads ambush/landing targets from Challenge 1 `valid_landing_zones`
- **Not** pyhulax built-in auto-land — optional ArUco near pads is separate visual aid

**Laptop dry-run:**
```bash
python scripts/dry_run_challenge1.py --fast   # produces landing_pad_report.json
python scripts/dry_run_challenge2.py --fast # 3 fake HULAs lawnmower-search a 5-robot convoy
```

The Challenge 2 dry-run spawns a simulated convoy (`challenge2_swarm/sim/ground_robots.py`) and
reports how many of the 5 robots the swarm collectively found, with snapshots in `output/snapshots/`.

## Practice: object detection

- **ArUco / AprilTag / QR:** no training — `detection/aruco_depth.py`
- **RoboMaster bodies:** label data → `python scripts/train_yolo.py`
- **Mapping drone NPU:** export trained model with organizer ONNX/RKNN scripts on Discord

## Testing

Two tiers. **Tier 1** runs on any laptop (no drone/camera/ROS2):

```bash
pip install opencv-contrib-python numpy PyYAML pytest
python -m pytest tests/ -v          # unit tests (incl. full dry-run)
python scripts/aruco_demo.py        # visual ArUco check -> output/aruco_demo.png
python scripts/occupancy_demo.py    # visual occupancy grid -> output/occupancy_demo.png
python scripts/dry_run_challenge1.py --fast   # full simulated mission (~1 min)
```

**Dry-run** (`scripts/dry_run_challenge1.py`) fakes UWB navigation + down-facing
camera, runs the same survey loop as the real drone, and writes:

- `output/challenge1/landing_pad_report.json` (`simulated: true`)
- `output/challenge1/arena_map.png`
- `output/challenge1/occupancy_wpNN.png`
- `output/challenge1/dry_run_preview_wpNN.png` (camera view with ArUco boxes)

**Tier 2** (needs hardware) — bring-up checks on the real machines:

- Mapping drone: confirm UWB topic publishes (`ros2 topic echo /uwb_tag`), MAVSDK connects, RealSense streams, offboard arms.
- Swarm: confirm Dola finds drone IPs on WiFi, `pyhulax` connects, video frames arrive.

## Tuning navigation

Edit `navigation` section in `config/challenge.yaml` (`kp_xy`, `max_vel_xy`, thresholds). Match organizer `kolomee.py` defaults first, then tune in arena.

## Important rules from organizers

- Use **offboard + velocity setpoints**, not MAVSDK position goto for mapping drone.
- Keep sending setpoints while in offboard (see `prime_offboard()`).
- Swarm loop must stay **non-blocking** — use per-drone states, not long `sleep()` for one drone.
