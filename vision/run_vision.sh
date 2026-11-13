#!/usr/bin/env bash
# exp05 vision: people around the robot on http://localhost:8107 (GET /people, /events, page /).
#   ./run_vision.sh            sim: sensor sim (people + lidar + camera, joints mirrored from the exp05 server :8100)
#                               + vision service. Start exp05's ./run_sim.sh first (else the robot stays in rest pose).
#   ./run_vision.sh robot      real G1: head camera via PC2 + MID-360 via exp04's lidar_relay.sh (UDP 56301)
#   ./run_vision.sh webcam     laptop webcam only (no lidar: distance from the image, rough)
# Options (environment): PRESENCE=1  also POST /api/listen/presence to the exp05 server (2.2 listener gate)
#   SHOP=mockup  booths in the sensor sim     SCENARIO=demo|empty  people in the sensor sim
#   CAMERA=...  CAMERA_MOUNT=level|head_d435|tilt10|laptop|x,y,z,r,p,y   HFOV=69   MODEL=lite|full|heavy   CPU=1
# Ctrl+C stops what this script started.
set -euo pipefail
cd "$(dirname "$0")"
MODE=${1:-sim}
LOG=../logs/vision_$(date +%Y-%m-%d_%H-%M-%S); mkdir -p "$LOG"
ARGS=(--port 8107 --model "${MODEL:-full}" --hfov "${HFOV:-69}")
[[ "${CPU:-0}" == 1 ]] && ARGS+=(--cpu)
[[ "${PRESENCE:-0}" == 1 ]] && ARGS+=(--presence-url http://127.0.0.1:8100/api/listen/presence)
PIDS=()
case "$MODE" in
  sim)
    .venv/bin/python -u -m vision.sim_sensors --shop "../shops/${SHOP:-mockup}/shop.json" --scenario "${SCENARIO:-demo}" \
      --camera-mount "${CAMERA_MOUNT:-level}" --hfov "${HFOV:-69}" --port 8110 --lidar-port 56305 \
      > >(tee "$LOG/sensor_sim.log") 2>&1 &
    PIDS+=($!)
    ARGS+=(--camera "${CAMERA:-http://127.0.0.1:8110/color.jpg}" --camera-mount "${CAMERA_MOUNT:-level}" --lidar-port 56305)
    ;;
  robot)
    # colour: Unitree's camera service on PC2 (port 8000); lidar packets: exp04/g1/deploy/lidar_relay.sh start
    ARGS+=(--camera "${CAMERA:-unitree:http://192.168.123.164:8000/capture}" --camera-mount "${CAMERA_MOUNT:-head_d435}"
           --lidar-port 56301)
    ;;
  webcam)
    ARGS+=(--camera "${CAMERA:-webcam:0}" --camera-mount "${CAMERA_MOUNT:-laptop}" --max-fps 14 --joints-url "")
    ;;
  *) echo "usage: $0 [sim|robot|webcam]"; exit 1 ;;
esac
.venv/bin/python -u -m vision.service "${ARGS[@]}" > >(grep --line-buffered -v "^[WI]0000\|^INFO:\|^WARNING: All\|tensor.cc" | tee "$LOG/vision.log") 2>&1 &
PIDS+=($!)
trap 'kill ${PIDS[*]} 2>/dev/null; wait' INT TERM
echo "[run_vision] $MODE, logs in exp05/logs/$(basename "$LOG"); page http://localhost:8107/"
wait
