#!/usr/bin/env bash
# exp05 on the REAL G1 (S7). Read TESTING_ROBOT.md first: robot standing in Unitree's normal mode, a person with the
# Unitree remote next to it. exp05 only uses rt/arm_sdk (upper body, ramped) + the hand topic + the speaker.
#   NET=enx00e04c80fe28 HANDS=dex3 ./run_robot.sh        (defaults shown; HANDS=inspire once the RH56 are fitted)
set -euo pipefail
cd "$(dirname "$0")/g1"
NET=${NET:-enx00e04c80fe28}
HANDS=${HANDS:-dex3}
export EXP05_HANDS=$HANDS  # the motion (wai, gestures) is solved on the matching hand model
if (exec 3<>/dev/tcp/127.0.0.1/8100) 2>/dev/null; then
  echo "port 8100 is in use (the sim stack, run_sim.sh?): stop it first (Ctrl+C there, or kill -INT its run_sim.sh)"; exit 1
fi
ip link show "$NET" >/dev/null 2>&1 || { echo "network interface $NET not found (USB-Ethernet adapter plugged in?)"; exit 1; }
LOG=../logs/robot_$(date +%Y-%m-%d_%H-%M-%S); mkdir -p "$LOG"
AI=""
if [[ "${NO_AI:-0}" != 1 ]] && ! (exec 3<>/dev/tcp/127.0.0.1/8105) 2>/dev/null; then
  ../ai/run_ai.sh > >(tee "$LOG/ai.log") 2>&1 &
  AI=$!
fi
.venv/bin/python -u deploy/server.py --net "$NET" --hands "$HANDS" > >(tee "$LOG/server.log") 2>&1 &
SRV=$!
trap 'kill -TERM $SRV 2>/dev/null; sleep 3; [[ -n "$AI" ]] && kill $AI 2>/dev/null; wait' INT TERM
echo "[run_robot] $NET, hands $HANDS, logs in exp05/logs/$(basename "$LOG"); UI http://localhost:8100 (speed starts at 0.5)"
wait
