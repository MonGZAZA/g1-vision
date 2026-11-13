#!/usr/bin/env bash
# exp05 in simulation: sim robot (MuJoCo window) + operator server on http://localhost:8100. Ctrl+C stops both.
#   HEADLESS=1 ./run_sim.sh        no MuJoCo window
#   SHOP=mockup ./run_sim.sh       booths of shops/<SHOP>/shop.json around the robot (default mockup)
#   NO_AI=1 ./run_sim.sh           without the AI service (voice, Q&A); NO_LLM=1: voice only (saves ~3 GB GPU)
#   HANDS=inspire ./run_sim.sh     Inspire 5-finger hands (default dex3 = the robot now)
set -euo pipefail
cd "$(dirname "$0")/g1"
SHOP=${SHOP:-mockup}
export EXP05_HANDS=${HANDS:-dex3}  # read by the sim, the server and the motion code (deploy/robot_model.py)
LOG=../logs/$(date +%Y-%m-%d_%H-%M-%S); mkdir -p "$LOG"
SIM_ARGS=(--shop "../shops/$SHOP/shop.json"); [[ "${HEADLESS:-0}" == 1 ]] && SIM_ARGS+=(--headless)
.venv/bin/python -u deploy/sim_robot.py "${SIM_ARGS[@]}" > >(tee "$LOG/sim.log") 2>&1 &
SIM=$!
AI=""
if [[ "${NO_AI:-0}" != 1 ]] && ! (exec 3<>/dev/tcp/127.0.0.1/8105) 2>/dev/null; then
  ../ai/run_ai.sh > >(tee "$LOG/ai.log") 2>&1 &
  AI=$!
fi
sleep 2
.venv/bin/python -u deploy/server.py > >(tee "$LOG/server.log") 2>&1 &
SRV=$!
trap 'kill -TERM $SRV 2>/dev/null; sleep 3; kill $SIM $AI 2>/dev/null; wait' INT TERM
echo "[run_sim] hands $EXP05_HANDS, logs in exp05/logs/$(basename "$LOG"); UI http://localhost:8100"
wait
