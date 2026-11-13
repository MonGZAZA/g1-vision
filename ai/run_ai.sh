#!/usr/bin/env bash
# exp05 AI service (TTS + LLM + speech recognition) on 127.0.0.1:8105. NO_LLM=1 (or a file ai/.no_llm): no LLM (saves
# ~3 GB GPU). NO_ASR=1: no speech recognition (saves ~1.7 GB GPU).
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME="$PWD/models/hf" HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
ARGS=(); [[ "${NO_LLM:-0}" == 1 || -f .no_llm ]] && ARGS+=(--no-llm)  # .no_llm: written by install.sh --no-llm
[[ "${NO_ASR:-0}" == 1 ]] && ARGS+=(--no-asr)
exec .venv/bin/python -u service.py "${ARGS[@]}" "$@"
