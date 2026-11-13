#!/usr/bin/env bash
# exp05 student installer: Python environments + AI models (the large files are downloaded here, not shipped in the zip).
#
#   ./install.sh               full: robot sim + vision + voice (TTS) + speech recognition + offline LLM for Q&A (~19 GB disk, ~30-60 min)
#   ./install.sh --no-llm      the same without the LLM: Q&A answers from the FAQ only                         (~11 GB)
#   ./install.sh --no-ai       robot sim + vision only (motion, pointing, shop editor, people detection; no voice) (~2 GB)
#   ./install.sh --video       also download the example wai video (YouTube, research use only)
#   ./install.sh --check       only run the checks again
#
# Safe to run again: finished steps are skipped (uv and Hugging Face both cache what they already have).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
AI=1; LLM=1; VIDEO=0; ONLY_CHECK=0
for a in "$@"; do
  case "$a" in
    --no-llm) LLM=0 ;;
    --no-ai) AI=0; LLM=0 ;;
    --video) VIDEO=1 ;;
    --check) ONLY_CHECK=1 ;;
    -h|--help) sed -n 2,11p "$0"; exit 0 ;;
    *) echo "unknown option $a (see ./install.sh --help)"; exit 1 ;;
  esac
done

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[error]\033[0m %s\n' "$*"; exit 1; }

# ---------------------------------------------------------------- 1. system
say "1/6 System"
[[ "$(uname -s)" == Linux ]] || die "Linux only (Ubuntu 22.04 / 24.04 x86_64 tested). On Windows try WSL2 (not tested)."
[[ "$(uname -m)" == x86_64 ]] || warn "not x86_64: some wheels (cyclonedds, mediapipe) may be missing"
sudo ip link set lo multicast on 2>/dev/null || true
. /etc/os-release 2>/dev/null && echo "OS: ${PRETTY_NAME:-unknown}"
FREE_GB=$(df -Pk . | awk 'NR==2 {print int($4/1024/1024)}')
NEED=$(( AI == 0 ? 3 : (LLM == 1 ? 21 : 13) ))
echo "free disk here: ${FREE_GB} GB (need ~${NEED} GB)"
(( FREE_GB >= NEED )) || die "not enough disk space (use --no-llm or --no-ai, or unzip somewhere else)"
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/GPU: /'
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  if (( LLM == 1 && VRAM < 6000 )); then warn "GPU has < 6 GB: the LLM may not fit. Consider --no-llm."; fi
else
  echo "GPU: none found (NVIDIA driver). Voice runs on the CPU (slower); the LLM needs an NVIDIA GPU."
  if (( LLM == 1 )); then warn "no NVIDIA GPU: switching to --no-llm (Q&A answers from the FAQ only)"; LLM=0; fi
fi
MISSING=()
command -v paplay >/dev/null || command -v pw-play >/dev/null || command -v aplay >/dev/null || MISSING+=(pulseaudio-utils)
ldconfig -p 2>/dev/null | grep -q 'libGL.so.1' || MISSING+=(libgl1)
ldconfig -p 2>/dev/null | grep -q 'libEGL.so.1' || MISSING+=(libegl1)
command -v curl >/dev/null || MISSING+=(curl)
if (( ${#MISSING[@]} )); then
  warn "missing system packages: ${MISSING[*]}"
  echo "      install them with:  sudo apt install ${MISSING[*]}"
fi

# ---------------------------------------------------------------- 2. uv
say "2/6 uv (Python package manager)"
export PATH="$HOME/.local/bin:$PATH"
export UV_LINK_MODE=copy
if ! command -v uv >/dev/null; then
  command -v curl >/dev/null || die "curl is needed to install uv: sudo apt install curl"
  echo "installing uv into ~/.local/bin (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

if (( ONLY_CHECK == 0 )); then
  # -------------------------------------------------------------- 3. robot env
  say "3/6 Robot environment (g1/: MuJoCo, Unitree SDK, Python 3.10)"
  (cd g1 && uv sync --frozen)

  # -------------------------------------------------------------- vision env + models (2.3)
  say "Vision environment (vision/: MediaPipe pose, YuNet face, lidar tracking; ~1 GB) + models (44 MB)"
  (cd vision && uv sync --frozen)
  mkdir -p vision/models
  MP=https://storage.googleapis.com/mediapipe-models/pose_landmarker
  for m in lite full heavy; do
    f=vision/models/pose_landmarker_$m.task
    if [[ ! -s $f ]]; then
      curl -fL --retry 3 -o "$f.part" "$MP/pose_landmarker_$m/float16/latest/pose_landmarker_$m.task" && mv "$f.part" "$f"
    fi
  done
  f=vision/models/face_detection_yunet_2023mar.onnx
  if [[ ! -s $f ]]; then
    curl -fL --retry 3 -o "$f.part" \
      https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx && mv "$f.part" "$f"
  fi
  ls vision/models

  # -------------------------------------------------------------- 4. AI env
  if (( AI == 1 )); then
    say "4/6 AI environment (ai/: PyTorch CUDA 12.4, TTS, ASR, LLM, MediaPipe; ~7 GB, takes a while)"
    (cd ai && uv sync --frozen)
  else
    say "4/6 AI environment: skipped (--no-ai)"
  fi

  # -------------------------------------------------------------- 5. models
  if (( AI == 1 )); then
    say "5/6 AI models from Hugging Face (into ai/models/hf)"
    mkdir -p ai/models/hf ai/voices
    if (( LLM == 1 )); then rm -f ai/.no_llm; else touch ai/.no_llm; fi  # run_ai.sh then never loads (downloads) the LLM
    HF_HOME="$ROOT/ai/models/hf" LLM=$LLM ai/.venv/bin/python - <<'EOF'
import os
from huggingface_hub import snapshot_download
models = [
    ("VIZINTZOR/MMS-TTS-THAI-FEMALEV2", None, "Thai female voice (VITS, ~0.3 GB)"),
    ("hexgrad/Kokoro-82M", ["config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"], "English voice (Kokoro, ~0.3 GB)"),
    ("typhoon-ai/typhoon-whisper-turbo", ["*.json", "*.txt", "model.safetensors"], "speech recognition, Thai + English (Typhoon Whisper Turbo, ~1.6 GB)"),
]
if os.environ["LLM"] == "1":
    models.append(("scb10x/typhoon2.5-qwen3-4b", None, "Thai/English LLM (Typhoon 2.5, ~7.6 GB)"))
for repo, allow, what in models:
    print(f"  {what}: {repo}", flush=True)
    snapshot_download(repo, allow_patterns=allow)
print("  (the voice-cloning model F5-TTS-THAI, ~1.3 GB, downloads the first time you upload a reference voice)")
EOF
  else
    say "5/6 AI models: skipped (--no-ai)"
  fi

  # -------------------------------------------------------------- optional video
  if (( VIDEO == 1 )); then
    if (( AI == 1 )); then
      say "Example wai video (YouTube x07hnDvNcxM, research use only, not for redistribution)"
      mkdir -p videos/examples
      ai/.venv/bin/yt-dlp -f "bv*[vcodec^=avc1][height<=720]+ba/b[vcodec^=avc1]/b" --merge-output-format mp4 \
        -o "videos/examples/%(id)s.%(ext)s" "https://www.youtube.com/watch?v=x07hnDvNcxM" \
        || warn "video download failed (yt-dlp vs YouTube changes often): try 'uv lock --upgrade-package yt-dlp' in ai/"
    else
      warn "--video needs the AI environment (it uses yt-dlp from ai/.venv); skipped"
    fi
  fi
fi

# ---------------------------------------------------------------- 6. checks
say "6/6 Check (starts its own private sim; ~1 min)"
cd g1
OK=1
.venv/bin/python tools/sim_check.py 2>&1 | tail -1 || OK=0
../vision/.venv/bin/python -c "import mediapipe, cv2; print('vision env ok')" || OK=0
for f in pose_landmarker_full.task face_detection_yunet_2023mar.onnx; do [[ -s ../vision/models/$f ]] || { echo "missing vision/models/$f"; OK=0; }; done
if (( AI == 1 )); then
  ../ai/.venv/bin/python -c "import torch, transformers, kokoro, mediapipe; print('AI env ok, CUDA:', torch.cuda.is_available())" || OK=0
fi
cd ..
if (( OK )); then
  say "Done. Start the simulation with:"
  if (( AI == 0 )); then echo "    NO_AI=1 ./run_sim.sh"; else echo "    ./run_sim.sh"; fi
  (( AI == 1 && LLM == 0 )) && echo "    (LLM not installed: ai/.no_llm is set, Q&A answers from the FAQ only; './install.sh' adds the LLM later)"
  echo "then open http://localhost:8100   (the student guide: STUDENT_GUIDE.html)"
  echo "vision (2.3), in a second terminal:  cd vision && ./run_vision.sh   (or: ./run_vision.sh webcam)  -> http://localhost:8107"
else
  die "a check failed: see the output above and the Troubleshooting section of STUDENT_GUIDE.html"
fi
