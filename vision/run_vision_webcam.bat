@echo off
cd /d "%~dp0"
echo Starting exp05 vision with computer webcam (14 FPS, 480p) on http://localhost:8107/ ...
python -m vision.service --camera webcam:0 --camera-mount laptop --hfov 69 --max-fps 14 --port 8107 --joints-url "" %*
