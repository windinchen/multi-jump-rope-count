#!/bin/sh
set -x
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install numpy opencv-python ultralytics
echo "EXIT=$?"
