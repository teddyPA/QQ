#!/bin/bash
# FLIR Tracker — simple launcher
# Double-click this file to start the program.
# (Requires: right-click → Open the first time to bypass Gatekeeper)

SCRIPT_DIR="$HOME/Documents/QuitQuito/Cameras"

cd "$SCRIPT_DIR" || { echo "ERROR: Could not find $SCRIPT_DIR"; read -p "Press Enter to exit..."; exit 1; }

echo "================================================"
echo "   FLIR Mosquito Tracker v2"
echo "   Project: $SCRIPT_DIR"
echo "================================================"
echo ""

# Try to find Python with required packages
if command -v python3 &>/dev/null; then
    python3 flir_capture.py
elif command -v python &>/dev/null; then
    python flir_capture.py
else
    echo "ERROR: Python not found."
    echo "Install Python 3 from https://www.python.org"
fi

echo ""
read -p "Press Enter to close..."
