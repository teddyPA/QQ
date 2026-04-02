#!/bin/bash
# QQ_Cameras — simple launcher
# Double-click to start. First time: right-click → Open (Gatekeeper).

SCRIPT_DIR="$HOME/Documents/QuitQuito/Cameras"

cd "$SCRIPT_DIR" || {
    echo "ERROR: Folder not found: $SCRIPT_DIR"
    read -p "Press Enter to exit..."; exit 1
}

# Install missing dependencies silently on first run
python3.10 -c "import PIL" 2>/dev/null || {
    echo "Installing Pillow (required for camera preview)..."
    python3.10 -m pip install pillow --quiet
}

echo "================================================"
echo "   QQ_Cameras  v2.1  —  2026-04-01"
echo "================================================"
echo ""

python3.10 flir_capture.py

echo ""
read -p "Press Enter to close..."
