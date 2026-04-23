#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  install-onnx-local.sh
#  Sets up onnxruntime-web for local development on your laptop.
#
#  Run:  chmod +x install-onnx-local.sh && ./install-onnx-local.sh
#
#  What this does:
#    1. Checks Node.js / npm are available
#    2. Downloads onnxruntime-web via npm (includes the WASM binaries)
#    3. Copies the required files to your Flask static/js/ directory
#    4. Verifies the ONNX model file is in place
#    5. Starts a local dev server that serves the app correctly
#
#  File layout after install:
#    your-project/
#    ├── static/
#    │   ├── js/
#    │   │   ├── ort.min.js            ← ONNX Runtime (minified)
#    │   │   ├── ort-wasm.wasm         ← WASM binary
#    │   │   ├── ort-wasm-simd.wasm    ← WASM SIMD (faster on desktop)
#    │   │   ├── ort-wasm-threaded.wasm
#    │   │   ├── tcn-worker.js         ← copy manually from this package
#    │   │   └── tcn-stress-detector.js
#    │   └── models/
#    │       └── tcn_audio_model.onnx  ← copy from Google Drive
#    └── templates/
#        └── interview.html
# ══════════════════════════════════════════════════════════════════════════════

set -e   # exit on any error
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${GREEN}══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ONNX Runtime Web — Local Setup              ${NC}"
echo -e "${GREEN}══════════════════════════════════════════════${NC}"

# ── 1. Detect project root ────────────────────────────────────────────────────
# Run this script from your Flask project root.
PROJECT_ROOT="$(pwd)"
STATIC_JS="$PROJECT_ROOT/static/js"
STATIC_MODELS="$PROJECT_ROOT/static/models"

echo -e "\n${YELLOW}Project root:${NC} $PROJECT_ROOT"

# ── 2. Check Node.js ──────────────────────────────────────────────────────────
if ! command -v node &> /dev/null; then
  echo -e "${RED}✗ Node.js not found.${NC}"
  echo "  Install from https://nodejs.org (LTS version recommended)"
  echo "  Or via your package manager:"
  echo "    Ubuntu/Debian:  sudo apt install nodejs npm"
  echo "    macOS:          brew install node"
  echo "    Windows:        winget install OpenJS.NodeJS.LTS"
  exit 1
fi
echo -e "${GREEN}✓ Node.js $(node --version)${NC}"

# ── 3. Check Python / Flask ───────────────────────────────────────────────────
if ! command -v python3 &> /dev/null; then
  echo -e "${RED}✗ Python3 not found${NC}" && exit 1
fi
echo -e "${GREEN}✓ Python $(python3 --version)${NC}"

# ── 4. Create directory structure ─────────────────────────────────────────────
echo -e "\n${YELLOW}Creating static directories...${NC}"
mkdir -p "$STATIC_JS"
mkdir -p "$STATIC_MODELS"
echo -e "${GREEN}✓ $STATIC_JS${NC}"
echo -e "${GREEN}✓ $STATIC_MODELS${NC}"

# ── 5. Install onnxruntime-web via npm ────────────────────────────────────────
echo -e "\n${YELLOW}Installing onnxruntime-web...${NC}"

# Create a temporary npm workspace
TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"
npm init -y --quiet > /dev/null 2>&1
npm install onnxruntime-web@1.20.1 --save --quiet

# ── 6. Copy WASM + JS files to static/js/ ────────────────────────────────────
echo -e "\n${YELLOW}Copying ONNX Runtime files to $STATIC_JS...${NC}"

ORT_DIST="$TMP_DIR/node_modules/onnxruntime-web/dist"

# Core files needed for WASM execution in a browser
# Updated loop in Step 6:
for FILE in \
  "ort.min.js" \
  "ort.min.js.map" \
  "ort-wasm.wasm" \
  "ort-wasm-simd.wasm" \
  "ort-wasm-threaded.wasm" \
  "ort-wasm-threaded.mjs" \
  "ort-wasm-simd-threaded.wasm" \
  "ort-wasm-simd-threaded.mjs"  # <--- Add this line
do
  if [ -f "$ORT_DIST/$FILE" ]; then
    cp "$ORT_DIST/$FILE" "$STATIC_JS/$FILE"
    SIZE=$(du -sh "$STATIC_JS/$FILE" | cut -f1)
    echo -e "  ${GREEN}✓${NC} $FILE  ($SIZE)"
  else
    echo -e "  ${YELLOW}⚠${NC} $FILE not found (may not be needed)"
  fi
done

# Also copy the Web Worker-compatible build if it exists
if [ -f "$ORT_DIST/ort.webworker.min.js" ]; then
  cp "$ORT_DIST/ort.webworker.min.js" "$STATIC_JS/ort.webworker.min.js"
  echo -e "  ${GREEN}✓${NC} ort.webworker.min.js"
fi

# Clean up temp dir
cd "$PROJECT_ROOT"
rm -rf "$TMP_DIR"

# ── 7. Check for ONNX model ───────────────────────────────────────────────────
echo -e "\n${YELLOW}Checking for TCN ONNX model...${NC}"
MODEL_PATH="$STATIC_MODELS/tcn_audio_model.onnx"
if [ -f "$MODEL_PATH" ]; then
  SIZE=$(du -sh "$MODEL_PATH" | cut -f1)
  echo -e "${GREEN}✓ tcn_audio_model.onnx found ($SIZE)${NC}"
else
  echo -e "${YELLOW}⚠ tcn_audio_model.onnx not found at $MODEL_PATH${NC}"
  echo "  Download it from Google Drive and place it at:"
  echo "  $MODEL_PATH"
  echo ""
  echo "  In Colab, the file was saved to:"
  echo "  /content/drive/MyDrive/[your-folder]/tcn_audio_model.onnx"
fi

# ── 8. Check tcn-worker.js is in place ───────────────────────────────────────
echo -e "\n${YELLOW}Checking for worker files...${NC}"
for F in "tcn-worker.js" "tcn-stress-detector.js"; do
  if [ -f "$STATIC_JS/$F" ]; then
    echo -e "${GREEN}✓ $F${NC}"
  else
    echo -e "${YELLOW}⚠ $F missing — copy from the downloaded package to $STATIC_JS/${NC}"
  fi
done

# ── 9. Python dependencies ────────────────────────────────────────────────────
echo -e "\n${YELLOW}Installing/verifying Python dependencies...${NC}"
pip install flask flask-cors onnxruntime --quiet
echo -e "${GREEN}✓ Python packages ready${NC}"

# ── 10. Create a minimal local dev server if app.py doesn't exist ─────────────
if [ ! -f "$PROJECT_ROOT/app.py" ]; then
  echo -e "\n${YELLOW}No app.py found — creating minimal dev server...${NC}"
  cat > "$PROJECT_ROOT/app.py" << 'PYEOF'
"""
Minimal local dev server for testing tcn-stress-detector.js
"""
from flask import Flask, send_from_directory, render_template
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

@app.route('/')
def index():
    return render_template('interview.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    print("\n🚀 Dev server running at http://localhost:5000")
    print("   Press Ctrl+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
PYEOF
  echo -e "${GREEN}✓ app.py created${NC}"
fi

# ── 11. Summary ───────────────────────────────────────────────────────────────
echo -e "\n${GREEN}══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete!                              ${NC}"
echo -e "${GREEN}══════════════════════════════════════════════${NC}"

echo -e "\n${YELLOW}Files in $STATIC_JS:${NC}"
ls -lh "$STATIC_JS" 2>/dev/null || echo "  (empty)"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "  1. Copy tcn_audio_model.onnx to $STATIC_MODELS/"
echo "  2. Copy tcn-worker.js to $STATIC_JS/"
echo "  3. Copy tcn-stress-detector.js to $STATIC_JS/"
echo "  4. Add to your HTML template:"
echo '     <script src="/static/js/ort.min.js"></script>'
echo '     <script src="/static/js/tcn-stress-detector.js"></script>'
echo ""
echo "  5. Start the dev server:"
echo "     python3 app.py"
echo ""
echo -e "${YELLOW}IMPORTANT — HTTPS for microphone access:${NC}"
echo "  Chrome and mobile browsers require HTTPS for getUserMedia()."
echo "  For local testing, localhost is exempt (works on http://localhost)."
echo "  For testing on your phone on the same network, use:"
echo ""
echo "    python3 app.py  # then open http://YOUR_LAPTOP_IP:5000"
echo ""
echo "  Or use ngrok for HTTPS tunneling:"
echo "    pip install ngrok"
echo "    ngrok http 5000"
