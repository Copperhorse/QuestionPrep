#!/bin/bash

# --- CONFIGURATION ---
# Keep your specific paths for the USB stick
LLAMA_DIR="/media/copper/USB_STICK/Git/llama.cpp/build/bin/"
MODEL_PATH="/media/copper/USB_STICK/Models/LFM2.5-1.2B-Instruct-Q8_0.gguf"

# ✅ CHANGED: Point to the new location of the python module
# We don't point to a file anymore; we run the module name
PYTHON_MODULE="qp_pipeline.Enricher"
HOST="localhost"
PORT="8080"
PROJECT_ROOT=$(pwd)
LOG_FILE="$PROJECT_ROOT/data/server_log.txt"

# Ensure log dir exists
mkdir -p "$PROJECT_ROOT/data"

# 1. Cleanup old processes
echo "🧹 Cleaning up old processes..."
pkill -f "llama-server" || true
sleep 2

# 2. Start Llama Server
echo "🚀 Starting llama-server..."
cd "$LLAMA_DIR"

./llama-server \
    -m "$MODEL_PATH" \
    --host $HOST \
    --port $PORT \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
cd "$PROJECT_ROOT"

# 3. Startup Verification
echo "⏳ Waiting for server to initialize (PID: $SERVER_PID)..."
tail -f "$LOG_FILE" --pid=$SERVER_PID &

sleep 10
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ CRITICAL ERROR: Server process died immediately!"
    exit 1
fi

# 4. Health Check Loop
SERVER_READY=false
echo "🏥 Checking health endpoint..."
# 4. Health Check Loop (Wait for model to be READY)
echo "🏥 Waiting for model to finish loading (this may take a while from USB)..."
for i in {1..60}; do
    # We check the 'props' or 'models' endpoint. If it returns 200, we are good.
    # If it returns 503, the model is still loading.
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://$HOST:$PORT/v1/models")

    if [ "$STATUS" -eq 200 ]; then
        echo -e "\n✅ Model is LOADED and ready!"
        SERVER_READY=true
        break
    fi
    echo -n "."
    sleep 3
done
if [ "$SERVER_READY" = false ]; then
    echo "❌ Timeout waiting for server health check."
    cat "$LOG_FILE"
    kill $SERVER_PID
    exit 1
fi

# 5. Run Python Enrichment via uv
# ✅ CHANGED: Runs the module inside the virtual environment
echo "🐍 Starting Python Enrichment Pipeline..."
uv run -m $PYTHON_MODULE

# 6. Cleanup
echo "🛑 Enrichment finished. Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "✅ Done."
