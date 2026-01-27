#!/bin/bash
# >>> mamba initialize >>>
export MAMBA_EXE='/home/copper/Y/micromamba';
export MAMBA_ROOT_PREFIX='/home/copper/Y';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"
fi
unset __mamba_setup
# <<< mamba initialize <<<
micromamba activate FYP

# --- CONFIGURATION ---
LLAMA_DIR="/media/copper/USB_STICK/Git/llama.cpp/build/bin/"
MODEL_PATH="/media/copper/USB_STICK/Models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
PYTHON_SCRIPT="KnowledgeGraph/Enricher.py"
HOST="localhost"
PORT="8080"
PROJECT_ROOT=$(pwd)
LOG_FILE="$PROJECT_ROOT/server_log.txt"

# 1. Cleanup old processes
echo "🧹 Cleaning up old processes..."
pkill -f "llama-server" || true
# Wait a moment for ports to free up
sleep 2

# 2. Start Llama Server
echo "🚀 Starting llama-server..."
cd "$LLAMA_DIR"

# FIX: Direct redirection ensures SERVER_PID is the actual server, not 'tee'
./llama-server \
    -m "$MODEL_PATH" \
    --host $HOST \
    --port $PORT \
    --ctx-size 12000 \
    --n-gpu-layers 100 \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
cd "$PROJECT_ROOT" # Immediately jump back to root

# 3. Startup Verification
echo "⏳ Waiting for server to initialize (PID: $SERVER_PID)..."
# Tail the log in the background so you can still see startup info
tail -f "$LOG_FILE" --pid=$SERVER_PID &

sleep 10
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ CRITICAL ERROR: Server process died immediately!"
    cat "$LOG_FILE"
    exit 1
fi

# 4. Health Check Loop
SERVER_READY=false
echo "🏥 Checking health endpoint..."
for i in {1..30}; do
    if curl -s "http://$HOST:$PORT/health" > /dev/null; then
        echo "✅ Server is UP and responding!"
        SERVER_READY=true
        break
    fi
    echo -n "."
    sleep 2
done

if [ "$SERVER_READY" = false ]; then
    echo "❌ Timeout waiting for server health check."
    cat "$LOG_FILE"
    kill $SERVER_PID
    exit 1
fi

# 5. Run Python Enrichment
echo "🐍 Starting Python Enrichment Pipeline..."
python3 "$PYTHON_SCRIPT"

# 6. Cleanup
echo "🛑 Enrichment finished. Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
echo "✅ Done."
