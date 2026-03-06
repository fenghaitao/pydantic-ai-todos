#!/bin/bash
# Start both the agent (port 8000) and UI (port 3000) servers.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$SCRIPT_DIR/.."
PIDFILE="$SCRIPT_DIR/.server-pids"

# Kill processes *listening* on these ports (not just connected clients like browsers)
for port in 8000 3000; do
    pids=$(lsof -ti:"$port" -sTCP:LISTEN 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Killing existing listener(s) on port $port (PIDs: $pids)..."
        echo "$pids" | xargs kill -9
        sleep 0.5
    fi
done

# --- Agent ---
echo "Starting agent on port 8000..."
cd "$ROOT/agent" || exit 1
nohup uv run python -u main.py > /tmp/agent.log 2>&1 &
AGENT_PID=$!

# --- UI ---
echo "Starting UI on port 3000..."
cd "$ROOT" || exit 1
nohup env PORT=3000 npm run dev:ui > /tmp/ui.log 2>&1 &
UI_PID=$!

echo "$AGENT_PID $UI_PID" > "$PIDFILE"
echo "Agent PID: $AGENT_PID  (logs: /tmp/agent.log)"
echo "UI PID:    $UI_PID  (logs: /tmp/ui.log)"
echo "Run scripts/stop.sh to stop both servers."
