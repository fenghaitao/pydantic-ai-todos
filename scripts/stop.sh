#!/bin/bash
# Stop both servers started by start.sh.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$SCRIPT_DIR/.server-pids"

stop_port() {
    local port=$1
    local pids
    pids=$(lsof -ti:"$port" -sTCP:LISTEN 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Stopping listener(s) on port $port (PIDs: $pids)..."
        echo "$pids" | xargs kill -9
    else
        echo "Nothing listening on port $port."
    fi
}

stop_port 8000
stop_port 3000

rm -f "$PIDFILE"
echo "Done."
