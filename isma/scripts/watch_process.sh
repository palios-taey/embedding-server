#!/bin/bash
# watch_process.sh — Monitor a background PID and notify via tmux on exit.
#
# Usage:
#   watch_process.sh <pid> <label> [tmux_session]
#
# Example:
#   watch_process.sh 543664 "colbert_pilot" weaver
#
# On success (exit 0): sends green notification to tmux session
# On failure (exit !=0): sends red alert + last 20 lines of log (if log found)

PID=$1
LABEL=${2:-"background_process"}
TMUX_SESSION=${3:-"weaver"}
LOG_FILE="/tmp/${LABEL}.log"

if [ -z "$PID" ]; then
    echo "Usage: $0 <pid> <label> [tmux_session]"
    exit 1
fi

echo "[watch_process] Watching PID $PID ($LABEL) → notify $TMUX_SESSION"

# Wait for process to finish
while kill -0 "$PID" 2>/dev/null; do
    sleep 10
done

# Get exit code (from /proc if still available, otherwise assume gone = done)
wait "$PID" 2>/dev/null
EXIT_CODE=$?

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

if [ $EXIT_CODE -eq 0 ]; then
    MSG="✅ [$TIMESTAMP] $LABEL COMPLETE (pid=$PID, exit=0)"
else
    MSG="❌ [$TIMESTAMP] $LABEL FAILED (pid=$PID, exit=$EXIT_CODE)"
    if [ -f "$LOG_FILE" ]; then
        TAIL=$(tail -20 "$LOG_FILE" | tr '\n' '|')
        MSG="$MSG | Last log: $TAIL"
    fi
fi

# Notify via tmux
tmux send-keys -t "$TMUX_SESSION" "" 2>/dev/null
tmux send-keys -t "$TMUX_SESSION" "echo '$MSG'" Enter 2>/dev/null

# Also push to Redis if available
redis-cli -h 192.168.100.10 -p 6379 \
    LPUSH "weaver:process_notifications" \
    "{\"label\":\"$LABEL\",\"pid\":$PID,\"exit_code\":$EXIT_CODE,\"timestamp\":\"$TIMESTAMP\"}" \
    2>/dev/null

echo "$MSG"
