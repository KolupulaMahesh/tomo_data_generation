#!/usr/bin/env bash
# Run a (Isaac) command and watch it: kill the whole process group as soon as one of OUR files shows up in a
# traceback, when the log stops growing for STALL seconds (a hang), or after MAX seconds. Prints the outcome.
#   tools/run_watched.sh <log> <max_s> <stall_s> -- <command ...>
LOG=$1 MAX=$2 STALL=$3
shift 3
[ "$1" = "--" ] && shift
setsid "$@" > "$LOG" 2>&1 &
PID=$!
start=$(date +%s); last_size=0; last_change=$start; why=""
while kill -0 "$PID" 2>/dev/null; do
  sleep 5
  now=$(date +%s); size=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
  if [ "$size" != "$last_size" ]; then last_size=$size; last_change=$now; fi
  if grep -a -q 'File "/home/emagesoft/Desktop/Dataset_gen' "$LOG"; then sleep 3; why="error (traceback in our code)"; break; fi
  if [ $((now - last_change)) -ge "$STALL" ]; then why="no output for ${STALL}s (hung)"; break; fi
  if [ $((now - start)) -ge "$MAX" ]; then why="timeout after ${MAX}s"; break; fi
done
if [ -n "$why" ]; then
  kill -9 -- "-$PID" 2>/dev/null
  echo "[watch] KILLED: $why"
fi
wait "$PID" 2>/dev/null
echo "[watch] finished after $(( $(date +%s) - start ))s"
grep -a -E '^\[(m[0-9]|watch)\]|^Traceback|^[A-Za-z]+Error|  File "/home/emagesoft/Desktop/Dataset_gen' "$LOG" | tail -30 | cut -c1-300
