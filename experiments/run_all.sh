#!/usr/bin/env bash
# Run the planned experiments in order. Sequencing is the shell's job so a
# stalled watcher cannot stall the queue. Markers go to LOG for monitoring;
# full training output stays on the terminal.
set -u
cd /home/dom/Desktop/PvML
LOG=/tmp/pvml-runs.log
: > "$LOG"

RUNS=(
  experiments/tinystories-d32-l4-h16-ctx128-5k.py
  experiments/tinystories-d128-l6-h4-ctx128-5k.py
  experiments/tinystories-d256-l8-h8-ctx128-5k.py
  experiments/tinystories-d32-l4-h16-ctx128-20k.py
  experiments/tinystories-d128-l6-h4-ctx128-20k.py
)

for i in "${!RUNS[@]}"; do
  f="${RUNS[$i]}"
  name=$(basename "$f" .py)
  echo "START $((i+1))/${#RUNS[@]} $name $(date +%H:%M:%S)" | tee -a "$LOG"
  if .venv/bin/python "$f"; then
    echo "DONE $((i+1))/${#RUNS[@]} $name $(date +%H:%M:%S)" | tee -a "$LOG"
  else
    echo "FAILED $((i+1))/${#RUNS[@]} $name exit=$? $(date +%H:%M:%S)" | tee -a "$LOG"
    exit 1
  fi
done
echo "ALL RUNS COMPLETE $(date +%H:%M:%S)" | tee -a "$LOG"
