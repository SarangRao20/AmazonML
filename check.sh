#!/usr/bin/env bash
# Pipeline status. Safe to run any time, edits nothing.
cd "$(dirname "$0")"

# The live inference log is the newest testrun_*.log, NOT pipeline_driver.log:
# the driver is optional and may have failed hours ago while inference is
# still going. Reading a fixed filename is what made this report a dead run
# as "NOT RUNNING" while PID 272767 was happily burning 9 CPU-hours.
# pick the newest log deterministically by mtime, and pick the inference
# PID rather than the watchdog wrapper that merely mentions it on its
# command line
LIVE_LOG=$(find logs -maxdepth 1 -name 'testrun*.log' -printf '%T@ %p\n' 2>/dev/null \
          | sort -rn | head -1 | cut -d' ' -f2-)
INF_PID=$(pgrep -af 'run_test_inference.py' 2>/dev/null \
          | grep -v run_guarded | grep -v zsh | awk '{print $1}' | head -1)
DRV_PID=$(pgrep -f "bash ./run_pipeline.sh" | head -1)
ROWS=$( [ -f output/matching_results.tsv ] && echo $(( $(wc -l < output/matching_results.tsv) - 1 )) || echo 0)
NEED=1732544

if [ -n "$INF_PID" ]; then
  state="RUNNING"
elif [ -n "$DRV_PID" ]; then
  state="DRIVER"
elif [ "$ROWS" -ge "$NEED" ]; then
  state="COMPLETE"
else
  state="STOPPED"
fi

printf 'STATUS: %s\n\n' "$state"

case "$state" in
  RUNNING)
    et=$(ps -o etime= -p "$INF_PID" | tr -d ' ')
    rss=$(ps -o rss= -p "$INF_PID" | awk '{printf "%.2f", $1/1048576}')
    echo "  inference pid $INF_PID   elapsed $et   rss ${rss} GB"
    echo "  log: $LIVE_LOG"
    [ -n "$DRV_PID" ] && echo "  (driver also running, pid $DRV_PID)"
    ;;
  DRIVER)
    echo "  inference not running; a pipeline driver is: $DRV_PID"
    echo "  log: logs/pipeline_driver.log"
    ;;
  COMPLETE)
    echo "  submission assembled: $ROWS rows (expected $NEED)"
    echo "  next: ./run_pipeline.sh validate"
    ;;
  STOPPED)
    echo "  nothing running and no complete submission."
    echo "  partial country files:"
    ls -la output/parts/*.done 2>/dev/null | awk '{print "    "$9}'
    [ -n "$LIVE_LOG" ] && { echo "  last log lines:"; tail -4 "$LIVE_LOG" | sed 's/^/    /'; }
    ;;
esac

# progress, whenever there is a live log
if [ -n "$LIVE_LOG" ]; then
  echo
  line=$(grep -E 'chunk [0-9]+|\[[0-9]/3\]|done:|name_word|name_char|addr:|combo:|exact|BLOCKING' "$LIVE_LOG" 2>/dev/null | tail -3)
  [ -n "$line" ] && { echo "  progress:"; echo "$line" | sed 's/^/    /'; }
  free -g | awk 'NR==2{printf "\n  ram available: %s GB of %s GB\n", $7, $2}'
fi
