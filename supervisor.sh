#!/bin/zsh
# Night supervisor: wakes the agent on terminal states. Polls every 10 min
# until 11:30 IST. Exit code carries the verdict (notification on completion).
cd /home/aayush/code/clg/AmazonML
deadline=$(date -d "11:30" +%s)
while [ $(date +%s) -lt $deadline ]; do
  sleep 600
  if [ -f output/matching_results.tsv ]; then
    rows=$(($(wc -l < output/matching_results.tsv) - 1))
    if [ $rows -eq 1732544 ]; then echo "VERDICT: DONE rows=$rows"; exit 0; fi
  fi
  if grep -q "STOP:" checkpoints/watchdog.log 2>/dev/null; then
    echo "VERDICT: WATCHDOG-STOPPED"; tail -8 checkpoints/watchdog.log; exit 1
  fi
  if ! pgrep -f overnight.py >/dev/null && ! pgrep -f watchdog.sh >/dev/null; then
    echo "VERDICT: BOTH-DEAD"; tail -4 checkpoints/overnight.log; exit 2
  fi
done
rows=$(($(wc -l < output/matching_results.tsv 2>/dev/null) - 1))
echo "VERDICT: DEADLINE rows=$rows"; tail -4 checkpoints/overnight.log; exit 3
