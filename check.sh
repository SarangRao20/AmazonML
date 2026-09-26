#!/usr/bin/env bash
# Is the pipeline done? One line, safe to run any time, never edits anything.
cd "$(dirname "$0")"

DRIVER=logs/pipeline_driver.log

if [ ! -f "$DRIVER" ]; then
  echo "no driver log yet - has ./run_pipeline.sh been started?"
  exit 1
fi

# ---- finished? the driver prints this banner last -------------------------
if grep -q '^.*== Done' "$DRIVER"; then
  echo "=============================================="
  echo "  DONE - pipeline finished"
  echo "=============================================="
  grep -E 'GATE|BEST rule|BLOCKING|rows written|rows:' "$DRIVER" | tail -6 | sed 's/^/  /'
  echo
  echo "  submissions:"
  for d in output output_v2; do
    if [ -f "$d/matching_results.tsv" ]; then
      echo "    $d/  $(( $(wc -l < "$d/matching_results.tsv") - 1 )) rows"
    else
      echo "    $d/  absent"
    fi
  done
  echo
  echo "  final check:  ./run_pipeline.sh validate"
  exit 0
fi

# ---- died partway? ---------------------------------------------------------
if ! pgrep -f "bash ./run_pipeline.sh" >/dev/null 2>&1 \
   && ! pgrep -f "run_pipeline.sh" >/dev/null 2>&1; then
  echo "=============================================="
  echo "  NOT RUNNING and never printed Done."
  echo "  It died. Last lines:"
  echo "=============================================="
  tail -15 "$DRIVER"
  exit 1
fi

# ---- still going -----------------------------------------------------------
stage=$(sed 's/\x1b\[[0-9;]*m//g' "$DRIVER" | grep -E '^== ' | tail -1 | sed 's/^== //')
stage=${stage:-starting up}
running=$(pgrep -f "run_test_inference.py|train_model.py" | head -3 | tr '\n' ' ')
waiting=$(grep -c 'Waiting for it' "$DRIVER")

printf 'RUNNING  stage: %s\n' "$stage"
[ -n "$running" ] && printf '  active: %s\n' "$running"
[ "$waiting" -gt 0 ] && printf '  note: parked waiting for the earlier run to finish\n'

# most recent progress line from whatever is logging right now
for l in $(ls -t logs/pipeline/infer_*.log logs/pipeline/retrain_*.log \
             logs/testrun_*.log 2>/dev/null | head -3); do
  line=$(grep -E 'chunk [0-9]+|country part|Recall|BEST|Fold|epoch|\[[0-9]/3\]|done:' "$l" 2>/dev/null | tail -1)
  [ -n "$line" ] && printf '  %s\n    %s\n' "$(basename "$l")" "$line"
done

printf '\n  full log: %s\n' "$DRIVER"
echo "  (re-run this script whenever; safe to repeat)"
exit 0
