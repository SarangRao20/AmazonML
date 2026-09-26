#!/usr/bin/env bash
#
# One-command driver: retrain on the full per-country pool, then re-run test
# inference with the winner, then validate both submissions.
#
#   ./run_pipeline.sh                 # everything, gate on OOF F0.5
#   ./run_pipeline.sh retrain         # just the retrain
#   ./run_pipeline.sh infer           # just the test run
#   ./run_pipeline.sh validate        # just validate
#   ./run_pipeline.sh status          # what is running / what exists
#
# Overridable via environment: SAMPLE_FRAC MIN_F0 CAP BLOCKER
# ER_TRAIN_CHUNK MAX_AVAIL_MB
#
# The existing submission in output/ is never touched. The new run writes to
# output_v2/ so there is always a validated fallback to submit.
set -uo pipefail
cd "$(dirname "$0")"

PY=./venv_fresh/bin/python
SAMPLE_FRAC="${SAMPLE_FRAC:-0.08}"   # 0.08 x 2.21M S1 = ~177k entities
MIN_F0="${MIN_F0:-96.50}"           # gate: reject a retrain below this OOF F0.5
CAP="${CAP:-45}"
BLOCKER="${BLOCKER:-sparse}"
ER_TRAIN_CHUNK="${ER_TRAIN_CHUNK:-5000}"
export ER_TRAIN_CHUNK
MAX_AVAIL_MB="${MAX_AVAIL_MB:-1600}"

RETRAIN_OUT=models_v2
NEW_OUT=output_v2
INCUMBENT=models
LOGDIR=logs/pipeline
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOGDIR"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
die() { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- preflight
preflight() {
  say "Preflight"
  [ -x "$PY" ] || die "no interpreter at $PY (expected ./venv_fresh)"
  "$PY" - <<'EOF' || die "dependency check failed"
import sys
missing = []
for m in ("polars", "pandas", "numpy", "scipy", "sklearn", "sparse_dot_topn",
          "xgboost", "lightgbm", "catboost"):
    try:
        __import__(m)
    except Exception:
        missing.append(m)
if missing:
    print("missing:", ", ".join(missing)); sys.exit(1)
print(f"   python {sys.version.split()[0]}, all core imports present")
EOF
  df -BG . | awk 'NR==2{printf "   disk: %s free\n", $4}'
  free -g | awk 'NR==2{printf "   ram: %s GB total, %s GB available\n", $2, $7}'
  command -v nproc >/dev/null && info "cpus: $(nproc)"
  info "gate: use new model only if OOF macro F0.5 >= ${MIN_F0}%"
  info "plan: sample_frac=${SAMPLE_FRAC} cap=${CAP} blocker=${BLOCKER}"
}

# ------------------------------------------------- wait for a running job
infer_running() {
  pgrep -f "run_test_inference.py" >/dev/null 2>&1
}

wait_for_infer() {
  if ! infer_running; then return 0; fi
  say "A test run is already in flight"
  info "pid: $(pgrep -f run_test_inference.py | tr '\n' ' ')"
  info "log: $(ls -t logs/testrun_*.log 2>/dev/null | head -1)"
  info "Waiting for it rather than competing for RAM (two inference jobs"
  info "at once is what tripped the watchdog earlier). Ctrl-C to abort."
  while infer_running; do sleep 60; done
  info "previous run finished"
}

# ------------------------------------------------------------------ retrain
do_retrain() {
  say "Stage 1/3  Retrain on the full per-country pool"
  info "This is the stage that matters: the shipped model was trained"
  info "against val_sample's 518k pool while India's test pool is 4.72M,"
  info "and 18 of the 53 features are channel/rank/density counts."
  local log="$LOGDIR/retrain_$STAMP.log"
  info "log: $log   (expect 2.5-3 h)"

  $PY -u tools/run_guarded.py --min-avail-mb "$MAX_AVAIL_MB" --log "$log" --cwd . -- \
    $PY -u train_model.py \
        --blocker "$BLOCKER" \
        --max-candidates "$CAP" \
        --sample-frac "$SAMPLE_FRAC" \
        --out "$RETRAIN_OUT" \
        2>&1 | tee "$LOGDIR/retrain_$STAMP.tail"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || die "retrain exited $rc (see $log)"
  [ -f "$RETRAIN_OUT/decision_rule.json" ] || die "retrain produced no decision_rule.json"

  NEW_F0=$(grep -o 'macro F0.5=[0-9.]*' "$log" | tail -1 | cut -d= -f2)
  NEW_RECALL=$(grep -o 'pair recall=[0-9.]*%' "$log" | tail -1)
  NEW_COV=$(grep -o 'entity full coverage=[0-9.]*%' "$log" | tail -1)
  info "blocking: ${NEW_RECALL}  ${NEW_COV}"
  info "retrained OOF macro F0.5 = ${NEW_F0}%"
  printf 'new_oof_f0_5=%s\n' "$NEW_F0" > "$LOGDIR/last_retrain.txt"

  if [ -z "$NEW_F0" ]; then die "could not read OOF F0.5 from $log"; fi
  if "$PY" -c "import sys; sys.exit(0 if float('$NEW_F0') >= float('$MIN_F0') else 1)"; then
    info "GATE PASSED (>= ${MIN_F0}%) -> will use $RETRAIN_OUT"
    return 0
  fi
  info "GATE FAILED: ${NEW_F0}% < ${MIN_F0}%. The retrain is worse or"
  info "not better; the incumbent stays. Skipping the new test run."
  return 1
}

# ------------------------------------------------------------------- infer
do_infer() {
  say "Stage 2/3  Test inference"
  local model_dir="$1" out_dir="$2"
  local log="$LOGDIR/infer_$STAMP.log"
  info "model: $model_dir   out: $out_dir"
  info "log: $log   (expect ~4.7 h for all three countries)"
  mkdir -p "$out_dir"
  # --force: the .done markers are per-output-dir, but be explicit so a
  # rerun after a partial failure recomputes rather than silently skipping.
  $PY -u tools/run_guarded.py --min-avail-mb "$MAX_AVAIL_MB" --log "$log" --cwd . -- \
    $PY -u run_test_inference.py \
        --blocker "$BLOCKER" \
        --max-candidates "$CAP" \
        --model-dir "$model_dir" \
        --out "$out_dir" \
        --force \
        2>&1 | tee "$LOGDIR/infer_$STAMP.tail"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || die "inference exited $rc (see $log)"
  info "rows: $(( $(wc -l < "$out_dir/matching_results.tsv") - 1 ))"
}

# ---------------------------------------------------------------- validate
do_validate() {
  say "Stage 3/3  Validate"
  local rc_all=0
  for d in output output_v2; do
    [ -f "$d/matching_results.tsv" ] || { info "$d: absent, skipping"; continue; }
    info "--- $d ---"
    $PY utils/validate_submission.py \
        --matching "$d/matching_results.tsv" \
        --candidate "$d/candidate_pairs.tsv" \
        --test-dir dataset/test || rc_all=1
  done
  return $rc_all
}

status() {
  say "Status"
  infer_running && info "inference RUNNING: $(pgrep -f run_test_inference.py | tr '\n' ' ')" \
                    || info "inference not running"
  local l; l=$(ls -t logs/testrun_*.log 2>/dev/null | head -1)
  [ -n "$l" ] && { info "latest run log: $l"; tail -3 "$l" | sed 's/^/     /'; }
  for d in output output_v2; do
    if [ -f "$d/matching_results.tsv" ]; then
      n=$(( $(wc -l < "$d/matching_results.tsv") - 1 ))
      parts=$(ls "$d"/parts/*.done 2>/dev/null | wc -l)
      if [ "$parts" -gt 0 ] && [ "$parts" -lt 3 ] && [ "$n" -lt 1700000 ]; then
        info "$d: STALE - $n rows, only $parts country part(s) finished."
        info "       Parts in $d/parts are still being written; the final"
        info "       file is only assembled once every country completes."
      else
        info "$d: $n rows"
      fi
    else
      info "$d: absent"
    fi
  done
  [ -d "$RETRAIN_OUT" ] && info "$RETRAIN_OUT: present" || info "$RETRAIN_OUT: absent"
  [ -f logs/retrain_verdict.txt ] && cat logs/retrain_verdict.txt | sed 's/^/     /'
}

# -------------------------------------------------------------------- main
case "${1:-all}" in
  status)   status ;;
  preflight) preflight ;;
  retrain)  preflight; wait_for_infer; do_retrain ;;
  infer)    preflight; wait_for_infer; do_infer "$INCUMBENT" "$NEW_OUT" ;;
  validate) do_validate ;;
  all)
    preflight
    wait_for_infer
    if do_retrain; then
      do_infer "$RETRAIN_OUT" "$NEW_OUT"
    fi
    do_validate || true
    say "Done"
    info "submission A (incumbent, val_sample pool): output/"
    info "submission B (retrained, full pool):         output_v2/"
    info "Both validated above. See $LOGDIR for logs."
    ;;
  *) die "unknown stage '${1}'. Use: all | retrain | infer | validate | status | preflight" ;;
esac
