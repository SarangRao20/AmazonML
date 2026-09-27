#!/usr/bin/env bash
# Post-run finalisation: wait for the run, then stage every submission in
# order so each is validated before the next is built.
#
# It never submits anything. Submission is a manual action on the site, and
# each stage writes to its own directory so an earlier stage is never
# overwritten by a later one.
#
#   ./finalize.sh            wait, then run all stages
#   ./finalize.sh now        skip the wait, run now
#   ./finalize.sh status     just report
set -uo pipefail
cd "$(dirname "$0")"

PY=./venv_fresh/bin/python
NEED=1732544
US_DONE=output/parts/US.done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
die()  { printf '\033[1;31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

validate() {  # validate <dir>
  local d="$1"
  [ -f "$d/matching_results.tsv" ] || return 1
  $PY utils/validate_submission.py \
      --matching "$d/matching_results.tsv" \
      --candidate "$d/candidate_pairs.tsv" \
      --test-dir dataset/test 2>&1 | tail -3
}

report() {
  info "expected $NEED rows"
  for d in output_baseline output output_dedup output_singleton; do
    if [ -f "$d/matching_results.tsv" ]; then
      local n=$(( $(wc -l < "$d/matching_results.tsv") - 1 ))
      local e; e=$(awk -F'\t' 'NR>1 && $2==""' "$d/matching_results.tsv" | wc -l)
      local c; c=$(( $(wc -l < "$d/matching_results.tsv") - 1 - e ))
      local note=""
      [ "$n" -ne "$NEED" ] && note="   <-- STALE, not the current run (only assembled after the last country)"
      info "$(printf '%-18s' "$d/") $n rows   empty $e ($(awk "BEGIN{printf \"%.2f\", 100*$e/($n?$n:1)}")%)   mean $(awk "BEGIN{printf \"%.2f\", $c/($n?$n:1)}")$note"
    else
      info "$(printf '%-18s' "$d/") absent"
    fi
  done
}

status() {
  say "Status"
  if pgrep -f "[r]un_test_inference.py" >/dev/null; then
    local pid; pid=$(pgrep -af "[r]un_test_inference.py" | grep -v run_guarded | awk '{print $1}' | head -1)
    info "inference RUNNING (pid $pid) $(ps -o etime= -p "$pid" | tr -d ' ')"
    for f in output/parts/*.done; do
      [ -e "$f" ] && info "  $(basename "$f" .done): $(cat "$f")"
    done
    tail -2 logs/testrun3_032459.log 2>/dev/null | sed 's/^/    /'
  else
    info "inference not running"
  fi
  free -g | awk 'NR==2{printf "   ram available %sG of %sG\n", $7, $2}'
  report
}

if [ "${1:-}" = "status" ]; then status; exit 0; fi

# ---- wait ----------------------------------------------------------------
if [ "${1:-}" != "now" ]; then
  say "Waiting for the run to finish"
  info "polling for $US_DONE every 60s; Ctrl-C to abort"
  while [ ! -f "$US_DONE" ]; do
    if ! pgrep -f "[r]un_test_inference.py" >/dev/null; then
      [ -f "$US_DONE" ] || die "inference stopped but $US_DONE never appeared"
    fi
    sleep 60
  done
  # the process assembles output/ after the last country; wait for that too
  info "US.done seen, waiting for output/ assembly"
  while pgrep -f "[r]un_test_inference.py" >/dev/null; do sleep 20; done
fi

# ---- stage A: raw --------------------------------------------------------
say "Stage A - raw output"
n=$(( $(wc -l < output/matching_results.tsv 2>/dev/null || echo 1) - 1 ))
[ "$n" -eq "$NEED" ] || die "output/ has $n rows, expected $NEED"
info "$n rows"
validate output || die "raw output failed validation"
info "this is submission A"

# ---- stage B: 1-to-1 dedup ----------------------------------------------
say "Stage B - 1-to-1 target uniqueness"
# Ground truth is a perfect matching: 7,638,365 pairs claim 7,638,365
# distinct targets, so any target claimed twice is guaranteed to contain a
# false positive. Resolved by candidate rank, lowest first.
mkdir -p output_dedup
$PY tools/apply_deduplication.py \
    --input output/matching_results.tsv \
    --output output_dedup/matching_results.tsv || die "dedup failed"
cp output/candidate_pairs.tsv output_dedup/candidate_pairs.tsv
validate output_dedup || die "dedup output failed validation"
info "this is submission B"

# ---- report --------------------------------------------------------------
say "Summary"
report
info ""
info "submit A (output/) or B (output_dedup/)? B removes guaranteed false"
info "positives; see docs/HANDOFF.md for the reasoning. A is the fallback."
info ""
info "Stage C (singleton gate) needs France scores from Colab. When"
info "output_france/parts/France.scores.tsv is available:"
info "  $PY tools/apply_singleton_gate.py --scores <file> --dir output_dedup"
