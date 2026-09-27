#!/usr/bin/env bash
# Poll the Colab retrain session. Read-only: never touches the run.
#
# Prints the newest log rather than a fixed filename, because failed
# attempts leave earlier logs behind and reading a stale one reports an
# error that was already fixed.
SESSION="${1:-retrain1}"
cd "$(dirname "$0")/.."
timeout 200 uvx --from google-colab-cli colab exec -s "$SESSION" <<'PY' 2>&1 | tail -34
import glob, os, subprocess

alive = subprocess.run(["pgrep", "-f", "train_model.py"],
                       capture_output=True, text=True).stdout.strip()
print("PROC:", alive.replace("\n", " ") or "(not running)")

if not alive:
    v = "/content/AmazonML/logs/retrain_verdict.txt"
    if os.path.exists(v):
        print("--- VERDICT ---")
        print(open(v).read())

logs = sorted(glob.glob("/content/AmazonML/logs/*.log"),
              key=os.path.getmtime)
for p in logs[-2:]:
    lines = open(p, errors="replace").readlines()
    print(f"--- {os.path.basename(p)} ({len(lines)} lines) ---")
    print("".join(lines[-14:]))

d = "/content/AmazonML/models_v2"
if os.path.isdir(d):
    print("models_v2:", sorted(os.listdir(d)))
PY
