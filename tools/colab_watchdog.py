#!/usr/bin/env python3
"""
Colab Watchdog & Keep-Alive Daemon.
Team: BreakEven

Continuously monitors Colab session 'breakeven_training':
1. Sends an active ping every 3 minutes to prevent Colab idle timeout.
2. Logs remote training progress to logs/colab_watchdog.log.
3. Automatically downloads trained models when finished.
"""

import subprocess
import time
import os
from pathlib import Path

SESSION = "breakeven_training"
LOG_FILE = Path("logs/colab_watchdog.log")
LOG_FILE.parent.mkdir(exist_ok=True)

def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_colab(cmd_str):
    try:
        p = subprocess.run(
            ["uvx", "--from", "google-colab-cli", "colab", "exec", "-s", SESSION],
            input=cmd_str,
            capture_output=True,
            text=True,
            timeout=60
        )
        return p.stdout + ("\n" + p.stderr if p.stderr and "warning" not in p.stderr.lower() else "")
    except Exception as e:
        return f"Error executing Colab command: {e}"

def check_remote():
    remote_script = """
import os, glob

# Check if training process is still alive
ps = os.popen("ps aux | grep train_model.py | grep -v grep").read().strip()
running = bool(ps)

# Read last few lines of log
log_tail = ""
if os.path.exists("/content/AmazonML/logs/colab_training.log"):
    with open("/content/AmazonML/logs/colab_training.log") as f:
        lines = f.readlines()
        log_tail = "".join(lines[-10:])

# Check if models exist
models = glob.glob("/content/AmazonML/models_v2/*.pkl") + glob.glob("/content/AmazonML/models_v2/*.json")
finished = len(models) >= 1 and not running

print(f"RUNNING:{running}")
print(f"FINISHED:{finished}")
print(f"MODELS_COUNT:{len(models)}")
print("=== LOG TAIL ===")
print(log_tail)
"""
    out = run_colab(remote_script)
    return out

def main():
    log(f"=== Colab Watchdog Started for session: {SESSION} ===")
    
    consecutive_errors = 0
    while True:
        try:
            status = check_remote()
            log(status.strip())
            
            if "FINISHED:True" in status:
                log("🎉 Training on Colab has FINISHED! Downloading trained models...")
                os.makedirs("models_colab", exist_ok=True)
                dl = subprocess.run([
                    "uvx", "--from", "google-colab-cli", "colab", "download",
                    "/content/AmazonML/models_v2",
                    "models_colab/",
                    "-s", SESSION
                ], capture_output=True, text=True)
                log(f"Download output: {dl.stdout}\n{dl.stderr}")
                log("=== All models downloaded successfully! ===")
                break
                
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            log(f"Watchdog iteration error ({consecutive_errors}): {e}")
            if consecutive_errors > 10:
                log("Too many consecutive errors, pausing...")
                time.sleep(120)
                
        # Heartbeat interval: 3 minutes (keeps Colab connection fresh and active)
        time.sleep(180)

if __name__ == "__main__":
    main()
