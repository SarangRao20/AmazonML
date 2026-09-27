#!/usr/bin/env python3
import subprocess
import sys
import time

SESSION = "france_v2"

def run_colab(cmd):
    p = subprocess.run(
        ["uvx", "--from", "google-colab-cli", "colab", "exec", "-s", SESSION],
        input=cmd,
        capture_output=True,
        text=True
    )
    return p.stdout

def check_progress():
    script = """
import subprocess, os
log = ""
if os.path.exists("/content/amazon_ml/run_france.log"):
    with open("/content/amazon_ml/run_france.log") as f:
        lines = f.readlines()
        log = "".join(lines[-15:])
done = os.path.exists("/content/amazon_ml/output_france/parts/France.scores.tsv") and "FINISHED" in log
print(f"DONE:{done}")
print("=== LOG ===")
print(log)
"""
    out = run_colab(script)
    return out

if __name__ == "__main__":
    out = check_progress()
    print(out)
    if "DONE:True" in out:
        print("France is FINISHED on Colab! Downloading France.scores.tsv...")
        dl = subprocess.run([
            "uvx", "--from", "google-colab-cli", "colab", "download",
            "/content/amazon_ml/output_france/parts/France.scores.tsv",
            "output/parts/France.scores.tsv",
            "-s", SESSION
        ], capture_output=True, text=True)
        print(dl.stdout)
        print(dl.stderr)
        print("Download complete!")
