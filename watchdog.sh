#!/bin/zsh
# Watchdog: poll overnight.py every 120s, restart on death (max 4).
# On repeat featurize-stage deaths, halve FEAT_WORK (4->2->1) before relaunch.
cd /home/aayush/code/clg/AmazonML
VENV=~/code/ML_DL/.venv/bin/python
LOG=checkpoints/overnight.log
WDLOG=checkpoints/watchdog.log
COUNT_F=checkpoints/.restarts
FEAT_F=checkpoints/.featwork

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $WDLOG }

[ -f $COUNT_F ] || echo 0 > $COUNT_F
[ -f $FEAT_F ] || echo 4 > $FEAT_F
[ -f checkpoints/.inferchunk ] || echo 25000 > checkpoints/.inferchunk
[ -f checkpoints/.samplen ] || echo 50000 > checkpoints/.samplen
[ -f checkpoints/.traindeaths ] || echo 0 > checkpoints/.traindeaths

while true; do
  sleep 120
  if pgrep -f "overnight.py" > /dev/null; then
    continue
  fi
  n=$(cat $COUNT_F)
  if [ $n -ge 4 ]; then
    say "STOP: 4 restarts used, needs human. Last log:"; tail -5 $LOG >> $WDLOG
    exit 0
  fi
  tail_text=$(tail -3 $LOG)
  fw=$(cat $FEAT_F); ic=$(cat checkpoints/.inferchunk); sn=$(cat checkpoints/.samplen)
  # class 1: died during train featurization (recall printed, no feat parquet)? shrink workers
  if echo "$tail_text" | grep -q "train blocking recall" && [ ! -f checkpoints/train_feat_*.parquet ]; then
    if [ $fw -gt 1 ]; then fw=$(( fw / 2 )); echo $fw > $FEAT_F; fi
  # class 2: died during model training (fold/model lines, thresholds missing)? shrink sample once
  elif echo "$tail_text" | grep -qi "fold\|xgboost\|lightgbm\|catboost\|training" && [ ! -f checkpoints/thresholds.json ]; then
    td=$(cat checkpoints/.traindeaths); td=$(( td + 1 )); echo $td > checkpoints/.traindeaths
    if [ $td -ge 2 ] && [ $sn -gt 25000 ]; then sn=25000; echo $sn > checkpoints/.samplen; fi
  # class 3: died during inference (country chunk lines)? halve chunk size
  elif echo "$tail_text" | grep -q "chunk .* done\|vs pool"; then
    if [ $ic -gt 6250 ]; then ic=$(( ic / 2 )); echo $ic > checkpoints/.inferchunk; fi
  fi
  # resume-safe artifacts present?
  have="none"
  [ -f checkpoints/thresholds.json ] && have="thresholds"
  ls checkpoints/train_feat_*.parquet >/dev/null 2>&1 && have="$have+feats"
  n=$(( n + 1 )); echo $n > $COUNT_F
  say "RESTART #$n (feat=$fw infer_chunk=$ic sample=$sn resume=$have). Death site:"; echo "$tail_text" >> $WDLOG
  PYTHONFAULTHANDLER=1 FEAT_WORK=$fw INFER_CHUNK=$ic SAMPLE_N=$sn nohup $VENV -u overnight.py >> $LOG 2>&1 &
  say "relaunched pid $!"
done
