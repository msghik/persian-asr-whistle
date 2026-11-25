#!/bin/bash
# Run trainings + decode/eval back to back (each step is idempotent / resumable).
#   tmux new -s queue 'bash scripts/run_queue.sh "S2 fa 1h" "S1 fa 1h" "PL pl 1h" "S2 fa 10h" "S1 fa 10h"'
# Optional: WAIT_FOR=<tmux session> waits until that session has ended first.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=$HOME/venvs/asr/bin/python
DATA=${PERSIAN_ASR_DATA:-$HOME/persian_asr_data}
LOG=$DATA/logs/queue.log
if [ -n "${WAIT_FOR:-}" ]; then
  while tmux has-session -t "=$WAIT_FOR" 2>/dev/null; do sleep 60; done
fi
for job in "$@"; do
  read -r sys lang size <<<"$job"
  run_id="${lang}_${sys}_${size}_bpe500_s0"
  echo "$(date '+%F %T') START $run_id" >>"$LOG"
  ok=0
  for attempt in 1 2 3; do  # re-running resumes from the latest checkpoint
    "$PY" "$REPO/scripts/train.py" --system "$sys" --lang "$lang" --size "$size" >>"$DATA/logs/train_$run_id.log" 2>&1 && { ok=1; break; }
    echo "$(date '+%F %T') train attempt $attempt failed: $run_id" >>"$LOG"
  done
  [ $ok = 1 ] || { echo "$(date '+%F %T') FAILED training $run_id" >>"$LOG"; continue; }
  if [ ! -f "$DATA/exp/$run_id/decode/scores.json" ]; then
    if [ "$lang" = pl ]; then lms="pl_cat pl_clean"; else lms="fa_clean"; fi
    "$PY" "$REPO/scripts/decode_eval.py" --run-id "$run_id" --lm $lms >>"$DATA/logs/decode_$run_id.log" 2>&1 \
      || echo "$(date '+%F %T') FAILED decode $run_id" >>"$LOG"
  fi
  echo "$(date '+%F %T') DONE $run_id $(grep -h "WER" "$DATA/logs/decode_$run_id.log" | tr '\n' ' ')" >>"$LOG"
done
