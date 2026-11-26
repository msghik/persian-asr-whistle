#!/bin/bash
# Copy everything that is expensive to rebuild to Google Drive (rclone remote "gdrive"):
#   bash scripts/sync_drive.sh            # upload  ~/persian_asr_data -> gdrive:persian_asr
#   bash scripts/sync_drive.sh --essentials # ~3.2 GB (measured 2026-11-26): final best-3.pt models, LMs, tokenizers, prep text, hyps, logs
#   bash scripts/sync_drive.sh --restore  # download gdrive:persian_asr -> ~/persian_asr_data
# Raw CV archives/clips are NOT copied (MDC re-downloads them in ~2 min; re-extract with tar).
# Uses `rclone copy` (never deletes on the destination). Set up the remote once with `rclone config`.
set -euo pipefail
DATA=${PERSIAN_ASR_DATA:-$HOME/persian_asr_data}
REMOTE=${RCLONE_REMOTE:-gdrive:persian_asr}
EXCLUDES=(--exclude "raw/**" --exclude "cat_ws/exp" --exclude "cat_ws/cat" --exclude "cat_ws/utils"
          --exclude "**/decode/*/fst_*/lat/**")
if [ "${1:-}" = "--essentials" ]; then
  # Final models + everything needed to re-decode/re-score; skips rebuildable bulk (features ~15 GB,
  # intermediate checkpoints with optimizer state, saved log-probs, invalid runs).
  shift
  EXCLUDES+=(--exclude "exp/*/check/checkpoint.*" --exclude "exp/*/decode/*/ark/**" --exclude "exp/*/pkl/**"
             --exclude "prep/*/feats/**" --exclude "prep_quick/**" --exclude "exp_invalid/**" --exclude "init/*.pt"
             --exclude "graph/**" --exclude "lm/*/pkl/**")
  rclone copy "$DATA" "$REMOTE" --transfers 8 --checkers 16 -P "${EXCLUDES[@]}"
  exit 0
fi
if [ "${1:-}" = "--restore" ]; then
  rclone copy "$REMOTE" "$DATA" --transfers 8 --checkers 16 -P "${EXCLUDES[@]}"
  echo "restored into $DATA; re-create workspace links by running any scripts/train.py command"
else
  rclone copy "$DATA" "$REMOTE" --transfers 8 --checkers 16 -P --copy-links=false "${EXCLUDES[@]}"
  # Kaldi/KenLM builds (static) so the next server can skip the build:
  tar -C "$HOME/persian_asr_local/tools" -czf "$DATA/../kaldi_kenlm_tools.tgz" kaldi kenlm 2>/dev/null &&
    rclone copy "$DATA/../kaldi_kenlm_tools.tgz" "$REMOTE/tools/" -P || echo "tools tarball skipped"
fi
