#!/usr/bin/env bash
# ============================================================================
# Copy a finished training run off HPCC to a remote (Dropbox by default).
#
# WHY: SMU HPCC does not back up any data. Nothing under /storage is protected
# against disk failure or accidental deletion. The --resume checkpointing in
# train5_multistream.py protects against a job being interrupted; it does not
# protect against losing the filesystem. Different problem, different fix.
#
# USAGE
#   scripts/backup_run.sh                      # metrics only, newest run
#   scripts/backup_run.sh --weights            # metrics + model weights
#   scripts/backup_run.sh --dry-run            # show what would copy
#   scripts/backup_run.sh /path/to/run_dir     # a specific run
#   scripts/backup_run.sh --remote myremote    # a remote other than "dropbox"
#
# WHAT GETS COPIED
#   By default ONLY the JSON metrics (history.json, final.json). These are
#   aggregate numbers -- no records, no user-level data.
#
#   --weights additionally copies *.pt. Consider this deliberately: the gen-5
#   model contains nn.Embedding(num_users, d_model), i.e. one learned vector
#   per user. That is user-level derived data, not raw records but not purely
#   aggregate either. Whether it may go to third-party cloud storage depends
#   on your data agreement, so it is opt-in rather than default.
#
# ONE-TIME SETUP
#   rclone's OAuth step needs a browser, which a headless cluster does not
#   have. So configure on your LAPTOP:
#       rclone config          # create a remote named "dropbox"
#       rclone config file     # prints the path of rclone.conf
#   then copy it across and lock it down:
#       scp ~/.config/rclone/rclone.conf jinmiao@omega.smu.edu.sg:~/.config/rclone/
#       ssh jinmiao@omega.smu.edu.sg 'chmod 700 ~/.config/rclone && chmod 600 ~/.config/rclone/rclone.conf'
#   rclone.conf is a CREDENTIAL. Never commit it, never put it in a PBS script.
# ============================================================================

set -euo pipefail

RUNS_ROOT="${RUNS_ROOT:-/storage/home/jinmiao/ProductGPT/runs}"
REMOTE="${REMOTE:-dropbox}"
DEST_PREFIX="${DEST_PREFIX:-ProductGPT_Backup/runs}"
WITH_WEIGHTS=0
DRY_RUN=""
RUN_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights)  WITH_WEIGHTS=1; shift ;;
    --dry-run)  DRY_RUN="--dry-run"; shift ;;
    --remote)   REMOTE="$2"; shift 2 ;;
    -h|--help)  sed -n '2,40p' "$0"; exit 0 ;;
    *)          RUN_DIR="$1"; shift ;;
  esac
done

# ---------------------------------------------------------------- preflight
command -v rclone >/dev/null 2>&1 || {
  echo "ERROR: rclone not found on PATH." >&2; exit 2; }

if ! rclone listremotes 2>/dev/null | grep -qx "${REMOTE}:"; then
  echo "ERROR: rclone remote '${REMOTE}:' is not configured." >&2
  echo "Configured remotes:" >&2
  rclone listremotes >&2 || true
  echo "See the ONE-TIME SETUP notes at the top of this script." >&2
  exit 2
fi

# Newest run directory containing a history.json, unless one was named.
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(find "$RUNS_ROOT" -name history.json -printf '%T@ %h\n' 2>/dev/null \
             | sort -rn | head -1 | cut -d' ' -f2-)"
  [[ -n "$RUN_DIR" ]] || { echo "ERROR: no run with history.json under $RUNS_ROOT" >&2; exit 2; }
  echo "Newest run: $RUN_DIR"
fi
[[ -d "$RUN_DIR" ]] || { echo "ERROR: not a directory: $RUN_DIR" >&2; exit 2; }

# Name the destination after the run's path relative to RUNS_ROOT, so
# concurrent configurations do not overwrite each other.
REL="${RUN_DIR#"$RUNS_ROOT"/}"
REL="${REL//\//_}"
DEST="${REMOTE}:${DEST_PREFIX}/${REL}"

echo "Source : $RUN_DIR"
echo "Dest   : $DEST"
echo "Weights: $([[ $WITH_WEIGHTS -eq 1 ]] && echo yes || echo 'no (metrics only)')"
echo

# ---------------------------------------------------------------- copy
# "copy" not "sync": sync makes the destination mirror the source and can
# DELETE remote files that are missing locally. copy only ever adds/updates.
#
# --filter rather than --include/--exclude: rclone warns that mixing the
# latter two is parsed in an indeterminate order. Filter rules are applied
# top to bottom, so the trailing "- *" reliably drops anything not matched
# by an earlier "+" rule.
FILTERS=(--filter "+ *.json")
[[ $WITH_WEIGHTS -eq 1 ]] && FILTERS+=(--filter "+ *.pt")
FILTERS+=(--filter "- *")

rclone copy $DRY_RUN "$RUN_DIR" "$DEST" \
  "${FILTERS[@]}" \
  --progress \
  --transfers 4 \
  --checkers 8

echo
if [[ -n "$DRY_RUN" ]]; then
  echo "Dry run - nothing was transferred and no remote directory was created."
else
  echo "Done. Remote contents:"
  rclone ls "$DEST"
fi
