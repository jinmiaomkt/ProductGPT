#!/usr/bin/env bash
# ============================================================================
# Report SMU HPCC (OMEGA) job status -- but ONLY when something changes.
#
# WHY CHANGE-ONLY
# ---------------
# Training jobs run up to 24h. Polling hourly and printing every time produces
# ~24 near-identical "still running" lines per job, which is noise that buries
# the one line that matters. This script keeps a signature of the last observed
# state and prints only on a transition: queued -> running, running -> gone,
# a new job appearing, or the cluster becoming unreachable.
#
# Terminal states are ALWAYS reported. A job vanishing from the queue is the
# most important event and the easiest to miss, so it gets its own line with
# whatever the tail of its log says about why.
#
# USAGE
#   scripts/hpcc_status.sh --once            # one poll, for a scheduled task
#   scripts/hpcc_status.sh                   # loop forever, for a live watcher
#   scripts/hpcc_status.sh --interval 1800   # poll every 30 min (default 900s)
#   scripts/hpcc_status.sh --force           # print current state even if unchanged
#
# REQUIRES passwordless SSH to OMEGA. Set it up from THIS machine (not from an
# OMEGA session) with:
#     ssh-copy-id jinmiao@omega.smu.edu.sg
# Verify with:
#     ssh -o BatchMode=yes jinmiao@omega.smu.edu.sg echo ok
# BatchMode is used throughout so a missing key fails immediately instead of
# hanging on a password prompt.
# ============================================================================
set -uo pipefail

HOST="${HPCC_HOST:-omega.smu.edu.sg}"
USER_NAME="${HPCC_USER:-jinmiao}"
LOG_DIR="${HPCC_LOG_DIR:-/storage/home/jinmiao/ProductGPT/logs}"
STATE_FILE="${HPCC_STATE_FILE:-$HOME/.cache/hpcc_status.state}"
INTERVAL=900
ONCE=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)       ONCE=1; shift ;;
    --force)      FORCE=1; shift ;;
    --interval)   INTERVAL="$2"; shift 2 ;;
    --host)       HOST="$2"; shift 2 ;;
    --user)       USER_NAME="$2"; shift 2 ;;
    --state-file) STATE_FILE="$2"; shift 2 ;;
    -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$(dirname "$STATE_FILE")" 2>/dev/null || true

ts() { date +'%Y-%m-%dT%H:%M'; }

ssh_omega() {
  ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
      "${USER_NAME}@${HOST}" "$@" 2>/dev/null
}

# A non-interactive ssh session does NOT get the PBS module's PATH, so a bare
# `qstat` fails with "command not found" -- which would look exactly like an
# empty queue. Prepend the known location.
PBS_BIN="${HPCC_PBS_BIN:-/opt/pbs/bin}"

# One line per job: "<jobid> <name> <state> <elapsed>".
# PBS Pro's `qstat -u` table puts state in field 10 and elapsed in field 11.
poll() {
  ssh_omega "export PATH=\$PATH:${PBS_BIN}; qstat -u ${USER_NAME} 2>/dev/null \
    | awk 'NF>=10 && \$1 ~ /^[0-9]/ {split(\$1,a,\".\"); print a[1], \$4, \$10, \$11}' \
    | while read -r jid name st el; do
        prog=\$(grep -aE '\\[ep [0-9]+\\]|Traceback|Error|OOM|Killed' \
               ${LOG_DIR}/gen5_train_\${jid}.log 2>/dev/null \
               | tail -n 1 | tr -s ' ' | cut -c1-140)
        echo \"\$jid \$name \$st \$el | \${prog:-no progress line yet}\"
      done"
}

# Distinguish "queue empty" from "qstat missing": if this fails, the poller is
# blind and must say so rather than reporting an empty queue.
qstat_works() {
  ssh_omega "export PATH=\$PATH:${PBS_BIN}; command -v qstat >/dev/null"
}

# Last useful line from a job's log: a progress line, or a failure signature.
log_tail() {
  local jid="$1"
  ssh_omega "tail -n 400 ${LOG_DIR}/gen5_train_${jid}.log 2>/dev/null \
             | grep -Ea 'epoch|val_nll|Traceback|Error|FAILED|OOM|Killed|Finish time' \
             | tail -n 1"
}

# "<jid> <name> <state> <elapsed> | <progress>"  ->  "<jid> <name> <state> |<progress>"
# Drops elapsed, which changes on every poll and is not a real event.
sig_of() {
  local l="${1:-}" head prog a b c d
  [[ -z "$l" ]] && { echo ""; return; }
  head="${l%%|*}"; prog="${l#*|}"
  read -r a b c d <<<"$head"
  echo "$a $b $c |$prog"
}

emit_once() {
  local now cur prev
  now="$(ts)"

  if ! cur="$(poll)"; then
    cur=""
  fi

  # Distinguish "no jobs" from "could not reach the cluster": a successful
  # connection with an empty queue still returns 0 from ssh.
  if ! ssh_omega true; then
    cur="__UNREACHABLE__"
  elif ! qstat_works; then
    cur="__NO_QSTAT__"
  fi

  prev=""
  [[ -f "$STATE_FILE" ]] && prev="$(cat "$STATE_FILE")"

  if [[ "$cur" == "$prev" && $FORCE -eq 0 ]]; then
    return 0
  fi

  if [[ "$cur" == "__UNREACHABLE__" ]]; then
    echo "[$now] OMEGA unreachable (VPN down, key missing, or cluster offline)"
    printf '%s' "$cur" > "$STATE_FILE"
    return 0
  fi

  if [[ "$cur" == "__NO_QSTAT__" ]]; then
    echo "[$now] reached OMEGA but qstat is not on PATH -- poller is blind."
    echo "         set HPCC_PBS_BIN to the directory containing qstat."
    printf '%s' "$cur" > "$STATE_FILE"
    return 0
  fi

  # Jobs that were in the previous poll but are gone now: terminal states.
  if [[ -n "$prev" && "$prev" != "__UNREACHABLE__" && "$prev" != "__NO_QSTAT__" ]]; then
    while IFS= read -r line; do
      [[ -z "${line:-}" ]] && continue
      local jid name state elapsed why
      read -r jid name state elapsed <<<"${line%%|*}"
      [[ -z "${jid:-}" ]] && continue
      if ! grep -q "^${jid} " <<<"$cur"; then
        why="$(log_tail "$jid")"
        echo "[$now] $jid $name LEFT THE QUEUE (was $state, ran $elapsed) | ${why:-no log line found}"
      fi
    done <<<"$prev"
  fi

  # Current jobs whose state OR training progress changed, or that are new.
  # Including progress in the comparison is what turns this from a state
  # watcher into a training-progress feed: each new epoch line is a change,
  # so an hourly poll delivers one update per hour per running job.
  if [[ -n "$cur" ]]; then
    while IFS= read -r line; do
      [[ -z "${line:-}" ]] && continue
      local head prog jid name state elapsed before
      head="${line%%|*}"
      prog="${line#*|}"
      read -r jid name state elapsed <<<"$head"
      [[ -z "${jid:-}" ]] && continue
      # Compare on job + state + progress, deliberately EXCLUDING elapsed
      # time. Elapsed ticks on every poll, so including it would make every
      # poll a "change" and defeat the whole change-only design.
      before="$(sig_of "$(grep "^${jid} " <<<"$prev")")"
      if [[ "$before" != "$(sig_of "$line")" ]]; then
        if [[ "$state" == "R" ]]; then
          echo "[$now] $jid $name RUNNING ${elapsed} |${prog}"
        else
          echo "[$now] $jid $name state=$state (elapsed $elapsed)"
        fi
      fi
    done <<<"$cur"
  elif [[ -n "$prev" && "$prev" != "__UNREACHABLE__" && "$prev" != "__NO_QSTAT__" ]]; then
    echo "[$now] queue is now empty"
  elif [[ $FORCE -eq 1 ]]; then
    # An explicit check deserves an answer even when the answer is "nothing".
    echo "[$now] queue is empty (no jobs queued or running)"
  fi

  printf '%s' "$cur" > "$STATE_FILE"
}

if [[ $ONCE -eq 1 ]]; then
  emit_once
  exit 0
fi

while true; do
  emit_once
  sleep "$INTERVAL"
done
