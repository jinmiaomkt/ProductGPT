#!/usr/bin/env bash
# ============================================================================
# Hourly training feed, running ON OMEGA itself via cron.
#
# WHY ON THE CLUSTER RATHER THAN THE LAPTOP
# -----------------------------------------
# The Windows Task Scheduler version only fires while the ThinkPad is awake,
# online, and on a network that can reach OMEGA. A closed laptop reports
# nothing. Running the same poller on the login node removes all three
# conditions: the machine generating the jobs is the machine watching them.
#
# INSTALL (on OMEGA):
#   crontab -e        # then add, without the leading '# ':
#   0 * * * * /bin/bash $HOME/ProductGPT/work/scripts/hpcc_cron.sh >> $HOME/ProductGPT/logs/hpcc_cron.log 2>&1
#
# INSPECT / REMOVE:
#   crontab -l
#   crontab -r                      # removes ALL cron entries for this user
#   tail -f ~/ProductGPT/logs/hpcc_cron.log
#
# Requires ~/.telegram_env (see scripts/telegram_setup.sh). Without it the
# poller still writes to the log; only the Telegram push is skipped.
# ============================================================================
set -uo pipefail

REPO="${REPO:-$HOME/ProductGPT/work}"
cd "$REPO" || exit 0

# cron runs with a minimal environment and no PBS module, so hpcc_status.sh
# needs to be told where qstat lives. It already defaults to /opt/pbs/bin;
# this makes the dependency explicit and overridable.
export HPCC_PBS_BIN="${HPCC_PBS_BIN:-/opt/pbs/bin}"
export HPCC_STATE_FILE="${HPCC_STATE_FILE:-$HOME/.cache/hpcc_status.state}"

out="$(bash scripts/hpcc_status.sh --once 2>&1)"

[[ -z "$out" ]] && exit 0        # nothing changed; stay quiet

echo "[$(date +'%Y-%m-%dT%H:%M')] ---"
printf '%s\n' "$out"

# One Telegram message per poll, not per line, so four queued jobs do not
# become four notifications.
bash scripts/notify.sh "$out" || true
