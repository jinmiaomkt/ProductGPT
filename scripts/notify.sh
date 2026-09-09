#!/usr/bin/env bash
# ============================================================================
# Send a short notification to Telegram. Used by the PBS job script (on job
# end or failure) and by the local hourly status task.
#
#   bash scripts/notify.sh "message text"
#
# CREDENTIALS -- READ THIS
# -----------------------
# The bot token is a credential. It is read from the environment, NEVER from
# this repo, and it must never be committed. Put it in a file only you can
# read, on each machine that needs to send:
#
#   printf 'export TG_TOKEN=123456789:AA...\nexport TG_CHAT=987654321\n' > ~/.telegram_env
#   chmod 600 ~/.telegram_env
#
# This script sources ~/.telegram_env if it exists, so PBS jobs and scheduled
# tasks pick it up without it appearing in any command line (a token passed as
# an argument would be visible to anyone running `ps`).
#
# HOW TO GET THE TWO VALUES
#   1. In Telegram, message @BotFather and send /newbot. Answer its two
#      questions. It replies with the token -- that is TG_TOKEN.
#   2. Send any message (e.g. /start) to your new bot. A bot cannot start a
#      conversation, so this step is required.
#   3. Run:  curl -s "https://api.telegram.org/bot<TOKEN>/getUpdates"
#      and read the numeric "chat":{"id":...} out of the reply. That is
#      TG_CHAT.
#
# FAILURE POLICY: this script NEVER fails its caller. A missing token, no
# network, or a Telegram outage exits 0 quietly. A training run must not die
# because a notification could not be delivered.
# ============================================================================
set -uo pipefail

MSG="${1:-(no message)}"

[[ -f "$HOME/.telegram_env" ]] && . "$HOME/.telegram_env" 2>/dev/null

TG_TOKEN="${TG_TOKEN:-}"
TG_CHAT="${TG_CHAT:-}"

if [[ -z "$TG_TOKEN" || -z "$TG_CHAT" ]]; then
  # Not configured. Say so on stderr (which lands in the job log) and move on.
  echo "notify: TG_TOKEN/TG_CHAT not set; skipping Telegram" >&2
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "notify: curl not available; skipping Telegram" >&2
  exit 0
fi

# --data-urlencode keeps newlines and special characters intact.
code="$(curl -s -m 20 -o /dev/null -w '%{http_code}' \
     -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
     -d chat_id="${TG_CHAT}" \
     --data-urlencode "text=${MSG}" 2>/dev/null)" || code="000"

if [[ "$code" != "200" ]]; then
  echo "notify: Telegram returned HTTP ${code}" >&2
fi
exit 0
