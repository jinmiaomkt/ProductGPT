#!/usr/bin/env bash
# ============================================================================
# Finish Telegram notification setup: verify the token and discover the chat id.
#
# WHY THIS EXISTS
# ---------------
# The obvious way to get a chat id is
#     curl "https://api.telegram.org/bot<TOKEN>/getUpdates"
# which puts the token on the command line, where `ps` and the shell history
# both capture it. This script reads the token out of ~/.telegram_env instead,
# so it is never typed into a command and never reaches a history file.
#
# USAGE
#   1. Create the file with your token (paste it inside an editor, not on a
#      command line):
#
#        umask 077
#        printf 'export TG_TOKEN=\nexport TG_CHAT=\n' > ~/.telegram_env
#        nano ~/.telegram_env        # put the token after TG_TOKEN=
#
#   2. In Telegram, send /start to your bot. A bot cannot open a conversation,
#      so without this getUpdates returns an empty list.
#
#   3. bash scripts/telegram_setup.sh
#
# The script verifies the token, finds the chat id, writes it back into
# ~/.telegram_env, and sends a test message. Run it on every machine that
# needs to send -- OMEGA and the laptop -- since each has its own home dir.
# ============================================================================
set -uo pipefail

ENVFILE="$HOME/.telegram_env"

if [[ ! -f "$ENVFILE" ]]; then
  echo "ERROR: $ENVFILE does not exist. Create it first:" >&2
  echo "    umask 077; printf 'export TG_TOKEN=\\nexport TG_CHAT=\\n' > $ENVFILE" >&2
  echo "    nano $ENVFILE" >&2
  exit 1
fi

perms="$(stat -c '%a' "$ENVFILE" 2>/dev/null || echo '?')"
if [[ "$perms" != "600" && "$perms" != "?" ]]; then
  echo "WARNING: $ENVFILE is mode $perms; tightening to 600" >&2
  chmod 600 "$ENVFILE"
fi

# shellcheck disable=SC1090
. "$ENVFILE"
TG_TOKEN="${TG_TOKEN:-}"

if [[ -z "$TG_TOKEN" || "$TG_TOKEN" == *"<"* || "$TG_TOKEN" == *"PASTE"* ]]; then
  echo "ERROR: TG_TOKEN in $ENVFILE is empty or still a placeholder." >&2
  echo "       Edit the file and put the real token after TG_TOKEN=" >&2
  exit 1
fi

api() { curl -s -m 20 "https://api.telegram.org/bot${TG_TOKEN}/$1"; }

echo "==> verifying the token"
me="$(api getMe)"
if ! grep -q '"ok":true' <<<"$me"; then
  echo "FAILED. Telegram says:" >&2
  echo "  $me" >&2
  echo "If this is 401 Unauthorized the token is wrong or has been revoked." >&2
  exit 1
fi
echo "    ok: $(sed -n 's/.*"username":"\([^"]*\)".*/\1/p' <<<"$me")"

echo "==> looking for a chat id"
updates="$(api getUpdates)"

chat_id="$(python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
ids = []
for u in d.get("result", []):
    for k in ("message", "edited_message", "channel_post", "my_chat_member"):
        c = (u.get(k) or {}).get("chat")
        if c and c.get("id") is not None:
            ids.append(str(c["id"]))
print(ids[-1] if ids else "")
' <<<"$updates" 2>/dev/null)"

if [[ -z "$chat_id" ]]; then
  echo "    no chat found." >&2
  echo "    Send /start to your bot in Telegram, then run this again." >&2
  echo "    (A bot cannot message you until you have messaged it once.)" >&2
  exit 1
fi
echo "    found chat id: $chat_id"

# Rewrite TG_CHAT in place, leaving the token line untouched.
tmp="$(mktemp)"
grep -v '^export TG_CHAT=' "$ENVFILE" > "$tmp"
printf 'export TG_CHAT=%s\n' "$chat_id" >> "$tmp"
mv "$tmp" "$ENVFILE"
chmod 600 "$ENVFILE"
echo "==> wrote TG_CHAT to $ENVFILE"

echo "==> sending a test message"
if bash "$(dirname "$0")/notify.sh" "ProductGPT notifications are live on $(hostname)."; then
  echo "    sent. Check Telegram."
fi
