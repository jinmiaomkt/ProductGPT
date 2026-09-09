@echo off
REM ===========================================================================
REM Hourly OMEGA status check, for Windows Task Scheduler.
REM
REM Wraps scripts/hpcc_status.sh so Task Scheduler has a single thing to call
REM without fighting nested-quote escaping. Runs as the logged-in user, so it
REM picks up %USERPROFILE%\.ssh\id_ed25519 and can authenticate to OMEGA.
REM
REM The poller is change-only, so this log stays EMPTY while nothing happens.
REM An empty log means "no state changes", not "the task did not run" -- the
REM heartbeat line below distinguishes the two.
REM
REM REGISTER (one line, from any shell):
REM   schtasks /Create /TN "ProductGPT HPCC status" /SC HOURLY /F ^
REM            /TR "C:\Users\jinmiao\ProductGPT\ProductGPT\scripts\hpcc_status_task.cmd"
REM
REM INSPECT / REMOVE:
REM   schtasks /Query  /TN "ProductGPT HPCC status" /V /FO LIST
REM   schtasks /Run    /TN "ProductGPT HPCC status"
REM   schtasks /Delete /TN "ProductGPT HPCC status" /F
REM
REM NOTE: this only runs while the machine is awake and on a network that can
REM reach OMEGA. A closed laptop polls nothing. The PBS mail directive in
REM gen5_train_hpcc.pbs (-m ae) is the channel that does not depend on this.
REM ===========================================================================

cd /d "%~dp0.."

"C:\Program Files\Git\bin\bash.exe" -lc "mkdir -p logs; printf '%%s heartbeat\n' \"$(date +'%%Y-%%m-%%dT%%H:%%M')\" >> logs/hpcc_heartbeat.log; out=$(bash scripts/hpcc_status.sh --once 2>&1); if [ -n \"$out\" ]; then printf '%%s\n' \"$out\" >> logs/hpcc_status.log; bash scripts/notify.sh \"$out\"; fi"
