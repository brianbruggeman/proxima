#!/bin/sh
# Run a command in its own process group and kill the group on timeout.
#
# Usage: scripts/guard-run.sh SECONDS -- COMMAND [ARG...]
#
# The command is started by perl's setpgrp so it is a process-group leader; the
# watchdog kills -PGID, not just the parent. A killed shell or tool timeout can
# no longer leave a child holding the Metal lane.
set -eu

if [ "$#" -lt 3 ] || [ "$2" != "--" ]; then
    echo "usage: $0 SECONDS -- COMMAND [ARG...]" >&2
    exit 64
fi

seconds="$1"
shift 2

perl -e 'setpgrp(0, 0); exec @ARGV' "$@" &
pid=$!

(
    sleep "$seconds"
    kill -TERM -"$pid" 2>/dev/null || true
    sleep 5
    kill -KILL -"$pid" 2>/dev/null || true
) &
watchdog=$!

set +e
wait "$pid"
status=$?
set -e

kill "$watchdog" 2>/dev/null || true
wait "$watchdog" 2>/dev/null || true
exit "$status"
