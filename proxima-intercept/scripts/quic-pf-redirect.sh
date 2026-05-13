#!/usr/bin/env bash
# Redirect an app's outbound QUIC (UDP:443) for one host to the local quic-capture
# server, for a live transparent-QUIC capture. macOS / pf.
#
# Usage:
#   sudo ./quic-pf-redirect.sh on  api2.cursor.sh   # start redirect
#   sudo ./quic-pf-redirect.sh off api2.cursor.sh   # stop + restore
#
# Approach: poison /etc/hosts so the app sends QUIC to 127.0.0.1, then pf rdr
# loopback UDP:443 -> the quic-capture port. Run quic-capture first:
#   cargo run -p proxima-intercept --example quic-capture --features quic-intercept
#
# CAVEAT (honest, read before relying on this): with /etc/hosts poisoned, the
# proxy's own reoriginate_h3() DNS lookup ALSO resolves <host> to 127.0.0.1 and
# loops back into the proxy. To close the loop the proxy must reach the REAL
# upstream IP, bypassing the poisoned entry. Two operator options until
# reoriginate_h3 takes an explicit upstream addr:
#   1. capture the real IP first (`dig +short <host> @1.1.1.1`) and pass it to the
#      proxy out-of-band, or
#   2. use a transparent pf rdr that preserves the original destination (no hosts
#      poisoning) and have the proxy read it via the pf state table — not wired yet.
# So this script is sufficient for OBSERVE+dump of the request (the capture goal);
# a fully-relaying live run needs the real-upstream-IP path. Documented, not faked.

set -euo pipefail

QUIC_PORT="${PROXIMA_QUIC_PORT:-4433}"
HOSTS_FILE=/etc/hosts
HOSTS_MARK="# proxima-quic-redirect"
PF_ANCHOR=proxima-quic

action="${1:-}"
host="${2:-}"
if [[ -z "$action" || -z "$host" ]]; then
  echo "usage: sudo $0 on|off <host>" >&2
  exit 2
fi

case "$action" in
  on)
    if ! grep -q "$HOSTS_MARK" "$HOSTS_FILE"; then
      echo "127.0.0.1 $host $HOSTS_MARK" >> "$HOSTS_FILE"
      echo "poisoned $host -> 127.0.0.1 in $HOSTS_FILE"
    fi
    echo "rdr pass on lo0 inet proto udp from any to any port 443 -> 127.0.0.1 port $QUIC_PORT" \
      | pfctl -a "$PF_ANCHOR" -f - 2>/dev/null
    pfctl -e 2>/dev/null || true
    echo "pf rdr active: lo0 udp/443 -> 127.0.0.1:$QUIC_PORT (anchor $PF_ANCHOR)"
    echo "now run quic-capture on :$QUIC_PORT and launch the app; dumps land in PROXIMA_INTERCEPT_H2_DUMP or /tmp"
    ;;
  off)
    pfctl -a "$PF_ANCHOR" -F all 2>/dev/null || true
    if grep -q "$HOSTS_MARK" "$HOSTS_FILE"; then
      grep -v "$HOSTS_MARK" "$HOSTS_FILE" > "${HOSTS_FILE}.tmp" && mv "${HOSTS_FILE}.tmp" "$HOSTS_FILE"
      echo "removed $host poisoning from $HOSTS_FILE"
    fi
    echo "redirect off"
    ;;
  *)
    echo "unknown action: $action" >&2
    exit 2
    ;;
esac
