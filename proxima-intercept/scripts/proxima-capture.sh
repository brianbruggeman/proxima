#!/bin/bash
# proxima-capture — the ONLY privileged thing the agent needs, made explicit and
# auditable. It does exactly four operations, nothing else:
#   sniff <seconds> <outfile>  : tcpdump UDP/443 to a /tmp/proxima-*.pcap for N secs
#   read  <pcapfile>           : tcpdump -nr on a /tmp/proxima-*.pcap (read it back)
#   on    <host> <ip>          : redirect that host's QUIC to the local quic-capture
#                                server — poison /etc/hosts (host->127.0.0.1) and add
#                                a pf rule sending loopback udp/443 to 127.0.0.1:4433
#   off                        : undo both (restore /etc/hosts, flush the pf anchor)
# No other commands. No arbitrary tcpdump/pfctl. Scope it in sudoers to THIS file:
#   <you> ALL=(ALL) NOPASSWD: /usr/local/sbin/proxima-capture.sh
# Install root-owned + immutable so it can't be edited/replaced to escalate:
#   sudo install -o root -g wheel -m 755 <repo>/proxima-capture.sh /usr/local/sbin/proxima-capture.sh
#   sudo chflags schg /usr/local/sbin/proxima-capture.sh

set -euo pipefail

HOSTS=/etc/hosts
MARK="# proxima-capture"
PF_ANCHOR=proxima-quic
QUIC_PORT=4433
TCP_PORT=8443

usage() { echo "usage: proxima-capture.sh sniff <seconds> <outfile> | read <pcap> | on <host> <ip> | off" >&2; exit 2; }

valid_host() { [[ "$1" =~ ^[a-zA-Z0-9._-]+$ ]]; }
valid_ip()   { [[ "$1" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; }
valid_secs() { [[ "$1" =~ ^[0-9]{1,3}$ ]]; }
valid_pcap() { [[ "$1" == /tmp/proxima-*.pcap ]]; }

cmd="${1:-}"
case "$cmd" in
  sniff)
    secs="${2:-}"; out="${3:-}"
    valid_secs "$secs" || usage
    valid_pcap "$out" || { echo "outfile must be /tmp/proxima-*.pcap" >&2; exit 2; }
    /usr/sbin/tcpdump -n -w "$out" 'udp and port 443' &
    td=$!
    sleep "$secs"
    kill "$td" 2>/dev/null || true
    wait "$td" 2>/dev/null || true
    echo "sniff done: $out"
    ;;
  read)
    pcap="${2:-}"
    valid_pcap "$pcap" || { echo "pcap must be /tmp/proxima-*.pcap" >&2; exit 2; }
    exec /usr/sbin/tcpdump -n -r "$pcap"
    ;;
  on)
    host="${2:-}"; ip="${3:-}"
    valid_host "$host" || usage
    valid_ip "$ip" || usage
    # per-host guard, not a blanket MARK guard — else a 2nd host never gets poisoned
    if ! grep -q " $host $MARK\$" "$HOSTS"; then
      echo "127.0.0.1 $host $MARK" >> "$HOSTS"
    fi
    printf 'rdr pass on lo0 inet proto tcp from any to any port 443 -> 127.0.0.1 port %s\nrdr pass on lo0 inet proto udp from any to any port 443 -> 127.0.0.1 port %s\n' "$TCP_PORT" "$QUIC_PORT" \
      | /sbin/pfctl -a "$PF_ANCHOR" -f -
    /sbin/pfctl -e 2>/dev/null || true
    echo "redirect ON: $host -> 127.0.0.1, lo0 tcp/443->$TCP_PORT udp/443->$QUIC_PORT (real upstream $ip)"
    ;;
  off)
    /sbin/pfctl -a "$PF_ANCHOR" -F all 2>/dev/null || true
    if grep -q "$MARK" "$HOSTS"; then
      grep -v "$MARK" "$HOSTS" > "${HOSTS}.proxima.tmp" && mv "${HOSTS}.proxima.tmp" "$HOSTS"
    fi
    echo "redirect OFF: /etc/hosts restored, pf anchor flushed"
    ;;
  *) usage ;;
esac
