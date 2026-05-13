# 2026-07-01 — host-b incumbent matrix

Campaign: cores {1,2,4,8} × forms {rekt→incumbent, rekt→proxima,
incumbent→proxima, incumbent→incumbent} × protocols {h1,h2,h3}, plus h3
part-source sub-arms. Host: host-b (8-core Linux), server pinned
c0..N-1, client pinned c4-7, 3 runs × 8 s per cell, means reported,
0 errors unless noted. Incumbents: wrk+nginx (h1), h2load+nginx-h2c
(h2), h2load-h3+nginx-http3 (h3, quiche). Tools: rekt_load / rekt_h2
(rekt repo examples, built against this tree), rekt_h3_load +
bench_server{,_h2,_h3} (this tree + rekt repo). Script: `/tmp/campaign.sh`
(host-b); raw cells `/tmp/camp/*.out`.

Caveats up front: (a) the 8c rows are client/server-contaminated (12
pinned threads on 8 cores) — read 1-4c for clean signal; (b) per the
saturated-incumbent rule, client-vs-client comparisons are only valid in
columns where the server has headroom; (c) server-side RSS/CPU sampling
mis-instrumented (`ps` caught the `time` wrapper) — client-side CPU/RSS
is complete in the raw cells; (d) h3 shows a trickle of errors at
saturation in BOTH owned and source arms (≤23 per ~3.3M requests,
≤0.0007%) — not source-specific.

## h1 — requests/sec (mean of 3)

| cores | rekt→prox | wrk→prox | rekt→ngx | wrk→ngx | server axis (prox/ngx) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 65,066 | 65,399 | 131,272 | 131,620 | 0.50x |
| 2 | 113,019 | 113,037 | 233,940 | 241,425 | 0.47x |
| 4 | 225,408 | 224,292 | 475,793 | 485,445 | 0.46x |
| 8† | 245,857 | 243,520 | 274,380 | 318,901 | — |

Client axis: rekt ties wrk within ±1-3% on BOTH server columns (1-4c).
Server axis: **proxima h1 ≈ 0.46-0.50x nginx at every clean core count**
— the h1 server is the largest untouched gap in the whole matrix
(streaming-dispatch per trivial GET; the server-side analog of what
send_raw fixed on the client, already named in the rekt-vs-wrk log).
Client RSS at c4: rekt 6.5 MB vs wrk 4.9 MB.

## h2 — requests/sec (mean of 3)

| cores | rekt→prox | h2load→prox | rekt→ngx | h2load→ngx | server axis (prox/ngx) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 675,176 | 698,097 | 91,964 | 93,887 | **7.4x** |
| 2 | 1,223,268 | 1,264,073 | 172,328 | 173,349 | **7.3x** |
| 4 | 2,191,774 | 2,151,100 | 330,897 | 323,805 | **6.6x** |
| 8† | 2,231,946 | 2,087,069 | 503,009 | 383,035 | — |

Server axis: **proxima h2c ≈ 6.6-7.4x nginx-h2 per core** — the
strongest cell block proxima owns. Client axis: rekt ties h2load
(±3%) on both columns at 1-4c. Client RSS at c4: **rekt_h2 996 MB vs
h2load 57 MB (17.5x)** — see the RSS finding below.

## h3 — requests/sec (mean of 3)

| cores | rekt→prox | h2load→prox | rekt→ngx | h2load→ngx | server axis h2load col (prox/ngx) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 149,436 | 127,833 | 291,879 | 212,651 | 0.60x |
| 2 | 240,819 | 224,272 | 523,013 | 496,635 | 0.45x |
| 4 | 417,347 | 379,131 | 826,467 | 1,082,229 | 0.35x |
| 8† | 376,512 | 381,312 | 556,824 | 813,588 | — |

Server axis: **proxima h3 ≈ 0.35-0.60x nginx-quiche** — consistent with
the 0.38x standing; still the second-largest gap after h1. Client axis
(headroom column = nginx at c4): rekt 0.76x h2load — matches the 0.73x
standing. Notably rekt EXTRACTS MORE from the proxima server than
h2load does (+10-17% at 1-4c), and beats h2load against nginx at low
server cores (server-saturated column — not a client verdict per
caveat b). Client RSS at c4: **rekt_h3_load 822 MB vs h2load 60 MB**.

## h3 part-source sub-arms — rekt→proxima column (mean of 3)

| cores | owned/owned | client-src | server-src | both-src | both vs owned |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 149,436 | 162,697 | 134,270 | 128,026 | 0.86x |
| 2 | 240,819 | 252,158 | 261,544 | 273,664 | **+13.6%** |
| 4 | 417,347 | 414,886 | 439,785 | 440,276 | **+5.5%** |
| 8† | 376,512 | 378,758 | 414,411 | 439,392 | **+16.7%** |

Off-loopback verdict for the flip decision: at 2+ server cores the
source path WINS — server-source +5.4% to +10.1%, both-source +5.5% to
+16.7%. The c1 row inverts (both-src 0.86x) — single-core server is a
different regime (server fully saturated, the deferred decode possibly
competing with the recv loop on the one core); do not average it away,
it bounds where a default flip applies. Errors appear equally in owned
and source arms at saturation (caveat d) — no correctness signal
against source. **Flip recommendation: server default is justified at
multi-core; take it as its own reviewed change, and re-run the c1
regime before flipping anything there.**

## Cross-cutting findings

1. **Client memory**: rekt h2/h3 drivers carry ~0.8-1.0 GB RSS vs
   incumbents' ~60 MB (h1 rekt is fine at 6.5 MB). Suspect: prime
   runtime lane-pool eager allocation at 4 cores × deep lanes (the
   720 MB eager-alloc shape fixed once before in the dynamic-inbox
   work). UNDIAGNOSED — needs its own pass; the bill-mover here is
   "client library" positioning, not the bench.
2. **Priority order by cell value**: h1 server (0.46x, biggest volume
   claim), h3 server (0.35x at 4c), h3 client (0.76x at headroom).
   h2 is the moat cell block (6.6-7.4x) — publishable number.
3. rekt-vs-wrk/h2load client parity holds everywhere servers have
   headroom except h3 (0.76x — the known structural per-op alloc gap,
   partially closed by the source path).

## CORRECTIONS (2026-07-01, post-publication audit)

Reader flagged anomalies; audited with data. Three findings:

1. **h2 client axis RETRACTED — concurrency mismatch.** `rekt_h2`'s
   second arg is connections **per core** (its own usage line says so);
   the campaign passed 16, so every h2 rekt cell ran 64 connections
   (2048 streams) vs h2load's 16 (512). The h2 "clients tie" read was
   invalid as published. Re-run at MATCHED parity (4/core = 16 total,
   server 4c, 3 runs each): rekt 1904-2442k vs h2load 1849-2076k —
   a genuine tie within run noise at ~2M req/s. The h2 SERVER-axis
   numbers stand (server-bound in both columns; the corrected-parity
   runs reproduce the same ~2M plateau the 64-conn cells hit). h1
   (25/core×4 = wrk -c100) and h3 (16 total = -c16, verified in
   source) were correctly matched all along.
2. **h3 low-core scaling is confounded — super-linear nginx scaling
   (212k→497k→1082k for 1→2→4 workers) is physically implausible.**
   Probe: re-pinning the 1-worker nginx from c0 to c1 produced 208k
   then 306k (+44% on run 2) — low-core h3 cells are contaminated by
   core-placement/softirq effects and show large run spread. Treat
   the h3 1c/2c rows (BOTH columns — proxima's worker was also on c0)
   as unreliable for scaling reads; the c4 row is credible (h2load→ngx
   1082k matches the prior session's independent ~1.12M standing).
   The h3 source-arm deltas at c2 are therefore weak evidence; the c4
   (+5.5%) and c8 (+16.7%) source wins and the c1 inversion stand as
   same-cell A/B (identical placement both arms).
3. **Port-squatting check done retroactively**: no stale nginx exists
   on the host now, and nothing in the campaign killed non-campaign
   nginxes — so no squatter was alive during the runs (the 12:59
   nginx.pid predates the campaign; its process is gone). Gap
   acknowledged: the script never verified exclusive binds before
   benching (the standing gotcha) — future campaigns must `ss -lun`
   per port before each block.

4. **Governor audited — cleared**: all 8 CPUs on host-b are in the
   `performance` governor (cpu0 observed at 4.6 GHz of 4.9 max,
   i7-9700K), so the campaign was not throttle-confounded.
5. **Absolute-number scoping (the "~2-3M on h2" question)**: h2load's
   own audit of the h2 c1 cell shows the exchanges are real — 5.53M
   requests/8 s on ONE server core, 100% 2xx, but the wire cost is
   degenerate: identical responses drive HPACK to full dynamic-table
   indexing (~1 B/request of header traffic, 90% savings reported by
   h2load) + a 2-byte body ≈ ~30 B/request total. Normalized, 691k/core
   at 4.6 GHz ≈ 150k/GHz-core vs TechEmpower plaintext leaders'
   ~86k/GHz-core — same order once you account for loopback (no NIC),
   no Date-header regeneration, no realistic headers, and a 2-byte
   identical body. These absolutes are NOT TechEmpower-comparable
   claims and must not be quoted as such; only within-matrix ratios
   carry meaning, and even the 7.4x-vs-nginx is scoped to this
   degenerate tiny-identical-response h2c shape (nginx pays full
   general-server per-request work; its h2 stack is untuned for
   deep-multiplex tiny requests). A TE-comparable claim would need the
   TE plaintext workload shape (realistic headers, Date, 13-B body,
   real network) — not run here.

---

## SERVER-AXIS RE-MEASURE 2026-07-19 (host-b) — supersedes the h1/h3 cells above

Re-ran the server axis (proxima server ÷ nginx, same client each column) on
**host-b, HEAD 0a7c45b2** (native-QUIC h3 server + both handshake bugs fixed
since 2026-07-01: coalesced-datagram drop, stale-`now` reap). nginx **1.30.3
`--with-http_v3_module`**. **Response parity: every server returns 200 + 2-byte
body `"ok"`.** Server `taskset -c 0..N-1`, client `-c 4-7`, 10s windows, 1 warmup
+ 3 measured. **0 errors and 0 ProtocolViolation in every cell.** All numbers
below reconciled against `~/rektbench/fanout/results/server-matrix-2026-07/` raw
per-trial logs.

| proto | cores | proxima mean | nginx mean | ratio (prox/ngx) | vs 2026-07-01 |
|---|---|---|---|---|---|
| h3 | 1 / 2 / 4 | 150,306 / 246,907 / 351,976 | 311,534 / 498,800 / 747,049 | **0.48 / 0.50 / 0.47x** | 0.35–0.60x → ~flat ratio, now 0-error |
| h1 | 1 / 2 / 4 | 125,854 / 236,544 / 357,827 | 126,031 / 238,044 / 396,773 | **1.00 / 0.99 / 0.90x** | 0.46–0.50x → **now parity** |
| h2c | 1 / 2 / 4 | 693,912 / 1,292,669 / 1,878,517* | 87,447 / 162,297 / 305,655 | **7.94 / 7.96 / 6.15x** | 6.6–7.4x → holds |

*h2 c4 CoV 16% (range 1,444,538–2,153,571, client-side scheduling swing at 2M+
rps; nginx c4 rock-stable at 305k). Moat holds regardless.

Three findings that change the read of the earlier table:

1. **The h1 "0.46–0.50x biggest gap" was half a concurrency-cliff artifact.**
   proxima h1 **ties nginx at ≤~40–50 conns/core** but **collapses above ~100
   conns/core** — single-core sweep: peak ~138k → c100 54k (with 207k client
   read-errors) → c200 20 rps, while nginx holds ~131k flat. The 2026-07-01 run
   drove c100 at 1 core, landing proxima IN its collapse zone. Held at 40
   conns/core (its stable range), h1 server is **~parity**. The real remaining h1
   issue is the **cliff** (accept/connection-handling saturation above ~100
   conns/core), not steady-state throughput.

2. **The native-h3 server's payoff is correctness + latency, not throughput.**
   Ratio barely moved (0.47x at c4), but it is now **0-error** (baseline had a
   ≤0.0007% trickle) and **lower per-request latency than nginx** — c4 p99
   4.46ms vs 7.64ms, TTFB 10.68ms vs 23.10ms. proxima h3 is **latency-competitive,
   parallelism-behind**: faster per request, less aggregate throughput per core
   (a scaling/batching gap). NB the old cell measured a since-replaced quinn
   server vs nginx-quiche; this is native-proxima vs nginx openssl-http3, so only
   the ratio is comparable.

3. **h3 ALPN strictness:** h2load-h3 must be forced with `--h3` or proxima's
   native handshake rejects it (nginx and `curl --http3` both work). Both columns
   used `--h3` so the comparison is fair; noted as a native-h3 interop edge.

Server CPU%/RSS not sampled this run (same gap the 2026-07-01 campaign flagged).
