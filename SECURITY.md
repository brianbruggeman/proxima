# Security Policy

## Supported Versions

Proxima is pre-1.0. Security fixes land on `main` first. Until a stable release
line exists, there is no guaranteed backport policy for older commits or tags.

## Reporting a Vulnerability

Please do not open a public issue for a suspected vulnerability.

Preferred reporting path:

1. Use GitHub private vulnerability reporting for this repository, if enabled.
2. If private reporting is not available, contact the maintainer through an
   existing private channel, or open a public issue that asks for a security
   contact without including technical details.

Include enough detail to reproduce and assess the issue:

- affected crate, feature flag, command, listener, or protocol path;
- platform and target architecture;
- minimal config or packet/request shape;
- impact, such as credential disclosure, request smuggling, sandbox escape,
  denial of service, unsafe memory behavior, or policy bypass;
- whether any public exploit, capture, or third-party disclosure already exists.

Do not include real credentials, API keys, private captures, production traffic,
or customer data. Use synthetic fixtures whenever possible.

## Scope

Security-sensitive areas include, but are not limited to:

- TLS, QUIC, HTTP/1, HTTP/2, HTTP/3, WebSocket, proxy protocol, and wire codecs;
- auth, client-auth, request signing, replay caches, and secret handling;
- interception, CONNECT proxying, certificate generation, and recording/replay;
- process, PTY, libc-shim, VM, DPDK, NVMe, pmem, and other host-boundary code;
- config loading, hot-swap, control-plane, MCP, and pipeline execution.

Public benchmark-only issues are usually not security vulnerabilities unless
they expose secrets, enable code execution, corrupt memory, or materially change
the behavior of production code.

## Response Expectations

This is an independent project with no commercial support SLA. The maintainer
will make a best-effort acknowledgement, triage, and fix plan. When practical,
fixes should include a regression test, a short impact note, and any required
fixture redaction.

Coordinated disclosure is preferred. Please give the maintainer a reasonable
opportunity to ship a fix before publishing exploit details.
