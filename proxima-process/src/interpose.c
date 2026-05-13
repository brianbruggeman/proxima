/*
 * proxima-process libc-interpose shim.
 *
 * Compiled into libproxima_process_shim.{dylib,so} by build.rs.
 * Loaded into a child via DYLD_INSERT_LIBRARIES (macOS) or
 * LD_PRELOAD (Linux); intercepts a small set of libc symbols and
 * routes them through proxima-process's dispatch chain over the
 * fd named by PROXIMA_DISPATCH_FD.
 *
 * Two modes:
 *
 *   1. STATIC FALLBACK (env unset): the intercepted call returns
 *      a fixed spoofed value ("proxima-shimmed" for gethostname).
 *      Used by the libc_shim_smoke off/on test that does not wire
 *      a dispatch chain.
 *
 *   2. DISPATCH ROUND-TRIP (env set): reads PROXIMA_DISPATCH_FD,
 *      encodes a `ChildRequest::Read { path, max_bytes, offset }`
 *      via postcard, frames it with a u32-BE length prefix, writes
 *      to the fd, reads back the framed `ChildResponse`, decodes,
 *      and returns the bytes the dispatch chain provided.
 *
 * Wire format (LOCKED 2026-05-23 per
 * `proxima.decision.libc_shim_vm_parity`):
 *   [u32_be length][postcard payload]
 *
 * Postcard encoding rules (smoke subset):
 *   - varint(u32/u64) — LEB128
 *   - varint(i32)     — zigzag-LEB128
 *   - String / Vec<u8> — varint(len) + raw bytes
 *   - bool            — single byte (0/1)
 *   - enum discriminant — varint(u32) in derive order
 *
 * Variant discriminants (see protocol.rs module doc):
 *   ChildRequest::Read = 0, Write = 1, Open = 2, Close = 3, Stat = 4
 *   ChildResponse::Read(_) = 0, Write(_) = 1, Open{} = 2, Close = 3,
 *                    Stat{} = 4, Error{errno} = 5
 *
 * The .dylib MUST NOT contain any proxima_process Rust symbols —
 * only the C interpose entries — so the rlib that pty-tester (and
 * every other consumer) links never ends up with these exports.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Debug logging to stderr — gated on PROXIMA_SHIM_DEBUG env. Use
 * during development to diagnose dispatch RT failures; keep off in
 * production so the shim has zero stderr noise. */
#define SHIM_DEBUG(...) do { \
    if (getenv("PROXIMA_SHIM_DEBUG") != NULL) { \
        fprintf(stderr, "[proxima-shim] " __VA_ARGS__); \
        fputc('\n', stderr); \
    } \
} while (0)

#define DISPATCH_FD_ENV "PROXIMA_DISPATCH_FD"

#define CHILD_REQUEST_READ 0
#define CHILD_RESPONSE_READ 0
#define CHILD_RESPONSE_ERROR 5

/* Canonical proc-style paths the shim addresses. The dispatch
 * chain matches on these — adding new symbols below adds new
 * paths but does NOT change the protocol (`ChildRequest::Read`
 * handles them all). */
#define PATH_HOSTNAME  "/proc/sys/kernel/hostname"
#define PATH_OSTYPE    "/proc/sys/kernel/ostype"
#define PATH_OSRELEASE "/proc/sys/kernel/osrelease"
#define PATH_VERSION   "/proc/sys/kernel/version"
#define PATH_MACHINE   "/proc/sys/kernel/machine"

/* Static fallback values for the env-unset path. */
static const char SPOOFED_HOSTNAME[] = "proxima-shimmed";
#define SPOOFED_HOSTNAME_LEN 15
static const char SPOOFED_OSTYPE[]    = "ProximaOS";
static const char SPOOFED_OSRELEASE[] = "1.0.0";
static const char SPOOFED_VERSION[]   = "shim-stage-2";
static const char SPOOFED_MACHINE[]   = "proxima-arch";

/* ---- LEB128 varint encoder ----
 * Writes the value into `out` and returns the number of bytes
 * written. Caller must guarantee out has at least 10 bytes (max
 * varint for u64). */
static size_t leb128_encode_u64(uint64_t value, unsigned char *out) {
    size_t written = 0;
    while (value >= 0x80) {
        out[written++] = (unsigned char)((value & 0x7f) | 0x80);
        value >>= 7;
    }
    out[written++] = (unsigned char)(value & 0x7f);
    return written;
}

/* Decodes a LEB128 u64 starting at `in`. Returns bytes consumed
 * via *consumed; returns -1 on overflow. */
static int leb128_decode_u64(const unsigned char *in, size_t in_len,
                              uint64_t *out, size_t *consumed) {
    uint64_t result = 0;
    int shift = 0;
    size_t i = 0;
    while (i < in_len && i < 10) {
        unsigned char byte = in[i++];
        result |= (uint64_t)(byte & 0x7f) << shift;
        if ((byte & 0x80) == 0) {
            *out = result;
            *consumed = i;
            return 0;
        }
        shift += 7;
    }
    return -1;
}

/* Zigzag-encode an i32 for postcard (signed-int varint). */
static uint32_t zigzag_encode_i32(int32_t value) {
    return ((uint32_t)value << 1) ^ (uint32_t)(value >> 31);
}

/* Zigzag-decode a u32 back to i32. */
static int32_t zigzag_decode_i32(uint32_t value) {
    return (int32_t)((value >> 1) ^ -(int32_t)(value & 1));
}

/* ---- frame I/O helpers ---- */

/* Write all `len` bytes from `buf` to `fd`, looping past short
 * writes. Returns 0 on success, -1 on error. */
static int write_all(int fd, const unsigned char *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = write(fd, buf + total, len - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        total += (size_t)n;
    }
    return 0;
}

/* Read exactly `len` bytes into `buf` from `fd`, looping past
 * short reads. Returns 0 on success, -1 on EOF or error. */
static int read_all(int fd, unsigned char *buf, size_t len) {
    size_t total = 0;
    while (total < len) {
        ssize_t n = read(fd, buf + total, len - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) return -1;
        total += (size_t)n;
    }
    return 0;
}

/* Frame + send: [u32_be length][payload]. */
static int send_frame(int fd, const unsigned char *payload, uint32_t len) {
    unsigned char prefix[4];
    prefix[0] = (unsigned char)((len >> 24) & 0xff);
    prefix[1] = (unsigned char)((len >> 16) & 0xff);
    prefix[2] = (unsigned char)((len >> 8) & 0xff);
    prefix[3] = (unsigned char)(len & 0xff);
    if (write_all(fd, prefix, 4) < 0) return -1;
    if (write_all(fd, payload, len) < 0) return -1;
    return 0;
}

/* Read frame: [u32_be length][payload]. Caller's buf must be at
 * least max_len bytes; payload length placed in *out_len. */
static int recv_frame(int fd, unsigned char *buf, size_t max_len, size_t *out_len) {
    unsigned char prefix[4];
    if (read_all(fd, prefix, 4) < 0) return -1;
    uint32_t len = ((uint32_t)prefix[0] << 24)
                 | ((uint32_t)prefix[1] << 16)
                 | ((uint32_t)prefix[2] << 8)
                 |  (uint32_t)prefix[3];
    if (len > max_len) return -1;
    if (read_all(fd, buf, len) < 0) return -1;
    *out_len = len;
    return 0;
}

/* ---- generic dispatch round-trip for ChildRequest::Read ----
 *
 * Encode + send a ChildRequest::Read { path, max_bytes, offset: 0 },
 * read back the framed ChildResponse, decode, and write the bytes
 * the dispatch chain provided into `out`.
 *
 * Returns 0 on success (bytes written to `out`, len in *out_len).
 * Returns -1 on any error; caller can fall back to static or fail.
 * Returns -2 if the dispatch chain returned ChildResponse::Error;
 * *err_errno populated with the errno value to surface.
 */
static int dispatch_read_path(int dispatch_fd,
                               const char *path, size_t path_len,
                               unsigned char *out, size_t max_bytes,
                               size_t *out_len, int *err_errno) {
    /* Encode ChildRequest::Read { path, max_bytes, offset: 0 }.
     * Max encoded size: 1 (discriminant) + 10 (path-len varint)
     * + path_len (path bytes) + 5 (max_bytes varint) + 10 (offset
     * varint). For our smoke paths (<60 bytes), 128-byte buffer
     * is ample. */
    unsigned char req[256];
    if (path_len + 32 > sizeof(req)) {
        SHIM_DEBUG("path too long for stack-bounded request buffer");
        return -1;
    }
    size_t pos = 0;
    req[pos++] = CHILD_REQUEST_READ;                          /* discriminant */
    pos += leb128_encode_u64((uint64_t)path_len, req + pos);  /* path length varint */
    memcpy(req + pos, path, path_len);                        /* path bytes */
    pos += path_len;
    pos += leb128_encode_u64((uint64_t)max_bytes, req + pos); /* max_bytes */
    pos += leb128_encode_u64(0, req + pos);                   /* offset */

    SHIM_DEBUG("sending frame: %zu bytes payload to fd %d", pos, dispatch_fd);
    if (send_frame(dispatch_fd, req, (uint32_t)pos) < 0) {
        SHIM_DEBUG("send_frame failed: errno=%d (%s)", errno, strerror(errno));
        return -1;
    }
    SHIM_DEBUG("frame sent ok");

    /* Read response. Bound: 1 discriminant + 10 varint + max_bytes
     * + 1 eof bool + safety = max_bytes + 32. */
    size_t resp_capacity = max_bytes + 32;
    unsigned char *resp = (unsigned char *)malloc(resp_capacity);
    if (resp == NULL) return -1;
    size_t resp_len = 0;
    if (recv_frame(dispatch_fd, resp, resp_capacity, &resp_len) < 0) {
        SHIM_DEBUG("recv_frame failed: errno=%d (%s)", errno, strerror(errno));
        free(resp);
        return -1;
    }
    SHIM_DEBUG("frame received: %zu bytes payload", resp_len);
    if (resp_len < 1) {
        SHIM_DEBUG("response too short");
        free(resp);
        return -1;
    }

    unsigned char discriminant = resp[0];
    size_t cursor = 1;

    if (discriminant == CHILD_RESPONSE_READ) {
        /* ReadResponse { bytes: Vec<u8>, eof: bool } */
        uint64_t bytes_len = 0;
        size_t consumed = 0;
        if (leb128_decode_u64(resp + cursor, resp_len - cursor,
                              &bytes_len, &consumed) < 0) {
            free(resp);
            return -1;
        }
        cursor += consumed;
        if (cursor + bytes_len > resp_len) {
            free(resp);
            return -1;
        }
        size_t copy_len = (bytes_len <= max_bytes) ? (size_t)bytes_len : max_bytes;
        memcpy(out, resp + cursor, copy_len);
        *out_len = copy_len;
        free(resp);
        return 0;
    }

    if (discriminant == CHILD_RESPONSE_ERROR) {
        /* Error { errno: i32 } as zigzag-LEB128 */
        uint64_t raw = 0;
        size_t consumed = 0;
        if (leb128_decode_u64(resp + cursor, resp_len - cursor,
                              &raw, &consumed) < 0) {
            free(resp);
            return -1;
        }
        *err_errno = zigzag_decode_i32((uint32_t)raw);
        free(resp);
        return -2;
    }

    /* Unknown discriminant — fail. */
    free(resp);
    return -1;
}

/* ---- common dispatch helper for "fetch one path into one buffer" ----
 *
 * Tries the dispatch path; on error, copies `fallback` (NUL-terminated,
 * fallback_len bytes) into `out`. Returns 0 on success with bytes
 * written to *out_len; returns -1 on error (errno set if dispatch
 * returned ChildResponse::Error). NUL-terminates the buffer at
 * out[*out_len]; caller must ensure cap >= 1. */
static int fetch_one(int dispatch_fd_or_neg,
                     const char *path, size_t path_len,
                     const char *fallback, size_t fallback_len,
                     unsigned char *out, size_t cap, size_t *out_len) {
    if (cap == 0) { errno = ENAMETOOLONG; return -1; }
    if (dispatch_fd_or_neg >= 0) {
        size_t buf_used = 0;
        int err_errno = 0;
        size_t request_cap = (cap - 1 < cap) ? cap - 1 : cap;
        int result = dispatch_read_path(dispatch_fd_or_neg,
                                         path, path_len,
                                         out, request_cap,
                                         &buf_used, &err_errno);
        if (result == 0) {
            out[buf_used] = '\0';
            *out_len = buf_used;
            return 0;
        }
        if (result == -2) { errno = err_errno; return -1; }
        /* result == -1: dispatch failed; fall through to static. */
    }
    if (fallback_len + 1 > cap) { errno = ENAMETOOLONG; return -1; }
    memcpy(out, fallback, fallback_len);
    out[fallback_len] = '\0';
    *out_len = fallback_len;
    return 0;
}

/* Read PROXIMA_DISPATCH_FD env, return parsed fd or -1 if unset/invalid. */
static int dispatch_fd_from_env(void) {
    const char *fd_str = getenv(DISPATCH_FD_ENV);
    if (fd_str == NULL || *fd_str == '\0') return -1;
    int fd = atoi(fd_str);
    return (fd >= 0) ? fd : -1;
}

/* ---- gethostname interpose ---- */

static int shim_gethostname(char *name, size_t len) {
    if (name == NULL) { errno = EFAULT; return -1; }
    if (len == 0)    { errno = ENAMETOOLONG; return -1; }

    int dispatch_fd = dispatch_fd_from_env();
    SHIM_DEBUG("gethostname intercept; dispatch_fd=%d", dispatch_fd);

    unsigned char buf[256];
    size_t buf_used = 0;
    size_t cap = (len < sizeof(buf)) ? len : sizeof(buf);
    if (fetch_one(dispatch_fd, PATH_HOSTNAME, sizeof(PATH_HOSTNAME) - 1,
                  SPOOFED_HOSTNAME, SPOOFED_HOSTNAME_LEN,
                  buf, cap, &buf_used) < 0) {
        return -1;
    }
    if (buf_used + 1 > len) { errno = ENAMETOOLONG; return -1; }
    memcpy(name, buf, buf_used);
    name[buf_used] = '\0';
    return 0;
}

/* ---- uname interpose ----
 *
 * uname(2) returns a struct utsname with 5 fields: sysname,
 * nodename, release, version, machine. Each gets ONE dispatch RT
 * (5 `ChildRequest::Read` calls total) — no new protocol variant
 * needed, parity stays intact. Dispatch chain matches the
 * canonical /proc/sys/kernel/* paths.
 *
 * struct utsname's field sizes differ across platforms (macOS
 * 256-byte chars, Linux glibc 65-byte chars + optional
 * domainname). We #include <sys/utsname.h> to get the platform's
 * exact layout and use sizeof() on the actual fields. */
#include <sys/utsname.h>

/* Helper macro: write one field of struct utsname via fetch_one.
 * Bails the whole shim_uname on field-level error. */
#define UNAME_FIELD(field, path_macro, fallback_const)                                    \
    do {                                                                                  \
        size_t used = 0;                                                                  \
        if (fetch_one(dispatch_fd, path_macro, sizeof(path_macro) - 1,                    \
                      fallback_const, sizeof(fallback_const) - 1,                         \
                      (unsigned char *)buf->field, sizeof(buf->field), &used) < 0) {      \
            return -1;                                                                    \
        }                                                                                 \
    } while (0)

static int shim_uname(struct utsname *buf) {
    if (buf == NULL) { errno = EFAULT; return -1; }
    int dispatch_fd = dispatch_fd_from_env();
    SHIM_DEBUG("uname intercept; dispatch_fd=%d", dispatch_fd);

    UNAME_FIELD(sysname,  PATH_OSTYPE,    SPOOFED_OSTYPE);
    UNAME_FIELD(nodename, PATH_HOSTNAME,  SPOOFED_HOSTNAME);
    UNAME_FIELD(release,  PATH_OSRELEASE, SPOOFED_OSRELEASE);
    UNAME_FIELD(version,  PATH_VERSION,   SPOOFED_VERSION);
    UNAME_FIELD(machine,  PATH_MACHINE,   SPOOFED_MACHINE);
    return 0;
}

#ifdef __APPLE__

#define DYLD_INTERPOSE(_replacement, _replacee)                                            \
    __attribute__((used)) static struct {                                                  \
        const void *replacement;                                                           \
        const void *replacee;                                                              \
    } _interpose_##_replacee                                                               \
        __attribute__((section("__DATA,__interpose"))) = {                                 \
        (const void *)(unsigned long)&_replacement,                                        \
        (const void *)(unsigned long)&_replacee,                                           \
    };

extern int gethostname(char *name, size_t len);
DYLD_INTERPOSE(shim_gethostname, gethostname)

extern int uname(struct utsname *buf);
DYLD_INTERPOSE(shim_uname, uname)

#else /* Linux / other ELF: LD_PRELOAD shadow */

int gethostname(char *name, size_t len) {
    return shim_gethostname(name, len);
}

int uname(struct utsname *buf) {
    return shim_uname(buf);
}

#endif
