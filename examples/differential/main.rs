//! Differential testing — feed identical bytes to proxima's H1 request-head
//! parser AND an independently constructed `httparse::Request`, then assert
//! they agree. `proxima_protocols::http1_codec::h1` delegates its
//! grammar-level parsing to `httparse` internally (see its module docs), so
//! the two never disagree on RFC 7230 grammar by construction. The genuinely
//! independent surface is proxima's own wrapper: `ParserLimits` — a resource
//! budget (max method / path / header-line bytes, max header count)
//! `httparse` has no concept of. Where that budget is tighter than what the
//! grammar alone would accept, the two parsers diverge on purpose, and this
//! example documents each case instead of asserting it away.
//!
//! Builds on: `proxima-protocols`' sans-IO request-head parser
//! (`proxima-protocols/src/http1_codec/h1.rs`) — the codec `proxima-http`'s
//! H1 connection state machine sits on top of.
//!
//! Run:
//!     cargo run --example differential

use proxima_protocols::http1_codec::h1::{HttpVersion, ParserLimits, Status, parse_head_with_limits};

// mirrors proxima_protocols::http1_codec::h1's private `MAX_HEADERS` array
// cap so the oracle sees the same header-slot budget proxima's parser is
// built against.
const SHARED_HEADER_SLOT_CAP: usize = 100;

#[derive(Debug, Clone, PartialEq, Eq)]
enum ParseOutcome {
    Partial,
    Rejected,
    Complete {
        method: Vec<u8>,
        path: Vec<u8>,
        version: HttpVersion,
        headers: Vec<(Vec<u8>, Vec<u8>)>,
        consumed: usize,
    },
}

impl ParseOutcome {
    fn label(&self) -> &'static str {
        match self {
            Self::Partial => "partial",
            Self::Rejected => "rejected",
            Self::Complete { .. } => "complete",
        }
    }
}

enum Expectation {
    Agree,
    Diverges { why: &'static str },
}

struct Case {
    name: &'static str,
    input: &'static [u8],
    limits: ParserLimits,
    expect: Expectation,
}

const DEFAULT_LIMITS: ParserLimits = ParserLimits {
    max_method_bytes: 16,
    max_path_bytes: 8192,
    max_header_line_bytes: 8192,
    max_headers: 100,
};

const TIGHT_PATH_BUDGET: ParserLimits = ParserLimits {
    max_path_bytes: 16,
    ..DEFAULT_LIMITS
};

const TIGHT_HEADER_LINE_BUDGET: ParserLimits = ParserLimits {
    max_header_line_bytes: 20,
    ..DEFAULT_LIMITS
};

const SHARED_TWO_HEADER_SLOTS: ParserLimits = ParserLimits {
    max_headers: 2,
    ..DEFAULT_LIMITS
};

const CORPUS: &[Case] = &[
    Case {
        name: "GET root, no headers",
        input: b"GET / HTTP/1.1\r\n\r\n",
        limits: DEFAULT_LIMITS,
        expect: Expectation::Agree,
    },
    Case {
        name: "GET path+query, two headers",
        input: b"GET /a/b?x=1 HTTP/1.1\r\nHost: example.com\r\nUser-Agent: differential/1\r\n\r\n",
        limits: DEFAULT_LIMITS,
        expect: Expectation::Agree,
    },
    Case {
        name: "POST with content-length, body left unconsumed",
        input: b"POST /submit HTTP/1.1\r\nHost: example.com\r\nContent-Length: 5\r\n\r\nhello",
        limits: DEFAULT_LIMITS,
        expect: Expectation::Agree,
    },
    Case {
        name: "OPTIONS asterisk-form request-target",
        input: b"OPTIONS * HTTP/1.1\r\nHost: example.com\r\n\r\n",
        limits: DEFAULT_LIMITS,
        expect: Expectation::Agree,
    },
    Case {
        name: "HTTP/1.0 request",
        input: b"GET / HTTP/1.0\r\n\r\n",
        limits: DEFAULT_LIMITS,
        expect: Expectation::Agree,
    },
    Case {
        name: "header name case preserved verbatim",
        input: b"GET / HTTP/1.1\r\nCoNtEnT-lEnGtH: 0\r\n\r\n",
        limits: DEFAULT_LIMITS,
        expect: Expectation::Agree,
    },
    Case {
        name: "partial request line",
        input: b"GET /hello HT",
        limits: DEFAULT_LIMITS,
        expect: Expectation::Agree,
    },
    Case {
        name: "partial headers, no blank-line terminator",
        input: b"GET / HTTP/1.1\r\nHost: example.com\r\n",
        limits: DEFAULT_LIMITS,
        expect: Expectation::Agree,
    },
    Case {
        name: "unsupported HTTP/2.0 version",
        input: b"GET / HTTP/2.0\r\n\r\n",
        limits: DEFAULT_LIMITS,
        expect: Expectation::Agree,
    },
    Case {
        name: "control byte in method token",
        input: b"G\x00ET / HTTP/1.1\r\n\r\n",
        limits: DEFAULT_LIMITS,
        expect: Expectation::Agree,
    },
    Case {
        name: "control byte in header name",
        input: b"GET / HTTP/1.1\r\nBad\x00Name: x\r\n\r\n",
        limits: DEFAULT_LIMITS,
        expect: Expectation::Agree,
    },
    Case {
        name: "header count exceeds a shared 2-slot cap",
        input: b"GET / HTTP/1.1\r\nA: 1\r\nB: 2\r\nC: 3\r\n\r\n",
        limits: SHARED_TWO_HEADER_SLOTS,
        expect: Expectation::Agree,
    },
    Case {
        name: "method token longer than proxima's default 16-byte budget",
        input: b"EXTENSIONMETHODNAME / HTTP/1.1\r\n\r\n",
        limits: DEFAULT_LIMITS,
        expect: Expectation::Diverges {
            why: "httparse has no maximum method length; proxima's ParserLimits does",
        },
    },
    Case {
        name: "request-target within grammar, over a tightened path budget",
        input: b"GET /aaaaaaaaaaaaaaaaaaaa HTTP/1.1\r\n\r\n",
        limits: TIGHT_PATH_BUDGET,
        expect: Expectation::Diverges {
            why: "httparse accepts any request-target length; proxima's budget is a DoS guard, not grammar",
        },
    },
    Case {
        name: "header line within grammar, over a tightened line budget",
        input: b"GET / HTTP/1.1\r\nX-Long-Header-Name: still-a-valid-value\r\n\r\n",
        limits: TIGHT_HEADER_LINE_BUDGET,
        expect: Expectation::Diverges {
            why: "httparse has no per-header-line length cap; proxima's budget is a DoS guard, not grammar",
        },
    },
];

// independently constructed httparse invocation — never observes proxima's
// own call, so agreement isn't just proxima calling into itself twice.
fn oracle_parse(input: &[u8], header_slots: usize) -> ParseOutcome {
    let mut header_storage = vec![httparse::EMPTY_HEADER; header_slots];
    let mut request = httparse::Request::new(&mut header_storage);
    match request.parse(input) {
        Ok(httparse::Status::Partial) => ParseOutcome::Partial,
        Err(_) => ParseOutcome::Rejected,
        Ok(httparse::Status::Complete(consumed)) => {
            let version = match request.version {
                Some(0) => HttpVersion::Http10,
                Some(1) => HttpVersion::Http11,
                _ => return ParseOutcome::Rejected,
            };
            ParseOutcome::Complete {
                method: request.method.unwrap_or_default().as_bytes().to_vec(),
                path: request.path.unwrap_or_default().as_bytes().to_vec(),
                version,
                headers: request
                    .headers
                    .iter()
                    .map(|header| (header.name.as_bytes().to_vec(), header.value.to_vec()))
                    .collect(),
                consumed,
            }
        }
    }
}

fn proxima_parse(input: &[u8], limits: ParserLimits) -> ParseOutcome {
    match parse_head_with_limits(input, limits) {
        Ok(Status::Partial) => ParseOutcome::Partial,
        Err(_) => ParseOutcome::Rejected,
        Ok(Status::Complete { head, consumed }) => ParseOutcome::Complete {
            method: head.method.to_vec(),
            path: head.path.to_vec(),
            version: head.version,
            headers: head
                .headers
                .iter()
                .map(|header| (header.name().to_vec(), header.value().to_vec()))
                .collect(),
            consumed,
        },
    }
}

fn main() {
    println!("differential: proxima's h1 parser vs httparse, same bytes, independent parse\n");
    println!("{:<58} {:<10} {:<10} agree?", "input", "proxima", "oracle");

    let mut agreed = 0usize;
    let mut documented_divergences = 0usize;

    for case in CORPUS {
        let header_slots = case.limits.max_headers.min(SHARED_HEADER_SLOT_CAP);
        let oracle_outcome = oracle_parse(case.input, header_slots);
        let proxima_outcome = proxima_parse(case.input, case.limits);
        let agree = oracle_outcome == proxima_outcome;

        println!(
            "{:<58} {:<10} {:<10} {}",
            case.name,
            proxima_outcome.label(),
            oracle_outcome.label(),
            if agree { "yes" } else { "documented-no" }
        );

        match case.expect {
            Expectation::Agree => {
                assert!(
                    agree,
                    "case {:?}: expected agreement but proxima={:?} oracle={:?}",
                    case.name, proxima_outcome, oracle_outcome
                );
                agreed += 1;
            }
            Expectation::Diverges { why } => {
                assert!(
                    !agree,
                    "case {:?}: expected a documented divergence but the two parsers agreed \
                     ({proxima_outcome:?}) — the divergence closed, update the docs",
                    case.name
                );
                documented_divergences += 1;
                println!("  documented divergence: {} — {why}", case.name);
            }
        }
    }

    println!(
        "\n{agreed} of {} inputs: proxima and httparse agree on the parsed result (or both reject)",
        CORPUS.len()
    );
    println!(
        "{documented_divergences} documented divergence(s): proxima's resource-budget limits reject \
         inputs httparse's bare grammar accepts — a human-adjudicated tradeoff, not a bug"
    );
}
