//! Shared runtime-selection arg parsing for `#[proxima::test]` and
//! `#[proxima::main]`. Both attributes accept `runtime` / `flavor` /
//! `worker_threads`; keeping that parsing in one place is what makes the two
//! attributes consistent by construction (principle 2). `#[proxima::main]`
//! additionally accepts `cores` / `affinity` for the prime/adaptive path —
//! that surface is main-only (see `main_attr.rs`), since `#[proxima::test]`'s
//! `worker_threads` genuinely means a tokio thread-pool size, not a core
//! count.

use proc_macro2::Span;
use syn::{Error, Expr, ExprLit, Lit};

/// Which runtime backend the generated entry point drives the body on. The
/// adaptive `Default` selects prime when the prime runtime feature is
/// compiled (mirrored at the call site by `run`), else tokio.
pub enum RuntimeKind {
    /// adaptive default — `run` (prime when compiled, else tokio).
    Default,
    /// explicit prime — `run_prime`.
    Prime,
    /// explicit tokio current-thread.
    Tokio,
    /// tokio multi-thread with optional worker count.
    TokioMultiThread { workers: Option<usize> },
}

pub fn extract_str_lit(expr: &Expr, field: &str) -> Result<String, Error> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Str(lit_str),
            ..
        }) => Ok(lit_str.value()),
        _ => Err(Error::new_spanned(
            expr,
            format!("expected string literal for `{field}`"),
        )),
    }
}

pub fn extract_usize_lit(expr: &Expr, field: &str) -> Result<usize, Error> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(lit_int),
            ..
        }) => lit_int.base10_parse::<usize>().map_err(|error| {
            Error::new_spanned(expr, format!("invalid integer for `{field}`: {error}"))
        }),
        _ => Err(Error::new_spanned(
            expr,
            format!("expected integer literal for `{field}`"),
        )),
    }
}

pub fn extract_bool_lit(expr: &Expr, field: &str) -> Result<bool, Error> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Bool(lit_bool),
            ..
        }) => Ok(lit_bool.value()),
        _ => Err(Error::new_spanned(
            expr,
            format!("expected boolean literal for `{field}`"),
        )),
    }
}

/// Parse a `runtime = "prime" | "tokio"` value into the matching variant.
pub fn parse_runtime_value(value: &Expr) -> Result<RuntimeKind, Error> {
    match extract_str_lit(value, "runtime")?.as_str() {
        "prime" => Ok(RuntimeKind::Prime),
        "tokio" => Ok(RuntimeKind::Tokio),
        other => Err(Error::new_spanned(
            value,
            format!("unknown runtime `{other}`; expected \"prime\" or \"tokio\""),
        )),
    }
}

/// Fold a parsed `flavor` / `worker_threads` into `runtime`, applying the same
/// rules both attributes share: `flavor` is mutually exclusive with
/// `runtime = "prime"`; `current_thread`/`multi_thread` lower a `Default`/
/// `Tokio` runtime to the matching tokio flavor; a bare `worker_threads`
/// implies multi-thread.
pub fn fold_flavor(
    runtime: RuntimeKind,
    flavor: Option<String>,
    worker_threads: Option<usize>,
) -> Result<RuntimeKind, Error> {
    if let Some(flavor_str) = flavor {
        return match (flavor_str.as_str(), &runtime) {
            (_, RuntimeKind::Prime) => Err(Error::new(
                Span::call_site(),
                "`runtime = \"prime\"` and `flavor` are mutually exclusive",
            )),
            ("current_thread", RuntimeKind::Default | RuntimeKind::Tokio) => Ok(RuntimeKind::Tokio),
            ("multi_thread", RuntimeKind::Default | RuntimeKind::Tokio) => {
                Ok(RuntimeKind::TokioMultiThread {
                    workers: worker_threads,
                })
            }
            (other, _) => Err(Error::new(
                Span::call_site(),
                format!(
                    "unknown flavor `{other}`; expected \"current_thread\" or \"multi_thread\""
                ),
            )),
        };
    }
    if worker_threads.is_some() && matches!(runtime, RuntimeKind::Default | RuntimeKind::Tokio) {
        return Ok(RuntimeKind::TokioMultiThread {
            workers: worker_threads,
        });
    }
    Ok(runtime)
}

/// Best-effort length of an `affinity` literal IF it names an explicit
/// physical-core list — the two list-shaped forms
/// `prime::config::Affinity::from_str` parses into `Affinity::Cores`:
/// `"a,b,c"` (comma list) or `"a-b"` (inclusive range). Returns `None` for
/// every other shape (`""`/`"float"`/`"packed"`/a bare offset), which has no
/// fixed length to check `cores` against.
///
/// Used only for `#[proxima::main]`'s `cores`-vs-`affinity` compile-time
/// conflict check. It never re-implements `Affinity::from_str`'s grammar or
/// validation — a malformed list still parses fine here (or returns `None`,
/// skipping the check) and is caught, once, by the real `Affinity::from_str`
/// the generated code calls at runtime.
pub fn affinity_fixed_len(spec: &str) -> Option<usize> {
    let trimmed = spec.trim();
    if let Some((start, end)) = trimmed.split_once('-') {
        let start: usize = start.trim().parse().ok()?;
        let end: usize = end.trim().parse().ok()?;
        return (end >= start).then(|| end - start + 1);
    }
    if trimmed.contains(',') {
        return Some(trimmed.split(',').count());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::affinity_fixed_len;

    #[test]
    fn affinity_fixed_len_reads_comma_list() {
        assert_eq!(affinity_fixed_len("4,5,6,7"), Some(4));
    }

    #[test]
    fn affinity_fixed_len_reads_inclusive_range() {
        assert_eq!(affinity_fixed_len("4-7"), Some(4));
    }

    #[test]
    fn affinity_fixed_len_is_none_for_non_list_shapes() {
        assert_eq!(affinity_fixed_len(""), None);
        assert_eq!(affinity_fixed_len("float"), None);
        assert_eq!(affinity_fixed_len("packed"), None);
        assert_eq!(affinity_fixed_len("4"), None);
    }

    #[test]
    fn affinity_fixed_len_is_none_for_non_list_garbage() {
        assert_eq!(affinity_fixed_len("banana"), None);
    }

    // a descending range ("7-4") is a genuine `Affinity::from_str` grammar
    // error, not a length question — this helper only computes a length, so
    // it defers entirely (returns `None`) and lets the real parser report it
    // once, at runtime.
    #[test]
    fn affinity_fixed_len_is_none_for_descending_range() {
        assert_eq!(affinity_fixed_len("7-4"), None);
    }

    // a comma list with a non-numeric entry still LOOKS like a 3-item list
    // syntactically — the entry's validity is a grammar question this helper
    // never answers (it only counts, for the macro's length-conflict check);
    // `Affinity::from_str` reports the real parse error at runtime.
    #[test]
    fn affinity_fixed_len_counts_a_comma_list_even_with_a_bad_entry() {
        assert_eq!(affinity_fixed_len("4,x,6"), Some(3));
    }
}
