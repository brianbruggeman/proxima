//! `#[proxima::main]` — the production sibling of `#[proxima::test]`. Turns
//! `async fn main() -> R { body }` into a sync `fn main() -> R` that boots a
//! runtime and drives the body to completion via `proxima::runtime::run*`.
//!
//! Runtime-selection surface: adaptive default, `runtime = "prime"|"tokio"`,
//! `flavor = "current_thread"|"multi_thread"` (shared parsing with
//! `#[proxima::test]` in `runtime_args`), plus two vocabularies that do NOT
//! overlap:
//!
//! - `cores = N` / `affinity = "<spec>"` — the prime/adaptive path's own
//!   vocabulary, mirroring `prime::config::CoreSelection` (count) and
//!   `prime::config::Affinity` (placement) so `#[proxima::main]` speaks the
//!   same language prime's own config/builder surface does, rather than
//!   borrowing tokio's `worker_threads`. `affinity`'s grammar is exactly what
//!   `Affinity::from_str` parses (`"float"`/`"packed"`/a bare offset/
//!   `"a,b,c"`/`"a-b"`) — parsed once, at runtime, by that same function;
//!   the macro never re-implements it.
//! - `worker_threads = N` — ONLY valid with an explicit `runtime = "tokio"`
//!   or `flavor = "multi_thread"`, where it is genuinely a thread-pool size
//!   (folded via `fold_flavor`, shared with `#[proxima::test]`).
//!
//! Mixing the two vocabularies is a compile error: `cores` + `worker_threads`
//! together, `affinity` (or `cores`/`worker_threads` on the wrong side) with
//! an explicit tokio runtime, and `cores = N` disagreeing with an explicit
//! `affinity` core-list's length all fail to expand rather than silently
//! picking one.
//!
//! The booted runtime is published via `proxima::runtime::install_runtime`
//! so `App::new()` called from `main`'s body adopts it instead of building an
//! independent second one — one `#[proxima::main(cores = N)]` now means one
//! N-core runtime, not two runtimes with contradictory core counts. Bare
//! `#[proxima::main]` (no `cores`) resolves to
//! [`CoreSelection::Auto`](prime::config::CoreSelection::Auto) — all
//! physical cores, matching
//! [`PrimeConfig::default()`](prime::config::PrimeConfig) — so the
//! zero-argument form is prod-shaped, not a 1-core toy default.
//!
//! The return type `R` is preserved verbatim, so `()`, `Result<T, E>`, and
//! `std::process::ExitCode` all flow through unchanged — `run*` returns
//! `R`, and a runtime-boot failure is `.expect`-ed (a process that can't boot
//! its runtime is a hard start failure, mirroring `#[tokio::main]`).

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{Error, ItemFn, Meta, ReturnType, Token, parse2};

use crate::runtime_args::{
    RuntimeKind, affinity_fixed_len, extract_str_lit, extract_usize_lit, fold_flavor,
    parse_runtime_value,
};

/// Parsed `#[proxima::main]` args: which runtime kind boots, and (for
/// `Default`/`Prime`, which don't switch backends) the `cores`/`affinity`
/// that size and place it. `Tokio`/`TokioMultiThread` carry their own
/// `worker_threads` count inline (`workers`) since `fold_flavor` already
/// threads it through those variants.
fn parse_args(args: TokenStream) -> Result<(RuntimeKind, Option<usize>, Option<String>), Error> {
    let mut runtime = RuntimeKind::Default;
    let mut flavor: Option<String> = None;
    let mut worker_threads: Option<usize> = None;
    let mut cores: Option<usize> = None;
    let mut affinity: Option<String> = None;

    if args.is_empty() {
        return Ok((runtime, None, None));
    }
    let metas = Punctuated::<Meta, Token![,]>::parse_terminated.parse2(args)?;
    for meta in metas {
        match &meta {
            Meta::NameValue(nv) => {
                let key = nv.path.get_ident().map(ToString::to_string);
                match key.as_deref() {
                    Some("runtime") => runtime = parse_runtime_value(&nv.value)?,
                    Some("flavor") => flavor = Some(extract_str_lit(&nv.value, "flavor")?),
                    Some("worker_threads") => {
                        worker_threads = Some(extract_usize_lit(&nv.value, "worker_threads")?);
                    }
                    Some("cores") => cores = Some(extract_usize_lit(&nv.value, "cores")?),
                    Some("affinity") => affinity = Some(extract_str_lit(&nv.value, "affinity")?),
                    Some(other) => {
                        return Err(Error::new_spanned(
                            &nv.path,
                            format!(
                                "unknown #[proxima::main] arg `{other}`; expected runtime, flavor, worker_threads, cores, or affinity"
                            ),
                        ));
                    }
                    None => return Err(Error::new_spanned(&nv.path, "expected identifier key")),
                }
            }
            _ => return Err(Error::new_spanned(&meta, "expected `key = value` arg")),
        }
    }

    if cores.is_some() && worker_threads.is_some() {
        return Err(Error::new(
            Span::call_site(),
            "`cores` and `worker_threads` are mutually exclusive — `cores` sizes the prime/adaptive runtime, `worker_threads` sizes an explicit tokio thread pool",
        ));
    }

    // `cores`/`affinity` are the prime/adaptive path's own vocabulary; an
    // explicit tokio runtime (`runtime = "tokio"` or any `flavor`) only
    // understands `worker_threads`. Reject the wrong vocabulary on either
    // side rather than silently dropping it (principle 15).
    let tokio_explicit = flavor.is_some() || matches!(runtime, RuntimeKind::Tokio);
    if tokio_explicit && cores.is_some() {
        return Err(Error::new(
            Span::call_site(),
            "`cores` only applies to the prime/adaptive runtime path; use `worker_threads` with an explicit tokio runtime",
        ));
    }
    if tokio_explicit && affinity.is_some() {
        return Err(Error::new(
            Span::call_site(),
            "`affinity` only applies to the prime/adaptive runtime path; it is not valid with `runtime = \"tokio\"` or `flavor`",
        ));
    }
    if !tokio_explicit && worker_threads.is_some() {
        return Err(Error::new(
            Span::call_site(),
            "`worker_threads` only applies to an explicit tokio runtime (`runtime = \"tokio\"` or `flavor = \"multi_thread\"`); use `cores` to size the prime/adaptive runtime",
        ));
    }
    if let (Some(count), Some(spec)) = (cores, &affinity)
        && let Some(fixed_len) = affinity_fixed_len(spec)
        && fixed_len != count
    {
        return Err(Error::new(
            Span::call_site(),
            format!(
                "`cores = {count}` conflicts with `affinity = \"{spec}\"`, which names {fixed_len} cores"
            ),
        ));
    }

    // `#[proxima::main]`-specific reading of bare `cores`/`affinity` (no
    // `runtime = ...`, no `flavor = ...`): they size/place whichever backend
    // `Default`/`Prime` resolves to — they do NOT imply a backend switch.
    if flavor.is_none() && matches!(runtime, RuntimeKind::Default | RuntimeKind::Prime) {
        return Ok((runtime, cores, affinity));
    }

    let folded = fold_flavor(runtime, flavor, worker_threads)?;
    Ok((folded, None, None))
}

pub fn expand(args: TokenStream, item: TokenStream) -> Result<TokenStream, Error> {
    let func = parse2::<ItemFn>(item)?;
    let (runtime, cores, affinity) = parse_args(args)?;

    if func.sig.asyncness.is_none() {
        return Err(Error::new_spanned(
            &func.sig,
            "#[proxima::main] requires an `async fn`",
        ));
    }
    if func.sig.ident != "main" {
        return Err(Error::new_spanned(
            &func.sig.ident,
            "#[proxima::main] must be applied to `fn main`",
        ));
    }
    if let Some(generic) = func.sig.generics.params.first() {
        return Err(Error::new_spanned(
            generic,
            "#[proxima::main] does not support a generic main",
        ));
    }
    if let Some(input) = func.sig.inputs.first() {
        return Err(Error::new_spanned(
            input,
            "#[proxima::main] does not support `main` parameters",
        ));
    }

    let cores_arg = match cores {
        Some(count) => quote!(::core::option::Option::Some(#count)),
        None => quote!(::core::option::Option::None),
    };
    let affinity_arg = match &affinity {
        Some(spec) => quote!(::core::option::Option::Some(#spec)),
        None => quote!(::core::option::Option::None),
    };

    let run_call: TokenStream = match &runtime {
        // `Default`/`Prime` never switch backends on account of `cores`/
        // `affinity` — those args size/place whichever one runs, and the
        // booted runtime is published so `App::new()` inside the body
        // adopts it (see `crate::runtime::install_runtime`).
        RuntimeKind::Default => {
            quote!(::proxima::runtime::run_with_cores(#cores_arg, #affinity_arg, __proxima_main_body))
        }
        RuntimeKind::Prime => {
            quote!(::proxima::runtime::run_prime_with_cores(#cores_arg, #affinity_arg, __proxima_main_body))
        }
        RuntimeKind::Tokio => {
            quote!(::proxima::runtime::run_tokio(
                false,
                ::core::option::Option::None,
                __proxima_main_body
            ))
        }
        RuntimeKind::TokioMultiThread { workers } => {
            let workers_arg = match workers {
                Some(count) => quote!(::core::option::Option::Some(#count)),
                None => quote!(::core::option::Option::None),
            };
            quote!(::proxima::runtime::run_tokio(true, #workers_arg, __proxima_main_body))
        }
    };

    let attrs = &func.attrs;
    let vis = &func.vis;
    let output = &func.sig.output;
    let block = &func.block;
    let body_output = match output {
        ReturnType::Default => quote!(()),
        ReturnType::Type(_, ty) => quote!(#ty),
    };

    // a prime main needs the prime runtime feature compiled in; under the
    // adaptive `Default` the call site falls back to tokio, so no cfg gate is
    // forced here (mirrors #[proxima::test]'s Default).
    let prime_cfg: TokenStream = if matches!(runtime, RuntimeKind::Prime) {
        quote! {
            #[cfg(not(all(
                feature = "runtime-prime-executor",
                feature = "runtime-prime-inbox-alloc",
                feature = "runtime-prime-reactor",
                feature = "runtime-prime-bgpool"
            )))]
            compile_error!(
                "#[proxima::main(runtime = \"prime\")] requires the prime runtime features (enable `serve-prime` or the four `runtime-prime-*` features)"
            );
        }
    } else {
        quote!()
    };

    Ok(quote! {
        // `#[proxima::main]` IS the sanctioned backend-selection surface, so its
        // generated `run*` call self-allows the `disallowed_methods` guardrail
        // (clippy.toml) that flags hand-written direct `run_prime`/`run_tokio`.
        #[allow(clippy::disallowed_methods)]
        #(#attrs)*
        #vis fn main() #output {
            #prime_cfg
            let __proxima_main_body = async move #block;
            let __proxima_main_output: #body_output =
                #run_call.expect("proxima::main: build runtime");
            __proxima_main_output
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand_ok(args: &str, item: &str) -> String {
        let args: TokenStream = args.parse().expect("parse args");
        let item: TokenStream = item.parse().expect("parse item");
        expand(args, item).expect("expand").to_string()
    }

    fn expand_err(args: &str, item: &str) -> String {
        let args: TokenStream = args.parse().expect("parse args");
        let item: TokenStream = item.parse().expect("parse item");
        expand(args, item).expect_err("expected error").to_string()
    }

    #[test]
    fn default_uses_adaptive_run_with_cores() {
        let expanded = expand_ok("", "async fn main() {}");
        assert!(expanded.contains("fn main"));
        assert!(expanded.contains("runtime :: run_with_cores ("));
        // bare `#[proxima::main]` passes `cores = None, affinity = None` —
        // `run_with_cores` resolves `None` cores to `CoreSelection::Auto`
        // (all physical cores), the one intentional behavior change this seam
        // makes (was `unwrap_or(1)` under the old `worker_threads` surface).
        assert!(
            expanded.contains(
                "run_with_cores (:: core :: option :: Option :: None , :: core :: option :: Option :: None"
            )
        );
        assert!(!expanded.contains("run_prime"));
        assert!(!expanded.contains("run_tokio"));
    }

    #[test]
    fn prime_uses_run_prime_with_cores() {
        let expanded = expand_ok("runtime = \"prime\"", "async fn main() {}");
        assert!(expanded.contains("run_prime_with_cores"));
        assert!(expanded.contains("compile_error"));
    }

    #[test]
    fn tokio_current_thread_uses_run_tokio_false() {
        let expanded = expand_ok("runtime = \"tokio\"", "async fn main() {}");
        assert!(expanded.contains("run_tokio (false"));
    }

    #[test]
    fn multi_thread_flavor_uses_run_tokio_true() {
        let expanded = expand_ok("flavor = \"multi_thread\"", "async fn main() {}");
        assert!(expanded.contains("run_tokio (true"));
    }

    // a BARE `cores` (no `runtime =`, no `flavor =`) sizes whichever backend
    // `Default` resolves to — it must NOT force a switch to tokio, otherwise a
    // prime-first build silently drives main on tokio the moment a caller
    // names a core count (the double-runtime bug this seam fixes).
    #[test]
    fn cores_alone_sizes_default_without_switching_backend() {
        let expanded = expand_ok("cores = 4", "async fn main() {}");
        assert!(
            expanded.contains(
                "runtime :: run_with_cores (:: core :: option :: Option :: Some (4usize)"
            )
        );
        assert!(!expanded.contains("run_tokio"));
    }

    // `runtime = "prime"` + `cores` is the explicit form of the same rule:
    // cores for prime, no backend switch.
    #[test]
    fn explicit_prime_with_cores_sizes_prime() {
        let expanded = expand_ok("runtime = \"prime\", cores = 3", "async fn main() {}");
        assert!(expanded.contains(
            "run_prime_with_cores (:: core :: option :: Option :: Some (3usize)"
        ));
    }

    // `affinity` threads through as a `&str` literal — parsed once, at
    // runtime, by the real `Affinity::from_str` (never re-implemented here).
    #[test]
    fn affinity_threads_through_as_str_literal() {
        let expanded = expand_ok(
            "runtime = \"prime\", cores = 4, affinity = \"packed\"",
            "async fn main() {}",
        );
        assert!(expanded.contains(
            "run_prime_with_cores (:: core :: option :: Option :: Some (4usize) , :: core :: option :: Option :: Some (\"packed\")"
        ));
    }

    // explicit `runtime = "tokio"` + `worker_threads` keeps today's behavior:
    // an explicit backend opt-in folds into tokio multi-thread, workers wired
    // through `fold_flavor` unchanged.
    #[test]
    fn explicit_tokio_with_worker_threads_still_folds_multi_thread() {
        let expanded = expand_ok(
            "runtime = \"tokio\", worker_threads = 4",
            "async fn main() {}",
        );
        assert!(
            expanded
                .contains("run_tokio (true , :: core :: option :: Option :: Some (4usize)")
        );
    }

    // `cores` and `worker_threads` name overlapping concerns for two
    // different backends — mixing them is ambiguous, not additive.
    #[test]
    fn rejects_cores_and_worker_threads_together() {
        let err = expand_err("cores = 2, worker_threads = 4", "async fn main() {}");
        assert!(err.contains("mutually exclusive"));
    }

    // a BARE `worker_threads` (no explicit tokio runtime) used to size
    // whichever backend `Default` resolved to; that vocabulary now belongs to
    // `cores` — `worker_threads` alone is a compile error pointing at it.
    #[test]
    fn rejects_bare_worker_threads_without_explicit_tokio() {
        let err = expand_err("worker_threads = 4", "async fn main() {}");
        assert!(err.contains("only applies to an explicit tokio runtime"));
    }

    #[test]
    fn rejects_worker_threads_with_explicit_prime() {
        let err = expand_err(
            "runtime = \"prime\", worker_threads = 3",
            "async fn main() {}",
        );
        assert!(err.contains("only applies to an explicit tokio runtime"));
    }

    #[test]
    fn rejects_affinity_with_explicit_tokio_runtime() {
        let err = expand_err(
            "runtime = \"tokio\", affinity = \"packed\"",
            "async fn main() {}",
        );
        assert!(err.contains("only applies to the prime/adaptive runtime path"));
    }

    #[test]
    fn rejects_affinity_with_multi_thread_flavor() {
        let err = expand_err(
            "flavor = \"multi_thread\", affinity = \"packed\"",
            "async fn main() {}",
        );
        assert!(err.contains("only applies to the prime/adaptive runtime path"));
    }

    #[test]
    fn rejects_cores_conflicting_with_affinity_core_list_length() {
        let err = expand_err("cores = 2, affinity = \"4,5,6\"", "async fn main() {}");
        assert!(err.contains("conflicts with"));
    }

    #[test]
    fn rejects_cores_conflicting_with_affinity_range_length() {
        let err = expand_err("cores = 2, affinity = \"4-7\"", "async fn main() {}");
        assert!(err.contains("conflicts with"));
    }

    // matching lengths (or an affinity shape with no fixed length — a bare
    // offset, "packed", "float") are not a conflict.
    #[test]
    fn accepts_cores_matching_affinity_core_list_length() {
        let expanded = expand_ok("cores = 3, affinity = \"4,5,6\"", "async fn main() {}");
        assert!(expanded.contains("run_with_cores"));
    }

    #[test]
    fn accepts_cores_with_non_list_affinity() {
        let expanded = expand_ok("cores = 4, affinity = \"packed\"", "async fn main() {}");
        assert!(expanded.contains("run_with_cores"));
    }

    #[test]
    fn preserves_result_return_type() {
        let expanded = expand_ok(
            "runtime = \"tokio\"",
            "async fn main() -> Result<(), Box<dyn std::error::Error>> { Ok(()) }",
        );
        assert!(expanded.contains("-> Result < () , Box < dyn std :: error :: Error > >"));
    }

    // the recommended `#[proxima::main]` idiom (docs/lib.rs example): any
    // error type propagates verbatim, not just `ProximaError` — the macro
    // never inspects `R`. `Send + Sync` is the form that also works on the
    // prime backend (see `run_with_cores`'s `F::Output: Send` bound in
    // `src/runtime.rs`); this test proves the macro itself imposes no
    // additional constraint beyond preserving the annotated type.
    #[test]
    fn preserves_boxed_send_sync_error_return_type() {
        let expanded = expand_ok(
            "",
            "async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> { Ok(()) }",
        );
        assert!(expanded.contains(
            "-> Result < () , Box < dyn std :: error :: Error + Send + Sync > >"
        ));
    }

    #[test]
    fn preserves_exit_code_return_type() {
        let expanded = expand_ok(
            "flavor = \"multi_thread\"",
            "async fn main() -> std::process::ExitCode { std::process::ExitCode::SUCCESS }",
        );
        assert!(expanded.contains("-> std :: process :: ExitCode"));
    }

    #[test]
    fn rejects_non_async() {
        let err = expand_err("", "fn main() {}");
        assert!(err.contains("requires an `async fn`"));
    }

    #[test]
    fn rejects_non_main_ident() {
        let err = expand_err("", "async fn run() {}");
        assert!(err.contains("must be applied to `fn main`"));
    }

    #[test]
    fn rejects_main_parameters() {
        let err = expand_err("", "async fn main(arg: u8) {}");
        assert!(err.contains("does not support `main` parameters"));
    }

    #[test]
    fn rejects_prime_with_flavor() {
        let err = expand_err(
            "runtime = \"prime\", flavor = \"multi_thread\"",
            "async fn main() {}",
        );
        assert!(err.contains("mutually exclusive"));
    }

    #[test]
    fn rejects_unknown_runtime() {
        let err = expand_err("runtime = \"glommio\"", "async fn main() {}");
        assert!(err.contains("unknown runtime"));
    }

    #[test]
    fn rejects_unknown_arg() {
        let err = expand_err("start_paused = true", "async fn main() {}");
        assert!(err.contains("unknown #[proxima::main] arg"));
    }
}
