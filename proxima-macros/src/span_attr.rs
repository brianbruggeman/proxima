use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream, Parser};
use syn::punctuated::Punctuated;
use syn::{Error, Expr, ExprLit, Ident, ItemFn, Lit, LitStr, Meta, Token, parse_quote, parse2};

// Resolve `…::recorder::Recorder` for whatever crate invoked `#[span]`: a direct
// `proxima-telemetry` dep, the `proxima` umbrella re-export, or this crate
// itself. Declarative macros get this for free via `$crate`; proc-macros do not.
fn recorder_path() -> TokenStream {
    if let Ok(found) = crate_name("proxima-telemetry") {
        return match found {
            FoundCrate::Itself => quote!(crate::recorder::Recorder),
            FoundCrate::Name(name) => {
                let krate = Ident::new(&name, Span::call_site());
                quote!(::#krate::recorder::Recorder)
            }
        };
    }
    match crate_name("proxima") {
        Ok(FoundCrate::Itself) => quote!(crate::telemetry::recorder::Recorder),
        Ok(FoundCrate::Name(name)) => {
            let krate = Ident::new(&name, Span::call_site());
            quote!(::#krate::telemetry::recorder::Recorder)
        }
        Err(_) => quote!(::proxima_telemetry::recorder::Recorder),
    }
}

// Resolve `…::spanned::Spanned` the same way `recorder_path` resolves
// `Recorder` — this crate has no Cargo dependency on proxima-telemetry
// (proc-macro crates stay dependency-free); the path is only ever named in
// the TOKEN OUTPUT handed back to the invoking crate, which does depend on
// telemetry one way or another.
fn spanned_path() -> TokenStream {
    if let Ok(found) = crate_name("proxima-telemetry") {
        return match found {
            FoundCrate::Itself => quote!(crate::spanned::Spanned),
            FoundCrate::Name(name) => {
                let krate = Ident::new(&name, Span::call_site());
                quote!(::#krate::spanned::Spanned)
            }
        };
    }
    match crate_name("proxima") {
        Ok(FoundCrate::Itself) => quote!(crate::telemetry::spanned::Spanned),
        Ok(FoundCrate::Name(name)) => {
            let krate = Ident::new(&name, Span::call_site());
            quote!(::#krate::telemetry::spanned::Spanned)
        }
        Err(_) => quote!(::proxima_telemetry::spanned::Spanned),
    }
}

struct SpanArgs {
    name: Option<String>,
    level: Option<String>,
    recorder: Option<Expr>,
    parent: Option<Expr>,
    kind: Option<String>,
    fields: Vec<Field>,
    err: bool,
    budget: Option<Expr>,
}

/// One `fields(...)` entry: a static key and the expression whose value is
/// tagged onto the span. The value must be `Into<ScalarValue>` (proxima tags are
/// typed scalars, not `Debug` strings) — a non-convertible expr is a compile
/// error at the call site, which is the point.
struct Field {
    key: String,
    value: Expr,
}

impl Parse for Field {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        // `"dotted.key" = expr` — string-literal key (dots aren't valid idents).
        if input.peek(LitStr) {
            let key: LitStr = input.parse()?;
            input.parse::<Token![=]>()?;
            let value: Expr = input.parse()?;
            return Ok(Self {
                key: key.value(),
                value,
            });
        }
        // `ident = expr` or bare `ident` (capture the variable by its own name).
        let ident: Ident = input.parse()?;
        let key = ident.to_string();
        if input.peek(Token![=]) {
            input.parse::<Token![=]>()?;
            let value: Expr = input.parse()?;
            Ok(Self { key, value })
        } else {
            Ok(Self {
                key,
                value: parse_quote!(#ident),
            })
        }
    }
}

fn parse_args(args: TokenStream) -> Result<SpanArgs, Error> {
    let mut parsed = SpanArgs {
        name: None,
        level: None,
        recorder: None,
        parent: None,
        kind: None,
        fields: Vec::new(),
        err: false,
        budget: None,
    };

    if args.is_empty() {
        return Ok(parsed);
    }

    let metas = Punctuated::<Meta, Token![,]>::parse_terminated.parse2(args)?;

    for meta in metas {
        match &meta {
            Meta::NameValue(name_value) => {
                let key = name_value.path.get_ident().map(ToString::to_string);
                match key.as_deref() {
                    Some("name") => parsed.name = Some(extract_str_lit(&name_value.value, "name")?),
                    Some("level") => {
                        parsed.level = Some(extract_str_lit(&name_value.value, "level")?);
                    }
                    Some("kind") => parsed.kind = Some(extract_str_lit(&name_value.value, "kind")?),
                    Some("recorder") => parsed.recorder = Some(name_value.value.clone()),
                    Some("parent") => parsed.parent = Some(name_value.value.clone()),
                    Some("budget") => parsed.budget = Some(name_value.value.clone()),
                    Some(other) => {
                        return Err(Error::new_spanned(
                            &name_value.path,
                            format!(
                                "unknown #[span] arg `{other}`; expected name, level, kind, recorder, parent, budget, fields, or err"
                            ),
                        ));
                    }
                    None => {
                        return Err(Error::new_spanned(
                            &name_value.path,
                            "expected identifier key",
                        ));
                    }
                }
            }
            Meta::List(list) if list.path.is_ident("fields") => {
                let fields =
                    Punctuated::<Field, Token![,]>::parse_terminated.parse2(list.tokens.clone())?;
                parsed.fields.extend(fields);
            }
            Meta::Path(path) if path.is_ident("err") => parsed.err = true,
            _ => {
                return Err(Error::new_spanned(
                    &meta,
                    "expected `key = value`, `fields(...)`, or `err`",
                ));
            }
        }
    }

    Ok(parsed)
}

fn extract_str_lit(expr: &Expr, field: &str) -> Result<String, Error> {
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

fn validate_level(level: &str, span: Span) -> Result<(), Error> {
    let valid = ["trace", "debug", "info", "warn", "error"];
    if valid.contains(&level) {
        Ok(())
    } else {
        Err(Error::new(
            span,
            format!("unknown level `{level}`; expected one of: trace, debug, info, warn, error"),
        ))
    }
}

fn validate_kind(kind: &str, span: Span) -> Result<(), Error> {
    let valid = ["internal", "server", "client", "producer", "consumer"];
    if valid.contains(&kind) {
        Ok(())
    } else {
        Err(Error::new(
            span,
            format!(
                "unknown kind `{kind}`; expected one of: internal, server, client, producer, consumer"
            ),
        ))
    }
}

pub fn expand(args: TokenStream, item: TokenStream) -> Result<TokenStream, Error> {
    let func = parse2::<ItemFn>(item)?;
    let span_args = parse_args(args)?;

    let func_name = func.sig.ident.to_string();
    let span_name = span_args.name.unwrap_or_else(|| func_name.clone());

    let level_str = span_args.level.unwrap_or_else(|| "info".to_string());
    validate_level(&level_str, Span::call_site())?;

    if let Some(kind) = &span_args.kind {
        validate_kind(kind, Span::call_site())?;
    }

    if func.sig.constness.is_some() {
        return Err(Error::new_spanned(
            func.sig.constness,
            "#[span] cannot be applied to const fn",
        ));
    }

    // The builder chain is `::core`-only + recorder inherent methods. The only
    // named proxima type is `Recorder::current()` on the default (ambient) path.
    let kind_call = span_args
        .kind
        .map(|kind| quote! { .kind_str(#kind) })
        .unwrap_or_default();
    let field_calls: Vec<TokenStream> = span_args
        .fields
        .iter()
        .map(|field| {
            let key = &field.key;
            let value = &field.value;
            quote! { .tag(#key, #value) }
        })
        .collect();

    // C5 tail-sampling: `budget = <ns>` arms a force-keep on the trace if the span
    // overruns, regardless of head sampling.
    let budget_call = span_args
        .budget
        .map(|budget| quote! { .budget(#budget) })
        .unwrap_or_default();

    // An async span's current-scoping is driven per-poll by `Spanned` (see the
    // async body below), so its guard must NOT enter the current-span stack at
    // creation — `start_deferred` builds it un-entered. A sync span brackets the
    // stack over its own scope, so `start` enters it.
    let start_call = if func.sig.asyncness.is_some() {
        quote! { .start_deferred() }
    } else {
        quote! { .start() }
    };

    // Everything after the span is OPENED, applied to whatever builder
    // `.span(...)` / `.span_from_traceparent(...)` produced.
    let rest_chain = quote! {
        #kind_call
        #budget_call
        .module_path(::core::module_path!())
        .file_line(::core::line!(), ::core::column!())
        #(#field_calls)*
        #start_call
    };

    // Proxima carries span context as EXPLICIT DATA, never an ambient /
    // thread-local "current span" (a naive thread-local stack corrupts under
    // concurrently-interleaved async tasks sharing an executor thread — the
    // exact footgun this design avoids). `parent = <expr>` is that explicit
    // seam: `<expr>` is an `Option<&[u8]>` W3C traceparent — a request
    // boundary's `RequestContext::traceparent()`, or bytes carried by hand
    // from a caller's own span. `Some` continues that trace (inherits
    // trace_id, records parent_span_id) via `span_from_traceparent`; `None`,
    // or no `parent` arg at all, opens a fresh root via `.span(...)`.
    let open_expr = match &span_args.parent {
        Some(parent_expr) => quote! {
            match #parent_expr {
                ::core::option::Option::Some(__proxima_traceparent) => {
                    __rec.span_from_traceparent(#span_name, __proxima_traceparent)
                }
                ::core::option::Option::None => __rec.span(#span_name),
            }
        },
        None => quote! { __rec.span(#span_name) },
    };

    // The guard is always `Option<guard>` so the err + drop paths are uniform.
    // Explicit `recorder = <expr>` always resolves (Some); the default resolves
    // the process-wide ambient recorder via `Recorder::current()` — zero wiring
    // needed, and `None` (no recorder installed) just runs the body span-free,
    // exactly like the `info!` / `debug!` emit macros.
    let build_span = match span_args.recorder {
        Some(recorder_expr) => quote! {
            ::core::option::Option::Some({
                let __rec = #recorder_expr;
                let __proxima_builder = #open_expr;
                __proxima_builder #rest_chain
            })
        },
        None => {
            let recorder_ty = recorder_path();
            quote! {
                #recorder_ty::current().map(|__rec| {
                    let __proxima_builder = #open_expr;
                    __proxima_builder #rest_chain
                })
            }
        }
    };

    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let stmts = &func.block.stmts;

    // `err`: run the body, set error status on `Err`, then yield the value. The
    // body is moved into a closure (sync) / async block (async) so every return
    // path — including early `return` — flows through the status check. Requires
    // a `Result` return; the `Err(_)` arm is the compile-time enforcement.
    let body = if span_args.err {
        let outcome = if sig.asyncness.is_some() {
            quote! { async move { #(#stmts)* }.await }
        } else {
            quote! { (move || { #(#stmts)* })() }
        };
        quote! {
            let mut __proxima_span = #build_span;
            let __proxima_outcome = #outcome;
            if ::core::result::Result::is_err(&__proxima_outcome) {
                if let ::core::option::Option::Some(__proxima_guard) = &mut __proxima_span {
                    __proxima_guard.mark_error(#span_name);
                }
            }
            __proxima_outcome
        }
    } else if sig.asyncness.is_some() {
        // Wave D Phase 1: an instrumented async fn's span rides on the
        // SAME primitive a spawned task's span does — `telemetry::Spanned`
        // wraps the future (here, the fn's own body) instead of holding
        // the guard as a bare local. Behavior-identical to the prior local-
        // variable form (the guard still finishes exactly when the body
        // resolves — `Spanned::poll` drops it on `Ready`, same as a local's
        // natural drop at scope exit) — this only changes the mechanism
        // to the one shared, wrap-don't-embed vocabulary.
        let spanned_ty = spanned_path();
        quote! {
            match #build_span {
                ::core::option::Option::Some(__proxima_guard) => {
                    #spanned_ty::scoped(async move { #(#stmts)* }, __proxima_guard).await
                }
                ::core::option::Option::None => (async move { #(#stmts)* }).await,
            }
        }
    } else {
        quote! {
            let __proxima_span = #build_span;
            #(#stmts)*
        }
    };

    Ok(quote! {
        #(#attrs)*
        #vis #sig {
            #body
        }
    })
}
