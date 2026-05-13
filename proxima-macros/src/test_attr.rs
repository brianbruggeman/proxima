use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{Attribute, Error, Expr, FnArg, Ident, ItemFn, Meta, Pat, Path, Token, Type, parse2};

use crate::runtime_args::{
    RuntimeKind, extract_bool_lit, extract_str_lit, extract_usize_lit, fold_flavor,
    parse_runtime_value,
};

struct TestArgs {
    runtime: RuntimeKind,
    cassette: Option<String>,
    start_paused: bool,
}

fn parse_args(args: TokenStream) -> Result<TestArgs, Error> {
    let mut runtime = RuntimeKind::Default;
    let mut cassette = None;
    let mut start_paused = false;
    let mut flavor: Option<String> = None;
    let mut worker_threads: Option<usize> = None;

    if args.is_empty() {
        return Ok(TestArgs {
            runtime,
            cassette,
            start_paused,
        });
    }
    let metas = Punctuated::<Meta, Token![,]>::parse_terminated.parse2(args)?;
    for meta in metas {
        match &meta {
            Meta::NameValue(nv) => {
                let key = nv.path.get_ident().map(ToString::to_string);
                match key.as_deref() {
                    Some("runtime") => {
                        runtime = parse_runtime_value(&nv.value)?;
                    }
                    Some("flavor") => {
                        flavor = Some(extract_str_lit(&nv.value, "flavor")?);
                    }
                    Some("worker_threads") => {
                        worker_threads = Some(extract_usize_lit(&nv.value, "worker_threads")?);
                    }
                    Some("start_paused") => {
                        start_paused = extract_bool_lit(&nv.value, "start_paused")?;
                    }
                    Some("cassette") => {
                        cassette = Some(extract_str_lit(&nv.value, "cassette")?);
                    }
                    Some(other) => {
                        return Err(Error::new_spanned(
                            &nv.path,
                            format!(
                                "unknown #[proxima::test] arg `{other}`; expected runtime, flavor, worker_threads, start_paused, or cassette"
                            ),
                        ));
                    }
                    None => return Err(Error::new_spanned(&nv.path, "expected identifier key")),
                }
            }
            _ => return Err(Error::new_spanned(&meta, "expected `key = value` arg")),
        }
    }

    runtime = fold_flavor(runtime, flavor, worker_threads)?;

    Ok(TestArgs {
        runtime,
        cassette,
        start_paused,
    })
}

/// rstest features that need an async-driving harness rstest controls;
/// incompatible with our native sync-shell generation.
const UNSUPPORTED_ATTRS: &[&str] = &["future", "awt", "timeout", "once"];

fn attr_name(attr: &Attribute) -> Option<String> {
    attr.path()
        .segments
        .first()
        .map(|seg| seg.ident.to_string())
}

fn type_ends_with(ty: &Type, name: &str) -> bool {
    matches!(ty, Type::Path(path) if path.path.segments.last().is_some_and(|seg| seg.ident == name))
}

/// One fn-level `#[case(..)]` / `#[case::desc(..)]` row of argument values.
struct CaseRow {
    desc: Option<String>,
    args: Vec<Expr>,
}

/// How a single test-fn parameter is supplied per generated case.
enum ParamKind {
    /// value pulled from each `#[case(..)]` row, positionally among case params.
    Case,
    /// one dimension of a `#[values(..)]` cartesian matrix.
    Values(Vec<Expr>),
    /// one dimension of a `#[config_values(..)]` matrix: each literal is a config
    /// source (raw text or a path) built into a `ConfigFixture` per case.
    ConfigValues(Vec<Expr>),
    /// the injected `TestCtx` cassette handle.
    Ctx,
    /// a record/replay `PipeHandle` from `cx.cassette_pipe(inner)`; the held
    /// expr is the real upstream recorded on first run, ignored on replay.
    Cassette(Expr),
    /// a `ConfigFixture` = base config (the held expr) deep-merged with each
    /// `#[overlay_case(patch)]` row's patch. Consumes one row arg, like `Case`.
    Overlay(Expr),
    /// resolved by calling a fixture function (`name()` / `#[from(path)]` /
    /// `#[with(args)]`).
    Fixture(TokenStream),
}

struct ParamSpec {
    ident: Ident,
    ty: Type,
    kind: ParamKind,
}

fn parse_case_rows(attrs: &[Attribute]) -> Result<Vec<CaseRow>, Error> {
    let mut rows = Vec::new();
    for attr in attrs {
        // `#[case(..)]` rows AND `#[overlay_case(patch)]` rows share the row
        // machinery; arity (case-param count) keeps them from mixing.
        let name = attr_name(attr);
        if name.as_deref() != Some("case") && name.as_deref() != Some("overlay_case") {
            continue;
        }
        let segments = &attr.path().segments;
        let desc = if segments.len() >= 2 {
            Some(segments[1].ident.to_string())
        } else {
            None
        };
        let args = match &attr.meta {
            Meta::List(_) => attr
                .parse_args_with(Punctuated::<Expr, Token![,]>::parse_terminated)?
                .into_iter()
                .collect(),
            _ => Vec::new(),
        };
        rows.push(CaseRow { desc, args });
    }
    Ok(rows)
}

fn classify_param(input: &FnArg) -> Result<ParamSpec, Error> {
    let typed = match input {
        FnArg::Receiver(receiver) => {
            return Err(Error::new_spanned(
                receiver,
                "#[proxima::test] does not support a `self` parameter",
            ));
        }
        FnArg::Typed(typed) => typed,
    };
    let ident = match typed.pat.as_ref() {
        Pat::Ident(pat_ident) => pat_ident.ident.clone(),
        other => {
            return Err(Error::new_spanned(
                other,
                "#[proxima::test] parameters must be simple identifiers",
            ));
        }
    };
    let ty = typed.ty.as_ref().clone();

    let mut is_case = false;
    let mut values: Option<Vec<Expr>> = None;
    let mut config_values: Option<Vec<Expr>> = None;
    let mut cassette_inner: Option<Expr> = None;
    let mut overlay_base: Option<Expr> = None;
    let mut from_path: Option<Path> = None;
    let mut with_args: Option<Vec<Expr>> = None;

    for attr in &typed.attrs {
        match attr_name(attr).as_deref() {
            Some("case") => is_case = true,
            Some("values") => {
                values = Some(
                    attr.parse_args_with(Punctuated::<Expr, Token![,]>::parse_terminated)?
                        .into_iter()
                        .collect(),
                );
            }
            Some("config_values") => {
                config_values = Some(
                    attr.parse_args_with(Punctuated::<Expr, Token![,]>::parse_terminated)?
                        .into_iter()
                        .collect(),
                );
            }
            Some("cassette") => cassette_inner = Some(attr.parse_args::<Expr>()?),
            Some("overlay") => overlay_base = Some(attr.parse_args::<Expr>()?),
            Some("from") => from_path = Some(attr.parse_args::<Path>()?),
            Some("with") => {
                with_args = Some(
                    attr.parse_args_with(Punctuated::<Expr, Token![,]>::parse_terminated)?
                        .into_iter()
                        .collect(),
                );
            }
            Some(name) if UNSUPPORTED_ATTRS.contains(&name) => {
                return Err(Error::new_spanned(
                    attr,
                    format!(
                        "#[{name}] is unsupported under #[proxima::test]: the generated shell is sync"
                    ),
                ));
            }
            _ => {}
        }
    }

    let kind = if is_case {
        ParamKind::Case
    } else if let Some(values) = values {
        ParamKind::Values(values)
    } else if let Some(config_values) = config_values {
        ParamKind::ConfigValues(config_values)
    } else if let Some(inner) = cassette_inner {
        ParamKind::Cassette(inner)
    } else if let Some(base) = overlay_base {
        ParamKind::Overlay(base)
    } else if from_path.is_none() && with_args.is_none() && type_ends_with(&ty, "TestCtx") {
        ParamKind::Ctx
    } else {
        // resolve against the #[proxima::fixture]-generated struct: default()
        // with no override, partial_N(args) for #[with(..)]; #[from] renames the
        // source struct. always async (the fixture macro emits async methods).
        let source = from_path.unwrap_or_else(|| Path::from(ident.clone()));
        let call = match with_args {
            Some(args) => {
                let partial = Ident::new(&format!("partial_{}", args.len()), Span::call_site());
                quote!(#source::#partial(#(#args),*).await)
            }
            None => quote!(#source::default().await),
        };
        ParamKind::Fixture(call)
    };

    Ok(ParamSpec { ident, ty, kind })
}

fn cartesian(dim_lengths: &[usize]) -> Vec<Vec<usize>> {
    let mut combos = vec![Vec::new()];
    for &length in dim_lengths {
        let mut next = Vec::with_capacity(combos.len() * length.max(1));
        for combo in &combos {
            for index in 0..length {
                let mut extended = combo.clone();
                extended.push(index);
                next.push(extended);
            }
        }
        combos = next;
    }
    combos
}

pub fn expand(args: TokenStream, item: TokenStream) -> Result<TokenStream, Error> {
    let func = parse2::<ItemFn>(item)?;
    let test_args = parse_args(args)?;

    if func.sig.asyncness.is_none() {
        return Err(Error::new_spanned(
            &func.sig,
            "#[proxima::test] requires an `async fn`",
        ));
    }
    if let Some(generic) = func.sig.generics.params.first() {
        return Err(Error::new_spanned(
            generic,
            "#[proxima::test] does not support generic test functions",
        ));
    }
    for attr in &func.attrs {
        if let Some(name) = attr_name(attr)
            && UNSUPPORTED_ATTRS.contains(&name.as_str())
        {
            return Err(Error::new_spanned(
                attr,
                format!(
                    "#[{name}] is unsupported under #[proxima::test]; await inside the body or use PROXIMA_TEST_TIMEOUT_MS"
                ),
            ));
        }
    }

    let case_rows = parse_case_rows(&func.attrs)?;
    let params = func
        .sig
        .inputs
        .iter()
        .map(classify_param)
        .collect::<Result<Vec<_>, _>>()?;

    // `Overlay` params consume a row arg just like `Case`, so they count toward
    // the per-row arity (which is how `#[overlay_case]` rows stay distinct from
    // `#[case]` rows — a mismatch is a clear error).
    let case_param_count = params
        .iter()
        .filter(|param| matches!(param.kind, ParamKind::Case | ParamKind::Overlay(_)))
        .count();
    for row in &case_rows {
        if row.args.len() != case_param_count {
            return Err(Error::new(
                Span::call_site(),
                format!(
                    "#[case(..)] supplies {} value(s) but the function has {} #[case] parameter(s)",
                    row.args.len(),
                    case_param_count
                ),
            ));
        }
    }
    if case_param_count > 0 && case_rows.is_empty() {
        return Err(Error::new_spanned(
            &func.sig,
            "function has #[case] parameter(s) but no #[case(..)] rows",
        ));
    }

    let TestArgs {
        runtime,
        cassette,
        start_paused,
    } = test_args;

    let make_plan = |case: &str| -> TokenStream {
        match &cassette {
            Some(name) => quote! {
                ::proxima::test_support::Plan::with_cassette(::proxima::test_support::CassetteSpec {
                    name: #name,
                    case: #case,
                    manifest_dir: ::core::env!("CARGO_MANIFEST_DIR"),
                })
            },
            None => quote!(::proxima::test_support::Plan::new()),
        }
    };

    let other_attrs: Vec<&Attribute> = func
        .attrs
        .iter()
        .filter(|attr| {
            !matches!(
                attr_name(attr).as_deref(),
                Some("case") | Some("overlay_case")
            )
        })
        .collect();
    let vis = &func.vis;
    let ident = &func.sig.ident;
    let output = &func.sig.output;
    let block = &func.block;

    let inner_params = params.iter().map(|param| {
        let name = &param.ident;
        let ty = &param.ty;
        quote!(#name: #ty)
    });
    let inner_fn = quote! {
        async fn __proxima_inner(#(#inner_params),*) #output #block
    };

    // a #[cassette(..)] param needs a cassette declared on the test — else
    // cassette_pipe panics at runtime; catch it as a clear compile error.
    if params
        .iter()
        .any(|param| matches!(param.kind, ParamKind::Cassette(_)))
        && cassette.is_none()
    {
        return Err(Error::new(
            Span::call_site(),
            "a #[cassette(..)] parameter requires #[proxima::test(cassette = \"...\")]",
        ));
    }

    let entry_call: TokenStream = match (&runtime, start_paused) {
        (RuntimeKind::TokioMultiThread { workers }, paused) => {
            let workers_arg = match workers {
                Some(count) => quote!(::core::option::Option::Some(#count)),
                None => quote!(::core::option::Option::None),
            };
            let fn_path = if paused {
                quote!(::proxima::test_support::run_tokio_multi_thread_paused)
            } else {
                quote!(::proxima::test_support::run_tokio_multi_thread)
            };
            quote!(#fn_path(plan_arg, #workers_arg, body_arg))
        }
        (RuntimeKind::Default, true) | (RuntimeKind::Tokio, true) => {
            quote!(::proxima::test_support::run_tokio_current_thread_paused(
                plan_arg, body_arg
            ))
        }
        (RuntimeKind::Default, false) => quote!(::proxima::test_support::run(plan_arg, body_arg)),
        (RuntimeKind::Prime, _) => quote!(::proxima::test_support::run_prime(plan_arg, body_arg)),
        (RuntimeKind::Tokio, false) => {
            quote!(::proxima::test_support::run_tokio(plan_arg, body_arg))
        }
    };

    let driver = |plan: TokenStream, call_args: &[TokenStream]| {
        quote! {
            let plan_arg = #plan;
            let body_arg = move |_proxima_cx: ::proxima::test_support::TestCtx| async move {
                ::proxima::test_support::IntoTestOutcome::into_test_outcome(
                    __proxima_inner(#(#call_args),*).await,
                );
            };
            #entry_call;
        }
    };

    let has_cases = !case_rows.is_empty();
    let has_values = params.iter().any(|param| {
        matches!(
            param.kind,
            ParamKind::Values(_) | ParamKind::ConfigValues(_)
        )
    });

    let prime_cfg: TokenStream = if matches!(runtime, RuntimeKind::Prime) {
        quote!(#[cfg(feature = "test-prime")])
    } else {
        quote!()
    };

    // not parameterized: a single test fn (zero params, fixtures, and/or the
    // cassette TestCtx — but no #[case]/#[values]).
    if !has_cases && !has_values {
        let call_args: Vec<TokenStream> = params
            .iter()
            .map(|param| match &param.kind {
                ParamKind::Ctx => quote!(_proxima_cx),
                ParamKind::Cassette(inner) => {
                    quote!(::proxima::test_support::cassette_pipe(&_proxima_cx, #inner).await.expect("cassette pipe"))
                }
                ParamKind::Fixture(call) => call.clone(),
                _ => unreachable!("case/values do not reach the non-parameterized path"),
            })
            .collect();
        let body = driver(make_plan(""), &call_args);
        return Ok(quote! {
            #prime_cfg
            #(#other_attrs)*
            #[::core::prelude::v1::test]
            #vis fn #ident() {
                #inner_fn
                #body
            }
        });
    }

    // parameterized: a module of one #[test] per (case row x values combo),
    // mirroring rstest's `mod <fn> { fn <desc>() }` shape + terse naming.
    let value_dim_lengths: Vec<usize> = params
        .iter()
        .filter_map(|param| match &param.kind {
            ParamKind::Values(values) | ParamKind::ConfigValues(values) => Some(values.len()),
            _ => None,
        })
        .collect();
    let value_dim_names: Vec<Ident> = params
        .iter()
        .filter_map(|param| match &param.kind {
            ParamKind::Values(_) | ParamKind::ConfigValues(_) => Some(param.ident.clone()),
            _ => None,
        })
        .collect();
    let value_combos = cartesian(&value_dim_lengths);
    let effective_rows = if case_rows.is_empty() {
        vec![CaseRow {
            desc: None,
            args: Vec::new(),
        }]
    } else {
        case_rows
    };

    let mut generated = Vec::new();
    for (row_index, row) in effective_rows.iter().enumerate() {
        for value_combo in &value_combos {
            let mut case_cursor = 0usize;
            let mut value_cursor = 0usize;
            let mut call_args: Vec<TokenStream> = Vec::with_capacity(params.len());
            for param in &params {
                match &param.kind {
                    ParamKind::Case => {
                        let expr = &row.args[case_cursor];
                        case_cursor += 1;
                        call_args.push(quote!(#expr));
                    }
                    ParamKind::Overlay(base) => {
                        let patch = &row.args[case_cursor];
                        case_cursor += 1;
                        call_args.push(quote! {
                            ::proxima::test_support::ConfigFixture::from_raw_or_path(
                                #base, ::core::env!("CARGO_MANIFEST_DIR"),
                            ).await.expect("overlay base config").overlay_str(#patch)
                        });
                    }
                    ParamKind::Values(values) => {
                        let chosen = &values[value_combo[value_cursor]];
                        value_cursor += 1;
                        call_args.push(quote!(#chosen));
                    }
                    ParamKind::ConfigValues(values) => {
                        let chosen = &values[value_combo[value_cursor]];
                        value_cursor += 1;
                        call_args.push(quote! {
                            ::proxima::test_support::ConfigFixture::from_raw_or_path(
                                #chosen, ::core::env!("CARGO_MANIFEST_DIR"),
                            ).await.expect("config_values fixture")
                        });
                    }
                    ParamKind::Ctx => call_args.push(quote!(_proxima_cx)),
                    ParamKind::Cassette(inner) => call_args.push(
                        quote!(::proxima::test_support::cassette_pipe(&_proxima_cx, #inner).await.expect("cassette pipe")),
                    ),
                    ParamKind::Fixture(call) => call_args.push(call.clone()),
                }
            }

            let mut name_parts: Vec<String> = Vec::new();
            if let Some(desc) = &row.desc {
                name_parts.push(desc.clone());
            } else if case_param_count > 0 {
                name_parts.push(format!("case_{}", row_index + 1));
            }
            for (dim, choice) in value_combo.iter().enumerate() {
                name_parts.push(format!("{}_{}", value_dim_names[dim], choice + 1));
            }
            if name_parts.is_empty() {
                name_parts.push("case_1".to_string());
            }
            let case_name = name_parts.join("_");
            let case_ident = Ident::new(&case_name, Span::call_site());
            let body = driver(make_plan(&case_name), &call_args);
            generated.push(quote! {
                #prime_cfg
                #(#other_attrs)*
                #[test]
                fn #case_ident() {
                    #body
                }
            });
        }
    }

    Ok(quote! {
        #prime_cfg
        #[allow(non_snake_case)]
        #vis mod #ident {
            #[allow(unused_imports)]
            use super::*;
            #inner_fn
            #(#generated)*
        }
    })
}
