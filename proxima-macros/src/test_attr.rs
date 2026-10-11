use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::ext::IdentExt;
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{
    Attribute, Error, Expr, FnArg, Ident, ItemFn, Meta, Pat, Path, Token, Type, Visibility, parse2,
};

use crate::runtime_args::{
    RuntimeKind, extract_bool_lit, extract_str_lit, extract_usize_lit, fold_flavor,
    parse_runtime_value,
};

/// Scheduling needs a test declares for an external runner. `serial` and
/// `group` are mutually exclusive: serial already excludes every other test.
/// `process` composes with either (shared process per test binary).
struct Sched {
    serial: bool,
    process: bool,
    group: Option<String>,
}

struct TestArgs {
    runtime: RuntimeKind,
    cassette: Option<String>,
    start_paused: bool,
    sched: Sched,
}

/// mirrors `proxima_test::sched::SCHED_GROUP_CAP`
const SCHED_GROUP_CAP: usize = 64;

/// nextest identifier rules: `XID_Start`, then `XID_Continue` or `-`; a leading
/// `@` is reserved for tool-defined groups.
fn validate_group(name: &str, span: &impl quote::ToTokens) -> Result<(), Error> {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return Err(Error::new_spanned(span, "`group` name must not be empty"));
    };
    if first == '@' {
        return Err(Error::new_spanned(
            span,
            format!(
                "`group` name `{name}` must not start with `@`; that prefix is reserved for tool groups"
            ),
        ));
    }
    if !unicode_ident::is_xid_start(first) {
        return Err(Error::new_spanned(
            span,
            format!("`group` name `{name}` must start with an XID_Start character (a letter)"),
        ));
    }
    if let Some(bad) = chars.find(|ch| *ch != '-' && !unicode_ident::is_xid_continue(*ch)) {
        return Err(Error::new_spanned(
            span,
            format!(
                "`group` name `{name}` contains `{bad}`; only XID_Continue characters and `-` are allowed"
            ),
        ));
    }
    if name.len() > SCHED_GROUP_CAP {
        return Err(Error::new_spanned(
            span,
            format!(
                "`group` name is {} bytes; the scheduling record holds at most {SCHED_GROUP_CAP}",
                name.len()
            ),
        ));
    }
    Ok(())
}

fn parse_args(args: TokenStream) -> Result<TestArgs, Error> {
    let mut runtime = RuntimeKind::Default;
    let mut cassette = None;
    let mut start_paused = false;
    let mut flavor: Option<String> = None;
    let mut worker_threads: Option<usize> = None;
    let mut sched = Sched {
        serial: false,
        process: false,
        group: None,
    };

    if args.is_empty() {
        return Ok(TestArgs {
            runtime,
            cassette,
            start_paused,
            sched,
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
                    Some("group") => {
                        if sched.group.is_some() {
                            return Err(Error::new_spanned(&nv.path, "duplicate `group` arg"));
                        }
                        let name = extract_str_lit(&nv.value, "group")?;
                        validate_group(&name, &nv.value)?;
                        sched.group = Some(name);
                    }
                    Some(flag @ ("serial" | "process")) => {
                        return Err(Error::new_spanned(
                            &nv.path,
                            format!("`{flag}` is a bare flag; write `{flag}`, not `{flag} = ..`"),
                        ));
                    }
                    Some(other) => {
                        return Err(Error::new_spanned(
                            &nv.path,
                            format!(
                                "unknown #[proxima::test] arg `{other}`; expected runtime, flavor, worker_threads, start_paused, cassette, group, or the bare flags serial and process"
                            ),
                        ));
                    }
                    None => return Err(Error::new_spanned(&nv.path, "expected identifier key")),
                }
            }
            Meta::Path(path) if path.is_ident("serial") => {
                if sched.serial {
                    return Err(Error::new_spanned(path, "duplicate `serial` arg"));
                }
                sched.serial = true;
            }
            Meta::Path(path) if path.is_ident("process") => {
                if sched.process {
                    return Err(Error::new_spanned(path, "duplicate `process` arg"));
                }
                sched.process = true;
            }
            Meta::Path(path) => {
                return Err(Error::new_spanned(
                    path,
                    "unknown #[proxima::test] flag; the bare flags are `serial` and `process` (other args are `key = value`)",
                ));
            }
            Meta::List(_) => return Err(Error::new_spanned(&meta, "expected `key = value` arg")),
        }
    }

    runtime = fold_flavor(runtime, flavor, worker_threads)?;

    if sched.serial && sched.group.is_some() {
        return Err(Error::new(
            Span::call_site(),
            "`serial` and `group` are mutually exclusive: `serial` already runs the test with nothing else in flight, so a group adds nothing",
        ));
    }

    Ok(TestArgs {
        runtime,
        cassette,
        start_paused,
        sched,
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
        let call = if let Some(args) = with_args {
            let partial = Ident::new(&format!("partial_{}", args.len()), Span::call_site());
            quote!(#source::#partial(#(#args),*).await)
        } else {
            quote!(#source::default().await)
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

/// One `#[used]` record in the scheduling section, emitted inside the test fn
/// so it shares the fn's `cfg`s. `module_path!()` there is the module libtest
/// prints (a case fn sits in a module named for the test fn); the crate
/// segment is stripped by `SchedRecord::new`.
fn sched_record_static(sched: &Sched, leaf: &str) -> TokenStream {
    let record = quote!(::proxima::test_support::SchedRecord);
    let flag_path = quote!(::proxima::test_support::sched);
    let mut flags = vec![quote!(0u8)];
    if sched.serial {
        flags.push(quote!(#flag_path::SCHED_FLAG_SERIAL));
    }
    if sched.process {
        flags.push(quote!(#flag_path::SCHED_FLAG_PROCESS));
    }
    let group = sched.group.as_ref().map_or_else(
        || quote!(::core::option::Option::None),
        |name| quote!(::core::option::Option::Some(#name)),
    );
    let constructor = quote!(#record::new(#(#flags)|*, #group, ::core::module_path!(), #leaf));
    quote! {
        #[used]
        #[cfg_attr(target_vendor = "apple", unsafe(link_section = "__DATA,__prox_test,regular,no_dead_strip"))]
        #[cfg_attr(windows, unsafe(link_section = ".prxtst$m"))]
        #[cfg_attr(not(any(target_vendor = "apple", windows)), unsafe(link_section = ".proxima_test"))]
        static __PROXIMA_SCHED_RECORD: #record = #constructor;
    }
}

fn reject_unsupported_test_shape(func: &ItemFn) -> Result<(), Error> {
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
    Ok(())
}

fn check_case_arity(
    func: &ItemFn,
    case_rows: &[CaseRow],
    case_param_count: usize,
) -> Result<(), Error> {
    for row in case_rows {
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
    Ok(())
}

// a #[cassette(..)] param needs a cassette declared on the test — else
// cassette_pipe panics at runtime; catch it as a clear compile error.
fn require_cassette_for_params(params: &[ParamSpec], cassette: Option<&str>) -> Result<(), Error> {
    let wants_cassette = params
        .iter()
        .any(|param| matches!(param.kind, ParamKind::Cassette(_)));
    if wants_cassette && cassette.is_none() {
        return Err(Error::new(
            Span::call_site(),
            "a #[cassette(..)] parameter requires #[proxima::test(cassette = \"...\")]",
        ));
    }
    Ok(())
}

fn cassette_spec(cassette: Option<&str>, case: &str) -> TokenStream {
    if let Some(name) = cassette {
        quote! {
            ::core::option::Option::Some(::proxima::test_support::CassetteSpec {
                name: #name,
                case: #case,
                manifest_dir: ::core::env!("CARGO_MANIFEST_DIR"),
            })
        }
    } else {
        quote!(::core::option::Option::None)
    }
}

fn entry_call(runtime: &RuntimeKind, start_paused: bool) -> TokenStream {
    match (runtime, start_paused) {
        (RuntimeKind::TokioMultiThread { workers }, paused) => {
            let workers_arg = if let Some(count) = workers {
                quote!(::core::option::Option::Some(#count))
            } else {
                quote!(::core::option::Option::None)
            };
            let fn_path = if paused {
                quote!(::proxima::test_support::run_tokio_multi_thread_paused)
            } else {
                quote!(::proxima::test_support::run_tokio_multi_thread)
            };
            quote!(#fn_path(cassette_arg, #workers_arg, body_arg))
        }
        (RuntimeKind::Default | RuntimeKind::Tokio, true) => {
            quote!(::proxima::test_support::run_tokio_current_thread_paused(
                cassette_arg,
                body_arg
            ))
        }
        (RuntimeKind::Default, false) => {
            quote!(::proxima::test_support::run(cassette_arg, body_arg))
        }
        (RuntimeKind::Prime, _) => {
            quote!(::proxima::test_support::run_prime(cassette_arg, body_arg))
        }
        (RuntimeKind::Tokio, false) => {
            quote!(::proxima::test_support::run_tokio(cassette_arg, body_arg))
        }
    }
}

fn call_arg(
    param: &ParamSpec,
    row: &CaseRow,
    value_combo: &[usize],
    case_cursor: &mut usize,
    value_cursor: &mut usize,
) -> TokenStream {
    match &param.kind {
        ParamKind::Case => {
            let expr = &row.args[*case_cursor];
            *case_cursor += 1;
            quote!(#expr)
        }
        ParamKind::Overlay(base) => {
            let patch = &row.args[*case_cursor];
            *case_cursor += 1;
            quote! {
                ::proxima::test_support::ConfigFixture::from_raw_or_path(
                    #base, ::core::env!("CARGO_MANIFEST_DIR"),
                ).await.expect("overlay base config").overlay_str(#patch)
            }
        }
        ParamKind::Values(values) => {
            let chosen = &values[value_combo[*value_cursor]];
            *value_cursor += 1;
            quote!(#chosen)
        }
        ParamKind::ConfigValues(values) => {
            let chosen = &values[value_combo[*value_cursor]];
            *value_cursor += 1;
            quote! {
                ::proxima::test_support::ConfigFixture::from_raw_or_path(
                    #chosen, ::core::env!("CARGO_MANIFEST_DIR"),
                ).await.expect("config_values fixture")
            }
        }
        ParamKind::Ctx => quote!(_proxima_cx),
        ParamKind::Cassette(inner) => {
            quote!(::proxima::test_support::cassette_pipe(&_proxima_cx, #inner).await.expect("cassette pipe"))
        }
        ParamKind::Fixture(call) => call.clone(),
    }
}

fn call_args_for(params: &[ParamSpec], row: &CaseRow, value_combo: &[usize]) -> Vec<TokenStream> {
    let mut case_cursor = 0usize;
    let mut value_cursor = 0usize;
    params
        .iter()
        .map(|param| call_arg(param, row, value_combo, &mut case_cursor, &mut value_cursor))
        .collect()
}

fn case_name(
    row: &CaseRow,
    row_index: usize,
    case_param_count: usize,
    value_combo: &[usize],
    value_dim_names: &[Ident],
) -> String {
    let mut name_parts: Vec<String> = Vec::new();
    if let Some(desc) = &row.desc {
        name_parts.push(desc.clone());
    } else if case_param_count > 0 {
        name_parts.push(format!("case_{}", row_index + 1));
    }
    for (dim_name, choice) in value_dim_names.iter().zip(value_combo) {
        name_parts.push(format!("{dim_name}_{}", choice + 1));
    }
    if name_parts.is_empty() {
        name_parts.push("case_1".to_string());
    }
    name_parts.join("_")
}

fn value_dimensions(params: &[ParamSpec]) -> (Vec<usize>, Vec<Ident>) {
    params
        .iter()
        .filter_map(|param| match &param.kind {
            ParamKind::Values(values) | ParamKind::ConfigValues(values) => {
                Some((values.len(), param.ident.clone()))
            }
            _ => None,
        })
        .unzip()
}

/// The pieces every generated test shares, whether the fn is a single test or
/// a module of parameterized cases.
struct TestShell<'a> {
    vis: &'a Visibility,
    ident: &'a Ident,
    other_attrs: Vec<&'a Attribute>,
    inner_fn: TokenStream,
    prime_cfg: TokenStream,
    entry_call: TokenStream,
    cassette: Option<String>,
    sched: Sched,
}

impl TestShell<'_> {
    fn driver(&self, case_name: &str, call_args: &[TokenStream]) -> TokenStream {
        let cassette_expr = cassette_spec(self.cassette.as_deref(), case_name);
        let entry_call = &self.entry_call;
        quote! {
            let cassette_arg = #cassette_expr;
            let body_arg = move |_proxima_cx: ::proxima::test_support::TestCtx| async move {
                ::proxima::test_support::IntoTestOutcome::into_test_outcome(
                    __proxima_inner(#(#call_args),*).await,
                );
            };
            #entry_call;
        }
    }

    // not parameterized: a single test fn (zero params, fixtures, and/or the
    // cassette TestCtx — but no #[case]/#[values]).
    fn single(&self, params: &[ParamSpec]) -> TokenStream {
        let no_row = CaseRow {
            desc: None,
            args: Vec::new(),
        };
        let call_args = call_args_for(params, &no_row, &[]);
        let body = self.driver("", &call_args);
        let record = sched_record_static(&self.sched, &self.ident.unraw().to_string());
        let (prime_cfg, other_attrs) = (&self.prime_cfg, &self.other_attrs);
        let (vis, ident, inner_fn) = (self.vis, self.ident, &self.inner_fn);
        quote! {
            #prime_cfg
            #(#other_attrs)*
            #[::core::prelude::v1::test]
            #vis fn #ident() {
                #record
                #inner_fn
                #body
            }
        }
    }

    fn case_fn(&self, case_name: &str, call_args: &[TokenStream]) -> TokenStream {
        let case_ident = Ident::new(case_name, Span::call_site());
        let body = self.driver(case_name, call_args);
        let record = sched_record_static(&self.sched, case_name);
        let (prime_cfg, other_attrs) = (&self.prime_cfg, &self.other_attrs);
        quote! {
            #prime_cfg
            #(#other_attrs)*
            #[test]
            fn #case_ident() {
                #record
                #body
            }
        }
    }

    // parameterized: a module of one #[test] per (case row x values combo),
    // mirroring rstest's `mod <fn> { fn <desc>() }` shape + terse naming.
    fn parameterized(
        &self,
        params: &[ParamSpec],
        case_rows: Vec<CaseRow>,
        case_param_count: usize,
    ) -> TokenStream {
        let (value_dim_lengths, value_dim_names) = value_dimensions(params);
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
                let call_args = call_args_for(params, row, value_combo);
                let name = case_name(
                    row,
                    row_index,
                    case_param_count,
                    value_combo,
                    &value_dim_names,
                );
                generated.push(self.case_fn(&name, &call_args));
            }
        }

        let (prime_cfg, vis, ident, inner_fn) =
            (&self.prime_cfg, self.vis, self.ident, &self.inner_fn);
        quote! {
            #prime_cfg
            #[allow(non_snake_case)]
            #vis mod #ident {
                #[allow(unused_imports)]
                use super::*;
                #inner_fn
                #(#generated)*
            }
        }
    }
}

pub fn expand(args: TokenStream, item: TokenStream) -> Result<TokenStream, Error> {
    let func = parse2::<ItemFn>(item)?;
    let test_args = parse_args(args)?;
    reject_unsupported_test_shape(&func)?;

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
    check_case_arity(&func, &case_rows, case_param_count)?;

    let TestArgs {
        runtime,
        cassette,
        start_paused,
        sched,
    } = test_args;

    let other_attrs: Vec<&Attribute> = func
        .attrs
        .iter()
        .filter(|attr| !matches!(attr_name(attr).as_deref(), Some("case" | "overlay_case")))
        .collect();
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

    require_cassette_for_params(&params, cassette.as_deref())?;

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

    let shell = TestShell {
        vis: &func.vis,
        ident: &func.sig.ident,
        other_attrs,
        inner_fn,
        prime_cfg,
        entry_call: entry_call(&runtime, start_paused),
        cassette,
        sched,
    };

    if !has_cases && !has_values {
        return Ok(shell.single(&params));
    }
    Ok(shell.parameterized(&params, case_rows, case_param_count))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAIN: &str = "async fn loads_checkpoint() {}";

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

    const SERIAL_BITS: &str = ":: proxima :: test_support :: sched :: SCHED_FLAG_SERIAL";
    const PROCESS_BITS: &str = ":: proxima :: test_support :: sched :: SCHED_FLAG_PROCESS";
    const NO_GROUP: &str = ":: core :: option :: Option :: None";

    fn some_group(name: &str) -> String {
        format!(":: core :: option :: Option :: Some (\"{name}\")")
    }

    fn record_args(expanded: &str) -> &str {
        const OPEN: &str = "SchedRecord :: new (";
        let (_, tail) = expanded.split_once(OPEN).expect("record constructor call");
        let (args, _) = tail
            .split_once(", :: core :: module_path")
            .expect("module_path argument");
        args.trim_end()
    }

    #[test]
    fn bare_test_gets_a_record_with_no_flags_and_no_group() {
        let expanded = expand_ok("", PLAIN);
        assert_eq!(record_args(&expanded), format!("0u8 , {NO_GROUP}"));
        assert!(expanded.contains("\"loads_checkpoint\""));
        assert!(expanded.contains("link_section"));
    }

    #[test]
    fn serial_flag_sets_only_the_serial_bit() {
        let expanded = expand_ok("serial", PLAIN);
        assert_eq!(
            record_args(&expanded),
            format!("0u8 | {SERIAL_BITS} , {NO_GROUP}")
        );
    }

    #[test]
    fn group_carries_the_name_and_no_flag_bits() {
        let expanded = expand_ok("group = \"real-checkpoint\"", PLAIN);
        assert_eq!(
            record_args(&expanded),
            format!("0u8 , {}", some_group("real-checkpoint"))
        );
    }

    #[test]
    fn process_alone_sets_only_the_process_bit_with_no_group() {
        let expanded = expand_ok("process", PLAIN);
        assert_eq!(
            record_args(&expanded),
            format!("0u8 | {PROCESS_BITS} , {NO_GROUP}")
        );
    }

    #[test]
    fn process_with_group_carries_both() {
        let expanded = expand_ok("group = \"gpu\", process", PLAIN);
        assert_eq!(
            record_args(&expanded),
            format!("0u8 | {PROCESS_BITS} , {}", some_group("gpu"))
        );
    }

    #[test]
    fn serial_with_process_and_no_group_is_allowed() {
        let expanded = expand_ok("process, serial", PLAIN);
        assert_eq!(
            record_args(&expanded),
            format!("0u8 | {SERIAL_BITS} | {PROCESS_BITS} , {NO_GROUP}")
        );
    }

    #[test]
    fn sched_args_compose_with_runtime_args() {
        let expanded = expand_ok("runtime = \"tokio\", group = \"gpu\"", PLAIN);
        assert!(expanded.contains("run_tokio"));
        assert_eq!(
            record_args(&expanded),
            format!("0u8 , {}", some_group("gpu"))
        );
    }

    #[test]
    fn every_parameterized_case_gets_its_own_record_named_for_the_case() {
        let expanded = expand_ok(
            "serial",
            "#[case::small(1u32)] #[case::large(2u32)] async fn sized(#[case] value: u32) {}",
        );
        assert_eq!(expanded.matches("SchedRecord :: new (").count(), 2);
        assert!(expanded.contains("\"small\""));
        assert!(expanded.contains("\"large\""));
    }

    #[test]
    fn four_section_attributes_cover_every_object_format() {
        let expanded = expand_ok("serial", PLAIN);
        assert!(expanded.contains("__DATA,__prox_test,regular,no_dead_strip"));
        assert!(expanded.contains(".prxtst$m"));
        assert!(expanded.contains(".proxima_test"));
    }

    #[test]
    fn group_name_accepts_nextest_identifiers() {
        for name in ["gpu", "real-checkpoint", "g2", "Gruppe-\u{00fc}ber"] {
            assert!(
                validate_group(name, &Ident::new("anchor", Span::call_site())).is_ok(),
                "{name}"
            );
        }
    }

    #[test]
    fn empty_group_is_rejected() {
        assert!(expand_err("group = \"\"", PLAIN).contains("must not be empty"));
    }

    #[test]
    fn group_starting_with_at_is_rejected() {
        assert!(expand_err("group = \"@tool:x\"", PLAIN).contains("reserved for tool groups"));
    }

    #[test]
    fn group_starting_with_digit_or_dash_or_underscore_is_rejected() {
        for name in ["1gpu", "-gpu", "_gpu"] {
            let message = expand_err(&format!("group = \"{name}\""), PLAIN);
            assert!(message.contains("XID_Start"), "{name}: {message}");
        }
    }

    #[test]
    fn group_with_space_or_dot_is_rejected() {
        for name in ["real checkpoint", "real.checkpoint", "gpu/1"] {
            let message = expand_err(&format!("group = \"{name}\""), PLAIN);
            assert!(message.contains("XID_Continue"), "{name}: {message}");
        }
    }

    #[test]
    fn group_longer_than_the_record_holds_is_rejected() {
        let long = "g".repeat(SCHED_GROUP_CAP + 1);
        assert!(expand_err(&format!("group = \"{long}\""), PLAIN).contains("at most 64"));
    }

    #[test]
    fn group_must_be_a_string_literal() {
        assert!(expand_err("group = 7", PLAIN).contains("expected string literal for `group`"));
    }
    #[test]
    fn serial_with_a_value_is_rejected() {
        assert!(
            expand_err("serial = true", PLAIN)
                .contains("`serial` is a bare flag; write `serial`, not `serial = ..`")
        );
    }

    #[test]
    fn process_with_a_value_is_rejected() {
        assert!(
            expand_err("process = true", PLAIN)
                .contains("`process` is a bare flag; write `process`, not `process = ..`")
        );
    }

    #[test]
    fn unknown_bare_flag_is_rejected() {
        assert!(
            expand_err("parallel", PLAIN).contains("the bare flags are `serial` and `process`")
        );
    }

    #[test]
    fn unknown_key_lists_the_scheduling_args() {
        let message = expand_err("threads = 2", PLAIN);
        assert!(message.contains("unknown #[proxima::test] arg `threads`"));
        assert!(message.contains("group"));
        assert!(message.contains("serial and process"));
    }

    #[test]
    fn duplicate_serial_is_rejected() {
        assert!(expand_err("serial, serial", PLAIN).contains("duplicate `serial`"));
    }

    #[test]
    fn duplicate_process_is_rejected() {
        assert!(expand_err("process, process", PLAIN).contains("duplicate `process`"));
    }

    #[test]
    fn duplicate_group_is_rejected() {
        assert!(expand_err("group = \"a\", group = \"b\"", PLAIN).contains("duplicate `group`"));
    }

    #[test]
    fn serial_and_group_together_are_rejected_as_redundant() {
        let message = expand_err("serial, group = \"gpu\"", PLAIN);
        assert!(message.contains("mutually exclusive"));
    }

    #[test]
    fn serial_and_group_stay_rejected_when_process_is_also_present() {
        let message = expand_err("process, serial, group = \"gpu\"", PLAIN);
        assert!(message.contains("mutually exclusive"));
    }
}
