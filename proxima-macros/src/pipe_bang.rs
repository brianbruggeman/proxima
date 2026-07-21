//! `pipe!(<closure or pipe-expr>)` — the function-like sibling of
//! `#[proxima::piped]`: leaf-lift a closure into a concrete [`Pipe`]-tier
//! value INLINE, at an expression position, instead of a top-level `fn`.
//!
//! Same tier vocabulary as the attribute macro (`Tier::plan` in
//! `pipe_attr.rs` — reused verbatim, not re-derived): every emitted value
//! carries the FULL downward closure of tiers its shape qualifies for.
//! The difference is entirely in what gets wrapped:
//!
//! - the attribute macro relocates a named `fn`'s body into a fieldless
//!   unit struct's `call`;
//! - `pipe!` has no fn to relocate — a closure is an anonymous, unnameable
//!   type — so it mints a small tuple struct that OWNS the closure as a
//!   field (`struct Leaf<F>(F)`) and calls through it. This is the
//!   sanctioned closure-to-trait-bridge pattern (a fresh struct per call
//!   site, never a shared library wrapper): [`build_leaf`] is called by
//!   `pipe!` here, and again by `filter!`/`fanout!`/`fanin!`
//!   (`filter_bang.rs`/`fan_bang.rs`) for each closure-literal arm, but
//!   the struct itself is emitted FRESH at every call site, never a named,
//!   importable type in `proxima-primitives`.
//!
//! Closure shape convention (mirrors `pipe_attr`'s fn-shape convention,
//! `FutureShape`):
//!
//! - a plain (non-`async`) closure, `|input: In| -> Result<Out, Err> { .. }`,
//!   is wrapped in `core::future::ready` — `Fn(In) -> Result<Out, Err>`,
//!   zero-cost, unconditionally `Unpin`. Reaches all four tiers.
//! - an `async` closure (stable since 1.85, `#[stable(feature =
//!   "async_closure")]`), `async move |input: In| -> Result<Out, Err> { .. }`,
//!   is called straight through — `AsyncFn(In) -> Result<Out, Err>`, RPITIT
//!   passthrough, zero extra cost. Reaches `Pipe` only. It does NOT reach
//!   `UnpinPipe`/`SendPipe`/`UnpinSendPipe`: `UnpinPipe` would need `Box::pin`
//!   (zero-box is a hard constraint here, unlike the attribute macro's own
//!   opt-in `unpin, boxed` escape hatch — reach for that existing mechanism,
//!   `#[proxima::piped(unpin, boxed)]` on a hand-written `async fn`, if a
//!   boxed climb is genuinely needed); `send` would need naming
//!   `AsyncFnMut::CallRefFuture`, which is `#[unstable(feature =
//!   "async_fn_traits")]` on every stable toolchain as of this writing. Both
//!   are refused with the specific reason spelled out, never silently
//!   downgraded.
//! - the return type must be written out explicitly (`-> Result<Out, Err>`)
//!   on every closure lifted this way, exactly as a `#[proxima::piped]` fn
//!   must: nothing here does type inference, it reads the annotation.
//!
//! `pipe!(<already-a-pipe-expr>)` (no closure) passes the expression through
//! unchanged — there is nothing to lift.

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::parse::{Parse, ParseStream, Parser};
use syn::punctuated::Punctuated;
use syn::{Error, Expr, ExprClosure, Ident, Meta, Pat, ReturnType, Token, Type};

use crate::pipe_attr::{PipeArgs, Tier, pipe_trait_path, result_types_from_type};

/// `pipe!(expr [, send] [, unpin] [, boxed])` — the expression, then an
/// optional comma-separated tail of the same bare-ident tier args
/// `#[proxima::piped]` accepts (`name = ..` does not apply here: nothing
/// needs to move aside for a name, since the result is bound with `let`
/// like any other expression). Shared by every function-like leaf-lift
/// macro (`pipe!`/`filter!`/each `fanout!`/`fanin!` arm) via
/// [`parse_bang_input`].
struct BangInput {
    expr: Expr,
    args: PipeArgs,
}

impl Parse for BangInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let expr: Expr = input.parse()?;
        let mut args = PipeArgs {
            send: false,
            unpin: false,
            boxed: false,
            name: None,
        };
        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            let metas = Punctuated::<Meta, Token![,]>::parse_terminated(input)?;
            for meta in metas {
                match &meta {
                    Meta::Path(path) if path.is_ident("send") => args.send = true,
                    Meta::Path(path) if path.is_ident("unpin") => args.unpin = true,
                    Meta::Path(path) if path.is_ident("boxed") => args.boxed = true,
                    _ => {
                        return Err(Error::new_spanned(
                            &meta,
                            "unknown arg; expected `send`, `unpin`, or `boxed`",
                        ));
                    }
                }
            }
        }
        Ok(Self { expr, args })
    }
}

/// Parse `expr [, tier args]` out of a bang macro's raw input — the one
/// grammar `pipe!`/`filter!`/`fanout!`/`fanin!` all share for a single arm.
pub(crate) fn parse_bang_input(input: TokenStream) -> Result<(Expr, PipeArgs), Error> {
    let BangInput { expr, args } = Parser::parse2(BangInput::parse, input)?;
    Ok((expr, args))
}

pub fn expand(input: TokenStream) -> Result<TokenStream, Error> {
    let (expr, args) = parse_bang_input(input)?;
    expand_expr(expr, args, "pipe!")
}

/// Shared by `pipe!` and `filter!`: lift a closure literal via [`build_leaf`],
/// or pass any other expression through unchanged. `macro_name` only shapes
/// error text (`filter!` reads better in its own errors than `pipe!` would).
pub(crate) fn expand_expr(expr: Expr, args: PipeArgs, macro_name: &str) -> Result<TokenStream, Error> {
    match expr {
        Expr::Closure(closure) => {
            let struct_ident = Ident::new("__ProximaPipeLeaf", Span::call_site());
            let plan = build_leaf(&struct_ident, &closure, &args)?;
            let definition = &plan.definition;
            Ok(quote! {
                {
                    #definition
                    #struct_ident(#closure)
                }
            })
        }
        other => {
            if args.send || args.unpin || args.boxed {
                return Err(Error::new_spanned(
                    &other,
                    format!(
                        "{macro_name} tier args (`send`/`unpin`/`boxed`) only apply when lifting \
                         a closure literal; this expression is already a pipe value — pass it \
                         through with no extra args"
                    ),
                ));
            }
            Ok(quote!(#other))
        }
    }
}

/// The closure-to-trait bridge a leaf-lift macro mints: a small generic
/// tuple struct (`struct #name<F>(F)`) plus the tier impls its shape
/// qualifies for, FRESH per call site — never a named, reusable type in
/// `proxima-primitives`. `definition` is the struct + every impl block;
/// the caller constructs an instance with `#name(#the_closure_expr)`.
pub(crate) struct LeafPlan {
    pub(crate) definition: TokenStream,
    pub(crate) in_type: Type,
    pub(crate) out_type: Type,
}

/// Whether the closure's single parameter (if any) is present — mirrors
/// `pipe_attr::expand_fn_form`'s `has_input`, so the call expression is
/// `(self.0)(input)` or `(self.0)()` to match.
fn closure_in_type(closure: &ExprClosure) -> Result<(bool, Type), Error> {
    match closure.inputs.len() {
        0 => Ok((false, syn::parse_quote!(()))),
        1 => match &closure.inputs[0] {
            Pat::Type(pat_type) => Ok((true, (*pat_type.ty).clone())),
            other => Err(Error::new_spanned(
                other,
                "a closure lifted this way must spell out its parameter's type, e.g. \
                 `|input: u64| -> Result<u64, Err> { .. }` — the macro reads the annotation, \
                 it does not infer types",
            )),
        },
        _ => Err(Error::new_spanned(
            &closure.inputs,
            "a closure lifted this way takes zero or one parameter (In is the single \
             parameter, or `()` for a source); use a tuple to carry more than one value",
        )),
    }
}

/// Which shape this closure's body reaches, and how `call` invokes it —
/// the closure-literal mirror of `pipe_attr::FutureShape`. Distinct from it
/// (rather than shared) because the async-closure `send` refusal has no
/// `async fn` counterpart: an `async fn`'s `Send`-ness is provable on
/// stable (the compiler checks the concrete generated state machine); an
/// async CLOSURE's is not, because doing so requires naming
/// `AsyncFnMut::CallRefFuture`, `#[unstable(feature = "async_fn_traits")]`.
enum ClosureShape {
    /// plain closure: `Fn(In) -> Result<Out, Err>`, wrapped in
    /// `core::future::ready`. -> `UnpinPipe`/`UnpinSendPipe` (plus base
    /// `Pipe`/`SendPipe`, per `Tier::plan`'s downward closure).
    ReadyWrapped,
    /// async closure, no climb requested: `AsyncFn(In) -> Result<Out, Err>`,
    /// called straight through. -> `Pipe` only. There is no `boxed` climb
    /// here (unlike the attribute macro) — zero-box is a hard constraint on
    /// this leaf-lift bridge; reach for `#[proxima::piped(unpin, boxed)]` on
    /// a hand-written `async fn` if the `Unpin` tier is genuinely needed.
    Passthrough,
}

impl ClosureShape {
    fn climbs_to_unpin(&self) -> bool {
        !matches!(self, ClosureShape::Passthrough)
    }
}

/// Build the leaf-lift bridge for one closure literal: a fresh tuple struct
/// named `struct_ident`, plus the tier impls its shape and `args` qualify
/// for. Shared by `pipe!`, `filter!`, and each `fanout!`/`fanin!` arm.
pub(crate) fn build_leaf(
    struct_ident: &Ident,
    closure: &ExprClosure,
    args: &PipeArgs,
) -> Result<LeafPlan, Error> {
    let is_async = closure.asyncness.is_some();

    if is_async && args.send {
        return Err(Error::new_spanned(
            closure,
            "`send` cannot be combined with an async closure lifted this way: proving the \
             closure's own returned future is `Send` requires naming \
             `AsyncFnMut::CallRefFuture`, which is `#[unstable(feature = \"async_fn_traits\")]` \
             on stable Rust. Two ways around it: (1) write a plain (non-`async`) closure \
             instead — it reaches every tier, including `send`; or (2) hand-write an `async fn` \
             and lift it with `#[proxima::piped(send)]`, whose `Send`-ness the compiler checks \
             directly against the concrete generated state machine.",
        ));
    }

    if args.boxed {
        return Err(Error::new_spanned(
            closure,
            "`boxed` is not supported here: this leaf-lift bridge is zero-box by construction, \
             unlike `#[proxima::piped]`'s opt-in `unpin, boxed` escape hatch. If a boxed `Unpin` \
             climb is genuinely needed, hand-write the closure as an `async fn` and lift it with \
             `#[proxima::piped(unpin, boxed)]` instead.",
        ));
    }

    let shape = match (is_async, args.unpin) {
        (false, _) => ClosureShape::ReadyWrapped,
        (true, true) => {
            return Err(Error::new_spanned(
                closure,
                "`unpin` cannot be applied to an async closure as-is: its body compiles to a \
                 compiler-generated state machine, which is `!Unpin`, so it can't be polled in \
                 place, and this bridge does not offer the `boxed` escape hatch (zero-box is a \
                 hard constraint here). Use a plain closure instead, or hand-write an `async fn` \
                 and lift it with `#[proxima::piped(unpin, boxed)]`.",
            ));
        }
        (true, false) => ClosureShape::Passthrough,
    };

    let (has_input, in_type) = closure_in_type(closure)?;
    let return_type = match &closure.output {
        ReturnType::Type(_, ty) => ty.as_ref(),
        ReturnType::Default => {
            return Err(Error::new_spanned(
                closure,
                "a closure lifted this way must spell out `-> Result<Out, Err>` — the macro \
                 reads the annotation, it does not infer the pipe's Out/Err from the body",
            ));
        }
    };
    let (out_type, err_type) = result_types_from_type(return_type)?;

    let tiers = Tier::plan(shape.climbs_to_unpin(), args.send);

    let call_expr = if has_input {
        quote!((self.0)(input))
    } else {
        quote!((self.0)())
    };
    let call_body = match shape {
        ClosureShape::ReadyWrapped => quote!(::core::future::ready(#call_expr)),
        ClosureShape::Passthrough => call_expr,
    };

    let fn_param_type = if has_input {
        quote!(#in_type)
    } else {
        quote!()
    };
    let closure_trait = if is_async {
        quote!(::core::ops::AsyncFn(#fn_param_type) -> ::core::result::Result<#out_type, #err_type>)
    } else {
        quote!(::core::ops::Fn(#fn_param_type) -> ::core::result::Result<#out_type, #err_type>)
    };

    let tier_impls: Vec<TokenStream> = tiers
        .iter()
        .map(|tier| {
            let trait_path = pipe_trait_path(&tier.trait_ident());
            let future_bound = tier.future_bound();
            let is_send_tier = matches!(tier, Tier::SendPipe | Tier::UnpinSendPipe);
            // a `send` tier only exists here for the sync (ReadyWrapped)
            // shape — the async-closure `send` combination was already
            // refused above — so this extra bound never applies to an
            // opaque, unnameable async-closure future.
            let send_bound = if is_send_tier {
                quote!(+ ::core::marker::Send + ::core::marker::Sync + 'static)
            } else {
                quote!()
            };
            quote! {
                impl<__ProximaF> #trait_path for #struct_ident<__ProximaF>
                where
                    __ProximaF: #closure_trait #send_bound,
                {
                    type In = #in_type;
                    type Out = #out_type;
                    type Err = #err_type;

                    fn call(
                        &self,
                        input: #in_type,
                    ) -> impl ::core::future::Future<Output = ::core::result::Result<#out_type, #err_type>> #future_bound {
                        #call_body
                    }
                }
            }
        })
        .collect();

    let definition = quote! {
        #[allow(non_camel_case_types)]
        struct #struct_ident<__ProximaF>(__ProximaF);

        impl<__ProximaF: ::core::clone::Clone> ::core::clone::Clone for #struct_ident<__ProximaF> {
            fn clone(&self) -> Self {
                Self(::core::clone::Clone::clone(&self.0))
            }
        }

        #(#tier_impls)*
    };

    Ok(LeafPlan {
        definition,
        in_type,
        out_type,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand_ok(input: &str) -> String {
        let tokens: TokenStream = input.parse().expect("parse input");
        expand(tokens).expect("expand").to_string()
    }

    fn expand_err(input: &str) -> String {
        let tokens: TokenStream = input.parse().expect("parse input");
        expand(tokens).expect_err("expected error").to_string()
    }

    #[test]
    fn sync_closure_emits_pipe_and_unpin_pipe_tiers() {
        let expanded =
            expand_ok("|input: u64| -> Result<u64, Infallible> { Ok(input * 2) }");
        assert!(expanded.contains("Pipe for __ProximaPipeLeaf"));
        assert!(expanded.contains("UnpinPipe for __ProximaPipeLeaf"));
        assert!(!expanded.contains("SendPipe"));
        assert!(expanded.contains("core :: future :: ready"));
        assert!(expanded.contains("(self . 0) (input)"));
    }

    #[test]
    fn sync_closure_with_send_emits_all_four_tiers() {
        let expanded =
            expand_ok("|input: u64| -> Result<u64, Infallible> { Ok(input * 2) }, send");
        assert!(expanded.contains("Pipe for __ProximaPipeLeaf"));
        assert!(expanded.contains("SendPipe for __ProximaPipeLeaf"));
        assert!(expanded.contains("UnpinPipe for __ProximaPipeLeaf"));
        assert!(expanded.contains("UnpinSendPipe for __ProximaPipeLeaf"));
        assert!(expanded.contains(":: core :: marker :: Send + :: core :: marker :: Sync"));
    }

    #[test]
    fn zero_arg_sync_closure_derives_unit_in_type() {
        let expanded = expand_ok("|| -> Result<u8, Infallible> { Ok(7) }");
        assert!(expanded.contains("type In = () ;"));
        assert!(expanded.contains("(self . 0) ()"));
    }

    #[test]
    fn async_closure_emits_only_pipe_tier() {
        let expanded =
            expand_ok("async move |input: u64| -> Result<u64, Infallible> { Ok(input) }");
        assert!(expanded.contains("Pipe for __ProximaPipeLeaf"));
        assert!(!expanded.contains("SendPipe"));
        assert!(!expanded.contains("UnpinPipe"));
        assert!(!expanded.contains("core :: future :: ready"));
        assert!(expanded.contains("AsyncFn"));
    }

    #[test]
    fn async_closure_with_send_is_refused() {
        let err = expand_err(
            "async move |input: u64| -> Result<u64, Infallible> { Ok(input) }, send",
        );
        assert!(err.contains("`send` cannot be combined with an async closure"));
        assert!(err.contains("async_fn_traits"));
    }

    #[test]
    fn async_closure_with_unpin_is_refused_zero_box() {
        let err = expand_err(
            "async move |input: u64| -> Result<u64, Infallible> { Ok(input) }, unpin",
        );
        assert!(err.contains("cannot be applied to an async closure as-is"));
        assert!(err.contains("does not offer the `boxed` escape hatch"));
    }

    #[test]
    fn boxed_is_always_refused_sync_or_async() {
        let sync_err = expand_err("|input: u64| -> Result<u64, Infallible> { Ok(input) }, boxed");
        assert!(sync_err.contains("zero-box by construction"));

        let async_err = expand_err(
            "async move |input: u64| -> Result<u64, Infallible> { Ok(input) }, boxed",
        );
        assert!(async_err.contains("zero-box by construction"));
    }

    #[test]
    fn closure_missing_return_type_annotation_is_refused() {
        let err = expand_err("|input: u64| { Ok(input) }");
        assert!(err.contains("must spell out"));
    }

    #[test]
    fn closure_with_untyped_parameter_is_refused() {
        let err = expand_err("|input| -> Result<u64, Infallible> { Ok(input) }");
        assert!(err.contains("must spell out its parameter's type"));
    }

    #[test]
    fn closure_with_more_than_one_parameter_is_refused() {
        let err =
            expand_err("|a: u64, b: u64| -> Result<u64, Infallible> { Ok(a + b) }");
        assert!(err.contains("zero or one parameter"));
    }

    #[test]
    fn unknown_tier_arg_is_refused() {
        let err = expand_err("|input: u64| -> Result<u64, Infallible> { Ok(input) }, bogus");
        assert!(err.contains("unknown arg"));
    }

    #[test]
    fn a_non_closure_expression_passes_through_unchanged() {
        let expanded = expand_ok("some_existing_pipe_value");
        assert_eq!(expanded, "some_existing_pipe_value");
    }

    #[test]
    fn tier_args_on_a_non_closure_expression_are_refused() {
        let err = expand_err("some_existing_pipe_value, send");
        assert!(err.contains("already a pipe value"));
    }
}
