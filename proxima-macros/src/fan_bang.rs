//! `fanout!(a, b, ..)` / `fanin!(a, b, ..)` — variadic sibling of `pipe!`:
//! build a [`FanOut`]/[`FanIn`] over N arms in one call, where each arm is
//! EITHER a closure literal (leaf-lifted the same way `pipe!` does, via
//! [`build_leaf`]) or an already-built pipe expression, passed through.
//!
//! Variadic arity is the entire reason these two macros exist: `FanOut<S,
//! Policy>`/`FanIn<S, Strategy, N>` both hold a HOMOGENEOUS collection of one
//! concrete sink/source type `S` — but N closure literals are N distinct,
//! unnameable types, one per call site (`pipe!`'s own bridge, minted fresh
//! per arm here too). Reconciling "N distinct types" with "one S" without a
//! `Box<dyn _>` is a sum type: a macro-generated enum with one variant per
//! arm, each variant generic over that arm's own (possibly-anonymous) type.
//! `S` becomes that enum. Every arm — closure-lifted or pass-through alike —
//! is one variant; `FanOut`/`FanIn`'s existing broadcast/merge loops are
//! UNCHANGED (they were already generic over `S: Pipe`/`S: UnpinPipe`), they
//! just now iterate over enum values instead of one concrete struct.
//!
//! The enum's own `Pipe`/`SendPipe` impls dispatch with an ordinary `match`
//! inside one `async move { .. }` block — awaiting inside a single shared
//! async block unifies N distinct per-arm future types into ONE anonymous
//! future for the whole match, the same way [`AndThen`]'s two-stage `Pipe`
//! impl does today. Its `UnpinPipe`/`UnpinSendPipe` impls need a second,
//! hand-rolled poll-dispatch enum (one variant per arm, each holding that
//! arm's own `Unpin` future) because there is no `async move` block on that
//! tier to hide the union behind — same reason `FanOut` and `AndThen` each
//! hand-roll their own `Unpin`-tier `Future` next to their async-block form.
//! Zero boxes either way.
//!
//! `fanin!`'s arms carry one extra restriction `fanout!`'s don't: [`FanIn`]
//! requires `UnpinPipe<In = (), Err = Exhausted> + DropSafe` sources — a
//! synchronous, never-suspending merge loop, not an async one — so a
//! closure-literal arm inside `fanin!` must be a plain (non-`async`)
//! closure; an async closure arm is refused with that reason. Its per-arm
//! leaf struct is unconditionally `DropSafe` (justified narrowly: a
//! `core::future::ready`-backed leaf never suspends, so dropping it
//! mid-poll can never observe partial state — there IS no "mid-poll" for a
//! future that resolves on its first poll), and the enum propagates
//! `DropSafe` from its arms the same AND-propagation shape
//! `primitives.rs`'s `marker_propagation` module already uses for
//! [`AndThen`].
//!
//! [`FanOut`]: proxima_primitives::pipe::FanOut
//! [`FanIn`]: proxima_primitives::pipe::FanIn
//! [`AndThen`]: proxima_primitives::pipe::AndThen

use proc_macro2::{Span, TokenStream};
use proc_macro_crate::{FoundCrate, crate_name};
use quote::{format_ident, quote};
use syn::parse::{Parse, ParseStream, Parser};
use syn::punctuated::Punctuated;
use syn::{Error, Expr, Ident, Token};

use crate::pipe_attr::{PipeArgs, pipe_path};
use crate::pipe_bang::build_leaf;

/// Which fan macro is expanding — the only place their behaviour diverges
/// (see the module doc's "one extra restriction").
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum FanKind {
    Out,
    In,
}

struct FanArms(Punctuated<Expr, Token![,]>);

impl Parse for FanArms {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Punctuated::parse_terminated(input).map(FanArms)
    }
}

pub fn expand_fanout(input: TokenStream) -> Result<TokenStream, Error> {
    expand(input, FanKind::Out)
}

pub fn expand_fanin(input: TokenStream) -> Result<TokenStream, Error> {
    expand(input, FanKind::In)
}

/// Resolve `::proxima_core::markers::DropSafe`, reachable either as a direct
/// dependency (`proxima-core`, e.g. from inside `proxima-primitives`
/// itself) or through the `proxima` umbrella's `pub use proxima_core as
/// error;` re-export. Mirrors `pipe_attr::pipe_path`'s fallback chain.
fn drop_safe_path() -> TokenStream {
    if let Ok(found) = crate_name("proxima-core") {
        return match found {
            FoundCrate::Itself => quote!(crate::markers::DropSafe),
            FoundCrate::Name(name) => {
                let krate = Ident::new(&name, Span::call_site());
                quote!(::#krate::markers::DropSafe)
            }
        };
    }
    match crate_name("proxima") {
        Ok(FoundCrate::Itself) => quote!(crate::error::markers::DropSafe),
        Ok(FoundCrate::Name(name)) => {
            let krate = Ident::new(&name, Span::call_site());
            quote!(::#krate::error::markers::DropSafe)
        }
        Err(_) => quote!(::proxima_core::markers::DropSafe),
    }
}

fn expand(input: TokenStream, kind: FanKind) -> Result<TokenStream, Error> {
    let FanArms(arms) = Parser::parse2(FanArms::parse, input)?;
    if arms.is_empty() {
        return Err(Error::new(
            Span::call_site(),
            "expected at least one arm, e.g. `fanout!(a, b)` / `fanin!(a, b)`",
        ));
    }

    let default_args = PipeArgs {
        send: false,
        unpin: false,
        boxed: false,
        name: None,
    };

    let enum_ident = Ident::new("__ProximaFanArms", Span::call_site());

    let mut leaf_definitions = Vec::new();
    let mut generic_idents = Vec::new();
    let mut variant_idents = Vec::new();
    let mut arm_values = Vec::new();

    for (index, arm) in arms.into_iter().enumerate() {
        let generic_ident = format_ident!("__ProximaFanArm{index}", span = Span::call_site());
        let variant_ident = format_ident!("Arm{index}", span = Span::call_site());

        let inner_value = match &arm {
            Expr::Closure(closure) => {
                if kind == FanKind::In && closure.asyncness.is_some() {
                    return Err(Error::new_spanned(
                        closure,
                        "fanin! arms must be plain (non-`async`) closures: FanIn's merge loop \
                         polls each source synchronously in place and requires a genuinely \
                         `Unpin`, never-suspending future. Lift an async source with \
                         `#[proxima::piped(unpin, boxed)]` on a hand-written `async fn` first \
                         and pass the resulting value in as a pass-through arm instead.",
                    ));
                }
                let struct_ident =
                    format_ident!("__ProximaFanLeaf{index}", span = Span::call_site());
                let plan = build_leaf(&struct_ident, closure, &default_args)?;
                let definition = plan.definition;
                let extra_drop_safe = if kind == FanKind::In {
                    let drop_safe = drop_safe_path();
                    quote! {
                        impl<__ProximaF> #drop_safe for #struct_ident<__ProximaF> {}
                    }
                } else {
                    quote!()
                };
                leaf_definitions.push(quote! {
                    #definition
                    #extra_drop_safe
                });
                quote!(#struct_ident(#arm))
            }
            _ => quote!(#arm),
        };
        let value = quote!(#enum_ident::#variant_ident(#inner_value));

        generic_idents.push(generic_ident);
        variant_idents.push(variant_ident);
        arm_values.push(value);
    }

    let call_enum_ident = Ident::new("__ProximaFanArmsCall", Span::call_site());

    let enum_definition = quote! {
        #[allow(non_camel_case_types)]
        enum #enum_ident<#(#generic_idents),*> {
            #(#variant_idents(#generic_idents)),*
        }
    };

    let async_tier_impls = build_async_tier_impls(&enum_ident, &generic_idents, &variant_idents);
    let unpin_tier_impls = build_unpin_tier_impls(
        &enum_ident,
        &call_enum_ident,
        &generic_idents,
        &variant_idents,
    );
    let drop_safe_impl = if kind == FanKind::In {
        let drop_safe = drop_safe_path();
        quote! {
            impl<#(#generic_idents: #drop_safe),*> #drop_safe for #enum_ident<#(#generic_idents),*> {}
        }
    } else {
        quote!()
    };

    let arms_value = match kind {
        FanKind::Out => {
            let fan_out = pipe_path(quote!(FanOut));
            quote! {
                {
                    extern crate alloc;
                    #fan_out::all_or_nothing(alloc::vec![#(#arm_values),*])
                }
            }
        }
        FanKind::In => {
            let fan_in = pipe_path(quote!(FanIn));
            let select = pipe_path(quote!(Select));
            quote! {
                #fan_in::new(
                    [#(#arm_values),*],
                    #select::RoundRobin,
                )
            }
        }
    };

    Ok(quote! {
        {
            #(#leaf_definitions)*
            #enum_definition
            #async_tier_impls
            #unpin_tier_impls
            #drop_safe_impl
            #arms_value
        }
    })
}

/// `Pipe`/`SendPipe` for the generated enum: one `match` inside one
/// `async move { .. }` block, so N distinct per-arm future types unify into
/// the ONE opaque future the trait's RPITIT `call` returns. Zero boxes.
fn build_async_tier_impls(
    enum_ident: &Ident,
    generic_idents: &[Ident],
    variant_idents: &[Ident],
) -> TokenStream {
    let pipe_trait = pipe_path(quote!(Pipe));
    let send_pipe_trait = pipe_path(quote!(SendPipe));
    let pipe_arms = quote! {
        match self {
            #(#enum_ident::#variant_idents(inner) => #pipe_trait::call(inner, input).await,)*
        }
    };
    let send_arms = quote! {
        match self {
            #(#enum_ident::#variant_idents(inner) => #send_pipe_trait::call(inner, input).await,)*
        }
    };

    quote! {
        impl<#(#generic_idents),*, __ProximaIn, __ProximaOut, __ProximaErr>
            #pipe_trait for #enum_ident<#(#generic_idents),*>
        where
            #(#generic_idents: #pipe_trait<
                In = __ProximaIn,
                Out = __ProximaOut,
                Err = __ProximaErr,
            >,)*
            __ProximaErr: ::core::fmt::Debug + 'static,
        {
            type In = __ProximaIn;
            type Out = __ProximaOut;
            type Err = __ProximaErr;

            fn call(
                &self,
                input: __ProximaIn,
            ) -> impl ::core::future::Future<Output = ::core::result::Result<__ProximaOut, __ProximaErr>> {
                async move { #pipe_arms }
            }
        }

        impl<#(#generic_idents),*, __ProximaIn, __ProximaOut, __ProximaErr>
            #send_pipe_trait for #enum_ident<#(#generic_idents),*>
        where
            #(#generic_idents: #send_pipe_trait<
                In = __ProximaIn,
                Out = __ProximaOut,
                Err = __ProximaErr,
            >,)*
            __ProximaIn: ::core::marker::Send,
            __ProximaErr: ::core::fmt::Debug + ::core::marker::Send + 'static,
        {
            type In = __ProximaIn;
            type Out = __ProximaOut;
            type Err = __ProximaErr;

            fn call(
                &self,
                input: __ProximaIn,
            ) -> impl ::core::future::Future<Output = ::core::result::Result<__ProximaOut, __ProximaErr>> + ::core::marker::Send {
                async move { #send_arms }
            }
        }
    }
}

/// `UnpinPipe`/`UnpinSendPipe` for the generated enum: a hand-rolled
/// poll-dispatch enum (one variant per arm holding that arm's own already-
/// `Unpin` future), matched once in `poll` — no state machine beyond that,
/// since dispatching to exactly one already-in-flight future needs none.
/// Mirrors `FanOut`'s own `FanOutUnpinCall`/`AndThen`'s `AndThenUnpinCall`.
fn build_unpin_tier_impls(
    enum_ident: &Ident,
    call_enum_ident: &Ident,
    generic_idents: &[Ident],
    variant_idents: &[Ident],
) -> TokenStream {
    let unpin_pipe_trait = pipe_path(quote!(UnpinPipe));
    let unpin_send_pipe_trait = pipe_path(quote!(UnpinSendPipe));
    let call_generic_idents: Vec<Ident> = generic_idents
        .iter()
        .map(|ident| format_ident!("{ident}Call"))
        .collect();

    let call_enum_definition = quote! {
        #[allow(non_camel_case_types)]
        enum #call_enum_ident<#(#call_generic_idents),*> {
            #(#variant_idents(#call_generic_idents)),*
        }
    };

    let poll_arms = quote! {
        match self.get_mut() {
            #(#call_enum_ident::#variant_idents(inner) => ::core::pin::Pin::new(inner).poll(cx),)*
        }
    };

    let unpin_call_arms = quote! {
        match self {
            #(#enum_ident::#variant_idents(inner) =>
                #call_enum_ident::#variant_idents(#unpin_pipe_trait::call(inner, input)),)*
        }
    };
    let unpin_send_call_arms = quote! {
        match self {
            #(#enum_ident::#variant_idents(inner) =>
                #call_enum_ident::#variant_idents(#unpin_send_pipe_trait::call(inner, input)),)*
        }
    };

    quote! {
        #call_enum_definition

        impl<#(#call_generic_idents),*, __ProximaOut, __ProximaErr> ::core::future::Future
            for #call_enum_ident<#(#call_generic_idents),*>
        where
            #(#call_generic_idents: ::core::future::Future<
                Output = ::core::result::Result<__ProximaOut, __ProximaErr>,
            > + ::core::marker::Unpin,)*
        {
            type Output = ::core::result::Result<__ProximaOut, __ProximaErr>;

            fn poll(
                self: ::core::pin::Pin<&mut Self>,
                cx: &mut ::core::task::Context<'_>,
            ) -> ::core::task::Poll<Self::Output> {
                #poll_arms
            }
        }

        impl<#(#generic_idents),*, __ProximaIn, __ProximaOut, __ProximaErr>
            #unpin_pipe_trait for #enum_ident<#(#generic_idents),*>
        where
            #(#generic_idents: #unpin_pipe_trait<
                In = __ProximaIn,
                Out = __ProximaOut,
                Err = __ProximaErr,
            >,)*
            __ProximaIn: ::core::clone::Clone,
            __ProximaErr: ::core::fmt::Debug + 'static,
        {
            type In = __ProximaIn;
            type Out = __ProximaOut;
            type Err = __ProximaErr;

            fn call(
                &self,
                input: __ProximaIn,
            ) -> impl ::core::future::Future<Output = ::core::result::Result<__ProximaOut, __ProximaErr>> + ::core::marker::Unpin {
                #unpin_call_arms
            }
        }

        impl<#(#generic_idents),*, __ProximaIn, __ProximaOut, __ProximaErr>
            #unpin_send_pipe_trait for #enum_ident<#(#generic_idents),*>
        where
            #(#generic_idents: #unpin_send_pipe_trait<
                In = __ProximaIn,
                Out = __ProximaOut,
                Err = __ProximaErr,
            >,)*
            __ProximaIn: ::core::clone::Clone + ::core::marker::Send,
            __ProximaErr: ::core::fmt::Debug + ::core::marker::Send + 'static,
        {
            type In = __ProximaIn;
            type Out = __ProximaOut;
            type Err = __ProximaErr;

            fn call(
                &self,
                input: __ProximaIn,
            ) -> impl ::core::future::Future<Output = ::core::result::Result<__ProximaOut, __ProximaErr>> + ::core::marker::Send + ::core::marker::Unpin {
                #unpin_send_call_arms
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand_ok(kind: FanKind, input: &str) -> String {
        let tokens: TokenStream = input.parse().expect("parse input");
        expand(tokens, kind).expect("expand").to_string()
    }

    fn expand_err(kind: FanKind, input: &str) -> String {
        let tokens: TokenStream = input.parse().expect("parse input");
        expand(tokens, kind).expect_err("expected error").to_string()
    }

    #[test]
    fn fanout_lifts_two_closure_arms_into_one_enum() {
        let expanded = expand_ok(
            FanKind::Out,
            "|input: u64| -> Result<(), Infallible> { Ok(()) }, \
             |input: u64| -> Result<(), Infallible> { Ok(()) }",
        );
        assert!(expanded.contains("__ProximaFanLeaf0"));
        assert!(expanded.contains("__ProximaFanLeaf1"));
        assert!(expanded.contains("enum __ProximaFanArms"));
        assert!(expanded.contains("Arm0"));
        assert!(expanded.contains("Arm1"));
        assert!(expanded.contains("FanOut :: all_or_nothing"));
        assert!(!expanded.contains("Box :: pin"));
        assert!(!expanded.contains("dyn"));
    }

    #[test]
    fn fanout_passes_a_non_closure_arm_through_unchanged_as_a_variant() {
        let expanded = expand_ok(
            FanKind::Out,
            "existing_pipe, |input: u64| -> Result<(), Infallible> { Ok(()) }",
        );
        assert!(expanded.contains("Arm0 (existing_pipe)"), "{expanded}");
        assert!(expanded.contains("__ProximaFanLeaf1"));
    }

    #[test]
    fn fanout_emits_no_drop_safe_propagation() {
        // FanOut never requires DropSafe on its sinks — only FanIn does —
        // so fanout! has nothing to propagate.
        let expanded = expand_ok(
            FanKind::Out,
            "|input: u64| -> Result<(), Infallible> { Ok(()) }",
        );
        assert!(!expanded.contains("DropSafe"));
    }

    #[test]
    fn fanin_lifts_closure_arms_and_defaults_to_round_robin() {
        let expanded = expand_ok(
            FanKind::In,
            "|(): ()| -> Result<u8, Exhausted> { Ok(1) }, \
             |(): ()| -> Result<u8, Exhausted> { Ok(2) }",
        );
        assert!(expanded.contains("FanIn :: new"));
        assert!(expanded.contains("Select :: RoundRobin"));
        assert!(expanded.contains("DropSafe for __ProximaFanLeaf0"));
        assert!(expanded.contains("DropSafe for __ProximaFanLeaf1"));
        assert!(expanded.contains("DropSafe for __ProximaFanArms"));
    }

    #[test]
    fn fanin_rejects_an_async_closure_arm() {
        let err = expand_err(
            FanKind::In,
            "async move |(): ()| -> Result<u8, Exhausted> { Ok(1) }",
        );
        assert!(err.contains("fanin! arms must be plain (non-`async`) closures"));
    }

    #[test]
    fn empty_arm_list_is_refused() {
        let err = expand_err(FanKind::Out, "");
        assert!(err.contains("expected at least one arm"));
    }

    #[test]
    fn single_arm_is_accepted() {
        let expanded = expand_ok(
            FanKind::Out,
            "|input: u64| -> Result<(), Infallible> { Ok(()) }",
        );
        assert!(expanded.contains("Arm0"));
        assert!(!expanded.contains("Arm1"));
    }
}
