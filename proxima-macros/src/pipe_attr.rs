//! `#[proxima::piped]` — generates a [`Pipe`]/[`SendPipe`]/[`UnpinPipe`]/
//! [`UnpinSendPipe`] impl from a plain function, removing the hand-written
//! unit-struct-plus-impl boilerplate every leaf pipe otherwise repeats.
//!
//! The macro adds NO noun to the pipe algebra (still exactly four tiers,
//! `proxima-primitives/src/pipe/primitives.rs`) — it only picks which one
//! tier a given function belongs to and writes the impl:
//!
//! - `sig.asyncness` decides the `Unpin` axis for free: an `async fn`'s
//!   future is a compiler-generated state machine (`!Unpin`), so it emits
//!   [`Pipe`]; a plain `fn` is wrapped in `core::future::ready`, whose future
//!   IS `Unpin`, so it emits [`UnpinPipe`].
//! - `Send` is NEVER inferred — only `#[proxima::piped(send)]` climbs to
//!   [`SendPipe`] / [`UnpinSendPipe`]. Climbing tiers because the types
//!   happened to allow it would charge a cost the caller never asked for
//!   (see `examples/send/README.md`).
//! - Exactly one tier is emitted per function. A type implementing two tiers
//!   at once makes every call site ambiguous (E0034) — that is not a
//!   convenience, it is breakage.
//! - `#[proxima::piped(unpin)]` on a bare `async fn` is refused at compile
//!   time: a compiler-generated async block future is never `Unpin`. But
//!   `Unpin` constrains how the future is SPELLED, not whether it awaits —
//!   `Pin<Box<F>>` is `Unpin` for any `F` (`Box` is a fixed-address
//!   indirection, not self-referential), so an `async fn` still reaches the
//!   `Unpin` tier via `#[proxima::piped(unpin, boxed)]`, at the cost of one
//!   heap allocation per call. `boxed` is never inferred — same opt-in
//!   discipline as `send` — and is gated behind the invoking crate's own
//!   `alloc` Cargo feature, so it cannot appear in a bare no_std build.
//! - The generated struct is always fieldless, so it always derives `Clone`
//!   unconditionally — no `derive(...)` macro arg, `Clone` is the only bound
//!   any combinator over a leaf pipe ever needs (`RateLimit`, `Retry`,
//!   `Delay`, `Isolate`, `Diff`, `Transform`, `Validate` all require
//!   `Inner: Clone`).
//! - `#[proxima::piped(...)]` also accepts a plain inherent `impl Foo { fn
//!   call(..) { .. } }` block (an `async fn call(&self, In) -> Result<Out,
//!   Err>`, or a sync `fn call(&self, In) -> impl Future<..> + Unpin`), for a
//!   STATEFUL pipe whose struct already exists with its own fields. Same
//!   tier selection, same `send`/`unpin`/`boxed` args; `name = ..` does not
//!   apply (the trait wears the impl's own type name). See
//!   `expand_impl_form`.
//!
//! [`Pipe`]: proxima_primitives::pipe::Pipe

use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{
    Error, Expr, FnArg, GenericArgument, Ident, ImplItem, ItemFn, ItemImpl, Meta, Pat,
    PathArguments, ReturnType, Token, Type, TypeParamBound, Visibility, parse2,
};

/// Parsed `#[proxima::piped(...)]` args.
// `pub(crate)`: the function-like leaf-lift macros (`pipe!`/`fanout!`/
// `fanin!` in `pipe_bang.rs`/`fan_bang.rs`) accept the same
// `send`/`unpin`/`boxed` vocabulary and share this parser and its fields
// rather than re-inventing them.
pub(crate) struct PipeArgs {
    /// `send` — climb to the cross-core `SendPipe`/`UnpinSendPipe` form.
    /// Never inferred; the caller must opt in explicitly.
    pub(crate) send: bool,
    /// `unpin` — documents the (already-automatic) `Unpin` tier on a sync
    /// fn; on an `async fn` it requires `boxed` too (see module doc).
    pub(crate) unpin: bool,
    /// `boxed` — on an `async fn`, reach the `Unpin` tier by heap-allocating
    /// one `Pin<Box<F>>` per call instead of passing the future through.
    /// Never inferred; alloc-gated (see `FutureShape::BoxPinWrapped`).
    pub(crate) boxed: bool,
    /// `name = Ident` — override the generated struct's name.
    pub(crate) name: Option<Ident>,
}

pub(crate) fn parse_args(args: TokenStream) -> Result<PipeArgs, Error> {
    let mut parsed = PipeArgs {
        send: false,
        unpin: false,
        boxed: false,
        name: None,
    };

    if args.is_empty() {
        return Ok(parsed);
    }

    let metas = Punctuated::<Meta, Token![,]>::parse_terminated.parse2(args)?;
    for meta in metas {
        match &meta {
            Meta::Path(path) if path.is_ident("send") => parsed.send = true,
            Meta::Path(path) if path.is_ident("unpin") => parsed.unpin = true,
            Meta::Path(path) if path.is_ident("boxed") => parsed.boxed = true,
            Meta::NameValue(name_value) if name_value.path.is_ident("name") => {
                parsed.name = Some(extract_ident(&name_value.value)?);
            }
            _ => {
                return Err(Error::new_spanned(
                    &meta,
                    "unknown #[proxima::piped] arg; expected `send`, `unpin`, `boxed`, or `name = Ident`",
                ));
            }
        }
    }

    Ok(parsed)
}

fn extract_ident(expr: &Expr) -> Result<Ident, Error> {
    match expr {
        Expr::Path(expr_path) if expr_path.path.segments.len() == 1 => {
            Ok(expr_path.path.segments[0].ident.clone())
        }
        _ => Err(Error::new_spanned(
            expr,
            "expected a bare identifier for `name`, e.g. `name = Foo`",
        )),
    }
}

/// The fn's single parameter type, or `()` for a zero-arg fn (the source
/// form). More than one parameter, or a `self` receiver, is rejected — the
/// pipe contract is single-`In`, and a multi-arg fn should take a tuple or
/// struct `In` instead of gaining new macro surface.
fn extract_in_type(sig: &syn::Signature) -> Result<Type, Error> {
    // a `self` receiver is rejected regardless of arity — it is never a
    // valid `In` for a free-standing pipe fn, and reporting it ahead of the
    // arity check gives the caller the actually-relevant error.
    if let Some(FnArg::Receiver(receiver)) = sig.inputs.first() {
        return Err(Error::new_spanned(
            receiver,
            "#[proxima::piped] does not support a `self` parameter",
        ));
    }

    match sig.inputs.len() {
        0 => Ok(syn::parse_quote!(())),
        1 => match &sig.inputs[0] {
            FnArg::Receiver(receiver) => Err(Error::new_spanned(
                receiver,
                "#[proxima::piped] does not support a `self` parameter",
            )),
            FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
                Pat::Ident(_) | Pat::Wild(_) => Ok((*pat_type.ty).clone()),
                other => Err(Error::new_spanned(
                    other,
                    "#[proxima::piped] fn parameters must be a simple identifier",
                )),
            },
        },
        _ => Err(Error::new_spanned(
            &sig.inputs,
            "#[proxima::piped] fns take zero or one argument (In is the single parameter, \
             or () for a source); use a tuple or struct to carry more than one value",
        )),
    }
}

/// Pull `Out`/`Err` out of a `Result<Out, Err>` type written by hand — the
/// pipe contract's `Out`/`Err` pair, so there is exactly one thing to write
/// down, not two. Shared by the free-fn path (`sig.output` IS this type) and
/// the impl-block async path (`sig.output` is this type too, since an
/// `async fn`'s declared return type is the `Result`, not a `Future`).
pub(crate) fn result_types_from_type(return_type: &Type) -> Result<(Type, Type), Error> {
    let Type::Path(type_path) = return_type else {
        return Err(Error::new_spanned(
            return_type,
            "#[proxima::piped] fns must return Result<Out, Err>",
        ));
    };

    let Some(last_segment) = type_path.path.segments.last() else {
        return Err(Error::new_spanned(
            return_type,
            "#[proxima::piped] fns must return Result<Out, Err>",
        ));
    };

    if last_segment.ident != "Result" {
        return Err(Error::new_spanned(
            return_type,
            "#[proxima::piped] fns must return Result<Out, Err>",
        ));
    }

    let PathArguments::AngleBracketed(generics) = &last_segment.arguments else {
        return Err(Error::new_spanned(
            return_type,
            "#[proxima::piped] fns must return Result<Out, Err> with both type parameters written out",
        ));
    };

    let type_args: Vec<&Type> = generics
        .args
        .iter()
        .filter_map(|arg| match arg {
            GenericArgument::Type(ty) => Some(ty),
            _ => None,
        })
        .collect();

    match type_args.as_slice() {
        [out_ty, err_ty] => Ok(((*out_ty).clone(), (*err_ty).clone())),
        _ => Err(Error::new_spanned(
            return_type,
            "#[proxima::piped] fns must return Result<Out, Err> with both type parameters written out",
        )),
    }
}

/// The fn must return `Result<Out, Err>` — see [`result_types_from_type`].
fn extract_result_types(sig: &syn::Signature) -> Result<(Type, Type), Error> {
    let return_type = match &sig.output {
        ReturnType::Default => {
            return Err(Error::new_spanned(
                sig,
                "#[proxima::piped] fns must return Result<Out, Err>; this fn has no return type",
            ));
        }
        ReturnType::Type(_, ty) => ty.as_ref(),
    };
    result_types_from_type(return_type)
}

/// Pull `Out`/`Err` out of a hand-written `impl Future<Output = Result<Out,
/// Err>> + ..` return type — the shape the impl-block sync `call` form
/// declares (see `expand_impl_form`). The bounds alongside `Future` (`Send`,
/// `Unpin`, ...) are never inspected here: the generated trait impl writes
/// its OWN bound from `Tier::future_bound`, the same way the free-fn path
/// never reads bounds off a plain fn's `Result` return either.
fn future_output_result_types(sig: &syn::Signature) -> Result<(Type, Type), Error> {
    let return_type = match &sig.output {
        ReturnType::Default => {
            return Err(Error::new_spanned(
                sig,
                "#[proxima::piped] a sync `call` must return `impl Future<Output = Result<Out, Err>> + Unpin`; this method has no return type",
            ));
        }
        ReturnType::Type(_, ty) => ty.as_ref(),
    };

    let Type::ImplTrait(impl_trait) = return_type else {
        return Err(Error::new_spanned(
            return_type,
            "#[proxima::piped] a sync `call` must return `impl Future<Output = Result<Out, Err>> + Unpin`",
        ));
    };

    let future_bound = impl_trait.bounds.iter().find_map(|bound| match bound {
        TypeParamBound::Trait(trait_bound)
            if trait_bound
                .path
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "Future") =>
        {
            trait_bound.path.segments.last()
        }
        _ => None,
    });

    let Some(future_segment) = future_bound else {
        return Err(Error::new_spanned(
            return_type,
            "#[proxima::piped] a sync `call`'s `impl Future<..>` bound must name `Future` directly",
        ));
    };

    let PathArguments::AngleBracketed(generics) = &future_segment.arguments else {
        return Err(Error::new_spanned(
            return_type,
            "#[proxima::piped] a sync `call` must spell out `Future<Output = Result<Out, Err>>`",
        ));
    };

    let output_type = generics.args.iter().find_map(|arg| match arg {
        GenericArgument::AssocType(assoc) if assoc.ident == "Output" => Some(&assoc.ty),
        _ => None,
    });

    let Some(output_type) = output_type else {
        return Err(Error::new_spanned(
            return_type,
            "#[proxima::piped] a sync `call` must spell out `Future<Output = Result<Out, Err>>`",
        ));
    };

    result_types_from_type(output_type)
}

/// Resolve `…::pipe::#tail` for whatever crate invoked `#[proxima::piped]`: a
/// direct `proxima-primitives` dep, the `proxima` umbrella re-export, or this
/// crate itself. Mirrors `span_attr::recorder_path`.
pub(crate) fn pipe_path(tail: TokenStream) -> TokenStream {
    if let Ok(found) = crate_name("proxima-primitives") {
        return match found {
            FoundCrate::Itself => quote!(crate::pipe::#tail),
            FoundCrate::Name(name) => {
                let krate = Ident::new(&name, Span::call_site());
                quote!(::#krate::pipe::#tail)
            }
        };
    }
    match crate_name("proxima") {
        Ok(FoundCrate::Itself) => quote!(crate::pipe::#tail),
        Ok(FoundCrate::Name(name)) => {
            let krate = Ident::new(&name, Span::call_site());
            quote!(::#krate::pipe::#tail)
        }
        Err(_) => quote!(::proxima_primitives::pipe::#tail),
    }
}

pub(crate) fn pipe_trait_path(trait_ident: &Ident) -> TokenStream {
    pipe_path(quote!(#trait_ident))
}

/// Which of the four standalone tiers this fn maps to. `asyncness` decides
/// the `Unpin` axis for free (see module doc); `send` is read from the
/// explicit `#[proxima::piped(send)]` opt-in only.
///
/// A pipe always implements EVERY tier its declared bounds qualify for, not
/// just one — the higher tiers are additive constraints on top of the root
/// form, never a replacement for it (`proxima_primitives::pipe::primitives`'s
/// module doc). `Tier::plan` computes that downward closure: `Pipe` always;
/// `SendPipe` when `send`; `UnpinPipe` when the fn's future shape is
/// `Unpin`-capable; `UnpinSendPipe` when both. A bare `#[proxima::piped(send)]`
/// async fn ends up `Pipe` AND `SendPipe`; `#[proxima::piped(send, unpin,
/// boxed)]` ends up all four.
#[derive(PartialEq, Eq)]
pub(crate) enum Tier {
    Pipe,
    SendPipe,
    UnpinPipe,
    UnpinSendPipe,
}

impl Tier {
    /// The full downward closure of tiers a fn with this future shape and
    /// `send` opt-in implements. `Pipe` is always first (base tier).
    pub(crate) fn plan(climbs_to_unpin: bool, send: bool) -> Vec<Tier> {
        let mut tiers = vec![Tier::Pipe];
        if send {
            tiers.push(Tier::SendPipe);
        }
        if climbs_to_unpin {
            tiers.push(Tier::UnpinPipe);
        }
        if climbs_to_unpin && send {
            tiers.push(Tier::UnpinSendPipe);
        }
        tiers
    }

    pub(crate) fn trait_ident(&self) -> Ident {
        let name = match self {
            Tier::Pipe => "Pipe",
            Tier::SendPipe => "SendPipe",
            Tier::UnpinPipe => "UnpinPipe",
            Tier::UnpinSendPipe => "UnpinSendPipe",
        };
        Ident::new(name, Span::call_site())
    }

    /// Extra bounds on the returned `impl Future`, beyond `Output = ..`.
    pub(crate) fn future_bound(&self) -> TokenStream {
        match self {
            Tier::Pipe => quote!(),
            Tier::SendPipe => quote!(+ ::core::marker::Send),
            Tier::UnpinPipe => quote!(+ ::core::marker::Unpin),
            Tier::UnpinSendPipe => quote!(+ ::core::marker::Send + ::core::marker::Unpin),
        }
    }
}

/// How `call`'s body turns the fn's own return into the tier's required
/// future shape. This is orthogonal to [`Tier`] (which trait + `Send` bound):
/// `ReadyWrapped` and `BoxPinWrapped` both land on `UnpinPipe`/
/// `UnpinSendPipe`, just at a different, explicitly chosen price.
enum FutureShape {
    /// `async fn`, no climb requested: its own future, passed straight
    /// through — RPITIT passthrough, zero extra cost. -> `Pipe`/`SendPipe`.
    Passthrough,
    /// plain `fn`, wrapped in `core::future::ready` (`Ready<T>` is `Unpin`
    /// unconditionally). Zero cost, zero alloc — the ring-pop shape.
    /// -> `UnpinPipe`/`UnpinSendPipe`.
    ReadyWrapped,
    /// `async fn` with an explicit `#[proxima::piped(unpin, boxed)]`: boxed
    /// via `Box::pin`, which is `Unpin` for any `F` because `Box` is a
    /// fixed-address indirection, not self-referential. One heap allocation
    /// per call, paid only because the caller asked for it.
    /// -> `UnpinPipe`/`UnpinSendPipe`.
    BoxPinWrapped,
}

impl FutureShape {
    /// The trait-tier this shape lands on ignores its own distinction
    /// between `ReadyWrapped`/`BoxPinWrapped` — both are the `Unpin` tier,
    /// just spelled differently.
    fn climbs_to_unpin(&self) -> bool {
        !matches!(self, FutureShape::Passthrough)
    }
}

/// How the impl-block form's relocated `call` body reaches the tier's
/// required future shape. Distinct from [`FutureShape`]: the free-fn path's
/// `ReadyWrapped` wraps a bare `Result` in `core::future::ready`, but an
/// impl-block `call` never returns a bare `Result` from its sync form — it
/// already declares `-> impl Future<..> + Unpin` itself (see
/// `future_output_result_types`), so there is nothing to wrap there.
enum ImplShape {
    /// `async fn call(..) -> Result<Out, Err>`, no climb requested: the
    /// relocated body becomes `async move { #body }` — the minimal wrapper
    /// that turns the (unrewritten) block into the future the trait's `call`
    /// must return. -> `Pipe`/`SendPipe`.
    AsyncBlockWrapped,
    /// sync `fn call(..) -> impl Future<..> + Unpin`: the relocated body
    /// already IS the future-producing expression the trait needs, so it
    /// moves across unchanged. -> `UnpinPipe`/`UnpinSendPipe`.
    Direct,
    /// `async fn call(..)` with an explicit `#[proxima::piped(unpin, boxed)]`:
    /// same `Box::pin` climb `FutureShape::BoxPinWrapped` documents, applied
    /// to the `async move { #body }` wrapper instead of a preserved fn call.
    /// -> `UnpinPipe`/`UnpinSendPipe`.
    BoxPinWrapped,
}

impl ImplShape {
    /// Mirrors `FutureShape::climbs_to_unpin`: whether this shape lands on
    /// the `Unpin` tier rather than the tier `is_async` alone would imply.
    fn climbs_to_unpin(&self) -> bool {
        !matches!(self, ImplShape::AsyncBlockWrapped)
    }
}

/// Dispatch on item kind: a plain inherent `impl Foo { fn call(..) {..} }`
/// block (see `expand_impl_form`, the stateful form) is tried first — its
/// grammar (`impl ... { ... }`) never overlaps a fn's (`fn ... { ... }`), so
/// trying it first never mis-routes a free fn. Anything else falls through
/// to the original free-fn path unchanged.
pub fn expand(args: TokenStream, item: TokenStream) -> Result<TokenStream, Error> {
    match parse2::<ItemImpl>(item.clone()) {
        Ok(item_impl) => expand_impl_form(args, item_impl),
        Err(_) => expand_fn_form(args, item),
    }
}

fn expand_fn_form(args: TokenStream, item: TokenStream) -> Result<TokenStream, Error> {
    let mut func = parse2::<ItemFn>(item)?;
    let pipe_args = parse_args(args)?;

    if let Some(generic) = func.sig.generics.params.first() {
        return Err(Error::new_spanned(
            generic,
            "#[proxima::piped] does not support a generic fn",
        ));
    }

    let is_async = func.sig.asyncness.is_some();

    // point 4: refuse the impossible with a good error, rather than silently
    // picking a different tier than the one the caller asked for. six real
    // combinations of (is_async, unpin, boxed); three are refused, each with
    // the specific reason that combination is wrong, not one generic message.
    let shape = match (is_async, pipe_args.unpin, pipe_args.boxed) {
        (false, _, true) => {
            return Err(Error::new_spanned(
                &func.sig,
                "#[proxima::piped(boxed)] is redundant on a plain `fn`: `core::future::ready` \
                 is already `Unpin` and allocates nothing. Remove `boxed`.",
            ));
        }
        (false, _, false) => FutureShape::ReadyWrapped,
        (true, false, true) => {
            return Err(Error::new_spanned(
                &func.sig,
                "`boxed` only matters when climbing to the `Unpin` tier; add `unpin`: \
                 `#[proxima::piped(unpin, boxed)]`.",
            ));
        }
        (true, true, true) => FutureShape::BoxPinWrapped,
        (true, true, false) => {
            let async_token = func.sig.asyncness.as_ref().expect("checked is_async above");
            return Err(Error::new_spanned(
                async_token,
                "#[proxima::piped(unpin)] cannot be applied to an `async fn` as-is: its body \
                 compiles to a compiler-generated state machine, which is `!Unpin`, so it \
                 can't be polled in place. Three ways to get an `Unpin` pipe here, in order \
                 of cost: (1) use a plain `fn` instead — `#[proxima::piped]` wraps its return \
                 in `core::future::ready`, whose future IS `Unpin`, for free; (2) hand-write \
                 the `Future` as an `Unpin` poll struct that still returns `Poll::Pending` \
                 and registers a waker — `Unpin` constrains how the future is spelled, not \
                 whether it awaits; see `proxima_primitives::pipe::signal_source::SignalCall` \
                 for a worked example; or (3) keep this `async fn` and pay one allocation per \
                 call with `#[proxima::piped(unpin, boxed)]`.",
            ));
        }
        (true, false, false) => FutureShape::Passthrough,
    };

    let in_type = extract_in_type(&func.sig)?;
    let (out_type, err_type) = extract_result_types(&func.sig)?;

    // The pipe wears the fn's own name, so `mount("/", hello)` names the
    // function the reader actually wrote. A unit struct and a fn both land in
    // the value namespace, so the fn moves aside rather than collide (E0428);
    // the struct is the surface, and it is the only one the caller names.
    let wears_fn_name = pipe_args.name.is_none();
    let struct_name = match &pipe_args.name {
        Some(ident) => ident.clone(),
        None => func.sig.ident.clone(),
    };
    if wears_fn_name {
        let hidden = format!("__proxima_pipe_{}", func.sig.ident);
        func.sig.ident = Ident::new(&hidden, func.sig.ident.span());
    }

    let tiers = Tier::plan(shape.climbs_to_unpin(), pipe_args.send);

    let fn_ident = &func.sig.ident;
    let vis = &func.vis;
    let has_input = !func.sig.inputs.is_empty();

    let call_expr = if has_input {
        quote!(#fn_ident(__proxima_pipe_input))
    } else {
        quote!(#fn_ident())
    };
    let call_body = match shape {
        FutureShape::Passthrough => call_expr,
        FutureShape::ReadyWrapped => quote!(::core::future::ready(#call_expr)),
        // a local `extern crate alloc;` resolves `alloc::boxed::Box`
        // regardless of whether the invoking crate already names it at its
        // own crate root — the generated code carries no ambient assumption.
        FutureShape::BoxPinWrapped => quote! {{
            extern crate alloc;
            alloc::boxed::Box::pin(#call_expr)
        }},
    };

    // `boxed` is the only shape that allocates, so it is the only one gated
    // behind the invoking crate's `alloc` Cargo feature — without this, a
    // bare no_std build (no "alloc" feature enabled) never sees `Box::pin`
    // at all, matching how the rest of this tree gates alloc-only surface
    // (`proxima-primitives/src/pipe/mod.rs`'s `#[cfg(feature = "alloc")]`
    // module gates).
    let alloc_cfg = match shape {
        FutureShape::BoxPinWrapped => quote!(#[cfg(feature = "alloc")]),
        FutureShape::Passthrough | FutureShape::ReadyWrapped => quote!(),
    };

    // the struct is named for a snake_case fn on purpose; the lint is right in
    // general and wrong here.
    let case_allow = if wears_fn_name {
        quote!(#[allow(non_camel_case_types)])
    } else {
        quote!()
    };

    // one impl block per tier in the downward closure — same `call_body`,
    // same `alloc_cfg`, only the trait and its future bound change.
    let tier_impls: Vec<TokenStream> = tiers
        .iter()
        .map(|tier| {
            let trait_path = pipe_trait_path(&tier.trait_ident());
            let future_bound = tier.future_bound();
            quote! {
                #alloc_cfg
                impl #trait_path for #struct_name {
                    type In = #in_type;
                    type Out = #out_type;
                    type Err = #err_type;

                    fn call(
                        &self,
                        __proxima_pipe_input: #in_type,
                    ) -> impl ::core::future::Future<Output = ::core::result::Result<#out_type, #err_type>> #future_bound {
                        #call_body
                    }
                }
            }
        })
        .collect();

    Ok(quote! {
        #alloc_cfg
        #func

        #alloc_cfg
        #case_allow
        #[derive(::core::clone::Clone)]
        #vis struct #struct_name;

        #(#tier_impls)*
    })
}

/// `Self`'s plain type name (`impl Foo { .. }` -> `Foo`) — the impl-block
/// form never generates a struct, so this is the only place its name is
/// read, not minted.
fn struct_name_from_self_ty(self_ty: &Type) -> Result<Ident, Error> {
    let Type::Path(type_path) = self_ty else {
        return Err(Error::new_spanned(
            self_ty,
            "#[proxima::piped] on an impl block requires a plain type name for `Self` \
             (`impl Foo { .. }`)",
        ));
    };
    let Some(segment) = type_path.path.segments.last() else {
        return Err(Error::new_spanned(
            self_ty,
            "#[proxima::piped] on an impl block requires a plain type name for `Self`",
        ));
    };
    if !matches!(segment.arguments, PathArguments::None) {
        return Err(Error::new_spanned(
            self_ty,
            "#[proxima::piped] on an impl block does not support a generic `Self` type",
        ));
    }
    Ok(segment.ident.clone())
}

/// `call` must borrow `self`, never take it by value or mutably: a pipe is
/// called through a shared handle (`Arc`, a fan-out arm, ...), so `&mut self`
/// would never satisfy any real caller anyway — reject it here with the
/// specific reason, rather than let the trait impl fail to type-check.
fn validate_call_receiver(sig: &syn::Signature) -> Result<(), Error> {
    match sig.inputs.first() {
        Some(FnArg::Receiver(receiver))
            if receiver.reference.is_some() && receiver.mutability.is_none() =>
        {
            Ok(())
        }
        Some(FnArg::Receiver(receiver)) => Err(Error::new_spanned(
            receiver,
            "`call` must take `&self`, not `&mut self` or `self` by value",
        )),
        _ => Err(Error::new_spanned(
            &sig.ident,
            "`call` must take `&self` as its receiver",
        )),
    }
}

/// `call`'s single parameter after `&self` — the impl-block mirror of
/// `extract_in_type`, minus the zero-arg source case (a method always keeps
/// the `&self` slot, so there is no fn-with-no-parens shape to special-case;
/// a source pipe here spells its `In` out as `(): ()`, same as `BackendQueue`
/// does today).
fn extract_call_param(sig: &syn::Signature) -> Result<(Pat, Type), Error> {
    let mut remaining = sig.inputs.iter().skip(1);
    let Some(FnArg::Typed(pat_type)) = remaining.next() else {
        return Err(Error::new_spanned(
            &sig.inputs,
            "#[proxima::piped] `call` takes exactly one parameter after `&self` (In is the \
             single parameter); use a tuple or struct to carry more than one value",
        ));
    };
    if remaining.next().is_some() {
        return Err(Error::new_spanned(
            &sig.inputs,
            "#[proxima::piped] `call` takes exactly one parameter after `&self` (In is the \
             single parameter); use a tuple or struct to carry more than one value",
        ));
    }
    match pat_type.pat.as_ref() {
        Pat::Ident(_) | Pat::Wild(_) | Pat::Tuple(_) => {
            Ok(((*pat_type.pat).clone(), (*pat_type.ty).clone()))
        }
        other => Err(Error::new_spanned(
            other,
            "#[proxima::piped] `call`'s parameter must be a simple identifier, `_`, or `()`",
        )),
    }
}

/// The stateful counterpart to `expand_fn_form`: `#[proxima::piped(...)]` on a
/// plain inherent `impl Foo { fn call(..) { .. } }` block, for a pipe whose
/// struct already carries its own fields (a client, a pool, a counter)
/// instead of being the always-fieldless ZST the free-fn form generates. No
/// struct is generated here — `Foo` is relocated as-is into `impl #trait for
/// Foo`, plus a leftover `impl Foo { .. }` for any other item the block
/// carried alongside `call`.
fn expand_impl_form(args: TokenStream, item_impl: ItemImpl) -> Result<TokenStream, Error> {
    let pipe_args = parse_args(args)?;

    if let Some(name) = &pipe_args.name {
        return Err(Error::new_spanned(
            name,
            "#[proxima::piped(name = ..)] does not apply to an impl block; the trait wears the \
             impl's own type name",
        ));
    }

    if let Some(generic) = item_impl.generics.params.first() {
        return Err(Error::new_spanned(
            generic,
            "#[proxima::piped] does not support a generic impl",
        ));
    }

    if let Some((_, trait_path, _)) = &item_impl.trait_ {
        return Err(Error::new_spanned(
            trait_path,
            "#[proxima::piped] on an impl block only supports a bare inherent impl \
             (`impl Foo { .. }`), not `impl Trait for Foo`",
        ));
    }

    let struct_name = struct_name_from_self_ty(&item_impl.self_ty)?;

    let mut call_methods = Vec::new();
    let mut leftover = Vec::new();
    for item in &item_impl.items {
        match item {
            ImplItem::Fn(method) if method.sig.ident == "call" => call_methods.push(method),
            other => leftover.push(other),
        }
    }

    let [call_method] = call_methods.as_slice() else {
        return Err(Error::new_spanned(
            &item_impl,
            "#[proxima::piped] on an impl block requires exactly one method named `call`",
        ));
    };

    if !matches!(call_method.vis, Visibility::Inherited) {
        return Err(Error::new_spanned(
            &call_method.vis,
            "`call` must not carry an explicit visibility qualifier; the generated trait impl \
             supplies its own (rustc rejects visibility here as E0449, \"visibility qualifiers \
             are not permitted here\") — remove `pub`",
        ));
    }

    validate_call_receiver(&call_method.sig)?;
    let (in_pat, in_type) = extract_call_param(&call_method.sig)?;

    let is_async = call_method.sig.asyncness.is_some();

    // the same six-combination matrix `expand_fn_form` refuses from, but
    // landing on `ImplShape` instead of `FutureShape`: an impl-block `call`
    // never returns a bare `Result` (see `ImplShape`'s docs), so the
    // no-climb sync arm is `Direct`, not `ReadyWrapped`.
    let shape = match (is_async, pipe_args.unpin, pipe_args.boxed) {
        (false, _, true) => {
            return Err(Error::new_spanned(
                &call_method.sig,
                "#[proxima::piped(boxed)] is redundant on a sync `call` that already returns \
                 `impl Future<..> + Unpin`. Remove `boxed`.",
            ));
        }
        (false, _, false) => ImplShape::Direct,
        (true, false, true) => {
            return Err(Error::new_spanned(
                &call_method.sig,
                "`boxed` only matters when climbing to the `Unpin` tier; add `unpin`: \
                 `#[proxima::piped(unpin, boxed)]`.",
            ));
        }
        (true, true, true) => ImplShape::BoxPinWrapped,
        (true, true, false) => {
            let async_token = call_method
                .sig
                .asyncness
                .as_ref()
                .expect("checked is_async above");
            return Err(Error::new_spanned(
                async_token,
                "#[proxima::piped(unpin)] cannot be applied to an `async fn call` as-is: its \
                 body compiles to a compiler-generated state machine, which is `!Unpin`, so it \
                 can't be polled in place. Two ways to get an `Unpin` pipe here: (1) write \
                 `call` as a sync `fn` returning a hand-written `impl Future<..> + Unpin` (a \
                 poll struct that still returns `Poll::Pending` and registers a waker — see \
                 `proxima_primitives::pipe::signal_source::SignalCall` for a worked example); \
                 or (2) keep this `async fn` and pay one allocation per call with \
                 `#[proxima::piped(unpin, boxed)]`.",
            ));
        }
        (true, false, false) => ImplShape::AsyncBlockWrapped,
    };

    let (out_type, err_type) = if is_async {
        let return_type = match &call_method.sig.output {
            ReturnType::Type(_, ty) => ty.as_ref(),
            ReturnType::Default => {
                return Err(Error::new_spanned(
                    &call_method.sig,
                    "an `async fn call` must return Result<Out, Err>; this method has no \
                     return type",
                ));
            }
        };
        result_types_from_type(return_type)?
    } else {
        future_output_result_types(&call_method.sig)?
    };

    let tiers = Tier::plan(shape.climbs_to_unpin(), pipe_args.send);

    // the block is relocated exactly as written — none of these arms parse
    // or rewrite a single statement inside it, they only choose how it gets
    // wrapped to match the tier's required future shape.
    let body = &call_method.block;
    let call_body = match shape {
        ImplShape::AsyncBlockWrapped => quote!(async move #body),
        ImplShape::Direct => quote!(#body),
        // a local `extern crate alloc;` resolves `alloc::boxed::Box`
        // regardless of whether the invoking crate already names it at its
        // own crate root — mirrors `expand_fn_form`'s identical gate.
        ImplShape::BoxPinWrapped => quote! {{
            extern crate alloc;
            alloc::boxed::Box::pin(async move #body)
        }},
    };

    let alloc_cfg = match shape {
        ImplShape::BoxPinWrapped => quote!(#[cfg(feature = "alloc")]),
        ImplShape::AsyncBlockWrapped | ImplShape::Direct => quote!(),
    };

    let call_attrs = &call_method.attrs;
    let leftover_impl = if leftover.is_empty() {
        quote!()
    } else {
        quote! {
            impl #struct_name {
                #(#leftover)*
            }
        }
    };

    // one impl block per tier in the downward closure — see
    // `expand_fn_form`'s identical construction.
    let tier_impls: Vec<TokenStream> = tiers
        .iter()
        .map(|tier| {
            let trait_path = pipe_trait_path(&tier.trait_ident());
            let future_bound = tier.future_bound();
            quote! {
                #alloc_cfg
                impl #trait_path for #struct_name {
                    type In = #in_type;
                    type Out = #out_type;
                    type Err = #err_type;

                    #(#call_attrs)*
                    fn call(
                        &self,
                        #in_pat: #in_type,
                    ) -> impl ::core::future::Future<Output = ::core::result::Result<#out_type, #err_type>> #future_bound {
                        #call_body
                    }
                }
            }
        })
        .collect();

    Ok(quote! {
        #(#tier_impls)*

        #leftover_impl
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
    fn async_fn_emits_pipe_tier() {
        let expanded = expand_ok(
            "",
            "async fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }",
        );
        assert!(expanded.contains("struct double"));
        assert!(expanded.contains(": pipe :: Pipe for double"));
        assert!(!expanded.contains(": pipe :: SendPipe"));
        assert!(!expanded.contains(": pipe :: UnpinPipe"));
        assert!(expanded.contains("__proxima_pipe_double (__proxima_pipe_input)"));
        assert!(!expanded.contains("core :: future :: ready"));
    }

    #[test]
    fn sync_fn_emits_unpin_pipe_tier_and_wraps_ready() {
        let expanded = expand_ok(
            "",
            "fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }",
        );
        assert!(expanded.contains(": pipe :: UnpinPipe for double"));
        assert!(
            expanded
                .contains("core :: future :: ready (__proxima_pipe_double (__proxima_pipe_input))")
        );
        assert!(expanded.contains("+ :: core :: marker :: Unpin"));
    }

    #[test]
    fn send_on_async_fn_emits_send_pipe_tier() {
        let expanded = expand_ok(
            "send",
            "async fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }",
        );
        assert!(expanded.contains(": pipe :: SendPipe for double"));
        assert!(expanded.contains("+ :: core :: marker :: Send"));
        assert!(!expanded.contains("Unpin"));
    }

    #[test]
    fn send_on_sync_fn_emits_unpin_send_pipe_tier() {
        let expanded = expand_ok(
            "send",
            "fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }",
        );
        assert!(expanded.contains(": pipe :: UnpinSendPipe for double"));
        assert!(expanded.contains("+ :: core :: marker :: Send + :: core :: marker :: Unpin"));
        assert!(expanded.contains("core :: future :: ready"));
    }

    #[test]
    fn zero_arg_fn_derives_unit_in_type() {
        let expanded = expand_ok("", "fn always() -> Result<u64, Infallible> { Ok(7) }");
        assert!(expanded.contains("type In = () ;"));
        assert!(expanded.contains("always ()"));
    }

    #[test]
    fn name_arg_overrides_struct_name() {
        let expanded = expand_ok(
            "name = RingSource",
            "fn ring(_: ()) -> Result<u8, Infallible> { Ok(1) }",
        );
        assert!(expanded.contains("struct RingSource"));
        assert!(expanded.contains("for RingSource"));
        // an explicit name cannot collide with the fn, so the fn keeps its own
        // name and stays directly callable.
        assert!(expanded.contains("fn ring (_ : ())"));
        assert!(!expanded.contains("__proxima_pipe_ring "));
    }

    #[test]
    fn struct_wears_the_fn_name_verbatim() {
        let expanded = expand_ok(
            "",
            "fn ring_source(_: ()) -> Result<u8, Infallible> { Ok(1) }",
        );
        assert!(expanded.contains("struct ring_source"));
        assert!(expanded.contains("for ring_source"));
    }

    #[test]
    fn rejects_unpin_on_async_fn_naming_all_three_exits() {
        let err = expand_err(
            "unpin",
            "async fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }",
        );
        assert!(err.contains("cannot be applied to an `async fn`"));
        assert!(err.contains("!Unpin"));
        assert!(err.contains("use a plain `fn`"));
        assert!(err.contains("SignalCall"));
        assert!(err.contains("#[proxima::piped(unpin, boxed)]"));
    }

    #[test]
    fn rejects_boxed_on_sync_fn() {
        let err = expand_err(
            "boxed",
            "fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }",
        );
        assert!(err.contains("redundant on a plain `fn`"));
    }

    #[test]
    fn rejects_boxed_without_unpin_on_async_fn() {
        let err = expand_err(
            "boxed",
            "async fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }",
        );
        assert!(err.contains("only matters when climbing to the `Unpin` tier"));
    }

    #[test]
    fn unpin_boxed_on_async_fn_emits_unpin_pipe_via_box_pin() {
        let expanded = expand_ok(
            "unpin, boxed",
            "async fn recv(input: u64) -> Result<u64, Infallible> { Ok(input) }",
        );
        assert!(expanded.contains(": pipe :: UnpinPipe for recv"));
        assert!(expanded.contains("extern crate alloc"));
        assert!(
            expanded.contains(
                "alloc :: boxed :: Box :: pin (__proxima_pipe_recv (__proxima_pipe_input))"
            )
        );
        assert!(expanded.contains("cfg (feature = \"alloc\")"));
        assert!(expanded.contains("+ :: core :: marker :: Unpin"));
    }

    #[test]
    fn send_unpin_boxed_on_async_fn_emits_unpin_send_pipe_via_box_pin() {
        let expanded = expand_ok(
            "send, unpin, boxed",
            "async fn recv(input: u64) -> Result<u64, Infallible> { Ok(input) }",
        );
        assert!(expanded.contains(": pipe :: UnpinSendPipe for recv"));
        assert!(
            expanded.contains(
                "alloc :: boxed :: Box :: pin (__proxima_pipe_recv (__proxima_pipe_input))"
            )
        );
        assert!(expanded.contains("+ :: core :: marker :: Send + :: core :: marker :: Unpin"));
        assert!(expanded.contains("cfg (feature = \"alloc\")"));
    }

    #[test]
    fn boxed_shape_gates_the_original_fn_too_not_only_the_impl() {
        // if only the impl were gated, a no-alloc build would still see the
        // now-uncalled `async fn recv`, which is a `dead_code` warning under
        // this workspace's `deny(warnings)` — every generated item must
        // carry the same cfg. `unpin, boxed` with no `send` climbs to two
        // tiers (base `Pipe` always, plus `UnpinPipe`), so it's fn + struct +
        // 2 impls = 4, not 3.
        let expanded = expand_ok(
            "unpin, boxed",
            "async fn recv(input: u64) -> Result<u64, Infallible> { Ok(input) }",
        );
        let alloc_cfg_count = expanded.matches("cfg (feature = \"alloc\")").count();
        assert_eq!(
            alloc_cfg_count, 4,
            "fn, struct, and every tier impl must carry the alloc cfg gate"
        );
    }

    #[test]
    fn allows_redundant_unpin_on_sync_fn() {
        let expanded = expand_ok(
            "unpin",
            "fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }",
        );
        assert!(expanded.contains(": pipe :: UnpinPipe for double"));
    }

    #[test]
    fn rejects_unknown_arg() {
        let err = expand_err(
            "bogus",
            "fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }",
        );
        assert!(err.contains("unknown #[proxima::piped] arg"));
    }

    #[test]
    fn rejects_non_result_return_type() {
        let err = expand_err("", "fn double(input: u64) -> u64 { input * 2 }");
        assert!(err.contains("must return Result<Out, Err>"));
    }

    #[test]
    fn rejects_missing_return_type() {
        let err = expand_err("", "fn sink(input: u64) {}");
        assert!(err.contains("must return Result<Out, Err>"));
    }

    #[test]
    fn rejects_more_than_one_argument() {
        let err = expand_err(
            "",
            "fn double(a: u64, b: u64) -> Result<u64, Infallible> { Ok(a + b) }",
        );
        assert!(err.contains("zero or one argument"));
    }

    #[test]
    fn rejects_self_receiver() {
        let err = expand_err(
            "",
            "fn double(&self, input: u64) -> Result<u64, Infallible> { Ok(input * 2) }",
        );
        assert!(err.contains("does not support a `self` parameter"));
    }

    #[test]
    fn rejects_generic_fn() {
        let err = expand_err(
            "",
            "fn double<T>(input: T) -> Result<T, Infallible> { Ok(input) }",
        );
        assert!(err.contains("does not support a generic fn"));
    }

    #[test]
    fn original_fn_moves_aside_so_the_struct_can_wear_its_name() {
        let expanded = expand_ok(
            "",
            "fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }",
        );
        // the body survives verbatim, under a name that cannot collide...
        assert!(
            expanded
                .contains("fn __proxima_pipe_double (input : u64) -> Result < u64 , Infallible >")
        );
        // ...and `double` is now the pipe, calling it.
        assert!(expanded.contains("struct double"));
        assert!(expanded.contains("__proxima_pipe_double (__proxima_pipe_input)"));
    }

    // ---- affordance A: auto-Clone on the generated struct ----

    #[test]
    fn generated_struct_derives_clone() {
        let expanded = expand_ok(
            "",
            "fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }",
        );
        assert!(expanded.contains("derive (:: core :: clone :: Clone)"));
        // no `derive(...)` macro arg exists to opt out — Clone is unconditional.
        assert!(!expanded.contains("derive (:: core :: clone :: Clone , ::"));
    }

    #[test]
    fn generated_struct_derives_clone_regardless_of_tier() {
        for (args, item) in [
            (
                "send",
                "async fn double(input: u64) -> Result<u64, Infallible> { Ok(input * 2) }",
            ),
            ("", "fn always() -> Result<u64, Infallible> { Ok(7) }"),
        ] {
            let expanded = expand_ok(args, item);
            assert!(
                expanded.contains("derive (:: core :: clone :: Clone)"),
                "expected a Clone derive in: {expanded}"
            );
        }
    }

    // ---- affordance B: stateful impl-block form ----

    #[test]
    fn impl_block_form_emits_send_pipe() {
        let expanded = expand_ok(
            "send",
            "impl Backend {
                async fn call(&self, request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
                    Ok(Response::ok(\"ok\"))
                }

                fn helper(&self) -> u64 {
                    7
                }
            }",
        );
        assert!(expanded.contains(": pipe :: SendPipe for Backend"));
        assert!(expanded.contains("+ :: core :: marker :: Send"));
        assert!(expanded.contains("async move"));
        assert!(expanded.contains("Ok (Response :: ok (\"ok\"))"));
        // the helper method survives, relocated into a leftover inherent impl.
        assert!(expanded.contains("impl Backend"));
        assert!(expanded.contains("fn helper (& self) -> u64"));
        // no struct is generated — the impl-block form never mints one.
        assert!(!expanded.contains("struct Backend"));
    }

    #[test]
    fn impl_block_form_rejects_pub_call() {
        let err = expand_err(
            "send",
            "impl Backend {
                pub async fn call(&self, request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
                    Ok(Response::ok(\"ok\"))
                }
            }",
        );
        assert!(err.contains("explicit visibility"));
        assert!(err.contains("E0449"));
    }

    #[test]
    fn impl_block_form_rejects_generic_impl() {
        let err = expand_err(
            "",
            "impl<T> Backend<T> {
                async fn call(&self, input: T) -> Result<T, Infallible> { Ok(input) }
            }",
        );
        assert!(err.contains("does not support a generic impl"));
    }

    #[test]
    fn impl_block_form_rejects_trait_impl() {
        let err = expand_err(
            "send",
            "impl SendPipe for Backend {
                async fn call(&self, request: Request<Bytes>) -> Result<Response<Bytes>, ProximaError> {
                    Ok(Response::ok(\"ok\"))
                }
            }",
        );
        assert!(err.contains("bare inherent impl"));
        assert!(err.contains("not `impl Trait for Foo`"));
    }

    #[test]
    fn impl_block_form_rejects_missing_call_method() {
        let err = expand_err(
            "",
            "impl Backend {
                fn helper(&self) -> u64 { 7 }
            }",
        );
        assert!(err.contains("exactly one method named `call`"));
    }

    #[test]
    fn impl_block_form_rejects_mut_self_receiver() {
        let err = expand_err(
            "",
            "impl Counter {
                async fn call(&mut self, input: u64) -> Result<u64, Infallible> { Ok(input) }
            }",
        );
        assert!(err.contains("not `&mut self` or `self` by value"));
    }

    #[test]
    fn impl_block_form_rejects_name_arg() {
        let err = expand_err(
            "name = Other",
            "impl Backend {
                async fn call(&self, input: u64) -> Result<u64, Infallible> { Ok(input) }
            }",
        );
        assert!(err.contains("does not apply to an impl block"));
    }

    #[test]
    fn impl_block_form_sync_call_emits_unpin_pipe_directly() {
        let expanded = expand_ok(
            "",
            "impl BackendQueue {
                fn call(&self, (): ()) -> impl Future<Output = Result<u32, Exhausted>> + Unpin {
                    core::future::ready(Ok(1))
                }
            }",
        );
        assert!(expanded.contains(": pipe :: UnpinPipe for BackendQueue"));
        assert!(expanded.contains("+ :: core :: marker :: Unpin"));
        // the sync body is relocated as-is — never wrapped in `core::future::ready`
        // a second time (it already produces the future).
        assert!(expanded.contains("core :: future :: ready (Ok (1))"));
        assert!(!expanded.contains("async move"));
    }

    #[test]
    fn impl_block_form_boxed_climbs_async_call_to_unpin() {
        let expanded = expand_ok(
            "unpin, boxed",
            "impl Recv {
                async fn call(&self, input: u64) -> Result<u64, Infallible> { Ok(input) }
            }",
        );
        assert!(expanded.contains(": pipe :: UnpinPipe for Recv"));
        assert!(expanded.contains("alloc :: boxed :: Box :: pin (async move"));
        assert!(expanded.contains("cfg (feature = \"alloc\")"));
    }

    #[test]
    fn impl_block_form_rejects_unpin_without_boxed_on_async_call() {
        let err = expand_err(
            "unpin",
            "impl Recv {
                async fn call(&self, input: u64) -> Result<u64, Infallible> { Ok(input) }
            }",
        );
        assert!(err.contains("cannot be applied to an `async fn call`"));
    }

    #[test]
    fn impl_block_form_rejects_boxed_on_sync_call() {
        let err = expand_err(
            "boxed",
            "impl BackendQueue {
                fn call(&self, (): ()) -> impl Future<Output = Result<u32, Exhausted>> + Unpin {
                    core::future::ready(Ok(1))
                }
            }",
        );
        assert!(err.contains("redundant on a sync `call`"));
    }
}
