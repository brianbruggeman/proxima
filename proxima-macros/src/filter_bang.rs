//! `filter!(<predicate closure>)` — lift a closure into the decision-pipe
//! shape `filter.rs`'s own module doc names as the whole point of that
//! file: "a decision IS a pipe, `In -> Result<In, Err>` (`Ok` = admit, the
//! item survives; `Err` = reject)". `Predicate`/`FilterConfig` are ONE
//! instantiation of that shape (config-driven, `Request<Bytes>`-specific);
//! `filter!` is the general one, for any payload — the same generalisation
//! the `filter_is_generic_over_a_non_http_payload` test in `filter.rs`
//! already proves by hand with a `Threshold` struct. No new combinator: a
//! `filter!`-lifted closure is the SAME leaf-lift bridge `pipe!` builds
//! ([`build_leaf`] in `pipe_bang.rs`), constrained to `Out == In` — reject
//! at macro-expansion time, not at the trait level, since the constraint is
//! about the closure's OWN shape, not a bound `PipeExt::filter` could
//! express (it only requires `Pred::Out == Self::In`, never `Pred::In ==
//! Pred::Out`, so a mis-shaped predicate would otherwise type-check as an
//! ordinary transform and silently not behave like a decision).

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{Error, Expr, Ident};

use crate::pipe_bang::{build_leaf, expand_expr, parse_bang_input};

pub fn expand(input: TokenStream) -> Result<TokenStream, Error> {
    let (expr, args) = parse_bang_input(input)?;

    let Expr::Closure(closure) = &expr else {
        // no closure to check the shape of — an already-built pipe value is
        // trusted the same way `PipeExt::filter` trusts any `Pred: Pipe<Out
        // = Self::In>` argument today.
        return expand_expr(expr, args, "filter!");
    };

    // build first (this also runs every other shape check — arity, async
    // rules, the explicit `-> Result<Out, Err>` requirement), THEN check
    // the one constraint specific to `filter!`: a decision's `Out` is its
    // own `In`, not a transform to some other type.
    let struct_ident = Ident::new("__ProximaFilterLeaf", Span::call_site());
    let plan = build_leaf(&struct_ident, closure, &args)?;

    let in_type = &plan.in_type;
    let out_type = &plan.out_type;
    let in_tokens = quote!(#in_type).to_string();
    let out_tokens = quote!(#out_type).to_string();
    if in_tokens != out_tokens {
        return Err(Error::new_spanned(
            closure,
            format!(
                "filter! requires a decision shape `In -> Result<In, Err>` (Ok admits, \
                 returning the SAME value) — this closure's input is `{in_tokens}` but its \
                 admit type is `{out_tokens}`. Use `pipe!` directly for a general transform, \
                 or change the closure to return `Ok(input)` unchanged on admit."
            ),
        ));
    }

    let definition = &plan.definition;
    Ok(quote! {
        {
            #definition
            #struct_ident(#closure)
        }
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
    fn decision_shaped_closure_lifts_into_a_filter_leaf() {
        let expanded = expand_ok(
            "|input: u64| -> Result<u64, &'static str> { if input < 10 { Ok(input) } else { Err(\"too big\") } }",
        );
        assert!(expanded.contains("Pipe for __ProximaFilterLeaf"));
        assert!(expanded.contains("UnpinPipe for __ProximaFilterLeaf"));
    }

    #[test]
    fn mismatched_out_type_is_refused() {
        let err = expand_err("|input: u64| -> Result<bool, &'static str> { Ok(input > 0) }");
        assert!(err.contains("requires a decision shape"));
        assert!(err.contains("u64"));
        assert!(err.contains("bool"));
    }

    #[test]
    fn a_non_closure_expression_passes_through_unchanged() {
        let expanded = expand_ok("some_existing_predicate");
        assert_eq!(expanded, "some_existing_predicate");
    }

    #[test]
    fn every_other_pipe_bang_shape_error_still_applies() {
        // filter! delegates to the same `build_leaf` pipe! uses, so its other
        // refusals (arity, missing return type, async+send, ...) apply
        // unchanged — proven here for one representative case.
        let err = expand_err("|a: u64, b: u64| -> Result<u64, Infallible> { Ok(a + b) }");
        assert!(err.contains("zero or one parameter"));
    }
}
