//! One resolver for every path this crate names in its token OUTPUT.
//!
//! A proc-macro crate has no `$crate`, and it deliberately carries no Cargo
//! dependency on the crates it emits paths into — so every emitted path has
//! to be resolved against whatever the INVOKING crate actually depends on.
//! Three cases, always the same three: the invoking crate is the owning crate
//! itself (`crate::…`), it depends on the owning crate directly
//! (`::proxima_telemetry::…`), or it only has the `proxima` umbrella and
//! reaches the same item through a re-export at a different depth
//! (`::proxima::telemetry::…`).
//!
//! `direct_tail` and `umbrella_tail` differ exactly because the umbrella
//! re-exports at a different depth (`markers::DropSafe` vs
//! `error::markers::DropSafe`); where the re-export keeps the depth, pass the
//! same tail twice.

use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::Ident;

pub(crate) fn resolve(
    owning_crate: &str,
    direct_tail: &TokenStream,
    umbrella_tail: &TokenStream,
) -> TokenStream {
    if let Ok(found) = crate_name(owning_crate) {
        return match found {
            FoundCrate::Itself => quote!(crate::#direct_tail),
            FoundCrate::Name(name) => {
                let krate = Ident::new(&name, Span::call_site());
                quote!(::#krate::#direct_tail)
            }
        };
    }
    match crate_name("proxima") {
        Ok(FoundCrate::Itself) => quote!(crate::#umbrella_tail),
        Ok(FoundCrate::Name(name)) => {
            let krate = Ident::new(&name, Span::call_site());
            quote!(::#krate::#umbrella_tail)
        }
        // neither dep is declared: name the owning crate outright and let the
        // invoking crate's own missing-dependency error say so.
        Err(_) => {
            let krate = Ident::new(&owning_crate.replace('-', "_"), Span::call_site());
            quote!(::#krate::#direct_tail)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `crate_name` reads the CARGO_MANIFEST_DIR of whatever is compiling. Under
    // `cargo test -p proxima-macros` that is this crate: it declares no owning
    // crate (proc-macro crates stay dependency-free) but DOES dev-depend on the
    // `proxima` umbrella — so these exercise the umbrella arm, which is exactly
    // the arm a hardcoded `::proxima::…` path would silently agree with. What
    // they pin is that the umbrella TAIL is the one used there, at its own
    // re-export depth, rather than the owning crate's.
    #[test]
    fn a_missing_direct_dep_falls_through_to_the_umbrella_tail() {
        let resolved = resolve(
            "proxima-telemetry",
            &quote!(trace::SpanCarrier),
            &quote!(telemetry::trace::SpanCarrier),
        );
        assert_eq!(
            resolved.to_string(),
            ":: proxima :: telemetry :: trace :: SpanCarrier"
        );
    }

    // the two tails genuinely differ — `proxima` re-exports `proxima_core` as
    // `error`, one level deeper than the crate owns it.
    #[test]
    fn the_umbrella_tail_is_used_at_its_own_re_export_depth() {
        let resolved = resolve(
            "proxima-core",
            &quote!(markers::DropSafe),
            &quote!(error::markers::DropSafe),
        );
        assert_eq!(
            resolved.to_string(),
            ":: proxima :: error :: markers :: DropSafe"
        );
    }

    // where the umbrella re-exports at the same depth, both tails are the same
    // and the resolved path only swaps the crate name.
    #[test]
    fn an_identical_tail_resolves_to_the_umbrella_unchanged() {
        let tail = quote!(pipe::Pipe);
        let resolved = resolve("proxima-primitives", &tail, &tail);
        assert_eq!(resolved.to_string(), ":: proxima :: pipe :: Pipe");
    }
}
