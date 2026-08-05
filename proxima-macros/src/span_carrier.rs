//! `#[derive(SpanCarrier)]` — implement the trait for a struct that owns a
//! span slot, so the span rides with the payload as explicit data rather than
//! through an ambient current-span stack.

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::punctuated::Punctuated;
use syn::{Data, DeriveInput, Error, Field, Fields, Ident, Token, parse2};

use crate::crate_path;

pub fn expand(item: TokenStream) -> Result<TokenStream, Error> {
    let input = parse2::<DeriveInput>(item)?;
    let struct_name = &input.ident;
    let (impl_generics, type_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(named) => &named.named,
            _ => {
                return Err(Error::new_spanned(
                    &input.ident,
                    "#[derive(SpanCarrier)] only supports structs with named fields",
                ));
            }
        },
        _ => {
            return Err(Error::new_spanned(
                &input.ident,
                "#[derive(SpanCarrier)] only supports structs",
            ));
        }
    };

    let carrier_field = find_carrier_field(fields)?;
    let carrier_trait = crate_path::resolve(
        "proxima-telemetry",
        &quote!(trace::SpanCarrier),
        &quote!(telemetry::trace::SpanCarrier),
    );
    let span_id_type = crate_path::resolve(
        "proxima-telemetry",
        &quote!(id::SpanId),
        &quote!(telemetry::id::SpanId),
    );

    Ok(quote! {
        impl #impl_generics #carrier_trait
            for #struct_name #type_generics
            #where_clause
        {
            fn span_id(&self) -> ::core::option::Option<#span_id_type> {
                self.#carrier_field
            }

            fn set_span_id(
                &mut self,
                id: ::core::option::Option<#span_id_type>,
            ) {
                self.#carrier_field = id;
            }
        }
    })
}

fn find_carrier_field(fields: &Punctuated<Field, Token![,]>) -> Result<Ident, Error> {
    let mut attr_marked: Option<Ident> = None;
    let mut default_named: Option<Ident> = None;

    for field in fields {
        let Some(field_name) = field.ident.as_ref() else {
            continue;
        };

        let has_attr = field
            .attrs
            .iter()
            .any(|attr| attr.path().is_ident("span_id"));

        if has_attr {
            if attr_marked.is_some() {
                return Err(Error::new_spanned(
                    field_name,
                    "#[derive(SpanCarrier)] found multiple #[span_id] attributes; expected exactly one",
                ));
            }
            attr_marked = Some(field_name.clone());
        }

        if field_name == "span_id" {
            default_named = Some(field_name.clone());
        }
    }

    if let Some(marked) = attr_marked {
        return Ok(marked);
    }

    if let Some(named) = default_named {
        return Ok(named);
    }

    Err(Error::new(
        Span::call_site(),
        "#[derive(SpanCarrier)] requires a field named `span_id: Option<SpanId>` \
         or a field annotated with `#[span_id]`",
    ))
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
    fn a_field_named_span_id_is_the_default_carrier() {
        let expanded = expand_ok("struct Envelope { span_id: Option<SpanId>, payload: Vec<u8> }");
        assert!(expanded.contains("SpanCarrier for Envelope"));
        assert!(expanded.contains("self . span_id"));
    }

    #[test]
    fn the_span_id_attribute_names_a_differently_named_carrier() {
        let expanded =
            expand_ok("struct Request { #[span_id] trace_slot: Option<SpanId>, body: Bytes }");
        assert!(expanded.contains("self . trace_slot"));
        assert!(!expanded.contains("self . body"));
    }

    // both emitted paths go through `crate_path::resolve`, so the derive is
    // reachable from `proxima-telemetry` itself and from a crate that depends
    // on it directly — not only through the `proxima` umbrella. Resolved here
    // to the umbrella because that is this crate's own dev-dependency; the
    // other two arms are pinned in `crate_path`'s own tests.
    #[test]
    fn emitted_paths_go_through_the_shared_resolver() {
        let expanded = expand_ok("struct Envelope { span_id: Option<SpanId> }");
        assert!(expanded.contains(":: proxima :: telemetry :: trace :: SpanCarrier"));
        assert!(expanded.contains(":: proxima :: telemetry :: id :: SpanId"));
    }

    #[test]
    fn generics_are_carried_onto_the_impl() {
        let expanded = expand_ok("struct Envelope<T> { span_id: Option<SpanId>, data: T }");
        assert!(expanded.contains("impl < T >"));
        assert!(expanded.contains("for Envelope < T >"));
    }

    #[test]
    fn rejects_a_struct_with_no_carrier_field() {
        let err = expand_err("struct Envelope { payload: Vec<u8> }");
        assert!(err.contains("requires a field named `span_id: Option<SpanId>`"));
    }

    #[test]
    fn rejects_two_span_id_attributes() {
        let err = expand_err(
            "struct Envelope { #[span_id] first: Option<SpanId>, #[span_id] second: Option<SpanId> }",
        );
        assert!(err.contains("expected exactly one"));
    }

    #[test]
    fn rejects_a_tuple_struct() {
        let err = expand_err("struct Envelope(Option<SpanId>);");
        assert!(err.contains("named fields"));
    }

    #[test]
    fn rejects_an_enum() {
        let err = expand_err("enum Envelope { A, B }");
        assert!(err.contains("#[derive(SpanCarrier)] only supports structs"));
    }
}
