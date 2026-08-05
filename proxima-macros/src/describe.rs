//! `#[derive(Schema)]` — generate a `proxima_config::schema::Schema` from a
//! struct so the typed shape is the single source of truth (no hand-authored
//! IR to drift).
//! supports `#[schema(rename = "...")]` and `#[schema(skip)]` per field; mirrors
//! the field-name semantics of serde's rename so contracts match the wire.
//!
//! The emitted code holds at the ALLOC tier, not std: `proxima-config` gates
//! this derive behind `schema-derive = ["schema", ..]` and `schema` in turn
//! behind `alloc`, so a consumer can be `#![no_std]`. Every alloc-bearing
//! path is therefore spelled through a local `extern crate alloc;` rather
//! than `::std::…` or an unqualified `.to_string()` that would need the
//! caller to have `ToString` already in scope.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Type, parse2};

pub fn expand(input: TokenStream) -> Result<TokenStream, syn::Error> {
    let ast: DeriveInput = parse2(input)?;
    let name = &ast.ident;
    let name_str = name.to_string();

    let Data::Struct(data) = &ast.data else {
        return Err(syn::Error::new_spanned(
            name,
            "#[derive(Schema)] only supports structs",
        ));
    };
    let Fields::Named(named) = &data.fields else {
        return Err(syn::Error::new_spanned(
            name,
            "#[derive(Schema)] only supports structs with named fields",
        ));
    };

    // a struct-level `#[serde(default)]` makes every field absent-tolerant, the
    // same way it does for serde — so the schema must mark them all optional.
    let struct_default = has_serde_default(&ast.attrs);

    let mut fields = Vec::new();
    for field in &named.named {
        let ident = field
            .ident
            .as_ref()
            .ok_or_else(|| syn::Error::new_spanned(field, "named field must have an ident"))?;
        let (rename, skip) = field_attrs(field)?;
        if skip {
            continue;
        }
        let field_name = rename.unwrap_or_else(|| ident.to_string());
        let ty = &field.ty;
        // optional iff serde would tolerate the key being absent: an `Option<T>`,
        // a field (or struct) carrying `#[serde(default)]`. keeps the contract's
        // required-set identical to what the wire actually deserializes.
        let optional = is_option(ty) || struct_default || has_serde_default(&field.attrs);
        fields.push(quote! {
            ::proxima_config::schema::field(#field_name, <#ty as ::proxima_config::schema::Describe>::schema(), #optional)
        });
    }

    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics ::proxima_config::schema::Describe for #name #ty_generics #where_clause {
            fn schema() -> ::proxima_config::schema::Schema {
                extern crate alloc;
                ::proxima_config::schema::Schema::Struct {
                    name: alloc::string::ToString::to_string(#name_str),
                    fields: alloc::vec![ #(#fields),* ],
                }
            }
        }
    })
}

/// read the wire name + skip flag off a field. precedence for the name:
/// `#[schema(rename)]` > `#[serde(rename)]` > the field ident — so the schema
/// tracks the actual wire name (serde's) without a second annotation, and
/// `#[schema(rename)]` is the override when the two must differ.
fn field_attrs(field: &syn::Field) -> Result<(Option<String>, bool), syn::Error> {
    let mut schema_rename = None;
    let mut serde_rename = None;
    let mut skip = false;
    for attr in &field.attrs {
        if attr.path().is_ident("schema") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("skip") {
                    skip = true;
                    Ok(())
                } else if meta.path.is_ident("rename") {
                    let lit: syn::LitStr = meta.value()?.parse()?;
                    schema_rename = Some(lit.value());
                    Ok(())
                } else {
                    Err(meta.error("unknown #[schema(...)] key (expected rename or skip)"))
                }
            })?;
        } else if attr.path().is_ident("serde") {
            // pull `rename = "..."` out of serde, ignoring its other keys.
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("rename")
                    && let Ok(value) = meta.value()
                    && let Ok(lit) = value.parse::<syn::LitStr>()
                {
                    serde_rename = Some(lit.value());
                } else if meta.input.peek(syn::Token![=]) {
                    let _: syn::Expr = meta.value()?.parse()?;
                }
                Ok(())
            });
        }
    }
    Ok((schema_rename.or(serde_rename), skip))
}

/// true if the attrs carry `#[serde(default)]` or `#[serde(default = "...")]` —
/// the marker serde uses for "tolerate this key being absent."
fn has_serde_default(attrs: &[syn::Attribute]) -> bool {
    let mut found = false;
    for attr in attrs {
        if !attr.path().is_ident("serde") {
            continue;
        }
        // ignore unrelated serde keys; only `default` flips absent-tolerance.
        let _ = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("default") {
                found = true;
            }
            // swallow values (e.g. default = "fn", rename = "x") without erroring.
            if meta.input.peek(syn::Token![=]) {
                let _: syn::Expr = meta.value()?.parse()?;
            }
            Ok(())
        });
    }
    found
}

fn is_option(ty: &Type) -> bool {
    let Type::Path(path) = ty else {
        return false;
    };
    path.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "Option")
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

    // the derive is gated behind `schema-derive -> schema -> alloc`, never
    // `schema-std`, so a `::std::…` path in the OUTPUT is a hard cliff for
    // every no_std consumer — it fails with E0433 at the derive site.
    #[test]
    fn emitted_code_names_alloc_never_std() {
        let expanded = expand_ok("struct Memory { id: String }");
        assert!(expanded.contains("extern crate alloc"));
        assert!(expanded.contains("alloc :: vec !"));
        assert!(!expanded.contains("std :: vec"));
    }

    // an unqualified `.to_string()` resolves only where the CALLER already has
    // `ToString` in scope; a no_std consumer generally does not.
    #[test]
    fn struct_name_is_owned_through_a_fully_qualified_call() {
        let expanded = expand_ok("struct Memory { id: String }");
        assert!(expanded.contains("alloc :: string :: ToString :: to_string (\"Memory\")"));
    }

    #[test]
    fn option_fields_are_marked_optional() {
        let expanded = expand_ok("struct Memory { id: String, score: Option<f64> }");
        assert!(expanded.contains("field (\"id\" ,"));
        assert!(expanded.contains("field (\"score\" ,"));
        assert!(expanded.contains("false"));
        assert!(expanded.contains("true"));
    }

    #[test]
    fn schema_rename_overrides_the_field_ident() {
        let expanded = expand_ok("struct Memory { #[schema(rename = \"type\")] kind: String }");
        assert!(expanded.contains("field (\"type\" ,"));
        assert!(!expanded.contains("field (\"kind\" ,"));
    }

    #[test]
    fn schema_skip_omits_the_field() {
        let expanded = expand_ok("struct Memory { id: String, #[schema(skip)] cursor: u64 }");
        assert!(expanded.contains("field (\"id\" ,"));
        assert!(!expanded.contains("cursor"));
    }

    // the schema tracks the WIRE name, so serde's rename is honoured with no
    // second annotation — and `#[schema(rename)]` wins when the two disagree.
    #[test]
    fn serde_rename_is_honoured_and_schema_rename_wins() {
        let from_serde = expand_ok("struct Memory { #[serde(rename = \"ty\")] kind: String }");
        assert!(from_serde.contains("field (\"ty\" ,"));

        let both = expand_ok(
            "struct Memory { #[schema(rename = \"type\")] #[serde(rename = \"ty\")] kind: String }",
        );
        assert!(both.contains("field (\"type\" ,"));
        assert!(!both.contains("field (\"ty\" ,"));
    }

    #[test]
    fn serde_default_makes_a_field_absent_tolerant() {
        let per_field = expand_ok("struct Memory { #[serde(default)] retries: u32 }");
        assert!(per_field.contains("field (\"retries\" , < u32 as"));
        assert!(per_field.contains("true"));

        let struct_wide = expand_ok("#[serde(default)] struct Memory { retries: u32 }");
        assert!(struct_wide.contains("true"));
    }

    #[test]
    fn rejects_an_enum() {
        let err = expand_err("enum Memory { A, B }");
        assert!(err.contains("#[derive(Schema)] only supports structs"));
    }

    #[test]
    fn rejects_a_tuple_struct() {
        let err = expand_err("struct Memory(String);");
        assert!(err.contains("named fields"));
    }

    #[test]
    fn rejects_an_unknown_schema_key() {
        let err = expand_err("struct Memory { #[schema(bogus)] id: String }");
        assert!(err.contains("unknown #[schema(...)] key"));
    }
}
