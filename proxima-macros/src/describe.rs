//! `#[derive(Schema)]` — generate a `proxima_config::schema::Schema` from a
//! struct so the typed shape is the single source of truth (no hand-authored
//! IR to drift).
//! supports `#[schema(rename = "...")]` and `#[schema(skip)]` per field; mirrors
//! the field-name semantics of serde's rename so contracts match the wire.

use proc_macro2::TokenStream;
use quote::quote;
use syn::Data;
use syn::DeriveInput;
use syn::Fields;
use syn::Type;
use syn::parse2;

pub fn expand(input: TokenStream) -> Result<TokenStream, syn::Error> {
    let ast: DeriveInput = parse2(input)?;
    let name = &ast.ident;
    let name_str = name.to_string();

    let Data::Struct(data) = &ast.data else {
        return Err(syn::Error::new_spanned(
            name,
            "Describe derive supports structs only",
        ));
    };
    let Fields::Named(named) = &data.fields else {
        return Err(syn::Error::new_spanned(
            name,
            "Describe derive supports named-field structs only",
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
                ::proxima_config::schema::Schema::Struct {
                    name: #name_str.to_string(),
                    fields: ::std::vec![ #(#fields),* ],
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
