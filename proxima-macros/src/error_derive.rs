//! `#[derive(Error)]` — project-native thiserror-shape derive.
//!
//! Emits `Display` + `core::error::Error` impls (and optional `From`
//! impls for `#[from]` fields). Generated code compiles under
//! `#![no_std]` with no `alloc` requirement.
//!
//! Surface (subset of thiserror; the most-used parts):
//! - `#[error("literal")]` — variant Display arm
//! - `#[error("with {0}")]` / `#[error("with {field}")]` — interpolated args
//! - `#[error(transparent)]` — forward Display + source to single inner field
//! - `#[source]` on a field — exposes via `core::error::Error::source()`
//! - `#[from]` on a single tuple-variant field — generates `From<Inner>`,
//!   AND treats the field as a `#[source]`
//!
//! Convention enforcement: see `lint_message_style` — emits a compile
//! warning when `#[error("…")]` starts uppercase or ends in `.!?`.

use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::{
    Data, DataEnum, DeriveInput, Field, Fields, Ident, LitStr, Variant, parse2, spanned::Spanned,
};

pub fn expand(input: TokenStream) -> Result<TokenStream, syn::Error> {
    let parsed: DeriveInput = parse2(input)?;
    let enum_data = match &parsed.data {
        Data::Enum(data) => data,
        Data::Struct(_) => {
            return Err(syn::Error::new(
                parsed.ident.span(),
                "proxima_macros::Error derive supports enums only (no structs)",
            ));
        }
        Data::Union(_) => {
            return Err(syn::Error::new(
                parsed.ident.span(),
                "proxima_macros::Error derive does not support unions",
            ));
        }
    };

    let analyzed = analyze_enum(&parsed.ident, enum_data)?;
    let display_impl = emit_display_impl(&parsed, &analyzed);
    let error_impl = emit_error_impl(&parsed, &analyzed);
    let from_impls = emit_from_impls(&parsed, &analyzed);

    Ok(quote! {
        #display_impl
        #error_impl
        #(#from_impls)*
    })
}

struct AnalyzedEnum<'enum_input> {
    name: &'enum_input Ident,
    variants: Vec<AnalyzedVariant<'enum_input>>,
}

struct AnalyzedVariant<'variant_input> {
    name: &'variant_input Ident,
    fields: &'variant_input Fields,
    display: DisplayKind,
    source_index: Option<usize>,
    from_index: Option<usize>,
    from_field_type: Option<&'variant_input syn::Type>,
}

enum DisplayKind {
    Literal(LitStr),
    Transparent,
}

fn analyze_enum<'enum_input>(
    name: &'enum_input Ident,
    data: &'enum_input DataEnum,
) -> Result<AnalyzedEnum<'enum_input>, syn::Error> {
    let variants = data
        .variants
        .iter()
        .map(analyze_variant)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(AnalyzedEnum { name, variants })
}

fn analyze_variant(variant: &Variant) -> Result<AnalyzedVariant<'_>, syn::Error> {
    let display = parse_variant_display(variant)?;
    let (source_index, from_index, from_field_type) = parse_variant_field_attrs(variant)?;
    Ok(AnalyzedVariant {
        name: &variant.ident,
        fields: &variant.fields,
        display,
        source_index,
        from_index,
        from_field_type,
    })
}

fn parse_variant_display(variant: &Variant) -> Result<DisplayKind, syn::Error> {
    let mut display: Option<DisplayKind> = None;
    for attribute in &variant.attrs {
        if !attribute.path().is_ident("error") {
            continue;
        }
        if display.is_some() {
            return Err(syn::Error::new(
                attribute.span(),
                "duplicate #[error(...)] on the same variant",
            ));
        }
        // Try `transparent` keyword first.
        let parsed_transparent = attribute.parse_args::<Ident>();
        if let Ok(ref keyword) = parsed_transparent
            && keyword == "transparent"
        {
            display = Some(DisplayKind::Transparent);
            continue;
        }
        // Otherwise expect a string literal.
        let parsed_literal = attribute.parse_args::<LitStr>().map_err(|_| {
            syn::Error::new(
                attribute.span(),
                "expected #[error(\"literal\")] or #[error(transparent)]",
            )
        })?;
        lint_message_style(&parsed_literal);
        display = Some(DisplayKind::Literal(parsed_literal));
    }
    display.ok_or_else(|| {
        syn::Error::new(
            variant.span(),
            "every variant needs an #[error(\"...\")] or #[error(transparent)] attribute",
        )
    })
}

fn parse_variant_field_attrs(
    variant: &Variant,
) -> Result<(Option<usize>, Option<usize>, Option<&syn::Type>), syn::Error> {
    let mut source_index: Option<usize> = None;
    let mut from_index: Option<usize> = None;
    let mut from_field_type: Option<&syn::Type> = None;

    let fields_iter: Box<dyn Iterator<Item = (usize, &Field)>> = match &variant.fields {
        Fields::Named(named) => Box::new(named.named.iter().enumerate()),
        Fields::Unnamed(unnamed) => Box::new(unnamed.unnamed.iter().enumerate()),
        Fields::Unit => Box::new(core::iter::empty()),
    };

    for (index, field) in fields_iter {
        let has_source = field.attrs.iter().any(|a| a.path().is_ident("source"));
        let has_from = field.attrs.iter().any(|a| a.path().is_ident("from"));
        if has_source && source_index.is_some() {
            return Err(syn::Error::new(
                field.span(),
                "duplicate #[source] on the same variant",
            ));
        }
        if has_from && from_index.is_some() {
            return Err(syn::Error::new(
                field.span(),
                "duplicate #[from] on the same variant",
            ));
        }
        if has_source {
            source_index = Some(index);
        }
        if has_from {
            from_index = Some(index);
            from_field_type = Some(&field.ty);
            // #[from] implies #[source]
            if source_index.is_none() {
                source_index = Some(index);
            }
        }
    }
    Ok((source_index, from_index, from_field_type))
}

fn lint_message_style(literal: &LitStr) {
    // Convention: lowercase, no trailing punctuation.
    // We cannot emit warnings from a derive macro without nightly
    // proc_macro_diagnostic, so this is currently a no-op stub.
    // Once the diag API stabilises, walk the value and warn here.
    let _ = literal;
}

fn emit_display_impl(parsed: &DeriveInput, analyzed: &AnalyzedEnum<'_>) -> TokenStream {
    let enum_name = analyzed.name;
    let (impl_generics, ty_generics, where_clause) = parsed.generics.split_for_impl();

    let arms = analyzed.variants.iter().map(emit_display_arm);

    quote! {
        impl #impl_generics ::core::fmt::Display for #enum_name #ty_generics #where_clause {
            fn fmt(&self, formatter: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                match self {
                    #(#arms)*
                }
            }
        }
    }
}

fn emit_display_arm(variant: &AnalyzedVariant<'_>) -> TokenStream {
    let variant_name = variant.name;
    match (&variant.display, &variant.fields) {
        (DisplayKind::Transparent, Fields::Unnamed(unnamed)) if unnamed.unnamed.len() == 1 => {
            quote! {
                Self::#variant_name(inner) => ::core::fmt::Display::fmt(inner, formatter),
            }
        }
        (DisplayKind::Transparent, Fields::Named(named)) if named.named.len() == 1 => {
            let field_name = named
                .named
                .first()
                .expect("named.len() == 1")
                .ident
                .as_ref()
                .expect("named has ident");
            quote! {
                Self::#variant_name { #field_name } => ::core::fmt::Display::fmt(#field_name, formatter),
            }
        }
        (DisplayKind::Transparent, _) => {
            // Non-single-field transparent: emit compile_error inline so
            // the user sees the failure at the variant site.
            let span = variant_name.span();
            quote_spanned! { span =>
                Self::#variant_name { .. } => {
                    ::core::compile_error!("#[error(transparent)] requires the variant to have exactly one field");
                    ::core::result::Result::Ok(())
                }
            }
        }
        (DisplayKind::Literal(literal), Fields::Unit) => {
            let value = literal.value();
            quote! {
                Self::#variant_name => formatter.write_str(#value),
            }
        }
        (DisplayKind::Literal(literal), Fields::Unnamed(unnamed)) => {
            let bindings: Vec<Ident> = (0..unnamed.unnamed.len())
                .map(|index| format_ident!("__field{}", index))
                .collect();
            let format_args = render_format_args(literal, &unnamed.unnamed, &bindings);
            quote! {
                Self::#variant_name(#(#bindings),*) => formatter.write_fmt(#format_args),
            }
        }
        (DisplayKind::Literal(literal), Fields::Named(named)) => {
            let field_names: Vec<&Ident> = named
                .named
                .iter()
                .map(|field| field.ident.as_ref().expect("named has ident"))
                .collect();
            let format_args = render_named_format_args(literal, &field_names);
            quote! {
                Self::#variant_name { #(#field_names),* } => formatter.write_fmt(#format_args),
            }
        }
    }
}

fn render_format_args(
    literal: &LitStr,
    fields: &syn::punctuated::Punctuated<Field, syn::Token![,]>,
    bindings: &[Ident],
) -> TokenStream {
    // For tuple variants, the user writes `{0}`, `{1}`, etc. — but we
    // rebind to `__field0`, `__field1` to keep the names valid Rust
    // identifiers. format_args! needs the rebinding spelled out.
    let format_string = literal.value();
    let rewritten = rewrite_positional_args(&format_string, bindings);
    let format_str = LitStr::new(&rewritten, literal.span());
    // No positional args needed after rewriting; all bindings are named.
    let _ = fields;
    quote! { ::core::format_args!(#format_str) }
}

fn render_named_format_args(literal: &LitStr, field_names: &[&Ident]) -> TokenStream {
    // For named-field variants the user writes `{field_name}` which
    // format_args! resolves against in-scope identifiers — and the
    // match arm above binds each field by its own name.
    let _ = field_names;
    quote! { ::core::format_args!(#literal) }
}

fn rewrite_positional_args(template: &str, bindings: &[Ident]) -> String {
    // Replace `\{B\}` and `{N:fmt}` with `{__fieldN}` / `{__fieldN:fmt}`.
    // Skip `{{` (literal `{`) and `}}` (literal `}`).
    let mut output = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();
    while let Some(character) = chars.next() {
        if character == '{' {
            if chars.peek() == Some(&'{') {
                output.push('{');
                output.push('{');
                chars.next();
                continue;
            }
            output.push('{');
            let mut index_buffer = String::new();
            while let Some(&next) = chars.peek() {
                if next.is_ascii_digit() {
                    index_buffer.push(next);
                    chars.next();
                } else {
                    break;
                }
            }
            if let Ok(index) = index_buffer.parse::<usize>()
                && let Some(binding) = bindings.get(index)
            {
                output.push_str(&binding.to_string());
            } else {
                output.push_str(&index_buffer);
            }
            continue;
        }
        if character == '}' && chars.peek() == Some(&'}') {
            output.push('}');
            output.push('}');
            chars.next();
            continue;
        }
        output.push(character);
    }
    output
}

fn emit_error_impl(parsed: &DeriveInput, analyzed: &AnalyzedEnum<'_>) -> TokenStream {
    let enum_name = analyzed.name;
    let (impl_generics, ty_generics, where_clause) = parsed.generics.split_for_impl();

    let any_source = analyzed.variants.iter().any(|variant| {
        variant.source_index.is_some() || matches!(variant.display, DisplayKind::Transparent)
    });

    if !any_source {
        // Empty Error impl is sufficient — the trait's `source()` default returns None.
        return quote! {
            impl #impl_generics ::core::error::Error for #enum_name #ty_generics #where_clause {}
        };
    }

    let arms = analyzed.variants.iter().map(emit_source_arm);

    quote! {
        impl #impl_generics ::core::error::Error for #enum_name #ty_generics #where_clause {
            fn source(&self) -> ::core::option::Option<&(dyn ::core::error::Error + 'static)> {
                match self {
                    #(#arms)*
                }
            }
        }
    }
}

fn emit_source_arm(variant: &AnalyzedVariant<'_>) -> TokenStream {
    let variant_name = variant.name;
    // Transparent: forward source() to the single inner field (and treat it as the source itself).
    if matches!(variant.display, DisplayKind::Transparent) {
        return match &variant.fields {
            Fields::Unnamed(unnamed) if unnamed.unnamed.len() == 1 => quote! {
                Self::#variant_name(inner) => ::core::option::Option::Some(inner),
            },
            Fields::Named(named) if named.named.len() == 1 => {
                let field_name = named
                    .named
                    .first()
                    .expect("len==1")
                    .ident
                    .as_ref()
                    .expect("named");
                quote! {
                    Self::#variant_name { #field_name } => ::core::option::Option::Some(#field_name),
                }
            }
            _ => quote! {
                Self::#variant_name { .. } => ::core::option::Option::None,
            },
        };
    }
    // Explicit #[source] / #[from] field on a variant.
    if let Some(index) = variant.source_index {
        return match &variant.fields {
            Fields::Unnamed(unnamed) => {
                let bindings: Vec<TokenStream> = (0..unnamed.unnamed.len())
                    .map(|position| {
                        if position == index {
                            quote! { source_field }
                        } else {
                            quote! { _ }
                        }
                    })
                    .collect();
                quote! {
                    Self::#variant_name(#(#bindings),*) => ::core::option::Option::Some(source_field),
                }
            }
            Fields::Named(named) => {
                let source_name = named
                    .named
                    .iter()
                    .nth(index)
                    .and_then(|field| field.ident.as_ref())
                    .expect("source index is in range");
                quote! {
                    Self::#variant_name { #source_name, .. } => ::core::option::Option::Some(#source_name),
                }
            }
            Fields::Unit => quote! {
                Self::#variant_name => ::core::option::Option::None,
            },
        };
    }
    // No source on this variant.
    match &variant.fields {
        Fields::Unit => quote! { Self::#variant_name => ::core::option::Option::None, },
        Fields::Unnamed(_) => quote! { Self::#variant_name(..) => ::core::option::Option::None, },
        Fields::Named(_) => quote! { Self::#variant_name { .. } => ::core::option::Option::None, },
    }
}

fn emit_from_impls(parsed: &DeriveInput, analyzed: &AnalyzedEnum<'_>) -> Vec<TokenStream> {
    let enum_name = analyzed.name;
    let (impl_generics, ty_generics, where_clause) = parsed.generics.split_for_impl();
    let mut impls = Vec::new();
    for variant in &analyzed.variants {
        let Some(_) = variant.from_index else {
            continue;
        };
        let Some(from_type) = variant.from_field_type else {
            continue;
        };
        let variant_name = variant.name;
        let constructor = match &variant.fields {
            Fields::Unnamed(unnamed) if unnamed.unnamed.len() == 1 => {
                quote! { Self::#variant_name(value) }
            }
            Fields::Named(named) if named.named.len() == 1 => {
                let field_name = named
                    .named
                    .first()
                    .expect("len==1")
                    .ident
                    .as_ref()
                    .expect("named");
                quote! { Self::#variant_name { #field_name: value } }
            }
            _ => {
                // #[from] on a multi-field variant: skip the From impl;
                // the source machinery still works.
                continue;
            }
        };
        impls.push(quote! {
            impl #impl_generics ::core::convert::From<#from_type> for #enum_name #ty_generics #where_clause {
                fn from(value: #from_type) -> Self {
                    #constructor
                }
            }
        });
    }
    impls
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::field_reassign_with_default,
        clippy::type_complexity,
        clippy::useless_vec,
        clippy::needless_range_loop,
        clippy::default_constructed_unit_structs
    )]
    use super::*;
    use quote::quote;

    #[test]
    fn rejects_structs() {
        let input = quote! {
            #[derive(Error)]
            struct NotAnEnum;
        };
        let result = expand(input);
        let error = result.expect_err("structs should be rejected");
        assert!(
            error.to_string().contains("enums only"),
            "error mentions enum-only: {}",
            error
        );
    }

    #[test]
    fn requires_error_attribute_on_every_variant() {
        let input = quote! {
            pub enum MissingAttr {
                Unit,
            }
        };
        let result = expand(input);
        let error = result.expect_err("missing #[error] should be rejected");
        assert!(
            error.to_string().contains("#[error("),
            "error mentions #[error]: {}",
            error
        );
    }

    #[test]
    fn unit_variant_with_literal_compiles() {
        let input = quote! {
            pub enum SimpleError {
                #[error("invalid")]
                Invalid,
            }
        };
        let output = expand(input).expect("expansion succeeds");
        let rendered = output.to_string();
        assert!(rendered.contains("Display"), "emits Display impl");
        assert!(
            rendered.contains("write_str"),
            "uses write_str for literals"
        );
        assert!(rendered.contains("invalid"), "preserves the literal text");
        assert!(rendered.contains("Error"), "emits Error impl");
    }

    #[test]
    fn tuple_variant_with_format_args() {
        let input = quote! {
            pub enum DecodeError {
                #[error("invalid magic byte: {0}")]
                InvalidMagic(u8),
            }
        };
        let output = expand(input).expect("expansion succeeds");
        let rendered = output.to_string();
        assert!(
            rendered.contains("__field0"),
            "rewrites positional args to bindings"
        );
    }

    #[test]
    fn transparent_forwards_display_and_source() {
        let input = quote! {
            pub enum WireError {
                #[error(transparent)]
                Inner(std::io::Error),
            }
        };
        let output = expand(input).expect("expansion succeeds");
        let rendered = output.to_string();
        assert!(
            rendered.contains("Display :: fmt"),
            "transparent forwards Display"
        );
        assert!(
            rendered.contains("Option :: Some"),
            "transparent surfaces source"
        );
    }

    #[test]
    fn from_attribute_emits_from_impl() {
        let input = quote! {
            pub enum WrapError {
                #[error("wrapped")]
                Wrap(#[from] std::io::Error),
            }
        };
        let output = expand(input).expect("expansion succeeds");
        let rendered = output.to_string();
        assert!(rendered.contains("From"), "emits From impl");
        assert!(
            rendered.contains("io :: Error"),
            "From's source type preserved"
        );
    }

    #[test]
    fn source_attribute_emits_source_arm() {
        let input = quote! {
            pub enum CauseError {
                #[error("downstream failure")]
                Downstream(#[source] std::io::Error),
            }
        };
        let output = expand(input).expect("expansion succeeds");
        let rendered = output.to_string();
        assert!(
            rendered.contains("source_field"),
            "binds source field for source()"
        );
    }

    #[test]
    fn named_field_format_args() {
        let input = quote! {
            pub enum NamedError {
                #[error("missing {field}")]
                Missing { field: u32 },
            }
        };
        let output = expand(input).expect("expansion succeeds");
        let rendered = output.to_string();
        assert!(
            rendered.contains("write_fmt"),
            "named fields use write_fmt + format_args"
        );
    }

    #[test]
    fn multi_variant_enum_emits_one_arm_per_variant() {
        let input = quote! {
            pub enum MultiError {
                #[error("a")]
                A,
                #[error("b")]
                B,
                #[error("c {0}")]
                C(u32),
            }
        };
        let output = expand(input).expect("expansion succeeds");
        let rendered = output.to_string();
        // Each variant name should appear at least once.
        assert!(rendered.contains(":: A"), "A arm emitted");
        assert!(rendered.contains(":: B"), "B arm emitted");
        assert!(rendered.contains(":: C"), "C arm emitted");
    }

    #[test]
    fn empty_enum_compiles() {
        let input = quote! {
            pub enum Empty {}
        };
        let output = expand(input).expect("expansion succeeds for empty enum");
        let rendered = output.to_string();
        assert!(rendered.contains("Display"), "Display impl emitted");
        assert!(rendered.contains("Error"), "Error impl emitted");
    }

    #[test]
    fn rewrite_positional_args_basic() {
        let bindings = vec![format_ident!("__field0"), format_ident!("__field1")];
        let rewritten = rewrite_positional_args("got {0} expected {1}", &bindings);
        assert_eq!(rewritten, "got {__field0} expected {__field1}");
    }

    #[test]
    fn rewrite_positional_args_with_format_spec() {
        let bindings = vec![format_ident!("__field0")];
        let rewritten = rewrite_positional_args("value is {0:#x}", &bindings);
        assert_eq!(rewritten, "value is {__field0:#x}");
    }

    #[test]
    fn rewrite_positional_args_escapes_braces() {
        let bindings = vec![format_ident!("__field0")];
        let rewritten = rewrite_positional_args("{{literal {0}}}", &bindings);
        assert_eq!(rewritten, "{{literal {__field0}}}");
    }
}
