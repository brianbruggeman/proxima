use proc_macro2::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Error, Fields, parse2};

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

    Ok(quote! {
        impl #impl_generics ::proxima::telemetry::trace::SpanCarrier
            for #struct_name #type_generics
            #where_clause
        {
            fn span_id(&self) -> ::core::option::Option<::proxima::telemetry::id::SpanId> {
                self.#carrier_field
            }

            fn set_span_id(
                &mut self,
                id: ::core::option::Option<::proxima::telemetry::id::SpanId>,
            ) {
                self.#carrier_field = id;
            }
        }
    })
}

fn find_carrier_field(
    fields: &syn::punctuated::Punctuated<syn::Field, syn::Token![,]>,
) -> Result<proc_macro2::Ident, Error> {
    let mut attr_marked: Option<proc_macro2::Ident> = None;
    let mut default_named: Option<proc_macro2::Ident> = None;

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
        proc_macro2::Span::call_site(),
        "#[derive(SpanCarrier)] requires a field named `span_id: Option<SpanId>` \
         or a field annotated with `#[span_id]`",
    ))
}
