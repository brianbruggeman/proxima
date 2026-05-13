use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{
    Attribute, Error, Expr, FnArg, Ident, ItemFn, Meta, Pat, Path, ReturnType, Token, Type, parse2,
};

/// A fixture dependency: a parameter of the fixture fn, itself resolved as a
/// fixture (by name), or from a `#[from(path)]`, or a literal `#[default(expr)]`.
struct Dep {
    ident: Ident,
    ty: Type,
    default: Option<Expr>,
    from: Option<Path>,
}

fn attr_name(attr: &Attribute) -> Option<String> {
    attr.path()
        .segments
        .first()
        .map(|seg| seg.ident.to_string())
}

fn parse_dep(input: &FnArg) -> Result<Dep, Error> {
    let typed = match input {
        FnArg::Receiver(receiver) => {
            return Err(Error::new_spanned(
                receiver,
                "#[proxima::fixture] does not support a `self` parameter",
            ));
        }
        FnArg::Typed(typed) => typed,
    };
    let ident = match typed.pat.as_ref() {
        Pat::Ident(pat_ident) => pat_ident.ident.clone(),
        other => {
            return Err(Error::new_spanned(
                other,
                "#[proxima::fixture] parameters must be simple identifiers",
            ));
        }
    };
    let ty = typed.ty.as_ref().clone();
    let mut default = None;
    let mut from = None;
    for attr in &typed.attrs {
        match attr_name(attr).as_deref() {
            Some("default") => default = Some(attr.parse_args::<Expr>()?),
            Some("from") => from = Some(attr.parse_args::<Path>()?),
            _ => {}
        }
    }
    Ok(Dep {
        ident,
        ty,
        default,
        from,
    })
}

/// How each dependency is resolved inside `default()`/`partial_N()` for the deps
/// the caller did not supply: `#[default(expr)]` > `#[from(path)]::default()` >
/// `<ident>::default()` (resolve by parameter name, rstest semantics).
fn resolve_dep(dep: &Dep) -> TokenStream {
    let ident = &dep.ident;
    if let Some(expr) = &dep.default {
        quote!(let #ident = #expr;)
    } else if let Some(path) = &dep.from {
        quote!(let #ident = #path::default().await;)
    } else {
        quote!(let #ident = #ident::default().await;)
    }
}

fn parse_once(args: TokenStream) -> Result<bool, Error> {
    if args.is_empty() {
        return Ok(false);
    }
    let metas = Punctuated::<Meta, Token![,]>::parse_terminated.parse2(args)?;
    let mut once = false;
    for meta in metas {
        match meta.path().get_ident().map(ToString::to_string).as_deref() {
            Some("once") => once = true,
            other => {
                return Err(Error::new_spanned(
                    &meta,
                    format!(
                        "unknown #[proxima::fixture] arg `{}`; expected `once`",
                        other.unwrap_or("?")
                    ),
                ));
            }
        }
    }
    Ok(once)
}

pub fn expand(args: TokenStream, item: TokenStream) -> Result<TokenStream, Error> {
    let once = parse_once(args)?;
    let func = parse2::<ItemFn>(item)?;
    if let Some(generic) = func.sig.generics.params.first() {
        return Err(Error::new_spanned(
            generic,
            "#[proxima::fixture] does not support generic fixtures yet",
        ));
    }

    let vis = &func.vis;
    let name = &func.sig.ident;
    let output = &func.sig.output;
    let block = &func.block;
    let attrs = &func.attrs;

    let deps = func
        .sig
        .inputs
        .iter()
        .map(parse_dep)
        .collect::<Result<Vec<_>, _>>()?;

    // `#[proxima::fixture(once)]`: compute once per process, share as `&'static T`.
    // Deps are resolved inside the init future; the value is memoized via
    // `AsyncOnce` (awaited on the driving runtime, never blocks the prime worker).
    if once {
        let return_ty: Type = match output {
            ReturnType::Type(_, boxed) => (**boxed).clone(),
            ReturnType::Default => {
                return Err(Error::new_spanned(
                    &func.sig,
                    "#[proxima::fixture(once)] requires a return type",
                ));
            }
        };
        let resolves = deps.iter().map(resolve_dep);
        return Ok(quote! {
            #[allow(non_camel_case_types, clippy::empty_structs_with_brackets)]
            #vis struct #name {}

            #[allow(dead_code, clippy::unused_async)]
            impl #name {
                #(#attrs)*
                #vis async fn get() -> &'static #return_ty {
                    static CELL: ::proxima::test_support::AsyncOnce<#return_ty> =
                        ::proxima::test_support::AsyncOnce::new();
                    CELL.get_or_init(|| async {
                        #(#resolves)*
                        #block
                    })
                    .await
                }

                #vis async fn default() -> &'static #return_ty { Self::get().await }
            }
        });
    }

    // get(): the user's body verbatim, always as an `async fn` (sync bodies just
    // never suspend) so the call site can uniformly `.await` it regardless of
    // whether this fixture is sync or async — no cross-crate async detection.
    let get_params = deps.iter().map(|dep| {
        let ident = &dep.ident;
        let ty = &dep.ty;
        quote!(#ident: #ty)
    });
    let get = quote! {
        #(#attrs)*
        #vis async fn get(#(#get_params),*) #output #block
    };

    // default(): resolve every dep, then call get().
    let default_resolves = deps.iter().map(resolve_dep);
    let dep_idents: Vec<&Ident> = deps.iter().map(|dep| &dep.ident).collect();
    let default = quote! {
        #vis async fn default() #output {
            #(#default_resolves)*
            Self::get(#(#dep_idents),*).await
        }
    };

    // partial_N(first_n): the caller supplies the first N deps (via #[with(..)]);
    // the rest are resolved. partial_len == get with all deps explicit.
    let partials = (1..=deps.len()).map(|count| {
        let supplied = &deps[..count];
        let resolved = &deps[count..];
        let partial_name = Ident::new(&format!("partial_{count}"), Span::call_site());
        let params = supplied.iter().map(|dep| {
            let ident = &dep.ident;
            let ty = &dep.ty;
            quote!(#ident: #ty)
        });
        let resolves = resolved.iter().map(resolve_dep);
        let all_idents = deps.iter().map(|dep| &dep.ident);
        quote! {
            #vis async fn #partial_name(#(#params),*) #output {
                #(#resolves)*
                Self::get(#(#all_idents),*).await
            }
        }
    });

    Ok(quote! {
        #[allow(non_camel_case_types, clippy::empty_structs_with_brackets)]
        #vis struct #name {}

        #[allow(dead_code, clippy::unused_async)]
        impl #name {
            #get
            #default
            #(#partials)*
        }
    })
}
