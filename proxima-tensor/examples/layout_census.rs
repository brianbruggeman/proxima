use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use proc_macro2::{TokenStream, TokenTree};
use syn::parse::{ParseStream, Parser};
use syn::punctuated::Punctuated;
use syn::visit::{self, Visit};

const FLOAT_VARIANTS: [&str; 6] = [
    "BFloat16", "BFloat8", "Fp6E3M2", "Float16", "Float32", "Float64",
];
const CONTROL_SOURCE: &str =
    include_str!("../specs/float-layout-dtype/fixtures/census_control.rs.txt");
const MODULE_ROOT_STEMS: [&str; 3] = ["mod", "lib", "main"];

#[derive(Debug, thiserror::Error)]
enum CensusError {
    #[error("usage: layout_census --root <dir> --crates <a,b,..> | layout_census --control")]
    Usage,
    #[error("cannot read {path}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("cannot parse {path}")]
    Parse {
        path: PathBuf,
        #[source]
        source: syn::Error,
    },
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Sites {
    pattern: usize,
    expression: usize,
}

#[derive(Default)]
struct Classifier {
    sites: Sites,
    skipped_modules: Vec<String>,
    inside_dtype_impl: bool,
}

struct FileReport {
    path: PathBuf,
    sites: Sites,
    skipped_modules: Vec<String>,
    test_only: bool,
}

fn tokens_mention(tokens: TokenStream, word: &str) -> bool {
    tokens.into_iter().any(|tree| match tree {
        TokenTree::Ident(ident) => ident == word,
        TokenTree::Group(group) => tokens_mention(group.stream(), word),
        _ => false,
    })
}

fn is_test_gated(attributes: &[syn::Attribute]) -> bool {
    attributes.iter().any(|attribute| {
        let name = attribute
            .path()
            .segments
            .last()
            .map(|segment| segment.ident.to_string());
        match (name.as_deref(), &attribute.meta) {
            (Some("test"), _) => true,
            (Some("cfg"), syn::Meta::List(list)) => {
                tokens_mention(list.tokens.clone(), "test")
                    && !tokens_mention(list.tokens.clone(), "not")
            }
            _ => false,
        }
    })
}

fn names_dtype(self_type: &syn::Type) -> bool {
    matches!(self_type, syn::Type::Path(path)
        if path.path.segments.last().is_some_and(|segment| segment.ident == "DType"))
}

impl Classifier {
    fn names_float_variant(&self, path: &syn::Path) -> bool {
        let names: Vec<String> = path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect();
        match names.as_slice() {
            [.., owner, variant] => {
                FLOAT_VARIANTS.contains(&variant.as_str())
                    && (owner == "DType" || (owner == "Self" && self.inside_dtype_impl))
            }
            _ => false,
        }
    }

    fn scan_tokens(&mut self, tokens: TokenStream) {
        let trees: Vec<TokenTree> = tokens.into_iter().collect();
        for window in trees.windows(4) {
            if let [
                TokenTree::Ident(owner),
                TokenTree::Punct(_),
                TokenTree::Punct(_),
                TokenTree::Ident(variant),
            ] = window
                && owner == "DType"
                && FLOAT_VARIANTS.contains(&variant.to_string().as_str())
            {
                self.sites.expression += 1;
            }
        }
        for tree in trees {
            if let TokenTree::Group(group) = tree {
                self.scan_tokens(group.stream());
            }
        }
    }

    fn visit_matches_arguments(&mut self, tokens: &TokenStream) -> syn::Result<()> {
        let parse = |input: ParseStream| {
            let subject: syn::Expr = input.parse()?;
            input.parse::<syn::Token![,]>()?;
            let pattern = syn::Pat::parse_multi_with_leading_vert(input)?;
            let guard = if input.peek(syn::Token![if]) {
                input.parse::<syn::Token![if]>()?;
                Some(input.parse::<syn::Expr>()?)
            } else {
                None
            };
            input.parse::<Option<syn::Token![,]>>()?;
            Ok((subject, pattern, guard))
        };
        let (subject, pattern, guard) = parse.parse2(tokens.clone())?;
        self.visit_expr(&subject);
        self.visit_pat(&pattern);
        guard
            .iter()
            .for_each(|expression| self.visit_expr(expression));
        Ok(())
    }

    fn visit_call_arguments(&mut self, tokens: &TokenStream) {
        let parsed =
            Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated.parse2(tokens.clone());
        match parsed {
            Ok(arguments) => arguments
                .iter()
                .for_each(|expression| self.visit_expr(expression)),
            Err(_) => self.scan_tokens(tokens.clone()),
        }
    }
}

impl<'ast> Visit<'ast> for Classifier {
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if !is_test_gated(&node.attrs) {
            return visit::visit_item_mod(self, node);
        }
        if node.content.is_none() {
            self.skipped_modules.push(node.ident.to_string());
        }
    }

    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        if !is_test_gated(&node.attrs) {
            visit::visit_item_fn(self, node);
        }
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        if !is_test_gated(&node.attrs) {
            visit::visit_impl_item_fn(self, node);
        }
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        if is_test_gated(&node.attrs) {
            return;
        }
        let outer = self.inside_dtype_impl;
        self.inside_dtype_impl = names_dtype(&node.self_ty);
        visit::visit_item_impl(self, node);
        self.inside_dtype_impl = outer;
    }

    fn visit_pat(&mut self, node: &'ast syn::Pat) {
        match node {
            syn::Pat::Path(path) => {
                self.sites.pattern += usize::from(self.names_float_variant(&path.path));
            }
            syn::Pat::TupleStruct(pattern) => {
                self.sites.pattern += usize::from(self.names_float_variant(&pattern.path));
                visit::visit_pat(self, node);
            }
            syn::Pat::Struct(pattern) => {
                self.sites.pattern += usize::from(self.names_float_variant(&pattern.path));
                visit::visit_pat(self, node);
            }
            _ => visit::visit_pat(self, node),
        }
    }

    fn visit_expr_path(&mut self, node: &'ast syn::ExprPath) {
        self.sites.expression += usize::from(self.names_float_variant(&node.path));
    }

    fn visit_macro(&mut self, node: &'ast syn::Macro) {
        let is_matches = node
            .path
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "matches");
        if !is_matches {
            return self.visit_call_arguments(&node.tokens);
        }
        if let Err(error) = self.visit_matches_arguments(&node.tokens) {
            eprintln!("warn: matches! arguments unparsed, scanning tokens as expressions: {error}");
            self.scan_tokens(node.tokens.clone());
        }
    }
}

fn analyze_source(path: &Path, source: &str) -> Result<FileReport, CensusError> {
    let file = syn::parse_file(source).map_err(|error| CensusError::Parse {
        path: path.to_path_buf(),
        source: error,
    })?;
    let mut classifier = Classifier::default();
    classifier.visit_file(&file);
    Ok(FileReport {
        path: path.to_path_buf(),
        sites: classifier.sites,
        skipped_modules: classifier.skipped_modules,
        test_only: is_test_gated(&file.attrs),
    })
}

fn rust_files(directory: &Path) -> Result<Vec<PathBuf>, CensusError> {
    let read_error = |source| CensusError::Read {
        path: directory.to_path_buf(),
        source,
    };
    let mut entries = fs::read_dir(directory)
        .map_err(read_error)?
        .map(|entry| entry.map(|found| found.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(read_error)?;
    entries.sort();
    let mut files = Vec::new();
    for entry in entries {
        if entry.is_dir() {
            files.extend(rust_files(&entry)?);
        } else if entry.extension().is_some_and(|extension| extension == "rs") {
            files.push(entry);
        }
    }
    Ok(files)
}

fn child_directory(path: &Path) -> PathBuf {
    let parent = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    if MODULE_ROOT_STEMS.contains(&stem) {
        parent
    } else {
        parent.join(stem)
    }
}

fn excluded_paths(reports: &[FileReport]) -> Vec<PathBuf> {
    reports
        .iter()
        .flat_map(|report| {
            let children = child_directory(&report.path);
            report
                .skipped_modules
                .iter()
                .flat_map(move |name| [children.join(format!("{name}.rs")), children.join(name)])
        })
        .collect()
}

fn analyze_crate(root: &Path, name: &str) -> Result<(usize, Sites), CensusError> {
    let reports = rust_files(&root.join(name).join("src"))?
        .iter()
        .map(|path| {
            let source = fs::read_to_string(path).map_err(|source| CensusError::Read {
                path: path.clone(),
                source,
            })?;
            analyze_source(path, &source)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let excluded = excluded_paths(&reports);
    let counted: Vec<&FileReport> = reports
        .iter()
        .filter(|report| {
            !report.test_only && !excluded.iter().any(|path| report.path.starts_with(path))
        })
        .collect();
    let sites = counted
        .iter()
        .fold(Sites::default(), |total, report| Sites {
            pattern: total.pattern + report.sites.pattern,
            expression: total.expression + report.sites.expression,
        });
    Ok((counted.len(), sites))
}

fn flag_value(arguments: &[String], flag: &str) -> Option<String> {
    arguments
        .iter()
        .position(|argument| argument == flag)
        .and_then(|index| arguments.get(index + 1))
        .cloned()
}

fn run(arguments: &[String]) -> Result<(), CensusError> {
    if arguments.iter().any(|argument| argument == "--control") {
        let report = analyze_source(Path::new("census_control.rs.txt"), CONTROL_SOURCE)?;
        println!(
            "control census pattern_sites={} expression_sites={}",
            report.sites.pattern, report.sites.expression
        );
        return Ok(());
    }
    let root = flag_value(arguments, "--root").ok_or(CensusError::Usage)?;
    let crates = flag_value(arguments, "--crates").ok_or(CensusError::Usage)?;
    for name in crates.split(',') {
        let (files, sites) = analyze_crate(Path::new(&root), name)?;
        println!(
            "census crate={name} files={files} pattern_sites={} expression_sites={}",
            sites.pattern, sites.expression
        );
    }
    Ok(())
}

fn main() -> Result<(), CensusError> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    run(&arguments)
}
