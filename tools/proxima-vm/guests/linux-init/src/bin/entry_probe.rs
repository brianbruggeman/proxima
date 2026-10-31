fn main() {
    let arguments: Vec<String> = std::env::args().collect();
    let directory = std::env::current_dir().map(|path| path.display().to_string()).unwrap_or_default();
    let probe = std::env::var("OCI_PROBE").unwrap_or_default();
    let variables = std::env::vars().count();
    println!("entry-probe argv={arguments:?} cwd={directory} OCI_PROBE={probe} variables={variables}");
}
