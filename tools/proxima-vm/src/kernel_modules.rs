//! Kernel modules read from a host `/usr/lib/modules/<release>` tree, ordered
//! so every module follows the modules it depends on, ready to pack into an
//! initramfs with [`crate::linux::initramfs_with_modules`].

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use proxima_core::ProximaError;

/// One module file as the guest will `finit_module` it, e.g. `virtio_net`
/// stored as `virtio_net.ko.zst`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelModule {
    pub name: String,
    pub file_name: String,
    pub bytes: Vec<u8>,
}

fn module_name(path: &str) -> &str {
    let file = path.rsplit('/').next().unwrap_or(path);
    file.split(".ko").next().unwrap_or(file)
}

fn parse_dependencies(modules_dep: &str) -> HashMap<&str, Vec<&str>> {
    modules_dep
        .lines()
        .filter_map(|line| line.split_once(':'))
        .map(|(path, dependencies)| (path.trim(), dependencies.split_whitespace().collect()))
        .collect()
}

fn visit<'dep>(
    path: &'dep str,
    dependencies: &HashMap<&'dep str, Vec<&'dep str>>,
    ordered: &mut Vec<&'dep str>,
) -> Result<(), ProximaError> {
    if ordered.contains(&path) {
        return Ok(());
    }
    let children = dependencies
        .get(path)
        .ok_or_else(|| ProximaError::Config(format!("modules.dep has no entry for `{path}`")))?;
    for child in children.iter().rev() {
        visit(child, dependencies, ordered)?;
    }
    ordered.push(path);
    Ok(())
}

fn path_of<'dep>(name: &str, dependencies: &HashMap<&'dep str, Vec<&'dep str>>) -> Result<&'dep str, ProximaError> {
    dependencies
        .keys()
        .copied()
        .find(|path| module_name(path) == name)
        .ok_or_else(|| ProximaError::Config(format!("modules.dep lists no module named `{name}`")))
}

/// Reads `wanted` and everything they depend on from `release_dir` (for
/// example `/usr/lib/modules/7.2.11-arch1-1`), dependencies first.
pub fn load_modules(release_dir: &Path, wanted: &[&str]) -> Result<Vec<KernelModule>, ProximaError> {
    let dep_path = release_dir.join("modules.dep");
    let modules_dep = fs::read_to_string(&dep_path)
        .map_err(|error| ProximaError::Config(format!("cannot read {}: {error}", dep_path.display())))?;
    let dependencies = parse_dependencies(&modules_dep);
    let mut ordered = Vec::new();
    for name in wanted {
        visit(path_of(name, &dependencies)?, &dependencies, &mut ordered)?;
    }
    ordered.into_iter().map(|path| read_module(release_dir, path)).collect()
}

fn read_module(release_dir: &Path, path: &str) -> Result<KernelModule, ProximaError> {
    let full = release_dir.join(path);
    let bytes = fs::read(&full).map_err(|error| ProximaError::Config(format!("cannot read module {}: {error}", full.display())))?;
    let file_name = path.rsplit('/').next().unwrap_or(path).to_owned();
    Ok(KernelModule { name: module_name(path).to_owned(), file_name, bytes })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    const DEP: &str = "kernel/drivers/virtio/virtio_mmio.ko.zst:\n\
        kernel/drivers/net/virtio_net.ko.zst: kernel/drivers/net/net_failover.ko.zst kernel/net/core/failover.ko.zst\n\
        kernel/drivers/net/net_failover.ko.zst: kernel/net/core/failover.ko.zst\n\
        kernel/net/core/failover.ko.zst:\n";

    #[test]
    fn dependencies_precede_dependents() {
        let dependencies = parse_dependencies(DEP);
        let mut ordered = Vec::new();

        visit(path_of("virtio_net", &dependencies).unwrap(), &dependencies, &mut ordered).unwrap();

        let names: Vec<&str> = ordered.iter().map(|path| module_name(path)).collect();
        assert_eq!(names, ["failover", "net_failover", "virtio_net"]);
    }

    #[test]
    fn an_unknown_module_is_a_named_error() {
        let dependencies = parse_dependencies(DEP);

        let error = path_of("nope", &dependencies).expect_err("unknown module");

        assert!(error.to_string().contains("nope"), "{error}");
    }
}
