//! The Cargo half of the area detector: per-package transitive workspace-dep
//! closure, each member's directory, and which member owns a changed path.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

#[derive(serde::Deserialize)]
pub(crate) struct CargoMetadata {
    pub(crate) workspace_members: Vec<String>,
    pub(crate) packages: Vec<Package>,
    pub(crate) resolve: Option<Resolve>,
}

#[derive(serde::Deserialize)]
pub(crate) struct Package {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) manifest_path: String,
}

#[derive(serde::Deserialize)]
pub(crate) struct Resolve {
    pub(crate) nodes: Vec<Node>,
}

#[derive(serde::Deserialize)]
pub(crate) struct Node {
    pub(crate) id: String,
    #[serde(default)]
    pub(crate) deps: Vec<NodeDep>,
}

#[derive(serde::Deserialize)]
pub(crate) struct NodeDep {
    pub(crate) pkg: String,
    #[serde(default)]
    pub(crate) dep_kinds: Vec<DepKind>,
}

#[derive(serde::Deserialize)]
pub(crate) struct DepKind {
    #[serde(default)]
    pub(crate) kind: Option<String>,
}

/// Per-member transitive workspace-dependency closure and per-member
/// directory relative to `root`.
pub(crate) struct WorkspaceGraph {
    pub(crate) closure: HashMap<String, BTreeSet<String>>,
    pub(crate) dirs: BTreeMap<String, String>,
}

const ROOT_SRC_PREFIXES: [&str; 4] = ["src/", "benches/", "tests/", "examples/"];

impl WorkspaceGraph {
    pub(crate) fn from_metadata(metadata: &CargoMetadata, root: &Path) -> Self {
        let members: HashSet<&str> = metadata
            .workspace_members
            .iter()
            .map(String::as_str)
            .collect();
        let id_to_name: HashMap<&str, &str> = metadata
            .packages
            .iter()
            .map(|package| (package.id.as_str(), package.name.as_str()))
            .collect();

        let mut dirs = BTreeMap::new();
        for package in &metadata.packages {
            if !members.contains(package.id.as_str()) {
                continue;
            }
            let manifest = Path::new(&package.manifest_path);
            let Some(parent) = manifest.parent() else {
                continue;
            };
            let Ok(relative) = parent.strip_prefix(root) else {
                continue;
            };
            let rendered = relative.to_string_lossy().replace('\\', "/");
            dirs.insert(package.name.clone(), rendered);
        }

        let mut direct: HashMap<&str, BTreeSet<&str>> = HashMap::new();
        for node in metadata.resolve.iter().flat_map(|resolve| &resolve.nodes) {
            let Some(&name) = id_to_name.get(node.id.as_str()) else {
                continue;
            };
            if !members.contains(node.id.as_str()) {
                continue;
            }
            let mut deps = BTreeSet::new();
            for dep in &node.deps {
                if !members.contains(dep.pkg.as_str()) {
                    continue;
                }
                let dev_only = !dep.dep_kinds.is_empty()
                    && dep
                        .dep_kinds
                        .iter()
                        .all(|kind| kind.kind.as_deref() == Some("dev"));
                if dev_only {
                    continue;
                }
                if let Some(&dep_name) = id_to_name.get(dep.pkg.as_str()) {
                    deps.insert(dep_name);
                }
            }
            direct.insert(name, deps);
        }

        let mut closure = HashMap::new();
        for package in direct.keys() {
            resolve_closure(package, &direct, &mut closure, &mut HashSet::new());
        }

        let closure = closure
            .into_iter()
            .map(|(package, reachable)| {
                (
                    package.to_string(),
                    reachable.into_iter().map(str::to_string).collect(),
                )
            })
            .collect();
        Self { closure, dirs }
    }

    /// Map changed files to the workspace member whose directory is the
    /// longest matching path prefix. The root crate (directory `.`) owns only
    /// its own source trees, not repo infrastructure, which the area path
    /// globs and the global force-all set route instead.
    pub(crate) fn owning_packages(&self, files: &[String]) -> BTreeSet<String> {
        let mut ranked: Vec<(&String, &String)> = self.dirs.iter().collect();
        ranked.sort_by_key(|(_, directory)| std::cmp::Reverse(directory.len()));

        let mut owners = BTreeSet::new();
        for path in files {
            for (name, directory) in &ranked {
                if directory.is_empty() || directory.as_str() == "." {
                    if ROOT_SRC_PREFIXES
                        .iter()
                        .any(|prefix| path.starts_with(prefix))
                    {
                        owners.insert((*name).clone());
                        break;
                    }
                    continue;
                }
                if path.starts_with(directory.as_str())
                    && path.as_bytes().get(directory.len()) == Some(&b'/')
                {
                    owners.insert((*name).clone());
                    break;
                }
            }
        }
        owners
    }
}

fn resolve_closure<'a>(
    package: &'a str,
    direct: &HashMap<&'a str, BTreeSet<&'a str>>,
    closure: &mut HashMap<&'a str, BTreeSet<&'a str>>,
    visiting: &mut HashSet<&'a str>,
) -> BTreeSet<&'a str> {
    if let Some(done) = closure.get(package) {
        return done.clone();
    }
    if !visiting.insert(package) {
        return BTreeSet::new();
    }
    let mut reachable = BTreeSet::from([package]);
    for dep in direct.get(package).into_iter().flatten() {
        reachable.extend(resolve_closure(dep, direct, closure, visiting));
    }
    visiting.remove(package);
    closure.insert(package, reachable.clone());
    reachable
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{CargoMetadata, WorkspaceGraph};

    fn metadata() -> CargoMetadata {
        let json = r#"{
            "workspace_members": ["root 0.1.0 (path+file:///w)", "leaf 0.1.0 (path+file:///w/leaf)", "tests-only 0.1.0 (path+file:///w/tests-only)"],
            "packages": [
                {"id": "root 0.1.0 (path+file:///w)", "name": "root", "manifest_path": "/w/Cargo.toml"},
                {"id": "leaf 0.1.0 (path+file:///w/leaf)", "name": "leaf", "manifest_path": "/w/leaf/Cargo.toml"},
                {"id": "tests-only 0.1.0 (path+file:///w/tests-only)", "name": "tests-only", "manifest_path": "/w/tests-only/Cargo.toml"}
            ],
            "resolve": {"nodes": [
                {"id": "root 0.1.0 (path+file:///w)", "deps": [
                    {"pkg": "leaf 0.1.0 (path+file:///w/leaf)", "dep_kinds": [{"kind": null}]},
                    {"pkg": "tests-only 0.1.0 (path+file:///w/tests-only)", "dep_kinds": [{"kind": "dev"}]}
                ]},
                {"id": "leaf 0.1.0 (path+file:///w/leaf)", "deps": []},
                {"id": "tests-only 0.1.0 (path+file:///w/tests-only)", "deps": []}
            ]}
        }"#;
        serde_json::from_str(json).unwrap_or_else(|error| panic!("fixture json: {error}"))
    }

    #[test]
    fn closure_follows_normal_deps_and_skips_dev_only_deps() {
        let graph = WorkspaceGraph::from_metadata(&metadata(), Path::new("/w"));
        let root = graph
            .closure
            .get("root")
            .unwrap_or_else(|| panic!("root has a closure"));
        assert!(root.contains("leaf"), "a normal dep is reachable");
        assert!(
            !root.contains("tests-only"),
            "a dev-only dep must not drag the graph into every leaf's closure"
        );
        assert_eq!(
            graph
                .closure
                .get("leaf")
                .map(std::collections::BTreeSet::len),
            Some(1)
        );
    }

    #[test]
    fn owners_pick_the_longest_directory_prefix() {
        let graph = WorkspaceGraph::from_metadata(&metadata(), Path::new("/w"));
        let files = [
            "leaf/src/lib.rs".to_string(),
            "leaf/../other/src/lib.rs".to_string(),
            "src/main.rs".to_string(),
            "scripts/gate.sh".to_string(),
        ];
        let owners = graph.owning_packages(&files);
        assert!(owners.contains("leaf"));
        assert!(owners.contains("root"), "root owns src/ but not scripts/");
        assert_eq!(owners.len(), 2, "scripts/ belongs to no member");
    }
}
