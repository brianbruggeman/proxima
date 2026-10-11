use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

struct Group {
    sha: String,
    extents: String,
    entry: String,
}

fn parse(path: &Path) -> Vec<Group> {
    fs::read_to_string(path)
        .expect("describe")
        .lines()
        .filter_map(|line| line.strip_prefix("ab describe sha="))
        .map(|rest| {
            let extents_at = rest.find("extents=").expect("extents") + 8;
            let count_at = rest.find(" count=").expect("count");
            let entry_at = rest.find(" entry=").expect("entry");
            Group {
                sha: rest[..16].to_string(),
                extents: rest[extents_at..count_at].to_string(),
                entry: rest[entry_at + 7..].to_string(),
            }
        })
        .collect()
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    let [_, base_describe, other_describe, other_caps, base_caps, directory, tag] = &arguments[..] else {
        panic!("usage: pair_variants <base describe> <other describe> <other caps> <base caps> <out dir> <tag>");
    };
    fs::create_dir_all(directory).expect("out dir");
    let other_by_key: BTreeMap<(String, String), Group> = parse(Path::new(other_describe))
        .into_iter()
        .map(|group| ((group.entry.clone(), group.extents.clone()), group))
        .collect();
    let (mut written, mut identical) = (0usize, 0usize);
    for group in parse(Path::new(base_describe)) {
        let Some(other) = other_by_key.get(&(group.entry.clone(), group.extents.clone())) else {
            continue;
        };
        let base_text = fs::read_to_string(Path::new(base_caps).join(format!("pipeline_{}.metal", group.sha))).expect("base text");
        let other_text = fs::read_to_string(Path::new(other_caps).join(format!("pipeline_{}.metal", other.sha))).expect("other text");
        if base_text == other_text {
            identical += 1;
            continue;
        }
        fs::write(Path::new(directory).join(format!("{}.{tag}.metal", group.sha)), other_text).expect("write variant");
        written += 1;
    }
    eprintln!("variants written={written} identical_groups={identical}");
}
