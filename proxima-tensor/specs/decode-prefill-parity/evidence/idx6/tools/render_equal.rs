use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

struct Group {
    sha: String,
    extents: String,
    entry: String,
    count: String,
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
                count: rest[count_at + 7..entry_at].to_string(),
                entry: rest[entry_at + 7..].to_string(),
            }
        })
        .collect()
}

fn main() {
    let arguments: Vec<String> = env::args().collect();
    let [_, base_describe, tip_describe, base_caps, tip_caps, variants, tags] = &arguments[..] else {
        panic!("usage: render_equal <base describe> <tip describe> <base caps> <tip caps> <variant dir> <tag,tag>");
    };
    let tags: Vec<&str> = tags.split(',').collect();
    let tip_by_key: BTreeMap<(String, String), Group> = parse(Path::new(tip_describe))
        .into_iter()
        .map(|group| ((group.entry.clone(), group.extents.clone()), group))
        .collect();
    println!("| base sha | tip sha | extents | dispatches | tip text vs base text | equals variant (bytes) | entry |");
    println!("|---|---|---|---|---|---|---|");
    let (mut compared, mut equal_variant, mut changed_without_variant, mut missing) = (0usize, 0usize, 0usize, 0usize);
    for group in parse(Path::new(base_describe)) {
        let Some(tip) = tip_by_key.get(&(group.entry.clone(), group.extents.clone())) else {
            missing += 1;
            println!("| {} | missing | {} | {} | no tip group with this entry and extents | | {} |", &group.sha[..8], group.extents, group.count, group.entry);
            continue;
        };
        let base_text = fs::read_to_string(Path::new(base_caps).join(format!("pipeline_{}.metal", group.sha))).expect("base text");
        let tip_text = fs::read_to_string(Path::new(tip_caps).join(format!("pipeline_{}.metal", tip.sha))).expect("tip text");
        let mut matched: Vec<String> = Vec::new();
        for tag in &tags {
            let path = Path::new(variants).join(format!("{}.{tag}.metal", group.sha));
            if let Ok(variant) = fs::read_to_string(&path) {
                if variant == tip_text {
                    matched.push((*tag).to_string());
                }
            }
        }
        compared += 1;
        let differs = base_text != tip_text;
        equal_variant += usize::from(!matched.is_empty());
        changed_without_variant += usize::from(differs && matched.is_empty());
        println!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            &group.sha[..8],
            &tip.sha[..8],
            group.extents,
            group.count,
            if differs { "differs" } else { "identical" },
            if matched.is_empty() { "-".to_string() } else { matched.join("+") },
            &group.entry[..group.entry.len().min(60)]
        );
    }
    eprintln!("compared={compared} tip_equals_a_measured_variant={equal_variant} changed_without_a_measured_variant={changed_without_variant} missing={missing}");
}
