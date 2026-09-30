//! Haystack, needles and prompt for the needle-in-a-haystack harness
//! (`proxima-tensor/specs/long-context/SPEC.md` R16a1-R16a3).
//!
//! Every length here is measured through a caller-supplied `count` closure
//! (`&str -> tokens`), so the same search runs against the model's own
//! tokenizer in production and against a tokenizer-independent counter in
//! tests. The prompt built here is the one string both arms receive.

use crate::NiahError;

pub(crate) const SOURCE: &str = include_str!("../data/war_and_peace.txt");

const STORY_START: &str = "\nBOOK ONE: 1805\n";
const FIRST_GUESS_NUMERATOR: usize = 3;
const FIRST_GUESS_DENOMINATOR: usize = 4;
const TOKENS_PER_ANSWER_LINE: usize = 16;
const ANSWER_MARGIN_TOKENS: usize = 32;
const PROMPT_MARGIN_TOKENS: usize = 64;
const NUMBER_RANGE: core::ops::Range<u32> = 1_000_000..10_000_000;

const NOUNS: [&str; 64] = [
    "marmot",
    "trombone",
    "quartz",
    "lantern",
    "harpoon",
    "saffron",
    "obelisk",
    "gondola",
    "thimble",
    "cormorant",
    "anvil",
    "pomegranate",
    "sextant",
    "tapestry",
    "walrus",
    "zeppelin",
    "bramble",
    "cobbler",
    "dirigible",
    "eucalyptus",
    "falconer",
    "glacier",
    "hammock",
    "iguana",
    "juniper",
    "kayak",
    "lighthouse",
    "mandolin",
    "nutmeg",
    "orchard",
    "pelican",
    "quiver",
    "rhubarb",
    "scarecrow",
    "tambourine",
    "umbrella",
    "vineyard",
    "windmill",
    "xylophone",
    "yak",
    "artichoke",
    "barnacle",
    "chandelier",
    "dulcimer",
    "espresso",
    "flamingo",
    "gargoyle",
    "hedgehog",
    "inkwell",
    "jackfruit",
    "kiln",
    "lozenge",
    "monsoon",
    "narwhal",
    "oregano",
    "porcupine",
    "quince",
    "raccoon",
    "sundial",
    "turnip",
    "vermilion",
    "wheelbarrow",
    "yodeler",
    "zucchini",
];

/// One hidden fact: a random noun keyed to a random seven-digit number,
/// e.g. `marmot` -> `4830912`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Needle {
    pub(crate) noun: &'static str,
    pub(crate) number: u32,
}

impl Needle {
    pub(crate) fn sentence(&self) -> String {
        format!(
            "The special magic number for {} is {}.",
            self.noun, self.number
        )
    }
}

/// Where one needle went, in haystack-only token coordinates (the haystack
/// before any needle sentence is inserted).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Placement {
    pub(crate) word_index: usize,
    pub(crate) depth_target: usize,
    pub(crate) depth_measured: usize,
}

/// The prompt both arms receive, with the facts needed to score and audit it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Case {
    pub(crate) prompt: String,
    pub(crate) needles: Vec<Needle>,
    pub(crate) placements: Vec<Placement>,
    pub(crate) prompt_tokens: usize,
    pub(crate) max_new_tokens: usize,
}

pub(crate) fn story_words() -> Result<Vec<&'static str>, NiahError> {
    let start = SOURCE.find(STORY_START).ok_or(NiahError::HaystackMarker)?;
    let story = SOURCE.get(start..).ok_or(NiahError::HaystackMarker)?;
    Ok(story.split_whitespace().collect())
}

pub(crate) fn generate_needles(seed: u64, needle_count: usize) -> Result<Vec<Needle>, NiahError> {
    if needle_count == 0 || needle_count > NOUNS.len() {
        return Err(NiahError::NeedleCount {
            requested: needle_count,
            available: NOUNS.len(),
        });
    }
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut nouns = NOUNS;
    rng.shuffle(&mut nouns);
    Ok(nouns
        .into_iter()
        .take(needle_count)
        .map(|noun| Needle {
            noun,
            number: rng.u32(NUMBER_RANGE),
        })
        .collect())
}

pub(crate) fn max_new_tokens(needle_count: usize) -> usize {
    needle_count * TOKENS_PER_ANSWER_LINE + ANSWER_MARGIN_TOKENS
}

fn question(needles: &[Needle]) -> String {
    let nouns = needles
        .iter()
        .map(|needle| needle.noun)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Question: the text above hides {} sentences of the form \"The special magic number for <noun> is <number>.\" \
         Give the number for each of these nouns: {nouns}.\nAnswer, one line per noun as \"noun: number\":\n",
        needles.len()
    )
}

fn interpolate(numerator: usize, span_words: usize, span_tokens: usize) -> usize {
    let scaled = u128::try_from(numerator).unwrap_or(u128::MAX)
        * u128::try_from(span_words).unwrap_or(u128::MAX)
        / u128::try_from(span_tokens.max(1)).unwrap_or(1);
    usize::try_from(scaled).unwrap_or(usize::MAX)
}

fn next_guess(
    low: (usize, usize),
    high: Option<(usize, usize)>,
    target: usize,
    limit: usize,
) -> usize {
    match high {
        Some((high_words, high_tokens)) if high_words - low.0 > 1 => {
            let offset = interpolate(target - low.1, high_words - low.0, high_tokens - low.1);
            (low.0 + offset).clamp(low.0 + 1, high_words - 1)
        }
        Some(_) => low.0,
        None => interpolate(target, low.0, low.1)
            .max(low.0 + 1)
            .min(limit)
            .max(low.0),
    }
}

/// The word count `k` whose prefix measures `[target - tolerance, target]`
/// tokens, with that measured token count. When the bracket collapses first,
/// the longest prefix that stays at or under `target`. Never exceeds `target`.
pub(crate) fn prefix_for_target(
    words: &[&str],
    target: usize,
    tolerance: usize,
    count: &mut impl FnMut(&str) -> Result<usize, NiahError>,
) -> Result<(usize, usize), NiahError> {
    if words.is_empty() {
        return Err(NiahError::HaystackMarker);
    }
    let floor = target.saturating_sub(tolerance);
    let mut low = (0_usize, 0_usize);
    let mut high: Option<(usize, usize)> = None;
    let mut guess =
        (target * FIRST_GUESS_NUMERATOR / FIRST_GUESS_DENOMINATOR).clamp(1, words.len());
    loop {
        let prefix = words.get(..guess).unwrap_or(words).join(" ");
        let tokens = count(&prefix)?;
        if (floor..=target).contains(&tokens) {
            return Ok((guess, tokens));
        }
        if tokens > target {
            high = Some((guess, tokens));
        } else {
            low = (guess, tokens);
        }
        let following = next_guess(low, high, target, words.len());
        if following <= low.0 {
            return Ok(low);
        }
        guess = following;
    }
}

/// R16a2: the longest word prefix of `words` that is within 1% of
/// `requested` tokens and never above it.
pub(crate) fn trim_haystack(
    words: &[&str],
    requested: usize,
    count: &mut impl FnMut(&str) -> Result<usize, NiahError>,
) -> Result<(usize, usize), NiahError> {
    let (kept, tokens) = prefix_for_target(words, requested, requested / 100, count)?;
    if tokens < requested - requested / 100 {
        return Err(NiahError::HaystackTooShort {
            requested,
            available: tokens,
        });
    }
    Ok((kept, tokens))
}

/// R16a3: needle `i` of `needles` lands at token depth
/// `(i + 0.5) * requested / needles`, within `needle_tokens`.
pub(crate) fn place_needles(
    words: &[&str],
    requested: usize,
    needle_tokens: usize,
    needles: usize,
    count: &mut impl FnMut(&str) -> Result<usize, NiahError>,
) -> Result<Vec<Placement>, NiahError> {
    (0..needles)
        .map(|index| {
            let depth_target = (2 * index + 1) * requested / (2 * needles);
            let (word_index, depth_measured) =
                prefix_for_target(words, depth_target, needle_tokens / 2, count)?;
            Ok(Placement {
                word_index,
                depth_target,
                depth_measured,
            })
        })
        .collect()
}

pub(crate) fn assemble(words: &[&str], insertions: &[(usize, String)]) -> String {
    let mut text = String::new();
    let mut pending = insertions.iter().peekable();
    for (index, word) in words.iter().enumerate() {
        while let Some((_, sentence)) = pending.next_if(|(at, _)| *at == index) {
            text.push_str(sentence);
            text.push(' ');
        }
        text.push_str(word);
        text.push(' ');
    }
    text.truncate(text.trim_end().len());
    text
}

/// Builds the prompt for a `context`-token window: a haystack trimmed to fit
/// beside the needles, the question and the answer budget, with `needle_count`
/// needles inserted (none when `control`). The question always asks about the
/// same nouns, so the control differs from the real run only by the hidden
/// sentences.
pub(crate) fn build_case(
    words: &[&str],
    context: usize,
    needle_count: usize,
    control: bool,
    seed: u64,
    count: &mut impl FnMut(&str) -> Result<usize, NiahError>,
) -> Result<Case, NiahError> {
    let needles = generate_needles(seed, needle_count)?;
    let sentences: Vec<String> = needles.iter().map(Needle::sentence).collect();
    let question = question(&needles);
    let max_new = max_new_tokens(needle_count);
    let needle_tokens = sentences
        .iter()
        .map(|sentence| count(sentence))
        .collect::<Result<Vec<_>, _>>()?;
    let overhead =
        needle_tokens.iter().sum::<usize>() + count(&question)? + max_new + PROMPT_MARGIN_TOKENS;
    let requested = context
        .checked_sub(overhead)
        .ok_or(NiahError::ContextTooSmall { context, overhead })?;
    let (kept, _) = trim_haystack(words, requested, count)?;
    let haystack = words.get(..kept).unwrap_or(words);
    let longest_needle = needle_tokens.iter().copied().max().unwrap_or(0);
    let placements = place_needles(haystack, requested, longest_needle, needle_count, count)?;
    let insertions: Vec<(usize, String)> = if control {
        Vec::new()
    } else {
        placements
            .iter()
            .zip(sentences)
            .map(|(placement, sentence)| (placement.word_index, sentence))
            .collect()
    };
    let prompt = format!("{}\n\n{question}", assemble(haystack, &insertions));
    let prompt_tokens = count(&prompt)?;
    if prompt_tokens + max_new > context {
        return Err(NiahError::PromptExceedsContext {
            prompt_tokens,
            max_new,
            context,
        });
    }
    Ok(Case {
        prompt,
        needles,
        placements,
        prompt_tokens,
        max_new_tokens: max_new,
    })
}
