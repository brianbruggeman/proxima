//! The one scorer both arms share. A needle is found when its number
//! appears in the response as a whole digit run, so `4830912` matches inside
//! prose but not inside `48309120` or `1483091`.

use crate::haystack::Needle;

pub(crate) fn is_exact_match(response: &str, expected: &str) -> bool {
    response
        .split(|character: char| !character.is_ascii_digit())
        .any(|run| run == expected)
}

pub(crate) fn found_count(response: &str, needles: &[Needle]) -> usize {
    needles
        .iter()
        .filter(|needle| is_exact_match(response, &needle.number.to_string()))
        .count()
}
