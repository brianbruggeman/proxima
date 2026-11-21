//! Human-readable rendering for the CLI: a tensor as a bounded value list so
//! a serving-sized output stays one readable line.

/// Values shown before the summary elides the rest. A serving tensor holds
/// thousands and the first few are what a reader checks, so the first line
/// stays readable while the total still tells you the size.
const PRINTED_VALUES: usize = 8;

/// Renders `values` as a bracketed list, eliding past [`PRINTED_VALUES`] with
/// the total count so a truncated line never reads as the whole tensor.
///
/// ```text
/// [10.0, 11.0, 12.0, 13.0]
/// [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, ... 1024 values]
/// ```
#[must_use]
pub fn summarize_values(values: &[f32]) -> String {
    let shown = values.len().min(PRINTED_VALUES);
    let head = values[..shown]
        .iter()
        .map(|value| format!("{value:?}"))
        .collect::<Vec<_>>()
        .join(", ");
    if shown == values.len() {
        format!("[{head}]")
    } else {
        format!("[{head}, ... {} values]", values.len())
    }
}

#[cfg(test)]
mod tests {
    use super::summarize_values;

    #[test]
    fn prints_every_value_of_a_four_element_output() {
        assert_eq!(
            summarize_values(&[10.0, 11.0, 12.0, 13.0]),
            "[10.0, 11.0, 12.0, 13.0]"
        );
    }

    #[test]
    fn elides_past_the_printed_window_and_keeps_the_total() {
        let values: Vec<f32> = (0..12).map(|value| value as f32).collect();

        assert_eq!(
            summarize_values(&values),
            "[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, ... 12 values]"
        );
    }

    #[test]
    fn prints_an_empty_output_as_an_empty_list() {
        assert_eq!(summarize_values(&[]), "[]");
    }
}
