use super::*;

pub(super) const INDEX32_REACH_LIMIT: u64 = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IndexFamily {
    Body,
    ExpertPrologue,
    BlockDecode,
}

pub(super) fn index_family(resolved: &BoundOp, quantized: &[Option<Codec>]) -> Option<IndexFamily> {
    match &resolved.kind {
        BoundOpKind::Reduce {
            reduce_op,
            init,
            output_axes,
            epilogue_operands,
            ..
        } => {
            if tiled_gemm_block(resolved, quantized, *reduce_op, *init, output_axes).is_some()
                || dense_batched_gemm_block(resolved, quantized, *reduce_op, *init, output_axes)
                    .is_some()
            {
                return None;
            }
            if let Some(block) = packed_row_block(resolved, quantized) {
                return expert_divide_is_rendered(resolved, output_axes, &block)
                    .then_some(IndexFamily::ExpertPrologue);
            }
            let unpacked = quantized.iter().all(Option::is_none);
            (unpacked && !epilogue_operands.is_empty() && gather_count(resolved) == 0)
                .then_some(IndexFamily::Body)
        }
        BoundOpKind::Elementwise { .. } => {
            let gathered = gather_count(resolved) > 0;
            match (gathered, quantized.iter().all(Option::is_none)) {
                (true, true) => Some(IndexFamily::Body),
                (true, false) => Some(IndexFamily::BlockDecode),
                (false, _) => None,
            }
        }
        _ => None,
    }
}

fn expert_divide_is_rendered(resolved: &BoundOp, output_axes: &[u16], block: &PackedRowBlock) -> bool {
    let rows = codec_rows_per_simdgroup(block.codec) as u64;
    packed_row_block_token_total(block, &resolved.extents) <= 1
        && packed_row_direct_output_axis(resolved, output_axes).is_none()
        && packed_row_direct_grouped_axes(resolved, output_axes)
            .is_some_and(|(_, _, _, out_axis)| resolved.extents[out_axis as usize].is_multiple_of(rows))
}

pub(super) fn index_family_fits(resolved: &BoundOp, quantized: &[Option<Codec>], family: IndexFamily) -> bool {
    match family {
        IndexFamily::Body => body_reach_fits(resolved),
        IndexFamily::ExpertPrologue => output_total_fits_u32(resolved),
        IndexFamily::BlockDecode => block_decode_reach_fits(resolved, quantized),
    }
}

fn layout_reach(layout: &Layout, extents: &[u64]) -> Option<u64> {
    layout.strides.iter().zip(extents).try_fold(layout.base.unsigned_abs(), |reach, (stride, extent)| {
        reach.checked_add(extent.saturating_sub(1).checked_mul(stride.unsigned_abs())?)
    })
}

fn gathered_reach(layout: &Layout, lookup: Option<&Lookup>, extents: &[u64]) -> Option<u64> {
    let direct = layout_reach(layout, extents)?;
    let Some(lookup) = lookup else {
        return Some(direct);
    };
    let fetched = lookup.extent.saturating_sub(1).checked_mul(lookup.element_stride.unsigned_abs())?;
    let index_reach = layout_reach(&lookup.index_layout, extents)?;
    direct.checked_add(fetched)?.checked_add(index_reach)
}

fn body_reach_fits(resolved: &BoundOp) -> bool {
    let Some(iteration_space) = resolved.extents.iter().try_fold(1u64, |total, extent| total.checked_mul(*extent)) else {
        return false;
    };
    let mut reaches: Vec<Option<u64>> = resolved
        .operands()
        .iter()
        .map(|(_, layout, lookup)| gathered_reach(layout, lookup.as_ref(), &resolved.extents))
        .collect();
    if let BoundOpKind::Reduce { out_layout, epilogue_operands, .. } = &resolved.kind {
        reaches.push(layout_reach(out_layout, &resolved.extents));
        reaches.extend(
            epilogue_operands
                .iter()
                .map(|(_, layout, lookup)| gathered_reach(layout, lookup.as_ref(), &resolved.extents)),
        );
    }
    (1..=INDEX32_REACH_LIMIT).contains(&iteration_space)
        && reaches.iter().all(|reach| reach.is_some_and(|value| value <= INDEX32_REACH_LIMIT))
}

fn output_total_fits_u32(resolved: &BoundOp) -> bool {
    let BoundOpKind::Reduce { output_axes, .. } = &resolved.kind else {
        return false;
    };
    output_axes
        .iter()
        .try_fold(1u64, |total, axis| total.checked_mul(resolved.extents[*axis as usize]))
        .is_some_and(|total| (1..=u64::from(u32::MAX)).contains(&total))
}

fn block_decode_reach_fits(resolved: &BoundOp, quantized: &[Option<Codec>]) -> bool {
    resolved.operands().iter().zip(quantized).all(|((_, layout, lookup), codec)| {
        let non_negative = layout.base >= 0
            && layout.strides.iter().all(|stride| *stride >= 0)
            && lookup.as_ref().is_none_or(|gather| gather.element_stride >= 0);
        let within = codec.is_none()
            || gathered_reach(layout, lookup.as_ref(), &resolved.extents)
                .is_some_and(|reach| reach <= u64::from(u32::MAX));
        non_negative && within
    })
}

fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

pub(super) fn replace_word(text: &str, word: &str, replacement: &str) -> String {
    let bytes = text.as_bytes();
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    while let Some(found) = text[cursor..].find(word) {
        let start = cursor + found;
        let end = start + word.len();
        let isolated = (start == 0 || !is_word_byte(bytes[start - 1])) && bytes.get(end).is_none_or(|byte| !is_word_byte(*byte));
        output.push_str(&text[cursor..start]);
        output.push_str(if isolated { replacement } else { word });
        cursor = end;
    }
    output.push_str(&text[cursor..]);
    output
}

fn strip_long_suffix(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut output = String::with_capacity(text.len());
    for (index, &byte) in bytes.iter().enumerate() {
        let mut digits_start = index;
        while digits_start > 0 && bytes[digits_start - 1].is_ascii_digit() {
            digits_start -= 1;
        }
        let after_number = index > 0
            && bytes[index - 1].is_ascii_digit()
            && (digits_start == 0 || !(bytes[digits_start - 1].is_ascii_alphabetic() || bytes[digits_start - 1] == b'_'));
        let suffix = byte == b'L' && after_number && bytes.get(index + 1).is_none_or(|next| !is_word_byte(*next));
        if !suffix {
            output.push(char::from(byte));
        }
    }
    output
}

fn uniform_long_members(source: &str) -> Vec<String> {
    let Some(start) = source.find("struct Uniforms {") else {
        return Vec::new();
    };
    let block = &source[start..];
    let end = block.find("};").unwrap_or(block.len());
    block[..end]
        .lines()
        .filter_map(|line| line.trim().strip_prefix("long "))
        .filter_map(|rest| rest.split(['[', ';']).next())
        .map(|name| name.trim().to_string())
        .collect()
}

fn closing_bracket(bytes: &[u8], open: usize) -> usize {
    let mut depth = 0usize;
    for (offset, &byte) in bytes[open..].iter().enumerate() {
        match byte {
            b'[' => depth += 1,
            b']' => {
                depth -= 1;
                if depth == 0 {
                    return open + offset;
                }
            }
            _ => {}
        }
    }
    bytes.len() - 1
}

fn uniform_member_at(body: &str, at: usize, members: &[String]) -> Option<usize> {
    let tail = &body[at..];
    members
        .iter()
        .find(|name| tail.starts_with(name.as_str()) && tail.as_bytes().get(name.len()).is_none_or(|byte| !is_word_byte(*byte)))
        .map(String::len)
}

fn cast_uniform_reads(body: &str, members: &[String]) -> String {
    let bytes = body.as_bytes();
    let mut output = String::with_capacity(body.len() + 256);
    let mut cursor = 0;
    while cursor < bytes.len() {
        let starts_read = bytes[cursor] == b'u'
            && bytes.get(cursor + 1) == Some(&b'.')
            && (cursor == 0 || !is_word_byte(bytes[cursor - 1]));
        let member_len = starts_read.then(|| uniform_member_at(body, cursor + 2, members)).flatten();
        if let Some(length) = member_len {
            let mut end = cursor + 2 + length;
            while bytes.get(end) == Some(&b'[') {
                end = closing_bracket(bytes, end) + 1;
            }
            output.push_str(&format!("((int){})", &body[cursor..end]));
            cursor = end;
        } else {
            output.push(char::from(bytes[cursor]));
            cursor += 1;
        }
    }
    output
}

pub(super) fn narrow_body_to_int32(source: &str) -> String {
    let members = uniform_long_members(source);
    let Some(at) = source.find("kernel void ") else {
        return source.to_string();
    };
    let (head, body) = source.split_at(at);
    let body = cast_uniform_reads(body, &members);
    let body = replace_word(&body, "long", "int");
    format!("{head}{}", strip_long_suffix(&body))
}

pub(super) fn narrow_expert_prologue(source: &str) -> String {
    source
        .replace(
            "long expert_slot = group_first / group_out_extent;",
            "long expert_slot = (long)((uint)group_first / (uint)group_out_extent);",
        )
        .replace(
            "long out_row_base = group_first % group_out_extent;",
            "long out_row_base = (long)((uint)group_first % (uint)group_out_extent);",
        )
}

pub(super) fn narrow_block_decode(source: &str) -> String {
    let mut output = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(found) = rest.find("(off") {
        let (before, tail) = rest.split_at(found);
        output.push_str(before);
        match block_decode_read(tail) {
            Some((wide, narrow)) => {
                output.push_str(&narrow);
                rest = &tail[wide..];
            }
            None => {
                output.push_str("(off");
                rest = &tail[4..];
            }
        }
    }
    output.push_str(rest);
    output
}

fn block_decode_read(tail: &str) -> Option<(usize, String)> {
    let digits = tail[4..].bytes().take_while(u8::is_ascii_digit).count();
    let name = &tail[1..4 + digits];
    let after_name = &tail[4 + digits..];
    let divisor_text = after_name.strip_prefix(" / ")?;
    let divisor_len = divisor_text.bytes().take_while(u8::is_ascii_digit).count();
    let divisor = &divisor_text[..divisor_len];
    let closing = divisor_text[divisor_len..].strip_prefix(") * ")?;
    let size_len = closing.bytes().take_while(u8::is_ascii_digit).count();
    let size = &closing[..size_len];
    let consumed = 4 + digits + 3 + divisor_len + 4 + size_len;
    Some((consumed, format!("(long)((uint){name} / {divisor}u) * {size}")))
}

pub(super) fn narrow_block_remainders(source: &str) -> String {
    let mut output = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(found) = rest.find("(uint)(off") {
        let (before, tail) = rest.split_at(found);
        output.push_str(before);
        let digits = tail[10..].bytes().take_while(u8::is_ascii_digit).count();
        let after_name = &tail[10 + digits..];
        let parsed = after_name.strip_prefix(" % ").and_then(|divisor_text| {
            let length = divisor_text.bytes().take_while(u8::is_ascii_digit).count();
            divisor_text[length..].strip_prefix(')').map(|_| length)
        });
        match parsed {
            Some(length) => {
                let divisor = &after_name[3..3 + length];
                output.push_str(&format!("((uint)off{} % {divisor}u)", &tail[10..10 + digits]));
                rest = &after_name[3 + length + 1..];
            }
            None => {
                output.push_str("(uint)(off");
                rest = &tail[10..];
            }
        }
    }
    output.push_str(rest);
    output
}

pub(super) fn apply_index_family(family: IndexFamily, source: String) -> String {
    match family {
        IndexFamily::Body => narrow_body_to_int32(&source),
        IndexFamily::ExpertPrologue => narrow_expert_prologue(&source),
        IndexFamily::BlockDecode => narrow_block_remainders(&narrow_block_decode(&source)),
    }
}
