# nostd-gate.awk — cfg-aware std:: / thread_local! scanner for nostd-gate.sh.
#
# Reads production-code lines (caller truncates before `#[cfg(test)]`) and
# prints "<line-number>:<line>" for every candidate violation:
#   - a `std::`-qualified path that is NOT inside a `#[cfg(feature = "std")]`
#     (or `#[cfg(all(..., feature = "std", ...))]`) gated item or block
#   - any `thread_local!` site (qualified or bare), always — gating doesn't
#     exempt it, since raw TLS use is itself the tracked debt
#
# State machine: `pending_gate` is set by the nearest preceding std-feature
# cfg attribute and consumed by the next code line; if that line opens a
# brace-delimited scope (fn body, impl block, mod, bare block, thread_local!
# block, ...), the gate is pushed onto a depth stack so every line inside
# inherits it until the matching close brace pops it back off.

{
    line = $0

    # multi-line attribute continuation, e.g. #[cfg(all(\n  ...,\n))]
    if (attr_open) {
        attr_text = attr_text "\n" line
        opens = gsub(/\[/, "[", line)
        closes = gsub(/\]/, "]", line)
        attr_balance += opens - closes
        if (attr_balance <= 0) {
            attr_open = 0
            apply_attr(attr_text)
        }
        next
    }

    if (line ~ /^[ \t]*#!?\[/) {
        attr_open = 1
        attr_text = line
        opens = gsub(/\[/, "[", line)
        closes = gsub(/\]/, "]", line)
        attr_balance = opens - closes
        if (attr_balance <= 0) {
            attr_open = 0
            apply_attr(attr_text)
        }
        next
    }

    # comments and blank lines never carry code or consume the pending gate
    if (line ~ /^[ \t]*(\/\/|$)/) {
        next
    }

    current_gate = pending_gate
    if (!current_gate && depth > 0 && gate[depth]) {
        current_gate = 1
    }
    pending_gate = 0

    is_std = ($0 ~ /(^|[^A-Za-z0-9_])std::/)
    is_thread_local = ($0 ~ /thread_local!/)

    if (is_thread_local || (is_std && !current_gate)) {
        print NR ":" $0
    }

    opens = gsub(/\{/, "{", line)
    closes = gsub(/\}/, "}", line)
    net = opens - closes
    if (net > 0) {
        for (index_i = 0; index_i < net; index_i++) {
            depth++
            gate[depth] = current_gate
        }
    } else if (net < 0) {
        for (index_i = 0; index_i < -net; index_i++) {
            if (depth > 0) {
                depth--
            }
        }
    }
}

function apply_attr(text) {
    if (text ~ /feature[ \t]*=[ \t]*"std"/) {
        if (text ~ /not\(feature[ \t]*=[ \t]*"std"\)/) {
            pending_gate = 0
        } else {
            pending_gate = 1
        }
    }
}
