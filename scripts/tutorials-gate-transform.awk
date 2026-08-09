# tutorials-gate-transform.awk — rewrite one docs/tutorials/*.md file so
# `cargo test --doc` can compile-check every ```rust fenced block. See
# scripts/tutorials-gate.sh for the two-pass mechanism this feeds (why a
# SECOND pass exists at all) and why each transform below exists. The
# original docs/tutorials/*.md file is never modified — this writes a
# transformed copy for rustdoc to read.
#
# usage:
#   awk -v FILEBASE=name -f tutorials-gate-transform.awk docs/tutorials/FILE.md
#     pass 1: no cross-block context. Every block stands alone.
#   awk -v FILEBASE=name -v MANIFEST=path -f tutorials-gate-transform.awk FILE.md
#     pass 1, and also appends "FILEBASE<TAB>ordinal<TAB>generated-open-line"
#     per block to MANIFEST, so the gate script can translate pass 1's
#     rustdoc failures (reported by generated line) back into "which block,
#     by document order, in which file".
#   awk -v FILEBASE=name -v GOODFILE=path -f tutorials-gate-transform.awk FILE.md
#     pass 2: GOODFILE lists "FILEBASE<TAB>ordinal" for every block pass 1
#     proved compiles standalone. A block whose ordinal is NOT listed is
#     never fed forward as context — this is what keeps one broken block
#     from cascading into every block after it in the same file, which is
#     exactly what happened the first time this accumulated unconditionally
#     (measured: the pass count went DOWN, not up, when every prior block
#     was accumulated regardless of whether it compiled on its own).
#
# state machine over fence lines: 0=outside a fence, 1=inside a rust fence
# (buffered, not printed until its closing fence), 2=inside a non-rust fence
# (tagged or retagged, printed straight through). Closing fences are always
# bare ``` in CommonMark regardless of the opening tag, so only the OPENING
# line ever needs classifying. All output goes through out()/outn() so a
# running count of GENERATED lines stays accurate for MANIFEST — this file's
# line numbers diverge from the source's the moment the first hidden line is
# injected, and rustdoc reports failures by the generated file's line.
#
# Each rust block gets, as hidden `# `-prefixed lines a reader never sees
# but rustdoc still compiles:
#   1. `use proxima::tutorial_gate_prelude::*;` — many tutorial snippets are
#      excerpts that elide their own imports "for space" (the tutorials say
#      so); this brings the common cast (Pipe/SendPipe/Future/Debug/the
#      proxima attribute + declarative macros/...) into scope the same way
#      the doctest two paragraphs up its own file already does.
#   2. (pass 2 only) every EARLIER block in the same file that pass 1 proved
#      compiles on its own — most snippets are excerpts of a multi-block
#      worked example (`struct Increment` in one block, `let chain =
#      Increment.and_then(Halve)` three paragraphs later), and rustdoc
#      compiles every block as its own independent program, so without this
#      the second block cannot see the first. A LATER block that redefines
#      an EARLIER one's name (the tutorials' own "Before (...) / Today
#      (...)" comparisons, e.g. two different `struct Backend`) retires the
#      earlier definition from later context instead of leaving both in
#      scope, which would be a duplicate-definition error neither version
#      has on its own.
#   3. the whole block wrapped in `fn main() { block_on(async { .. }); }` —
#      a fragment ending mid-`.await` (continuing the surrounding prose) is
#      not valid inside rustdoc's own default SYNC `fn main` wrapper.
#
# Two classes of block cannot be made to compile standalone at all, no
# matter which pass, and are marked accordingly rather than silently left to
# fail or silently dropped:
#
#   - a bare associated-fn signature excerpted from a larger `impl`/`trait`
#     block one method at a time (02-listener-builder.md and
#     07-sugar-composition.md's teaching style for the builder axis traits)
#     — `self`/`&self`/`&mut self` is only legal inside an `impl`/`trait`,
#     and the tutorial excerpt never repeats the real type its methods
#     belong to. Detected mechanically: a `fn` taking `self` as its first
#     parameter with no `impl`/`trait` opening earlier in the SAME block.
#     Marked `ignore` (never compiled — nothing to compile it AGAINST).
#   - `server.run_until_signal().await` genuinely blocks forever waiting for
#     a process signal nothing in a test harness will ever send — confirmed
#     empirically (this hung the first gate run; SIGKILLed as a stuck
#     process). Marked `no_run`: still compiled, never executed.
#
# All three rules are mechanical, applied to every block, not a hand-picked
# list — a new tutorial block matching any of these shapes lands already
# correctly classified.

BEGIN {
    state = 0
    buf = ""
    ordinal = 0
    outline = 0
    # `proxima::prelude`'s own export list (src/lib.rs, `pub mod prelude`) —
    # every block already has these in real scope via
    # `use proxima::tutorial_gate_prelude::*;` (which itself globs
    # `crate::prelude::*`). A tutorial block that teaches one of these
    # traits by re-declaring it AND implementing it for the real
    # `proxima::Listener`/`Client` (02-listener-builder.md's whole style —
    # showing one axis method at a time) creates a second, differently-
    # named-but-identically-spelled trait live in the SAME scope as the
    # real one: `Listener::builder()` becomes genuinely ambiguous between
    # the real trait's impl and the excerpt's own (measured directly:
    # E0034 "multiple applicable items", one candidate cited as
    # `main::{closure#0}::ListenerBuilderEntry`, the other as the real
    # `proxima::ListenerBuilderEntry`). Such a block still compiles fine on
    # its own; it must simply never be fed forward into a LATER block that
    # also has the real trait in scope.
    split("AnyProtocol App ClassifyOutcome Client ClientBuilder ClientProtocol ClientProtocolExt ClientSecurityExt ClientTransportExt Listener ListenerBuilder ListenerBuilderEntry ListenerProtocolExt ListenerTransportExt ProbeVerdict", prelude_names_arr, " ")
    for (pn in prelude_names_arr) prelude_names[prelude_names_arr[pn]] = 1
    if (GOODFILE != "") {
        while ((getline gline < GOODFILE) > 0) {
            split(gline, parts, "\t")
            if (parts[1] == FILEBASE) good[parts[2]] = 1
        }
        close(GOODFILE)
    }
}

function out(line) {
    print line
    outline++
}

function is_bare_self_fn(text,    n, lines, i, seen_container) {
    n = split(text, lines, "\n")
    seen_container = 0
    for (i = 1; i <= n; i++) {
        if (lines[i] ~ /^[ \t]*(pub(\([a-z]+\))? )?(impl|trait)[ \t]/) seen_container = 1
        if (!seen_container && lines[i] ~ /fn[ \t]+[A-Za-z_][A-Za-z0-9_]*\([ \t]*(&(mut[ \t]+)?)?self([ \t]*[,)]|:)/) return 1
    }
    return 0
}

function extract_names(text,    n, lines, i, line, name) {
    delete defined_here
    n = split(text, lines, "\n")
    for (i = 1; i <= n; i++) {
        line = lines[i]
        if (match(line, /^(pub(\([a-z]+\))? )?(async )?(struct|enum|trait|fn|const|static|type)[ \t]+[A-Za-z_][A-Za-z0-9_]*/)) {
            name = line
            sub(/^(pub(\([a-z]+\))? )?(async )?(struct|enum|trait|fn|const|static|type)[ \t]+/, "", name)
            sub(/[^A-Za-z0-9_].*$/, "", name)
            if (name != "") defined_here[name] = 1
        }
        # `#[piped(..., name = X)]` (or `#[proxima::piped(...)]`) names the
        # macro-GENERATED type X — a real defined name, invisible to the
        # struct/enum/... scan above because it never appears as literal
        # source text; the macro's own `name = X` argument is the only place
        # it is spelled. Without this, the two macro-expanded and
        # hand-written forms of the same tutorial example (see
        # 00-foundations.md section 7, "Before (...) / Today (...)") both
        # stay "active" and collide the moment a later block accumulates
        # both — measured directly (E0428 "the name `Backend` is defined
        # multiple times", the macro's own expansion cited as the second
        # definition site). Scoped to a `piped(...)` line specifically — a
        # bare `name = ...` is an ordinary local variable assignment
        # (`let name = protocol.name().to_string();`, seen for real in
        # 02-listener-builder.md) and must not be read as a type definition.
        if (line ~ /piped\(/ && match(line, /name[ \t]*=[ \t]*[A-Za-z_][A-Za-z0-9_]*/)) {
            name = substr(line, RSTART, RLENGTH)
            sub(/^name[ \t]*=[ \t]*/, "", name)
            if (name != "") defined_here[name] = 1
        }
    }
}

state == 0 && /^```rust(,[A-Za-z_,]+)?[ \t]*$/ {
    state = 1
    buf = ""
    ordinal++
    next
}
state == 0 && /^```[A-Za-z0-9_+-]+[ \t]*$/ {
    out($0)
    state = 2
    next
}
state == 0 && /^```[ \t]*$/ {
    out("```text")
    state = 2
    next
}
state == 1 && /^```[ \t]*$/ {
    if (is_bare_self_fn(buf)) {
        out("```rust,ignore")
        out("# a bare associated-fn signature excerpted from its real impl/trait;")
        out("# self is only legal inside that impl, which this excerpt does not repeat.")
        n = split(buf, blines, "\n")
        for (i = 1; i <= n; i++) if (blines[i] != "" || i < n) out(blines[i])
        out("```")
        state = 0
        next
    }

    # deactivate any EARLIER chunk this block redefines BEFORE building this
    # block's own context — a block that redefines an earlier name must
    # never see the earlier definition in its OWN accumulated context
    # either, not only in later blocks' (this was the second bug the
    # `Backend`/`ProxyPipe` collisions surfaced: a "Today" block's context
    # still carried its own "Before" because deactivation used to run only
    # AFTER a block's context was already built).
    extract_names(buf)
    for (name in defined_here) {
        if (name in owner) chunk_active[owner[name]] = 0
        owner[name] = ordinal
    }

    context = ""
    if (GOODFILE != "") {
        for (i = 1; i < ordinal; i++) {
            if (chunk_active[i] && (i in good)) context = context chunk_text[i]
        }
    }

    tag = ((context buf) ~ /run_until_signal/) ? "```rust,no_run" : "```rust"
    if (MANIFEST != "") print FILEBASE "\t" ordinal "\t" (outline + 1) >> MANIFEST
    out(tag)
    out("# use proxima::tutorial_gate_prelude::*;")
    out("# fn main() { proxima::tutorial_gate_prelude::block_on(async {")

    m = split(context, clines, "\n")
    for (i = 1; i <= m; i++) if (clines[i] != "") out("# " clines[i])

    bn = split(buf, blines, "\n")
    for (i = 1; i <= bn; i++) if (blines[i] != "" || i < bn) out(blines[i])

    out("# }); }")
    out("```")

    chunk_text[ordinal] = buf
    chunk_active[ordinal] = 1
    for (name in defined_here) {
        if (name in prelude_names) chunk_active[ordinal] = 0
    }

    state = 0
    next
}
state == 2 && /^```[ \t]*$/ {
    out($0)
    state = 0
    next
}
state == 1 { buf = buf $0 "\n"; next }
{ out($0) }
