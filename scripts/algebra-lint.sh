#!/usr/bin/env bash
# algebra-lint — mechanically enforce the claims proxima makes about itself.
#
# proxima's claim is "everything is a Pipe: call(In) -> Result<Out, Err>", and
# its examples claim to await without polling and to run on proxima. A claim you
# do not execute is marketing. This runs them.
#
# Every check here exists because it was violated in real code, not in theory.
#
# usage: scripts/algebra-lint.sh          (exit 1 on findings)

set -uo pipefail
cd "$(dirname "$0")/.."

FINDINGS=0
say() { printf '%s\n' "$*"; }
finding() { FINDINGS=$((FINDINGS + 1)); printf '  FAIL %s\n' "$*"; }
ok() { printf '  ok   %s\n' "$*"; }

PIPE_DIR=proxima-primitives/src/pipe
EX=examples

# 1. a trait WE DECLARE in the pipe layer whose method is not pipe-shaped.
#    Only our own seams — impls of std/foreign traits (PartialEq::eq, Future::poll,
#    Stream::poll_next at the interop bridge) are not our claim to keep.
#    A pipe answers Result<Out, Err>. A seam answering `bool` throws away the item
#    AND the reason, so companions grow to carry them back (a way to hold the item,
#    a way to build the rejection, a way to pick drop-vs-error). A seam answering
#    `Poll<..>` is a second, competing readiness protocol.
# A seam that TOUCHES THE ITEM must be a pipe. One that only answers a control
# question and never sees the item is a strategy — a plain function — and is
# fine. The line is readable straight off the signature: does the method mention
# the trait's own payload type (a generic param, or Self::Assoc)?
#
#   Decide<In>::decide(&self, input: &In) -> bool   <- takes the item, answers
#       bool. DEFECT: the bool destroys the item and the reason, so companions
#       grow to carry them back (Rejectable, OnReject, FromFn).
#   FanInStrategy::index(step, start, n) -> usize   <- never sees an item.
#       A dial. Correctly not a pipe; making it one would build and poll a
#       future to compute a usize.

# Seams investigated and EXEMPTED, with cause. Each cost a real investigation;
# the reason IS the justification, not a suppression. Re-litigate by deleting a
# line and re-reading the cited code — not by trusting this list.
#
# The defect this check exists for is a DECISION on the data path that cannot
# carry its own answer: `Decide::decide(&self, &In) -> bool` threw away both the
# item and the reason, so Rejectable/OnReject/Filter grew to carry them back.
# A QUERY whose answer is complete is not that, and no regex can tell them apart
# — hence this list rather than a cleverer rule.
#
#   Clock::delay            returns the trait's own FUTURE, not a payload. Clock
#                           has no In/Out at all; nothing of "the item" is here.
#   KeyOf::build_rejection  no `self` parameter whatsoever — a static factory
#                           building a canned rejection from config alone.
#   KeyOf::rate_key         borrows the item to derive a key; the item SURVIVES
#                           and is passed to the inner pipe right after
#                           (rate_limit.rs:417,426). A query, not a decision.
#   ApplyOps::apply         consumes and FULLY returns the item — nothing thrown
#                           away. Called inline inside Transform::call, which is
#                           already the pipe. Result<_, Infallible> would add
#                           ceremony and remove nothing.
#   Replayable::fork        infallible 1->2 split; returns the item plus a real
#                           companion (a replay source that is used). Pipe is
#                           1-in/1-out; this does not fit and gains nothing.
#   BatchSource::drain_batch  N-item, sync, no-waker, no-alloc T0 drain. Its own
#                           doc rules it deliberately non-pipe; forcing 1:1 call
#                           semantics means allocating, fixing arity, or adding a
#                           waker — each breaks a claim the file makes.
#   DrainSink::accept       the documented borrowed-vs-owned split: the OWNED
#                           push sink already IS SendPipe<In=Item, Out=()>; the
#                           zero-copy one cannot be (its In is borrowed).
EXEMPT_SEAMS='Clock::fn delay|KeyOf::fn build_rejection|KeyOf::fn rate_key|ApplyOps::fn apply|Replayable::fn fork|BatchSource::fn drain_batch|DrainSink::fn accept'

say "pipe layer: seams that touch the item must be pipe-shaped"
while IFS= read -r hit; do
  finding "$hit"
done < <(awk '
  # remember the trait name and its generic params (the payload types)
  /^pub trait [A-Za-z_]/ {
    intrait = 1; line = $0
    tname = $3; sub(/[<:{].*/, "", tname)
    params = ""
    if (match(line, /<[^>]*>/)) {
      params = substr(line, RSTART + 1, RLENGTH - 2)
      gsub(/:[^,]*/, "", params); gsub(/ /, "", params)   # drop bounds
    }
    next
  }
  intrait && /^\}/ { intrait = 0; next }
  intrait && /^[[:space:]]+fn [a-z_]/ {
    sig = $0; gsub(/^[[:space:]]+|[[:space:]]+$/, "", sig)
    if (sig ~ /Result</ || sig ~ /impl Future/ || sig ~ /and_then/) next
    # does it touch the item? -> mentions Self::Assoc, or one of the trait params
    touches = (sig ~ /Self::/)
    if (!touches && params != "") {
      split(params, p, ",")
      for (i in p) if (p[i] != "" && sig ~ ("[^A-Za-z_]" p[i] "[^A-Za-z_0-9]")) touches = 1
    }
    if (touches) printf "%s:%d  %s::%s\n", FILENAME, FNR, tname, sig
  }
' "$PIPE_DIR"/*.rs "$PIPE_DIR"/*/*.rs 2>/dev/null | grep -vE "$EXEMPT_SEAMS")
[ "$FINDINGS" -eq 0 ] && ok "no data-path seam dodges the pipe shape"

# (the old grep check for "taught primitives implement Pipe" lived here. it is
# gone: `Filter` no longer exists — filtering is `predicate.and_then(inner)`,
# a chain, not a named combinator — and the check would have demanded a type
# back to satisfy itself. rustc already asserts this properly and cannot be
# fooled by a rename: see `algebra_claims` in proxima-primitives/src/pipe/mod.rs,
# which fails the BUILD if a taught primitive stops being a pipe.)

# 3. examples must use proxima, not work around it.
say ""
say "examples must use proxima"
BEFORE=$FINDINGS
check_ex() { # pattern, why
  local pat="$1" why="$2" hits
  hits=$(grep -rlnE "$pat" --include='*.rs' "$EX" 2>/dev/null | sort -u)
  if [ -n "$hits" ]; then
    while IFS= read -r f; do finding "$f — $why"; done <<< "$hits"
  fi
}
check_ex 'unsafe\s*\{' 'unsafe in an example; configure it properly instead'
check_ex 'futures::executor::block_on' "drives proxima's app with futures' executor; use #[proxima::main] and .await"
check_ex 'env::set_var' 'sets a global env var to configure proxima; use config or pass it explicitly'

# `thread::sleep` gets its own rule, not `check_ex`: a first cut flagged
# every `thread::sleep` in examples/, seven hits, and read each one's control
# flow wrong. Only ONE (`protocol_fleet.rs`, since fixed with `Notify` +
# `timeout`) was a real busy-wait defect. Two (`dpdk_tcp_connect.rs`,
# `init_telemetry.rs`) are not loops at all — a one-shot pacing delay and a
# wait on an external kernel process — so flagging "any thread::sleep" was
# flagging code that never polls for anything. The other five are the SAME
# bounded poll-connect loop, and it is not a workaround this repo forgot to
# fix: `src/listener/handle.rs:439-448` documents, against itself, that
# `App::serve` returns before its listener's first poll runs the real
# bind/listen syscalls, that closing the race is out of scope, and that
# "callers needing a synchronization point today poll-connect with a bounded
# retry loop" — this exact shape. A signal to await does not exist yet; the
# loop is the documented answer, not the defect this check exists to catch.
#
# Two mechanical rules, applied in order:
#   1. flag `thread::sleep` only when it sits inside a loop whose body ALSO
#      breaks/returns on a success condition — pacing and one-shot waits
#      (no enclosing loop, or a loop with no break-on-success) are cleared by
#      construction. Approximated by scanning the ~8 lines immediately above
#      the sleep for a `for `/`while `/`loop` header AND a `break`/`return`/
#      `Ok(` — every real loop-shaped site here has both within that span.
#   2. within that same span, if the loop is a `TcpStream::connect` retry
#      AND the repo documents the exact readiness gap it retries around
#      (grepped once, repo-wide, for the phrases `src/listener/handle.rs`
#      itself uses: "readiness race", "poll-connect", "before its `serve`"),
#      it is the sanctioned workaround, not a finding.
say ""
say "examples: thread::sleep is a defect only when polling for a signal that exists"
BEFORE=$FINDINGS
GAP_DOCUMENTED=0
if grep -rqE 'readiness race|poll-connect|before its `serve`' src 2>/dev/null; then
  GAP_DOCUMENTED=1
fi
while IFS= read -r hit; do
  sleep_file=$(cut -d: -f1 <<< "$hit")
  sleep_line=$(cut -d: -f2 <<< "$hit")
  window_start=$((sleep_line - 8))
  [ "$window_start" -lt 1 ] && window_start=1
  window=$(sed -n "${window_start},${sleep_line}p" "$sleep_file")
  is_loop_with_break=0
  if grep -qE '^[[:space:]]*(for |while |loop\b)' <<< "$window" \
     && grep -qE '\b(break|return|Ok\()' <<< "$window"; then
    is_loop_with_break=1
  fi
  if [ "$is_loop_with_break" -eq 0 ]; then
    continue
  fi
  if [ "$GAP_DOCUMENTED" -eq 1 ] && grep -q 'TcpStream::connect' <<< "$window"; then
    continue
  fi
  finding "$sleep_file:$sleep_line — busy-wait with sleep inside a break-on-success loop; proxima awaits readiness without polling"
done < <(grep -rnE 'thread::sleep' --include='*.rs' "$EX" 2>/dev/null)
[ "$FINDINGS" -eq "$BEFORE" ] && ok "no std workarounds in examples"

# 3b. the library is held to the same bar as the examples — harder, in fact.
#     This check exists because `unsafe` and `Box<dyn Future>` both landed in the
#     pipe layer while the lint was only watching examples/. A check written
#     against the last instance instead of the invariant catches the last
#     instance.
say ""
say "pipe layer: no unsafe; a file claiming no-alloc must not allocate"
BEFORE=$FINDINGS
# unsafe anywhere in the pipe layer. Precise: this is the algebra's core.
while IFS= read -r hit; do finding "$hit"; done < <(
  grep -rn 'unsafe[[:space:]]*{' --include='*.rs' "$PIPE_DIR" 2>/dev/null | grep -v '^[^:]*:[0-9]*://'
)
# Box<dyn ..> is LEGITIMATE at the alloc tier for an open dyn set (PipeFactory,
# alloc_tier's erasure) — the rules say so. Flagging it everywhere is noise, and
# a noisy check is one nobody runs. So use each file's OWN claim as the lint: if
# a module doc says no-alloc, it may not box. The file convicts itself.
for f in "$PIPE_DIR"/*.rs; do
  head -40 "$f" | grep -qiE '^//!.*(no-alloc|no_alloc)' || continue
  # heapless::Vec is fixed-capacity and stack-allocated — it does not allocate.
  while IFS= read -r hit; do
    finding "$f:$hit — this file's own module doc claims no-alloc"
  done < <(grep -n 'Box<dyn\|Box::pin\|alloc::vec\|\.to_vec()' "$f" 2>/dev/null \
             | grep -vE '^[0-9]+:[[:space:]]*//|heapless')
done
[ "$FINDINGS" -eq "$BEFORE" ] && ok "no unsafe; no-alloc files honour their own claim"

# 4. blanket impls: an implicit bridge over an open set of foreign types is
#    surface nobody agreed to. One explicit opt-in adapter instead.
#
#    A first cut of this check flagged `impl<P> DynPipe<..> for P where P: Pipe,
#    ...` and `impl<P> SendDynPipe<..> for P where P: SendPipe, ...`
#    (alloc_tier.rs) as blanket impls. Both are correct code: the rule bans a
#    blanket over an OPEN, UNBOUNDED set of foreign types, and these are bounded
#    by a trait THIS WORKSPACE declares — a type only qualifies by already
#    implementing `Pipe`/`SendPipe`, so it has already opted in. This is also
#    the one place `ProximaError: From<P::Err>` converts (alloc_tier.rs's own
#    module doc), the erasure boundary that lets a `dyn` handle exist without a
#    per-pipe wrapper type. Deleting it would force exactly the wrapper-to-host
#    minting §1/§20 rule out — it is the mechanism that PREVENTS minting, not an
#    instance of it. So the check now reads the where-clause: a candidate is a
#    finding only when the target generic's bound is empty (unbounded) or names
#    no trait this workspace declares. ROOT_TRAITS is read from the pipe
#    layer's own trait declarations, not hand-copied, so a new root trait
#    (`UnpinPipe`, `UnpinSendPipe`, ...) is picked up automatically.
say ""
say "no blanket impls over an unbounded or foreign-only target"
BEFORE=$FINDINGS
ROOT_TRAITS=$(grep -hoE '^pub trait [A-Za-z_][A-Za-z0-9_]*' "$PIPE_DIR"/primitives.rs 2>/dev/null \
                | awk '{print $3}' | paste -sd'|' -)
while IFS= read -r hit; do
  file=${hit%%:*}
  rest=${hit#*:}
  start_line=${rest%%:*}
  header=${rest#*:}
  # impl<..., T, ...> SomeTrait for T   (bare generic param as the target)
  target=$(sed -E 's/.*for[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*/\1/' <<< "$header")
  generics=$(sed -E 's/.*impl<([^>]*)>.*/\1/' <<< "$header")
  if ! grep -qE "(^|[,[:space:]])${target}([,:[:space:]]|$)" <<< "$generics"; then
    continue
  fi
  # gather the where-clause: from this impl's header line up to its opening `{`
  block=$(awk -v start="$start_line" 'NR >= start { print; if ($0 ~ /\{/) exit }' "$file")
  # the bound directly on the target type, e.g. "P: Pipe," or "P: SendPipe,"
  own_bound=$(grep -oE "(^|[^A-Za-z0-9_])${target}[[:space:]]*:[[:space:]]*[A-Za-z_][A-Za-z0-9_:]*" <<< "$block" \
                | head -1 | sed -E 's/.*://; s/^[[:space:]]+|[[:space:]]+$//g' )
  bounded_by_ours=0
  if [ -n "$ROOT_TRAITS" ] && grep -qE "^($ROOT_TRAITS)\$" <<< "$own_bound"; then
    bounded_by_ours=1
  fi
  if [ "$bounded_by_ours" -eq 0 ]; then
    finding "$header — target's own bound is '${own_bound:-<none>}', not one of this workspace's root traits ($ROOT_TRAITS)"
  fi
done < <(grep -rnE '^impl<[^>]+>\s+[A-Za-z_][A-Za-z0-9_]*(<[^>]*>)?\s+for\s+[A-Z][A-Za-z0-9_]*\s*$' \
           --include='*.rs' "$PIPE_DIR" 2>/dev/null)
[ "$FINDINGS" -eq "$BEFORE" ] && ok "every blanket impl is bounded by one of this workspace's own root traits"

# 5. the generated-code tell.
say ""
say "no === banner === decorations"
BEFORE=$FINDINGS
while IFS= read -r hit; do finding "$hit"; done < <(
  grep -rn '=== .* ===' --include='*.rs' --include='*.md' --include='*.sh' \
    "$EX" "$PIPE_DIR" scripts 2>/dev/null | grep -vE 'frame.rs|algebra-lint.sh'
)
[ "$FINDINGS" -eq "$BEFORE" ] && ok "no banners"

# 6. a type minted only to host an impl. This audit found seven zero-sized
#    `PhantomData` structs whose entire purpose was carrying a `Pipe` impl for
#    a free function beside them — each with zero callers outside its own
#    module and test. They have since been deleted; this check is the
#    mechanical trap for the next one. All seven had the SAME shape:
#
#        pub fn parse_complete(input: &[u8]) -> Result<ParsedGguf, _>  // the job
#        pub struct ParseComplete<'a>(PhantomData<&'a [u8]>);          // the host
#        impl Pipe for ParseComplete<'a> { .. calls parse_complete .. }
#
#    Two ways to do one job, the second existing only to satisfy a trait.
#    PhantomData is not the smell — a PhantomData-only type is the standard,
#    correct shape for a zero-sized type-parameter carrier (`JsonCodec`,
#    `Convert`, below). A first cut of this check fired on ANY PhantomData-only
#    struct with a trait impl, and condemned `Convert<From, To>`
#    (proxima-tensor/src/convert.rs) — wrong: `Convert`'s per-dtype-pair
#    conversion bodies live directly in its `impl Pipe for Convert<From, To>`
#    blocks; there is no sibling `convert(from) -> to` free function it
#    wraps. The real discriminator is a SIBLING FREE FUNCTION performing the
#    same job the impl claims to: the struct is a host, not an implementation,
#    exactly when a `pub fn` doing its work already exists beside it.
#
#    Detecting "the impl body is essentially a call to a sibling fn" needs a
#    real body-vs-signature diff, which awk cannot do reliably. The applied
#    approximation: fire only when the struct's own file also declares a
#    `pub fn` whose name is the snake_case of the struct name (`ParseComplete`
#    -> `parse_complete`, `WriteComplete` -> `write_complete`, `Encode` ->
#    `encode`, `Decode` -> `decode` — all four deleted types matched this).
#    No such sibling, no finding: the impl carries its own logic.
#
#    Shape, unchanged: `struct Name(PhantomData<..>);` (or a `{ }` body whose
#    only fields are `PhantomData`) with a trait `impl ... for Name`
#    somewhere in the same file. Scoped to library crate source only —
#    `examples/`, `tests/`, and `benches/` build local fixtures (see
#    algebra-lint's own header on that split), and a struct inside an in-file
#    `#[cfg(test)]` module is a test fixture, not library surface (the awk
#    companion tracks that by brace depth).
#
# Allow-list: for the rare case the sibling-fn heuristic still over-fires —
# same shape as every other allow-list in this file, one line of cause each.
say ""
say "library: no type minted only to host a sibling free function's impl"
BEFORE=$FINDINGS
PHANTOM_AWK="$(dirname "$0")/algebra-lint-phantom-host.awk"
declare -a PHANTOM_ALLOW_FILE PHANTOM_ALLOW_NAME PHANTOM_ALLOW_REASON

phantom_is_allowed() {
  local file="$1" name="$2" index
  for index in "${!PHANTOM_ALLOW_FILE[@]}"; do
    if [ "${PHANTOM_ALLOW_FILE[$index]}" = "$file" ] && [ "${PHANTOM_ALLOW_NAME[$index]}" = "$name" ]; then
      return 0
    fi
  done
  return 1
}

while IFS= read -r hit; do
  hit_file=${hit%%:*}
  hit_rest=${hit#*:}
  hit_line=${hit_rest%%:*}
  hit_name=${hit_rest#*:}
  if ! grep -qE "^impl(<[^>]*>)?[[:space:]]+[A-Za-z_][A-Za-z0-9_]*(<[^>]*>)?[[:space:]]+for[[:space:]]+${hit_name}(<|[[:space:]]|$)" "$hit_file" 2>/dev/null; then
    continue
  fi
  sibling_fn=$(sed -E 's/([a-z0-9])([A-Z])/\1_\2/g' <<< "$hit_name" | tr '[:upper:]' '[:lower:]')
  if ! grep -qE "^pub(\([a-z]+\))?[[:space:]]+fn[[:space:]]+${sibling_fn}[[:space:]]*\(" "$hit_file" 2>/dev/null; then
    continue
  fi
  if phantom_is_allowed "$hit_file" "$hit_name"; then
    continue
  fi
  finding "$hit_file:$hit_line: $hit_name — hosts an impl that wraps sibling free fn '${sibling_fn}', not allow-listed"
done < <(
  find proxima-* prime rt -name '*.rs' 2>/dev/null \
    | grep -vE '/(examples|tests|benches|target)/' \
    | xargs -I{} awk -f "$PHANTOM_AWK" {} 2>/dev/null
)
[ "$FINDINGS" -eq "$BEFORE" ] && ok "no PhantomData-only struct wraps a sibling free function's job"

say ""
if [ "$FINDINGS" -gt 0 ]; then
  say "algebra-lint: $FINDINGS finding(s)"
  exit 1
fi
say "algebra-lint: clean"
