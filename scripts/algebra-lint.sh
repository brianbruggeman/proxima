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
check_ex 'thread::sleep' 'busy-wait with sleep; proxima awaits readiness without polling'
check_ex 'futures::executor::block_on' "drives proxima's app with futures' executor; use #[proxima::main] and .await"
check_ex 'env::set_var' 'sets a global env var to configure proxima; use config or pass it explicitly'
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
say ""
say "no blanket impls"
BEFORE=$FINDINGS
while IFS= read -r hit; do
  # impl<..., T, ...> SomeTrait for T   (bare generic param as the target)
  target=$(sed -E 's/.*for[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*/\1/' <<< "$hit")
  generics=$(sed -E 's/.*impl<([^>]*)>.*/\1/' <<< "$hit")
  if grep -qE "(^|[,[:space:]])${target}([,:[:space:]]|$)" <<< "$generics"; then
    finding "$hit"
  fi
done < <(grep -rnE '^impl<[^>]+>\s+[A-Za-z_][A-Za-z0-9_]*(<[^>]*>)?\s+for\s+[A-Z][A-Za-z0-9_]*\s*$' \
           --include='*.rs' "$PIPE_DIR" 2>/dev/null)
[ "$FINDINGS" -eq "$BEFORE" ] && ok "no blanket impls in the pipe layer"

# 5. the generated-code tell.
say ""
say "no === banner === decorations"
BEFORE=$FINDINGS
while IFS= read -r hit; do finding "$hit"; done < <(
  grep -rn '=== .* ===' --include='*.rs' --include='*.md' --include='*.sh' \
    "$EX" "$PIPE_DIR" scripts 2>/dev/null | grep -vE 'frame.rs|algebra-lint.sh'
)
[ "$FINDINGS" -eq "$BEFORE" ] && ok "no banners"

say ""
if [ "$FINDINGS" -gt 0 ]; then
  say "algebra-lint: $FINDINGS finding(s)"
  exit 1
fi
say "algebra-lint: clean"
