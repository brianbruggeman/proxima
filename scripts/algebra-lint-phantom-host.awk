# algebra-lint-phantom-host.awk — finds a struct whose only fields are
# PhantomData, defined outside any `#[cfg(test)]` module. algebra-lint.sh
# then checks each hit for a trait `impl ... for` block, which is the shape
# that was found and deleted: seven such structs existed only to carry a
# `Pipe` impl for a free function beside them, with zero callers outside
# their own module and test.
#
# cfg(test) tracking is brace-depth based, mirroring nostd-gate.awk: a
# `#[cfg(test)]` line followed by a `mod` line opens a test region that lasts
# until that mod's closing brace, so struct defs inside `mod tests { .. }`
# are never candidates (a test fixture named `Identity<Value>` is not a
# library type).
BEGIN { depth = 0; test_depth = -1; pending_test = 0; collecting = 0 }
{
  line = $0
  in_test = (test_depth != -1 && depth > test_depth)

  if (!collecting && !in_test &&
      match(line, /^[[:space:]]*pub(\([a-z]+\))?[[:space:]]+struct[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/)) {
    name = line
    sub(/^[[:space:]]*pub(\([a-z]+\))?[[:space:]]+struct[[:space:]]+/, "", name)
    sub(/[<({;[:space:]].*/, "", name)
    if (line ~ /;[[:space:]]*$/) {
      check_candidate(name, line, FNR)
    } else if (line ~ /\{[[:space:]]*$/) {
      collecting = 1
      collect_name = name
      collect_start = FNR
      collect_body = ""
      next
    }
  }
  if (collecting) {
    collect_body = collect_body "\n" line
    if (line ~ /^\}/) {
      check_candidate(collect_name, collect_body, collect_start)
      collecting = 0
    }
    next
  }

  if (line ~ /^[[:space:]]*#\[cfg\(test\)\][[:space:]]*$/) {
    pending_test = 1
  } else if (pending_test && line ~ /mod[[:space:]]+[A-Za-z_]/) {
    test_depth = depth
    pending_test = 0
  } else if (line !~ /^[[:space:]]*#\[/) {
    pending_test = 0
  }

  opens = gsub(/\{/, "{", line)
  closes = gsub(/\}/, "}", line)
  depth += opens - closes
  if (test_depth != -1 && depth <= test_depth) test_depth = -1
}
function check_candidate(name, body, start_line,   fieldcount, ok, n, i, field) {
  n = split(body, arr, "\n")
  fieldcount = 0
  ok = 1
  for (i = 1; i <= n; i++) {
    field = arr[i]
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", field)
    if (field == "" || field == "}" || field ~ /^\/\// || field ~ /^#\[/) continue
    if (field ~ /^pub(\([a-z]+\))?[[:space:]]+struct/) {
      if (field !~ /PhantomData/) ok = 0
      continue
    }
    fieldcount++
    if (field !~ /PhantomData/) ok = 0
  }
  if (ok && (fieldcount >= 1 || body ~ /PhantomData/)) {
    printf "%s:%d:%s\n", FILENAME, start_line, name
  }
}
