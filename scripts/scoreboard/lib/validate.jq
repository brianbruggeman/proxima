def isnum: type == "number";
def unreported_obj: type == "object" and has("unreported");

def typeok($t; $k):
  if $t == "number" then isnum
  elif $t == "string" then (type == "string" and length > 0)
  elif $t == "bool" then type == "boolean"
  elif $t == "utc" then (type == "string" and test($k.timestamp_pattern))
  elif $t == "array" then (type == "array" and length > 0 and all(.[]; isnum))
  else false end;

def field_problem($r; $f; $k):
  ($r | getpath($f.path | split("."))) as $v
  | "run \($r.index // "?") field \($f.path)" as $where
  | if $v == null then "\($where): missing"
    elif ($v | unreported_obj) then
      if ($f.unreportable | not) or $r.arm == $k.proxima_arm then
        "\($where): unreported is not allowed here (proxima arm or a field that must be measured)"
      elif ($v.unreported | IN($k.unreported_reasons[]) | not) then
        "\($where): unreported reason '\($v.unreported)' is not in the closed list \($k.unreported_reasons | join(","))"
      else empty end
    elif ($v | typeok($f.type; $k)) then empty
    else "\($where): not a finite \($f.type) (\($v | tojson))" end;

$c[0] as $k
| map(select((.record // "run") == "not-run")) as $nr
| map(select((.record // "run") == "run")) as $runs
| ($runs | map(.modality) | unique) as $mods
| ($mods | if length == 1 then $k.modalities[.[0]] else null end) as $mk
| ($runs | map(select(.admitted == true and .exit_status == 0))) as $ok
| ($runs | map(.backend) | unique) as $mb
| ($nr | map(.backend)) as $nb
| (
  if ($runs | length) == 0 then "runs: no run records in the file" else empty end,

  if ($mods | length) != 1 then "modality: expected exactly one, found \($mods | tojson)"
  elif $mk == null then "modality: '\($mods[0])' is not in the contract" else empty end,

  ($runs | map(.session_id) | unique) as $sessions
  | if ($sessions | length) > 1 then "session_id: \($sessions | length) distinct values in one file \($sessions | tojson)" else empty end,

  ($runs | map(.index) | map(select(isnum))) as $idx
  | ( if ($idx | length) != ($runs | length) then "index: a run record has no numeric index" else empty end,
      ($idx | group_by(.) | .[] | select(length > 1) | "index: \(.[0]) appears \(length) times"),
      (range(0; ($idx | max // -1) + 1) as $i
        | select($idx | any(. == $i) | not)
        | "index: run \($i) is missing, indices must be contiguous from 0") ),

  ($runs[] as $r | $k.required_always[] as $f | field_problem($r; $f; $k)),

  (if $mk != null then
     ($ok[] as $r
       | ($k.required_common[], $mk.required[]) as $f
       | field_problem($r; $f; $k))
   else empty end),

  ($runs[] | select(.started_at != null and .ended_at != null and .ended_at < .started_at)
    | "run \(.index) field ended_at: earlier than started_at"),

  ($runs[] | select(.exit_status == 0 and (.quality.match == false or .quality.match_incumbent == false))
    | "run \(.index) backend \(.backend) arm \(.arm): quality verdict failed (match=\(.quality.match), match_incumbent=\(.quality.match_incumbent))"),

  ($mb[] | select(IN($k.backends[]) | not) | "backend '\(.)': not in the contract list \($k.backends | join(","))"),

  ($nr[] | select(.backend | IN($k.backends[]) | not) | "not-run record: backend '\(.backend)' not in the contract list"),
  ($nr[] | select((.reason // "") | IN($k.not_run_reasons[]) | not)
    | "backend \(.backend): not-run reason '\(.reason // "")' is not in the closed list \($k.not_run_reasons | join(","))"),
  ($k.backends[] as $b
    | select(($b | IN($mb[])) and ($b | IN($nb[])))
    | "backend \($b): both measured and marked not-run"),
  ($k.backends[] as $b
    | select(($b | IN($mb[]) | not) and ($b | IN($nb[]) | not))
    | "backend \($b): neither measured nor covered by a not-run record"),

  ($mb[] | select(IN($k.backends[])) as $b
    | ($runs | map(select(.backend == $b))) as $br
    | ($br | map(.arm) | unique) as $arms
    | ( if ($arms | any(. == $k.proxima_arm)) | not then "backend \($b): arm \($k.proxima_arm) missing"
        elif ($arms | length) < 2 then "backend \($b): incumbent arm missing, only \($arms | tojson) present"
        elif ($arms | length) > 2 then "backend \($b): more than two arms \($arms | tojson)"
        else empty end,
        ($arms[] as $a
          | ($br | map(select(.arm == $a))) as $ar
          | ($ar | map(select(.repeat == 0 and (.discarded_warmup | not) and .admitted == true and .exit_status == 0))) as $good
          | ( if ($good | length) < $k.min_n_per_arm
                then "backend \($b) arm \($a): n=\($good | length) admitted processes, minimum \($k.min_n_per_arm)" else empty end,
              if ($ar | any(.discarded_warmup == true)) | not
                then "backend \($b) arm \($a): no discarded_warmup record" else empty end,
              if $mk != null and (($good | map(.input.id) | unique | length) < $k.min_inputs[$mods[0]])
                then "backend \($b) arm \($a): \($good | map(.input.id) | unique | length) distinct inputs, minimum \($k.min_inputs[$mods[0]])" else empty end )),
        ($br | map(select(.repeat == 0 and (.discarded_warmup | not))) | sort_by(.started_at)
          | . as $seq | range(1; length) as $i
          | select($seq[$i].arm == $seq[$i - 1].arm)
          | "backend \($b): arms do not alternate at run index \($seq[$i].index) (\($seq[$i].arm) follows \($seq[$i - 1].arm))") )
  )
)
