def unreported_obj: type == "object" and has("unreported");

# nearest rank s[ceil(p*n)-1], with p an integer percent so no float rounding reaches ceil
def nearest_rank($sorted; $pct):
  ($sorted | length) as $n
  | $sorted[((($pct * $n) + 99) / 100 | floor) - 1];

def sample_cov($v):
  ($v | length) as $n
  | ($v | add / $n) as $mean
  | if $n < 2 then "n/a: n<2"
    elif $mean == 0 then "n/a: zero mean"
    else (($v | map(. - $mean | . * .) | add) / ($n - 1) | sqrt) / $mean * 100 end;

def stats($v; $k):
  ($v | sort) as $s
  | ($s | length) as $n
  | ($k.percentiles | map({key: ., value: nearest_rank($s; (ltrimstr("p") | tonumber))}) | from_entries)
    + {mean: ($s | add / $n), cov_pct: sample_cov($s), n: $n};

def lift($xs; f): ($xs | map(select(unreported_obj)) | first) // ($xs | f);

def rawval($r; $base; $stage):
  if $stage != null then $r.timings[$stage + "_us"]
  elif $base == "ttft_us" or $base == "ttlt_us" then $r.timings[$base]
  elif $base == "footprint_peak_bytes" or $base == "steady_rss_bytes" or $base == "gpu_allocated_peak_bytes"
    then $r.resources[$base]
  elif $base == "cpu_pct" then
    ($r.resources) as $x
    | if ($x.wall_us // 0) > 0 then ($x.cpu_user_us + $x.cpu_sys_us) / $x.wall_us * 100 else null end
  elif $base == "decode_tok_s" then
    lift([$r.timings.ttft_us, $r.timings.ttlt_us]; (.[1] - .[0]) as $dt
      | if $dt > 0 and $r.generated_tokens != null then ($r.generated_tokens - 1) / ($dt / 1000000) else null end)
  elif $base == "prefill_tok_s" then
    lift([$r.timings.ttft_us]; if .[0] > 0 and $r.prompt_tokens != null then $r.prompt_tokens / (.[0] / 1000000) else null end)
  elif $base == "rtf" then
    lift([$r.timings.ttlt_us, $r.timings.load_us]; (($r.input.duration_us // 0)) as $d
      | if $d > 0 then (.[0] - .[1]) / $d else null end)
  else null end;

def specs($k; $mod):
  $k.modalities[$mod] as $mk
  | ([$k.metrics | to_entries[]
        | select(.value.modalities | any(. == $mod))
        | .key as $base | .value as $m
        | ($m.temps | map(select(. as $t | $t == "all" or ($mk.temps | any(. == $t))))) as $temps
        | $temps[] as $t
        | {name: (if ($temps | length) > 1 then "\($base)_\($t)" else $base end),
           base: $base, stage: null, temp: $t, unit: $m.unit, derived: ($m.derived // false)}]
     + [$mk.stages[] as $st | $mk.temps[] as $t
        | {name: (if ($mk.temps | length) > 1 then "\($st)_us_\($t)" else "\($st)_us" end),
           base: null, stage: $st, temp: $t, unit: "us", derived: false}]);

def of_temp($recs; $t):
  if $t == "cold" then $recs | map(select(.repeat == 0))
  elif $t == "warm" then $recs | map(select(.repeat > 0))
  else $recs end;

def metric_values($recs; $spec):
  of_temp($recs; $spec.temp)
  | if $spec.base == "step_us"
    then map(.timings.step_us | if type == "array" then .[] else . end)
    else map(rawval(.; $spec.base; $spec.stage)) end;

def metric_result($vals; $spec; $where; $k):
  ($vals | map(select(unreported_obj))) as $un
  | ($vals | map(select(type == "number"))) as $nums
  | if ($vals | length) == 0 then {error: "\($where) metric \($spec.name): no values"}
    elif ($vals | length) != ($un | length) + ($nums | length)
      then {error: "\($where) metric \($spec.name): \(($vals | length) - ($un | length) - ($nums | length)) records have no value"}
    elif ($un | length) == ($vals | length) then {unreported: $un[0].unreported, unit: $spec.unit}
    elif ($un | length) > 0 then {error: "\($where) metric \($spec.name): mix of measured and unreported values"}
    else stats($nums; $k) + {unit: $spec.unit, derived: $spec.derived} end;

def arm_key($a; $k): if $a == $k.proxima_arm then "proxima" else "incumbent" end;

def ratios($arms):
  if $arms.proxima != null and $arms.incumbent != null then
    $arms.proxima.metrics | to_entries
    | map(. as $e
        | ($arms.incumbent.metrics[$e.key]) as $inc
        | {key: $e.key,
           value: (if ($e.value.p50? != null) and ($inc.p50? != null) and $inc.p50 != 0
                   then $e.value.p50 / $inc.p50 else null end)})
    | from_entries
  else {} end;

def measured($b; $br; $specs; $k):
  ($br | map(.arm) | unique | map(. as $a
    | "backend \($b) arm \($a)" as $where
    | ($br | map(select(.arm == $a))) as $ar
    | {key: arm_key($a; $k),
       value: {name: $a,
               metrics: ($specs | map(. as $s | {key: $s.name, value: metric_result(metric_values($ar; $s); $s; $where; $k)}) | from_entries)}})
   | from_entries) as $arms
  | {status: "measured", arms: $arms, ratio_p50: ratios($arms)};

$c[0] as $k
| map(select((.record // "run") == "not-run")) as $nr
| map(select((.record // "run") == "run")) as $runs
| if ($runs | length) == 0 then ("no run records in the file\n" | halt_error(1)) else . end
| ($runs | map(.modality) | unique) as $mods
| if ($mods | length) != 1 or ($k.modalities[$mods[0]] == null)
    then ("modality: expected exactly one contract modality, found \($mods | tojson)\n" | halt_error(1)) else . end
| $mods[0] as $mod
| specs($k; $mod) as $specs
| ($runs | map(select(.admitted == true and .exit_status == 0 and (.discarded_warmup | not)))) as $good
| {
    contract_version: $k.version,
    modality: $mod,
    model: $runs[0].model,
    session_id: $runs[0].session_id,
    host: $runs[0].host,
    not_applicable: ([$k.metrics | to_entries[] | select(.value.modalities | any(. == $mod) | not) | {key: .key, value: "n/a: \($mod)"}] | from_entries),
    backends: ($k.backends | map(. as $b
      | ($good | map(select(.backend == $b))) as $br
      | ($nr | map(select(.backend == $b)) | first) as $n
      | if ($br | length) > 0 then {key: $b, value: measured($b; $br; $specs; $k)}
        elif $n != null then {key: $b, value: {status: "not-run", reason: $n.reason, detail: $n.detail}}
        elif ($runs | any(.backend == $b)) then {key: $b, value: {status: "measured-none-admitted"}}
        else {key: $b, value: {status: "missing"}} end) | from_entries)
  }
| ([.. | objects | select(has("error")) | .error]
   + [.backends | to_entries[] | select(.value.status == "measured-none-admitted") | "backend \(.key): runs exist but none are admitted, non-warmup, exit 0"]) as $errs
| if ($errs | length) > 0 then (($errs | join("\n")) + "\n" | halt_error(1)) else . end

