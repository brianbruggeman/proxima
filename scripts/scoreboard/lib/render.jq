def fixed($v; $d): ($v * pow(10; $d) | round) as $r
  | (pow(10; $d)) as $p
  | ($r | if . < 0 then -. else . end) as $abs
  | (if $r < 0 then "-" else "" end)
    + ($abs / $p | floor | tostring)
    + (if $d > 0 then "." + ((($abs % $p) | tostring) | ("0" * ($d - length)) + .) else "" end);

def dashes($n): [range(0; $n) | "-"];

def unit_text($v; $unit):
  if $unit == "us" then fixed($v / 1000; 1) + " ms"
  elif $unit == "tok/s" then fixed($v; 1) + " tok/s"
  elif $unit == "bytes" then fixed($v / 1000000; 1) + " MB"
  elif $unit == "%" then fixed($v; 1) + "%"
  else fixed($v; 3) end;

def stat_cells($m; $k):
  if $m == null then dashes(($k.percentiles | length) + 3)
  elif $m.unreported != null then (["unreported: \($m.unreported)"] + dashes(($k.percentiles | length) + 2))
  else
    ($k.percentiles | map(. as $p
        | unit_text($m[$p]; $m.unit)
          + (if $k.separating_n[$p] != null and $m.n < $k.separating_n[$p] then " =max" else "" end)))
    + [unit_text($m.mean; $m.unit),
       (if ($m.cov_pct | type) == "number" then fixed($m.cov_pct; 2) + "%" else $m.cov_pct end),
       ($m.n | tostring)]
  end;

$c[0] as $k
| . as $s
| (["backend", "metric"]
   + ($k.percentiles | map("proxima " + .)) + ["proxima mean", "proxima cov", "proxima n"]
   + ($k.percentiles | map("incumbent " + .)) + ["incumbent mean", "incumbent cov", "incumbent n"]
   + ["ratio p50", "runs"]) as $header
| ("| " + ($header | join(" | "))),
  ("|" + ($header | map("---") | join("|"))),
  ($k.backends[] as $b
    | $s.backends[$b] as $bk
    | if $bk.status == "measured" then
        ($bk.arms.proxima.metrics | keys_unsorted[]) as $name
        | $bk.arms.proxima.metrics[$name] as $pm
        | $bk.arms.incumbent.metrics[$name] as $im
        | ($bk.ratio_p50[$name]) as $r
        | ([$b, ($name + (if $pm.derived == true then " (derived, not gated)" else "" end))]
           + stat_cells($pm; $k) + stat_cells($im; $k)
           + [(if $r == null then "-" else fixed($r; 3) + "x" end), $cite])
        | "| " + join(" | ")
      elif $bk.status == "not-run" then
        ([$b, "not-run: \($bk.reason)"] + dashes(($k.percentiles | length) * 2 + 7) + [$cite])
        | "| " + join(" | ")
      else
        ([$b, "missing"] + dashes(($k.percentiles | length) * 2 + 7) + [$cite])
        | "| " + join(" | ")
      end),
  ($s.not_applicable | to_entries[] as $e
    | ([$s.backends | to_entries[] | select(.value.status == "measured") | .key][] as $b
       | ([$b, $e.key, $e.value] + dashes(($k.percentiles | length) * 2 + 6) + [$cite])
       | "| " + join(" | ")))
