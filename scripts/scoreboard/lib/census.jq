def has_number_unit:
  test("(^|[^A-Za-z0-9_.])[0-9][0-9,]*(\\.[0-9]+)?\\s?(ms|µs|us|MB|GB|tok/s|s|x|%)([^A-Za-z0-9_]|$)");

def cell_count:
  sub("^\\s*\\|"; "") | sub("\\|\\s*$"; "") | [splits("(?<!\\\\)\\|")] | length;

def is_separator: test("^\\s*\\|?[\\s:|-]*-[\\s:|-]*\\|?\\s*$");

def cited($line): any($valid[]; . as $t | $line | contains($t));

[inputs] as $lines
| (reduce range(0; $lines | length) as $i ({fence: false, rows: []};
    ($lines[$i]) as $l
    | if ($l | test("^\\s*```")) then .fence = (.fence | not)
      elif .fence then .
      else .rows += [{n: ($i + 1), text: $l, kind: (if ($l | test("^\\s*\\|")) then "table" else "prose" end)}] end)
  | .rows) as $rows
| (reduce $rows[] as $r ({blocks: [], prev: -1};
    if $r.kind == "table" then
      if .prev == ($r.n - 1) then .blocks[-1] += [$r] else .blocks += [[$r]] end | .prev = $r.n
    else . end)
  | .blocks) as $blocks
| ($blocks | map(. as $b
    | ($b[0] | .text | cell_count) as $width
    | {header: $b[0],
       width: $width,
       data: ($b[1:] | if length > 0 and (.[0].text | is_separator) then .[1:] else . end)})) as $tables
| ($tables | map(.data[]) ) as $data_rows
| ($tables | map(. as $t | $t.data[] | select((.text | cell_count) != $t.width)
    | "defect: line \(.n): \(.text | cell_count) cells, header at line \($t.header.n) has \($t.width)")) as $row_defects
| ($blocks | map(select(length > 1 and (.[1].text | is_separator) and ((.[1].text | cell_count) != (.[0].text | cell_count)))
    | "defect: line \(.[1].n): separator has \(.[1].text | cell_count) cells, header at line \(.[0].n) has \(.[0].text | cell_count)")) as $sep_defects
| ($sep_defects + $row_defects) as $defects
| ($rows | map(select(.kind == "prose" and (.text | has_number_unit)))) as $prose
| ($data_rows | map(select(cited(.text) | not))) as $bad_rows
| ($prose | map(select(cited(.text) | not))) as $bad_prose
| ($defects[]),
  "tables=\($tables | length) data_rows=\($data_rows | length) prose_number_lines=\($prose | length) cited=\(($data_rows | length) - ($bad_rows | length) + ($prose | length) - ($bad_prose | length)) defects=\($defects | length)",
  ($bad_rows[] | "uncited row line \(.n)"),
  ($bad_prose[] | "uncited prose line \(.n)"),
  "uncited=\(($bad_rows | length) + ($bad_prose | length))"
