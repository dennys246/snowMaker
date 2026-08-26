# snowMaker
The snowMaker repo is the primary repository used for segmenting and preprocessing a custom snowpack dataset collected by Denny Schaedig in the Rocky Mountains. 

## Metadata tables

The dataset is one **image table** plus four **side tables**, one per grain. The
grain of a value is the file it is in, so it cannot be miscounted:

| file | one row per | key |
|---|---|---|
| `metadata/preprocessed.jsonl`, `metadata/raw.jsonl` | photograph | `file_path` (`site`, `column`, `core`, `segment`) |
| `metadata/pits.jsonl` | pit (site-day) | `site` |
| `metadata/layers.jsonl` | snow layer | `site`, `column`, `layer_index` |
| `metadata/cores.jsonl` | core, or unsampled ladder rung | `site`, `column`, `core` |
| `metadata/ect.jsonl` | ECT **fracture** | `site`, `column`, `test_index`, `fracture_index` |

Each side table is declared once, in its own module — [`pits.py`](pits.py),
[`layers.py`](layers.py), [`cores.py`](cores.py), [`ect.py`](ect.py) — as a
`TableSpec` of fields, domains and rules on the machinery in
[`schema.py`](schema.py). [`card.py`](card.py) renders the dataset card's YAML
front matter from those specs, so the card cannot drift from the code. The full
design, and the notice to snowGradient, is in
[`docs/SCHEMA_V2.md`](docs/SCHEMA_V2.md).

Rules that apply everywhere:

- **`null` means not measured.** It is never a default and never coalesced.
  Not-measured, measured-as-absent and measured-as-zero are three states.
- **An untested column has no row in `ect.jsonl`.** That is not the same as an
  `X` (tested, no fracture in 30 taps).
- **Out-of-domain values are rejected, never coerced.** Unknown columns are
  rejected too, so a typo cannot make a value vanish.
- **Values that must be constant within a key raise on disagreement**, never
  first-value-wins.
- **Raw fields only.** No stability score, no P/N boolean, no derived ordinal.

### Field intake

Drop the field-protocol CSVs next to `site_logs.csv` in a site's intake folder;
each is optional, and a site dug before the protocol simply has none:

```
intake/site_8/site_pits.csv     one row
intake/site_8/site_layers.csv   one row per layer
intake/site_8/site_cores.csv    one row per ladder rung, sampled or skipped
intake/site_8/site_ect.csv      one row per fracture
```

Columns are exactly the table's columns (`python -c "import ect; print(ect.SPEC.columns)"`).
Leave a cell blank for not measured. Booleans are `true`/`false`; `layer_ids`
is `3;4;5`; `coordinates` is `"39.79812, -105.77641"` at the receiver's full
precision (ingest flags fewer than four decimals and never rounds). Dates are
`YYYY-MM-DD`; `time_of_day` is `HH:MM+HH:MM`, local time with UTC offset.

```csv
site,column,test_index,fracture_index,ect_result,ect_taps,ect_tap_convention,ect_failure_depth_cm,ect_failure_grain,ect_hardness_above,ect_hardness_below,ect_fracture_character,ect_slope_angle_deg,ect_column_depth_cm,ect_suspected_pwl_below,ect_operator
8,1,1,1,N,8,initiating,25,FC,1F,4F,SC,31,90,false,collector-01
8,1,1,2,P,22,initiating,70,DH,P,F,SP,31,90,false,collector-01
8,2,1,1,X,,,,,,,,30,90,true,collector-01
```

Field notation `ECTP12` is `ect_result=P`, `ect_taps=12`. An `X` row keeps
`ect_column_depth_cm` and `ect_suspected_pwl_below` so a deep persistent slab
can be censored rather than read as stable.

`valve.intake()` validates and merges these into the master tables in
`intake/`; `valve.update_metadata()` writes the image tables and then the side
tables, checking every cross-table link (a photographed core needs a `cores`
row, a `layer_ids` entry needs its layer, every site needs a `pits` row) before
anything is written.

### Migration (2026-08-25)

The seven `ect_*` columns that sat on the image table, null on every row, moved
to `ect.jsonl`, and `pits.jsonl` / `cores.jsonl` were created from the pit- and
core-level values the image rows repeat:

```bash
python scripts/migrate_side_tables.py <dataset_dir> --check   # report only
python scripts/migrate_side_tables.py <dataset_dir>           # write
```

The script proves the image tables lost nothing but the seven null columns and
that every field the protocol adds is `null` on legacy rows — never `"X"`,
`0`, `""` or `false`. Re-running is safe; it never overwrites a side table.

`layers.jsonl` and `ect.jsonl` are **N/A — not assessed** until the first pit
of the 2026–27 season: present, empty, documented on the card, but not declared
as `load_dataset()` configs (`datasets` refuses an empty config). When the first
row lands, regenerate the card's front matter and paste it in:

```bash
.venv/bin/python -c "import card; print(card.front_matter_yaml(card.tables_with_rows('<dataset_dir>')))"
```

## Publishing to the Hub

`intake.upload_metadata(dataset_dir)` uploads only `metadata/*.jsonl` and
`README.md` — the images already on the Hub are untouched, so it is a few
megabytes rather than the ~96 GB the full repo weighs. It defaults to a dry run;
`dry_run=False` is the explicit confirmation to write.

```python
import intake
intake.upload_metadata("path/to/rocky_mountain_snowpack")                  # dry run
intake.upload_metadata("path/to/rocky_mountain_snowpack", dry_run=False)   # push
```

It validates every table and every cross-table link first, then compares the
local copy against the Hub and **refuses to upload if the Hub holds rows the
local copy does not**, since that would delete them. A stale working copy is the
usual cause. Pass `allow_shrink=True` only when rows are being deliberately
withdrawn.

## Tests

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

The round-trip tests write one row of each ECT class, a multi-fracture test, a
repeat test, a full layer profile, every core recovery state and an untested
column, then read them back through both `load_dataset()` and a direct JSONL
read and assert both match the card.
