# rmdig metadata schema v2 — side tables

Status: **implemented in snowMaker, applied to the local dataset clone, decisions
confirmed 2026-08-25; Hub push pending.** This document is the notice to
snowGradient of what changes.

## Why

snowGradient's post-mortem on the breakability campaign named the causes:

1. "Breakability" is within-layer cohesion at a depth (the corer enters the
   wall horizontally), not a weak-layer count, and it is dominated by depth.
2. `collector` had one distinct value across all 845 core rows.
3. Site 0's "advised not to use" lived in free text and was used anyway.
4. `segment_count` is a photography artefact, not a field count.
5. Everything coarser than an image was repeated across image rows, so the
   grain of a value was out-of-band knowledge and n was inflatable by ~100x.

Two new requirements — an 8–16 record layer profile per column, and ECTs that
commonly produce several fractures — do not fit a repeated-column image table
at all. The seven `ect_*` columns added on 2026-08-19 were still 100% null, so
moving them was free; it becomes a breaking migration with the first real test.

## Layout

```
metadata/preprocessed.jsonl   image rows — unchanged except ect_* removed
metadata/raw.jsonl            image rows — unchanged except ect_* removed
metadata/pits.jsonl           one row per site-day                 key (site)
metadata/layers.jsonl         one row per snow layer               key (site, column, layer_index)
metadata/cores.jsonl          one row per core / ladder rung       key (site, column, core)
metadata/ect.jsonl            one row per ECT fracture             key (site, column, test_index, fracture_index)
```

Each is a Hugging Face config: `load_dataset("rmdig/rocky_mountain_snowpack",
"pits")` etc. The image table stays config `default` with splits `train`
(preprocessed) and `raw`. Every side-table column is `Value`/`List`, never
`ClassLabel`.

Columns, domains and rules are the single source of truth in the snowMaker
modules `pits.py`, `layers.py`, `cores.py`, `ect.py`; the dataset card's YAML
is generated from them by `card.py`. The card body documents every column, its
grain, and what `load_dataset()` returns versus a direct JSONL read.

## What changes for snowGradient — read before pulling

### Breaking: `ect_*` columns are gone from the image table

`snowgradient.ect` (keyed to the 2026-08-19 layout) will find no `ect_result`
… `ect_operator` on `dataset["train"]`. They are in the `ect` config, at
fracture grain, with these differences from the old columns:

| old (image table, per (site, column)) | new (`ect.jsonl`, per fracture) |
|---|---|
| seven columns, null = not tested | **no row** = not tested |
| one result per column | `test_index` and `fracture_index`; N8 at 25 cm then P22 at 70 cm is two rows of one test |
| `ect_operator` nullable | required |
| — | `ect_tap_convention` (`initiating` / `propagating`), required with `ect_taps` |
| — | `ect_fracture_character`, `ect_slope_angle_deg`, `ect_column_depth_cm`, `ect_suspected_pwl_below` |
| `X` forbids every other field | `X` permits `ect_column_depth_cm` and `ect_suspected_pwl_below` (right-censoring) |

To reproduce "one ECT per column" from the new table: group by
`(site, column, test_index)`. To count tested columns: distinct `(site, column)`
in `ect.jsonl`. To find untested columns: `(site, column)` pairs present in
`cores.jsonl` and absent from `ect.jsonl`.

### Non-breaking, but relevant

- **Grain is the file.** Do not count image rows as observations of anything
  coarser than an image. `pits.jsonl` has one row per pit; join on `site`.
- **`method_deviation`** is the structured replacement for site 0's free-text
  warning. Filter on it. Site 0 is `true`.
- **`pits.coordinates` is float64** and carries full source precision
  (site 0 remains 2-decimal as recorded). The image table's `coordinates`
  stays float32 — unchanged, because changing a published type is breaking.
- **`pits.date` is `date32`**: `load_dataset()` gives `datetime.date`, the
  JSONL gives `"YYYY-MM-DD"`. It is not `string` because pyarrow infers ISO
  dates as timestamps before the cast and would hand you `"2026-12-20 00:00:00"`.
- **`cores.core_temperature_c` is degrees C**; the image table's
  `core_temperature` stays degrees F (unchanged). 10 of 38 legacy readings are
  above 0 °C and are flagged as suspect on ingest, not removed.
- **`layers.jsonl` and `ect.jsonl` are N/A — not assessed** until the first
  pit of the 2026–27 season. The files exist and are empty (direct read: zero
  rows); their schemas are documented and enforced, but they are **not
  declared as configs** until they hold a row, because `datasets` refuses an
  empty config. `load_dataset(..., "layers")` raises (unknown config) until
  then. The intake run that writes the first row adds the config.
- **Image-table types are untouched.** `datatype` and `wind_loading` are still
  `ClassLabel` (ints from `load_dataset()`, strings in the JSONL). The
  `datatype.isin(["core", 0])` filter downstream remains dead-but-harmless;
  nothing here flips the JSONL back to ints.

## Semantics that are easy to get wrong

- `null` is *not measured*. `[]` is rejected. `""` is rejected. `0` is a value.
- `cores.depth_skipped_reason` non-null means the rung was **not sampled**; such
  a row has no count, no recovery, no temperature, and may have no image rows.
- `recovery_quality`: `intact` / `partial` carry a `breakability_field_count`;
  `disintegrated` (could not be counted) and `not_recovered` carry none.
  Legacy cores have `null` — not recorded — which is none of the four.
- `breakability_field_count` is counted at extraction. It is **not** derived
  from `segment`, and nothing in the pipeline derives it.
- ECT `PV` has no tap. `P`/`N` need `ect_taps` in 1–30 and a tap convention.
  `X` is one row and the only row of its test.

## Validation (enforced on ingest, on write and before upload)

Per record: domains, nullability, cross-field rules above. Per table: unique
keys; per-test constancy in `ect`; layer contiguity (overlap = error, gap =
warning). Cross-table: every `layer_ids` entry exists in the same column; every
image row's site has a `pits` row and its core a `cores` row; skipped rungs
have no images; the pit-level values the image table repeats agree with
`pits.jsonl`.

## Decisions (confirmed by Denny, 2026-08-25)

1. `method_deviation` for legacy sites 1–6 is `false`: the pilot (site 0)
   established the standard method and no later note records a deviation.
2. `collector_id` for legacy sites is the name string `"Denny Schaedig"`
   verbatim. The AvApp should emit the same string for Denny.
3. `aspect_deg_true` is null on legacy pits. The image table's `slope_face`
   holds a compass reading whose true/magnetic convention was not recorded, so
   it was not copied.
4. `layers` and `ect` are **N/A — not assessed**: documented on the card,
   files present but empty, not declared as `load_dataset()` configs until the
   first record lands (`card.tables_with_rows()` decides which configs the
   front matter declares).
