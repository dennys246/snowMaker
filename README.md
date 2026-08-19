# snowMaker
The snowMaker repo is the primary repository used for segmenting and preprocessing a custom snowpack dataset collected by Denny Schaedig in the Rocky Mountains. 

## Extended column test (ECT) intake

Snow stability is recorded per snow column via the Extended Column Test. The
schema — value domains, cross-field rules and the Hugging Face feature types —
lives in [`ect.py`](ect.py) and is the single place it is written down.

Drop a `site_ects.csv` next to `site_logs.csv` in a site's intake folder, one
row per snow column:

```csv
site,column,ect_result,ect_taps,ect_failure_depth_cm,ect_failure_grain,ect_hardness_above,ect_hardness_below,ect_operator
5,1,P,12,62.3,FC,1F,F,Denny Schaedig
5,2,X,,,,,,Denny Schaedig
```

- `ect_result` is `PV` / `P` / `N` / `X` — the letter only. Field notation
  `ECTP12` is recorded as `ect_result=P`, `ect_taps=12`.
- Leave every cell blank for a column that was not tested. **Blank is not `X`**:
  blank means the test was not run, `X` means it ran and nothing fractured in
  30 taps.
- Values must be identical for every row of the same `(site, column)`; a
  disagreement raises rather than quietly taking the first one.

Sites collected before the protocol simply have no `site_ects.csv`, and their
rows carry `null` across all seven columns. To add the columns to metadata
written before the schema landed:

```bash
python scripts/backfill_ect.py <dataset_dir> --check   # report only
python scripts/backfill_ect.py <dataset_dir>           # write
```

The backfill verifies that row counts and every pre-existing column value are
unchanged, and that nothing was filled with `"X"` or `0`.

Field procedure and the power analysis behind the encoding are in the
snowGradient repo at `docs/plans/ECT_PROTOCOL.md`.

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

It validates the ECT schema first, then compares the local copy against the Hub
and **refuses to upload if the Hub holds rows the local copy does not**, since
that would delete them. A stale working copy is the usual cause. Pass
`allow_shrink=True` only when rows are being deliberately withdrawn.

## Tests

```bash
python -m pytest
```
