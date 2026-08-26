"""
Move the Rocky Mountain snowpack metadata to the side-table layout.

Before 2026-08-25 every value coarser than an image was repeated across image
rows, and seven ``ect_*`` columns sat on the image table at ``(site, column)``
grain -- null on every row. This script:

1. Removes the seven ``ect_*`` columns from ``raw.jsonl`` and
   ``preprocessed.jsonl``, refusing if any of them holds a value (a real test
   would need ``test_index``/``fracture_index`` and must be moved by hand).
2. Creates ``pits.jsonl`` -- one row per site, from the pit-level values the
   image rows already repeat (coordinates, date, time, depth, slope angle,
   collector), plus the structured ``method_deviation`` flag.
3. Creates ``cores.jsonl`` -- one row per (site, column, core) the image rows
   reference, carrying the ladder depth the pipeline already assigned and the
   site_temps reading in degrees C.
4. Creates ``layers.jsonl`` and ``ect.jsonl`` empty: no layer profile and no
   ECT has been recorded yet, and an absent row is how "not measured" is
   written at those grains.

Every field the field protocol adds is backfilled to ``null`` -- never ``"X"``,
``0``, ``""`` or ``false`` -- and the script asserts that, along with proving the
image tables lost nothing but the seven null columns: row counts and every
other value are compared before and after, and a mismatch aborts before
anything is written.

Two legacy values are set from the record rather than left null, because they
are required and the record is unambiguous:

``method_deviation``
    Site 0's notes say, verbatim, "Pilot using slightly modified coring method
    that took too large segment for snow profile picturing, advised not to use
    in higher level models" -- a recorded deviation, so site 0 is ``true`` with
    that sentence as its note. Sites 1-6 were dug with the standard method that
    the pilot established and their notes record no deviation, so they are
    ``false`` (measured as absent, not unmeasured).

``collector_id``
    The image table's ``collector`` value, verbatim.

``date`` / ``time_of_day`` are re-expressed as ISO 8601 (``1/12/25`` ->
``2025-01-12``; ``11:40 AM MST`` -> ``11:40-07:00``). ``core_temperature_c``
is the site_temps reading converted from degrees F, rounded to one decimal --
every legacy reading is an integer number of degrees C written down in F, so
this recovers what was read, not more digits than the instrument had.

Usage:

    python scripts/migrate_side_tables.py <dataset_dir> --check   # report only
    python scripts/migrate_side_tables.py <dataset_dir>           # write

Re-running is safe: image tables already migrated are left alone, and an
existing side table is verified to contain every row this script would create
(so rows collected since are never clobbered) and otherwise left untouched.
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import schema
import pits, layers, cores, ect
from intake import IMAGE_TABLES, SIDE_TABLES, strip_legacy_ect


# Sites whose notes record a deviation from the standard method, with the note
# verbatim. Every other legacy site was dug with the standard method.
LEGACY_METHOD_DEVIATIONS = {
    0: (
        "Pilot using slightly modified coring method that took too large segment "
        "for snow profile picturing, advised not to use in higher level models."
    ),
}

# The only zones the legacy 'time' column ever used. Anything else is an error,
# not a guess.
LEGACY_UTC_OFFSETS = {"MST": "-07:00", "MDT": "-06:00"}

# Values that would silently misrepresent an unmeasured field if a backfill ever
# produced them in place of null.
FORBIDDEN_BACKFILL_VALUES = ("X", 0, 0.0, "", "null", "None", False, [])


class MigrationError(RuntimeError):
    """Raised when the migration would change something it must not."""


# --- Legacy value conversion --------------------------------------------------

def legacy_date(text):
    """``1/12/25`` (US month/day/two-digit year) -> ``2025-01-12``."""
    try:
        return datetime.strptime(text.strip(), "%m/%d/%y").strftime("%Y-%m-%d")
    except ValueError:
        raise MigrationError(f"legacy date {text!r} is not M/D/YY") from None


def legacy_time(text):
    """``11:40 AM MST`` -> ``11:40-07:00``."""
    parts = text.strip().split()
    if len(parts) != 3 or parts[2] not in LEGACY_UTC_OFFSETS:
        raise MigrationError(
            f"legacy time {text!r} is not 'H:MM AM|PM MST|MDT'; add its zone to "
            f"LEGACY_UTC_OFFSETS if it is a real zone"
        )
    try:
        clock = datetime.strptime(f"{parts[0]} {parts[1]}", "%I:%M %p").strftime("%H:%M")
    except ValueError:
        raise MigrationError(f"legacy time {text!r} is not 'H:MM AM|PM'") from None
    return f"{clock}{LEGACY_UTC_OFFSETS[parts[2]]}"


def constant(rows, column, where):
    """The single value ``column`` takes across ``rows``, or an error."""
    distinct = {json.dumps(row[column], sort_keys = True) for row in rows}
    if len(distinct) != 1:
        raise MigrationError(
            f"{where}: {column} is not constant across its image rows: {sorted(distinct)}"
        )
    return rows[0][column]


# --- Building the side tables -------------------------------------------------

def build_pits(image_rows):
    """One pits row per site, from the values the image rows repeat."""
    by_site = {}
    for row in image_rows:
        by_site.setdefault(row["site"], []).append(row)

    records = []
    for site, rows in sorted(by_site.items()):
        where = f"site {site}"
        note = LEGACY_METHOD_DEVIATIONS.get(site)
        record = {
            "site": site,
            "collector_id": constant(rows, "collector", where),
            "corer_id": None,
            "coordinates": constant(rows, "coordinates", where),
            "aspect_deg_true": None,
            "slope_angle_deg": float(constant(rows, "slope_angle", where)),
            "date": legacy_date(constant(rows, "date", where)),
            "time_of_day": legacy_time(constant(rows, "time", where)),
            "ground_cover": None,
            "snowpack_depth_cm": float(constant(rows, "snowpack_depth", where)),
            "method_deviation": note is not None,
            "method_deviation_note": note,
        }
        records.append(pits.SPEC.validate(record, context = f"pits {where}: "))
    return records


def build_cores(image_rows):
    """One cores row per (site, column, core) the image rows reference."""
    by_core = {}
    for row in image_rows:
        by_core.setdefault((row["site"], row["column"], row["core"]), []).append(row)

    records = []
    for (site, column, core), rows in sorted(by_core.items()):
        where = f"core ({site}, {column}, {core})"
        depth = float(constant(rows, "core_depth", where))
        reading = constant(rows, "core_temperature", where)
        record = cores.legacy_core_row(site, column, core, reading)
        if record["core_depth_cm"] != depth:
            raise MigrationError(
                f"{where}: image rows say core_depth={depth} but the ladder gives "
                f"{record['core_depth_cm']}"
            )
        records.append(record)
    return records


# --- Checks -------------------------------------------------------------------

def fingerprint(row):
    """A stable snapshot of every non-ECT field, for the unchanged-values check."""
    return json.dumps(
        {key: value for key, value in row.items() if key not in ect.LEGACY_IMAGE_COLUMNS},
        sort_keys = True,
    )


def is_forbidden(value):
    """Would this value silently misrepresent an unmeasured field?"""
    for forbidden in FORBIDDEN_BACKFILL_VALUES:
        # Compare by type as well, so 0 does not match False by accident
        if type(value) is type(forbidden) and value == forbidden:
            return True
    return False


def check_backfilled_nulls(spec, records, must_be_null):
    """Every field the protocol adds is null on every legacy row, or the run aborts."""
    for record in records:
        for column in must_be_null(record):
            value = record[column]
            if is_forbidden(value):
                raise MigrationError(
                    f"{spec.name} {dict(zip(spec.key, spec.key_of(record)))}: backfilled "
                    f"{column} to {value!r}, which reads as a measurement; an "
                    f"unmeasured field is null"
                )
            if value is not None:
                raise MigrationError(
                    f"{spec.name} {dict(zip(spec.key, spec.key_of(record)))}: backfilled "
                    f"{column} to {value!r}; an unmeasured field must be null"
                )


def pit_nulls(record):
    fields = ["corer_id", "aspect_deg_true", "ground_cover"]
    if record["site"] not in LEGACY_METHOD_DEVIATIONS:
        fields.append("method_deviation_note")
    return fields


def core_nulls(record):
    return ["depth_skipped_reason", "breakability_field_count", "recovery_quality", "layer_ids"]


def check_side_table_file(spec, path, built):
    """
    Reconcile an existing side-table file with what this run would build.

    Returns ``"absent"``, ``"identical"`` or ``"superset"``. Raises if the file is
    missing a row this run builds or holds a different version of one -- the
    file is never overwritten.
    """
    if not os.path.exists(path):
        return "absent"
    existing = {spec.key_of(row): row for row in schema.read_jsonl(path)}
    for record in built:
        key = spec.key_of(record)
        if key not in existing:
            raise MigrationError(
                f"{path} exists but lacks {dict(zip(spec.key, key))}; it will not be "
                f"overwritten -- reconcile by hand"
            )
        if existing[key] != record:
            raise MigrationError(
                f"{path} holds a different {dict(zip(spec.key, key))} than this "
                f"migration builds; it will not be overwritten.\n"
                f"  file:  {existing[key]}\n  built: {record}"
            )
    return "identical" if len(existing) == len(built) else "superset"


# --- Main ---------------------------------------------------------------------

def migrate(dataset_dir, check_only = False, log = print):
    """
    Run the migration. Returns a report dict; raises MigrationError/SchemaError
    before writing anything if any check fails.
    """
    if not dataset_dir.endswith("/"):
        dataset_dir += "/"
    metadata_dir = f"{dataset_dir}metadata/"

    # 1. Image tables: strip the null ect_* columns and prove nothing else moved
    image_tables = {}
    image_reports = {}
    for name in IMAGE_TABLES:
        path = f"{metadata_dir}{name}.jsonl"
        if not os.path.exists(path):
            raise MigrationError(f"{path} not found")
        before = schema.read_jsonl(path)
        before_prints = [fingerprint(row) for row in before]
        carrying = sum(1 for row in before if any(c in row for c in ect.LEGACY_IMAGE_COLUMNS))

        after = [strip_legacy_ect(dict(row), context = f"{name}.jsonl row {i}: ") for i, row in enumerate(before)]

        if len(after) != len(before):
            raise MigrationError(f"{path}: row count changed from {len(before)} to {len(after)}")
        for position, (original, stripped) in enumerate(zip(before_prints, after)):
            if fingerprint(stripped) != original:
                raise MigrationError(
                    f"{path} row {position}: a non-ECT value changed.\n"
                    f"  before: {original}\n  after:  {fingerprint(stripped)}"
                )
        for row in after:
            leftover = [c for c in ect.LEGACY_IMAGE_COLUMNS if c in row]
            if leftover:
                raise MigrationError(f"{path}: {leftover} still present after strip")

        image_tables[name] = after
        image_reports[name] = {"rows": len(after), "stripped": carrying}

    all_image_rows = [row for rows in image_tables.values() for row in rows]

    # 2-4. Side tables
    built = {
        "pits": build_pits(all_image_rows),
        "layers": [],
        "cores": build_cores(all_image_rows),
        "ect": [],
    }
    check_backfilled_nulls(pits.SPEC, built["pits"], pit_nulls)
    check_backfilled_nulls(cores.SPEC, built["cores"], core_nulls)
    for record in built["cores"]:
        reading = constant(
            [r for r in all_image_rows if (r["site"], r["column"], r["core"]) == cores.SPEC.key_of(record)],
            "core_temperature", "core",
        )
        if (reading is None) != (record["core_temperature_c"] is None):
            raise MigrationError(
                f"cores {cores.SPEC.key_of(record)}: temperature presence changed "
                f"({reading!r} -> {record['core_temperature_c']!r})"
            )

    warnings = []
    for spec in SIDE_TABLES:
        warnings.extend(spec.validate_table(built[spec.name], context = f"{spec.name}: "))
    warnings.extend(schema.validate_dataset(built, image_rows = all_image_rows, context = "migration: "))

    side_reports = {}
    for spec in SIDE_TABLES:
        path = f"{metadata_dir}{spec.name}.jsonl"
        state = check_side_table_file(spec, path, built[spec.name])
        side_reports[spec.name] = {"rows": len(built[spec.name]), "state": state}

    # Write
    if not check_only:
        for name, rows in image_tables.items():
            if image_reports[name]["stripped"]:
                schema.write_jsonl(f"{metadata_dir}{name}.jsonl", rows)
        for spec in SIDE_TABLES:
            if side_reports[spec.name]["state"] == "absent":
                schema.write_jsonl(f"{metadata_dir}{spec.name}.jsonl", built[spec.name])

    # Report
    verb = "would strip" if check_only else "stripped"
    log(f"\n{'image table':<20}{'rows':>8}{'  ect_* ' + verb:>22}")
    for name, report in image_reports.items():
        log(f"{name + '.jsonl':<20}{report['rows']:>8}{report['stripped']:>22}")
    verb = "would write" if check_only else "wrote"
    log(f"\n{'side table':<20}{'rows':>8}  state")
    for name, report in side_reports.items():
        action = {"absent": verb, "identical": "already present, unchanged",
                  "superset": "present with newer rows, unchanged"}[report["state"]]
        log(f"{name + '.jsonl':<20}{report['rows']:>8}  {action}")
    for warning in warnings:
        log(f"WARNING: {warning}")
    log(
        "\nImage row counts and every non-ECT value verified unchanged. Every field "
        "the field protocol adds is null on legacy rows -- never \"X\", 0, \"\" or false."
    )
    return {"image": image_reports, "side": side_reports, "warnings": warnings}


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[1])
    parser.add_argument("dataset_dir", help = "Root of the rocky_mountain_snowpack dataset")
    parser.add_argument("--check", action = "store_true", help = "Verify without writing anything")
    arguments = parser.parse_args(argv)
    migrate(arguments.dataset_dir, check_only = arguments.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
