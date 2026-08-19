"""
Backfill the seven ECT columns onto the existing Rocky Mountain snowpack metadata.

Every row collected before the extended column test protocol landed gets ``null``
across all seven columns, meaning *not tested*. That is a different measurement
from ``"X"``, which means the test ran and the column did not fracture in 30 taps,
and it is a different measurement from ``0``, which is not a legal tap count at
all. This script asserts that distinction survives, rather than trusting it.

The script also proves it changed nothing else: row counts and every pre-existing
column value are compared before and after, and a mismatch aborts before anything
is written.

Usage:

    python scripts/backfill_ect.py <dataset_dir>            # backfill and write
    python scripts/backfill_ect.py <dataset_dir> --check    # report only, no writes
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ect


# The metadata files that carry dataset rows. 'raw' and 'preprocessed' are the
# full manifests; the three splits are subsets of 'preprocessed'.
METADATA_FILES = ('raw', 'preprocessed', 'train', 'test', 'validation')

# Values that would silently corrupt the not-tested/no-fracture distinction if a
# backfill ever produced them in place of null.
FORBIDDEN_BACKFILL_VALUES = ('X', 0, 0.0, '', 'null', 'None', False)


class BackfillError(RuntimeError):
    """Raised when a backfill would change something it must not."""


def read_rows(path):
    """Read a JSONL metadata file into a list of dicts, preserving order."""
    rows = []
    with open(path, 'r', encoding = 'utf-8') as handle:
        for position, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise BackfillError(f"{path} line {position + 1} is not valid JSON: {error}") from None
    return rows


def fingerprint(row):
    """A stable snapshot of every non-ECT field, for the unchanged-values check."""
    return json.dumps(
        {key: value for key, value in row.items() if key not in ect.ECT_COLUMNS},
        sort_keys = True,
    )


def is_forbidden(value):
    """Would this value silently misrepresent an untested column?"""
    for forbidden in FORBIDDEN_BACKFILL_VALUES:
        # Compare by type as well, so 0 does not match False by accident
        if type(value) is type(forbidden) and value == forbidden:
            return True
    return False


def backfill_file(path, check_only):
    """
    Backfill one metadata file and verify the result.

    Arguments:
        - path (str) - Path to the JSONL metadata file
        - check_only (bool) - Report without writing

    Returns a summary dict for the report.
    """
    before = read_rows(path)
    before_fingerprints = [fingerprint(row) for row in before]
    before_missing = sum(
        1 for row in before if any(column not in row for column in ect.ECT_COLUMNS)
    )

    after = [ect.backfill_row(dict(row)) for row in before]

    # Row count unchanged
    if len(after) != len(before):
        raise BackfillError(
            f"{path}: row count changed from {len(before)} to {len(after)}"
        )

    # Every pre-existing column value unchanged
    for position, (original, backfilled) in enumerate(zip(before_fingerprints, after)):
        if fingerprint(backfilled) != original:
            raise BackfillError(
                f"{path} row {position}: a pre-existing column value changed.\n"
                f"  before: {original}\n"
                f"  after:  {fingerprint(backfilled)}"
            )

    # Backfilled columns are null, never "X", 0, False or an empty string
    for position, (original, backfilled) in enumerate(zip(before, after)):
        for column in ect.ECT_COLUMNS:
            if column in original:
                continue  # was already recorded; not this script's business
            value = backfilled[column]
            if is_forbidden(value):
                raise BackfillError(
                    f"{path} row {position}: backfilled {column} to {value!r}, which "
                    f"reads as a measurement. 'X' means tested with no fracture in 30 "
                    f"taps and 0 is not a legal tap count; an untested column is null."
                )
            if value is not None:
                raise BackfillError(
                    f"{path} row {position}: backfilled {column} to {value!r}; an "
                    f"untested column must be null."
                )

    # The schema rules must hold over the whole file
    for position, row in enumerate(after):
        ect.validate_ect(row, context = f"{path} row {position}: ")
    ect.validate_group_constancy(after, context = f"{path}: ")

    tested = sum(1 for row in after if row['ect_result'] is not None)

    if not check_only and before_missing:
        temporary = f"{path}.ect-backfill.tmp"
        with open(temporary, 'w', encoding = 'utf-8') as handle:
            for row in after:
                handle.write(json.dumps(row) + '\n')
        os.replace(temporary, path)

    return {
        'path': path,
        'rows': len(after),
        'backfilled': before_missing,
        'tested': tested,
    }


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[1])
    parser.add_argument('dataset_dir', help = "Root of the rocky_mountain_snowpack dataset")
    parser.add_argument(
        '--check',
        action = 'store_true',
        help = "Verify without writing anything",
    )
    arguments = parser.parse_args(argv)

    dataset_dir = arguments.dataset_dir
    if not dataset_dir.endswith('/'):
        dataset_dir += '/'

    summaries = []
    for name in METADATA_FILES:
        path = f"{dataset_dir}metadata/{name}.jsonl"
        if not os.path.exists(path):
            print(f"  {name + '.jsonl':<22} not present, skipping")
            continue
        summaries.append(backfill_file(path, arguments.check))

    if not summaries:
        print(f"No metadata files found under {dataset_dir}metadata/")
        return 1

    verb = "would backfill" if arguments.check else "backfilled"
    print(f"\n{'file':<24}{'rows':>8}{'  ' + verb:>20}{'with ECT':>12}")
    for summary in summaries:
        print(
            f"{os.path.basename(summary['path']):<24}"
            f"{summary['rows']:>8}"
            f"{summary['backfilled']:>22}"
            f"{summary['tested']:>12}"
        )
    print(
        f"\nRow counts and all pre-existing column values verified unchanged. "
        f"Backfilled columns are null (not tested), never \"X\" or 0."
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
