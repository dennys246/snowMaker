"""
Round-trip tests for the ECT columns.

Writes one row of each result class (PV / P / N / X) plus an untested row, then
reads them back through both ``datasets.load_dataset()`` and a direct read of the
JSONL, and asserts both give exactly what the dataset card claims.

The card's claims are the thing under test, not the library's behaviour in the
abstract: the 2026-08-17 ``datatype`` incident happened because the two access
paths disagreed and nothing said so. So this file also pins the *disagreement* the
card documents for ``datatype`` and ``coordinates``, to catch it changing silently.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ect

datasets = pytest.importorskip("datasets", reason = "datasets is needed for the load_dataset half")


# One row per result class, plus a row from before the ECT protocol existed.
# Two rows share (site 4, column 1) to exercise the (site, column) grain.
ROWS = [
    {
        'file_path': 'preprocessed/cores/image_1.png',
        'datatype': 'core',
        'site': 4, 'column': 1, 'core': 1, 'segment': -1,
        'coordinates': [39.66412, -105.87903],
        # Propagated during isolation, before any tap: no tap count exists.
        'ect_result': 'PV', 'ect_taps': None,
        'ect_failure_depth_cm': 48.0, 'ect_failure_grain': 'DH',
        'ect_hardness_above': 'P', 'ect_hardness_below': 'F',
        'ect_operator': 'Denny Schaedig',
    },
    {
        'file_path': 'preprocessed/cores/image_2.png',
        'datatype': 'profile',
        'site': 4, 'column': 1, 'core': 2, 'segment': -1,
        'coordinates': [39.66412, -105.87903],
        # Same snow column as the row above, so the seven values must repeat.
        'ect_result': 'PV', 'ect_taps': None,
        'ect_failure_depth_cm': 48.0, 'ect_failure_grain': 'DH',
        'ect_hardness_above': 'P', 'ect_hardness_below': 'F',
        'ect_operator': 'Denny Schaedig',
    },
    {
        'file_path': 'preprocessed/cores/image_3.png',
        'datatype': 'core',
        'site': 5, 'column': 1, 'core': 1, 'segment': -1,
        'coordinates': [39.62881, -105.94017],
        # ECTP12 -- fractured on tap 12 and propagated the full column.
        'ect_result': 'P', 'ect_taps': 12,
        'ect_failure_depth_cm': 62.3, 'ect_failure_grain': 'FC',
        'ect_hardness_above': '1F', 'ect_hardness_below': 'F',
        'ect_operator': 'Denny Schaedig',
    },
    {
        'file_path': 'preprocessed/cores/image_4.png',
        'datatype': 'magnified_profile',
        'site': 6, 'column': 1, 'core': 1, 'segment': -1,
        'coordinates': [39.59104, -105.64322],
        # ECTN23 -- fractured on tap 23 but did not propagate across.
        'ect_result': 'N', 'ect_taps': 23,
        'ect_failure_depth_cm': 105.5, 'ect_failure_grain': 'RG',
        'ect_hardness_above': 'K', 'ect_hardness_below': '4F',
        'ect_operator': 'A. Partner',
    },
    {
        'file_path': 'preprocessed/cores/image_5.png',
        'datatype': 'crystal_card',
        'site': 7, 'column': 1, 'core': 1, 'segment': -1,
        'coordinates': [39.60225, -105.87741],
        # ECTX -- tested, no fracture in 30 taps. Nothing fractured, so there is
        # nothing to describe: every other ECT column is null.
        'ect_result': 'X', 'ect_taps': None,
        'ect_failure_depth_cm': None, 'ect_failure_grain': None,
        'ect_hardness_above': None, 'ect_hardness_below': None,
        'ect_operator': 'Denny Schaedig',
    },
    {
        'file_path': 'preprocessed/cores/image_6.png',
        'datatype': 'core',
        'site': 0, 'column': 1, 'core': 1, 'segment': -1,
        # Site 0's known two-decimal-place coordinate
        'coordinates': [39.66, -105.88],
        # Not tested: null across all seven, which is NOT the same as 'X'
        **ect.null_ect(),
    },
]

CARD = """---
dataset_info:
  features:
    - name: file_path
      dtype: string
    - name: datatype
      dtype:
        class_label:
          names:
            0: 'core'
            1: 'profile'
            2: 'magnified_profile'
            3: 'crystal_card'
    - name: site
      dtype: int64
    - name: column
      dtype: int64
    - name: core
      dtype: int64
    - name: segment
      dtype: int64
    - name: coordinates
      dtype:
        list:
          dtype: float32
{ect_features}
configs:
- config_name: default
  data_files:
  - split: preprocessed
    path: "metadata/preprocessed.jsonl"
---
Fixture card for the ECT round-trip test.
"""


@pytest.fixture
def dataset_dir(tmp_path):
    """A minimal dataset repo: a card declaring the ECT features, plus the rows."""
    root = tmp_path / "rmsnow"
    (root / "metadata").mkdir(parents = True)
    (root / "README.md").write_text(
        CARD.format(ect_features = ect.card_feature_yaml()), encoding = 'utf-8'
    )
    with open(root / "metadata" / "preprocessed.jsonl", 'w', encoding = 'utf-8') as handle:
        for row in ROWS:
            handle.write(json.dumps(row) + '\n')
    return root


def read_jsonl(dataset_dir):
    """The direct-read path: whatever JSON literals are physically in the file."""
    path = dataset_dir / "metadata" / "preprocessed.jsonl"
    with open(path, 'r', encoding = 'utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_load_dataset(dataset_dir, tmp_path):
    """The load_dataset path: the card's declared features applied to the same file."""
    loaded = datasets.load_dataset(
        str(dataset_dir),
        split = 'preprocessed',
        cache_dir = str(tmp_path / "hf-cache"),
        download_mode = 'force_redownload',
    )
    return loaded, list(loaded)


# --- What the card claims the two paths agree on ----------------------------

def test_ect_columns_are_value_not_classlabel(dataset_dir, tmp_path):
    """The card says the ECT columns are Value(...), which is why they agree."""
    loaded, _ = read_load_dataset(dataset_dir, tmp_path)
    for column, dtype in ect.ECT_FEATURE_DTYPES.items():
        feature = loaded.features[column]
        assert isinstance(feature, datasets.Value), (
            f"{column} is {feature!r}; a ClassLabel would return int codes from "
            f"load_dataset() while the JSONL holds strings"
        )
        assert feature.dtype == dtype


def test_ect_values_identical_on_both_paths(dataset_dir, tmp_path):
    """Every ECT value reads back the same through load_dataset() and the JSONL."""
    _, loaded_rows = read_load_dataset(dataset_dir, tmp_path)
    direct_rows = read_jsonl(dataset_dir)

    assert len(loaded_rows) == len(direct_rows) == len(ROWS)

    for position, (loaded, direct, expected) in enumerate(zip(loaded_rows, direct_rows, ROWS)):
        for column in ect.ECT_COLUMNS:
            assert loaded[column] == direct[column], (
                f"row {position}: {column} differs between load_dataset() "
                f"({loaded[column]!r}) and the JSONL ({direct[column]!r})"
            )
            assert loaded[column] == expected[column], (
                f"row {position}: {column} round-tripped to {loaded[column]!r}, "
                f"expected {expected[column]!r}"
            )


def test_null_and_X_survive_as_different_values(dataset_dir, tmp_path):
    """The distinction the whole column set rests on, checked on both paths."""
    _, loaded_rows = read_load_dataset(dataset_dir, tmp_path)
    direct_rows = read_jsonl(dataset_dir)

    for rows, label in ((loaded_rows, 'load_dataset'), (direct_rows, 'jsonl')):
        by_site = {row['site']: row for row in rows}

        no_fracture = by_site[7]
        assert no_fracture['ect_result'] == 'X', (
            f"{label}: a tested column with no fracture in 30 taps must stay 'X'"
        )
        # Tested but nothing fractured, so there is nothing to describe
        for column in ect.ECT_FRACTURE_COLUMNS:
            assert no_fracture[column] is None
        assert no_fracture['ect_operator'] == 'Denny Schaedig'

        not_tested = by_site[0]
        for column in ect.ECT_COLUMNS:
            assert not_tested[column] is None, (
                f"{label}: an untested column must be null across all seven, got "
                f"{column}={not_tested[column]!r}"
            )
        # The corruption this guards against
        assert not_tested['ect_result'] != 'X'
        assert not_tested['ect_taps'] != 0
        assert not_tested['ect_taps'] is not False


def test_each_result_class_round_trips(dataset_dir, tmp_path):
    """PV, P, N and X each survive both paths with their tap-count semantics."""
    _, loaded_rows = read_load_dataset(dataset_dir, tmp_path)
    direct_rows = read_jsonl(dataset_dir)

    for rows, label in ((loaded_rows, 'load_dataset'), (direct_rows, 'jsonl')):
        by_site = {row['site']: row for row in rows}

        # PV: propagated during isolation, before any tap
        assert by_site[4]['ect_result'] == 'PV', label
        assert by_site[4]['ect_taps'] is None, label
        assert by_site[4]['ect_failure_grain'] == 'DH', label

        # P12: fractured on tap 12 and propagated
        assert by_site[5]['ect_result'] == 'P', label
        assert by_site[5]['ect_taps'] == 12, label
        # float64, so the JSON literal survives exactly rather than becoming
        # 62.29999923706055 the way a float32 column would
        assert by_site[5]['ect_failure_depth_cm'] == 62.3, label

        # N23: fractured on tap 23 but did not propagate
        assert by_site[6]['ect_result'] == 'N', label
        assert by_site[6]['ect_taps'] == 23, label

        # X: no fracture in 30 taps
        assert by_site[7]['ect_result'] == 'X', label
        assert by_site[7]['ect_taps'] is None, label

        assert {row['ect_result'] for row in rows} == {'PV', 'P', 'N', 'X', None}, label


def test_ect_constant_within_site_column(dataset_dir, tmp_path):
    """Two rows of the same snow column carry the same test, on both paths."""
    _, loaded_rows = read_load_dataset(dataset_dir, tmp_path)
    direct_rows = read_jsonl(dataset_dir)

    for rows, label in ((loaded_rows, 'load_dataset'), (direct_rows, 'jsonl')):
        ect.validate_group_constancy(rows, context = f"{label}: ")
        pair = [row for row in rows if (row['site'], row['column']) == (4, 1)]
        assert len(pair) == 2, label
        for column in ect.ECT_COLUMNS:
            assert pair[0][column] == pair[1][column], label


# --- What the card claims the two paths disagree on -------------------------

def test_documented_disagreements_still_hold(dataset_dir, tmp_path):
    """
    ``datatype`` and ``coordinates`` differ between the paths, and the card says so.

    Pinned here so a change to that behaviour shows up as a failing test rather
    than as zero matched pairs somewhere downstream, which is how the 2026-08-17
    ``datatype`` change presented.
    """
    _, loaded_rows = read_load_dataset(dataset_dir, tmp_path)
    direct_rows = read_jsonl(dataset_dir)

    # datatype: ClassLabel index vs. the string in the file
    assert [row['datatype'] for row in loaded_rows] == [0, 1, 0, 2, 3, 0]
    assert [row['datatype'] for row in direct_rows] == [
        'core', 'profile', 'core', 'magnified_profile', 'crystal_card', 'core'
    ]

    # coordinates: float32 in the card, so load_dataset() loses precision the
    # JSONL keeps. This is why elevation is derived from the JSONL fix.
    loaded_lat = loaded_rows[2]['coordinates'][0]
    direct_lat = direct_rows[2]['coordinates'][0]
    assert direct_lat == 39.62881
    assert loaded_lat != direct_lat
    assert abs(loaded_lat - direct_lat) < 1e-5


# --- Validation rules -------------------------------------------------------

def test_valid_records_accepted():
    for row in ROWS:
        ect.validate_ect(row, context = f"{row['file_path']}: ")


@pytest.mark.parametrize("record, reason", [
    ({'ect_result': 'PV', 'ect_taps': 3}, "PV means it went before any tap"),
    ({'ect_result': 'X', 'ect_taps': 30}, "X means nothing fractured"),
    ({'ect_result': 'X', 'ect_failure_grain': 'FC'}, "X has no failure interface"),
    ({'ect_result': 'X', 'ect_failure_depth_cm': 40.0}, "X has no failure depth"),
    ({'ect_result': 'X', 'ect_hardness_above': 'P'}, "X has no interface to bracket"),
    ({'ect_result': 'P'}, "P needs a tap count"),
    ({'ect_result': 'N'}, "N needs a tap count"),
    ({'ect_result': 'P', 'ect_taps': 0}, "taps start at 1"),
    ({'ect_result': 'P', 'ect_taps': 31}, "taps stop at 30"),
    ({'ect_taps': 12}, "a tap count with no result"),
    ({'ect_failure_depth_cm': 40.0}, "a depth with no result"),
    ({'ect_operator': 'Denny Schaedig'}, "an operator with no result"),
])
def test_invalid_records_rejected(record, reason):
    values = ect.null_ect()
    values.update(record)
    with pytest.raises(ect.ECTValidationError):
        ect.validate_ect(values, context = f"{reason}: ")


@pytest.mark.parametrize("raw", [
    {'ect_result': 'ECTP'},        # field notation, not the stored letter
    {'ect_result': 'P1'},          # taps belong in ect_taps
    {'ect_result': 'p'},           # case matters
    {'ect_result': 'PN'},
    {'ect_result': 0},
    {'ect_result': 'PV', 'ect_failure_grain': 'facets'},   # not an ICSSG code
    {'ect_result': 'PV', 'ect_hardness_above': 'medium'},  # not a hardness class
    {'ect_result': 'P', 'ect_taps': 'twelve'},
    {'ect_result': 'P', 'ect_taps': 12.5},
    {'ect_result': 'P', 'ect_taps': 12, 'ect_failure_depth_cm': -3.0},
])
def test_out_of_domain_values_rejected_not_coerced(raw):
    with pytest.raises(ect.ECTValidationError):
        ect.normalize_ect(raw, context = "out of domain: ")


def test_blank_csv_cells_read_as_not_tested():
    """pandas hands back NaN and '' for blank cells; both mean not tested."""
    import math

    raw = {column: '' for column in ect.ECT_COLUMNS}
    raw['ect_taps'] = float('nan')
    raw['ect_failure_depth_cm'] = math.nan
    assert ect.normalize_ect(raw) == ect.null_ect()


def test_group_constancy_violation_is_loud():
    """A disagreement raises rather than letting the first value win."""
    rows = [
        {'site': 4, 'column': 1, **ect.null_ect(), 'ect_result': 'X'},
        {'site': 4, 'column': 1, **ect.null_ect(), 'ect_result': 'PV'},
    ]
    with pytest.raises(ect.ECTValidationError) as caught:
        ect.validate_group_constancy(rows)
    message = str(caught.value)
    assert 'ect_result' in message
    assert 'site=4' in message and 'column=1' in message


def test_backfill_adds_nulls_without_touching_recorded_values():
    untested = {'site': 0, 'column': 1}
    ect.backfill_row(untested)
    assert all(untested[column] is None for column in ect.ECT_COLUMNS)

    recorded = {'site': 5, 'column': 1, 'ect_result': 'X'}
    ect.backfill_row(recorded)
    assert recorded['ect_result'] == 'X', "backfill must not overwrite a real result"
    assert recorded['ect_taps'] is None
