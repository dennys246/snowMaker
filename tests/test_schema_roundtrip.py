"""
Round-trip tests for the side tables.

Writes a small dataset -- a pit with and without a method deviation, a full
layer profile, cores of every recovery quality plus a skipped ladder rung, one
ECT row of each result class (PV / P / N / X), a multi-fracture test, a repeat
test on the same column and an untested column -- then reads it back through
both ``datasets.load_dataset()`` and a direct read of the JSONL, and asserts
both give exactly what the dataset card claims.

The card's claims are the thing under test, not the library's behaviour in the
abstract: the 2026-08-17 ``datatype`` incident happened because the two access
paths disagreed and nothing said so. So this file also pins the *disagreement*
the card documents for the image table's ``datatype`` and ``coordinates``.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import card
import cores
import ect
import layers
import pits
import schema

datasets = pytest.importorskip("datasets", reason = "datasets is needed for the load_dataset half")

SIDE_TABLES = (pits.SPEC, layers.SPEC, cores.SPEC, ect.SPEC)


# --- Fixture rows -----------------------------------------------------------

PITS = [
    {
        'site': 8, 'collector_id': 'collector-01', 'corer_id': 'apple-corer-40mm',
        'coordinates': [39.66412, -105.87903], 'aspect_deg_true': 112.5,
        'slope_angle_deg': 31.0, 'date': '2026-12-20', 'time_of_day': '09:45-07:00',
        'ground_cover': 'talus', 'snowpack_depth_cm': 145.0,
        'method_deviation': True, 'method_deviation_note': 'Corer swapped after rung 3',
    },
    {
        'site': 9, 'collector_id': 'collector-02', 'corer_id': None,
        'coordinates': [39.62881, -105.94017], 'aspect_deg_true': None,
        'slope_angle_deg': 24.0, 'date': '2027-01-05', 'time_of_day': '13:10-07:00',
        'ground_cover': None, 'snowpack_depth_cm': 98.0,
        'method_deviation': False, 'method_deviation_note': None,
    },
]

# A full profile of site 8 column 1, top down and contiguous
LAYERS = [
    {'site': 8, 'column': 1, 'layer_index': 1, 'depth_top_cm': 0.0, 'depth_bottom_cm': 15.0,
     'hand_hardness': 'F', 'grain_type': 'PP', 'grain_size_mm': 1.0, 'density_kg_m3': 120.0,
     'temperature_c': -8.5, 'lwc': 'D'},
    {'site': 8, 'column': 1, 'layer_index': 2, 'depth_top_cm': 15.0, 'depth_bottom_cm': 60.0,
     'hand_hardness': 'P', 'grain_type': 'RG', 'grain_size_mm': 0.5, 'density_kg_m3': 280.0,
     'temperature_c': -6.0, 'lwc': 'D'},
    {'site': 8, 'column': 1, 'layer_index': 3, 'depth_top_cm': 60.0, 'depth_bottom_cm': 85.0,
     'hand_hardness': '4F', 'grain_type': 'FC', 'grain_size_mm': 1.5, 'density_kg_m3': 230.0,
     'temperature_c': -3.5, 'lwc': None},
    # The slab-over-depth-hoar configuration: P above, F below
    {'site': 8, 'column': 1, 'layer_index': 4, 'depth_top_cm': 85.0, 'depth_bottom_cm': 145.0,
     'hand_hardness': 'F', 'grain_type': 'DH', 'grain_size_mm': 3.0, 'density_kg_m3': None,
     'temperature_c': -1.0, 'lwc': None},
]

CORES = [
    {'site': 8, 'column': 1, 'core': 1, 'core_depth_cm': 10.0, 'depth_skipped_reason': None,
     'breakability_field_count': 1, 'recovery_quality': 'intact', 'core_temperature_c': -8.0,
     'layer_ids': [1]},
    {'site': 8, 'column': 1, 'core': 2, 'core_depth_cm': 20.0, 'depth_skipped_reason': None,
     'breakability_field_count': 3, 'recovery_quality': 'partial', 'core_temperature_c': -6.0,
     'layer_ids': [2]},
    # A rung that was not sampled: the reason is recorded, nothing else is
    {'site': 8, 'column': 1, 'core': 3, 'core_depth_cm': 30.0, 'depth_skipped_reason': 'ice_layer',
     'breakability_field_count': None, 'recovery_quality': None, 'core_temperature_c': None,
     'layer_ids': [2]},
    # Fell apart: could not be counted, and that is recorded as such
    {'site': 8, 'column': 1, 'core': 7, 'core_depth_cm': 70.0, 'depth_skipped_reason': None,
     'breakability_field_count': None, 'recovery_quality': 'disintegrated', 'core_temperature_c': -3.0,
     'layer_ids': [3]},
    # A legacy-shaped core: nothing the protocol adds was measured
    {'site': 9, 'column': 1, 'core': 1, 'core_depth_cm': 10.0, 'depth_skipped_reason': None,
     'breakability_field_count': None, 'recovery_quality': None, 'core_temperature_c': None,
     'layer_ids': None},
    {'site': 9, 'column': 1, 'core': 2, 'core_depth_cm': 20.0, 'depth_skipped_reason': None,
     'breakability_field_count': None, 'recovery_quality': 'not_recovered', 'core_temperature_c': None,
     'layer_ids': None},
]

TEST_8_1 = {'ect_tap_convention': 'initiating', 'ect_slope_angle_deg': 31.0,
            'ect_column_depth_cm': 90.0, 'ect_suspected_pwl_below': False,
            'ect_operator': 'collector-01'}

ECT = [
    # Two fractures from one test: ECTN8 at 25 cm, then ECTP22 at 70 cm
    {'site': 8, 'column': 1, 'test_index': 1, 'fracture_index': 1, 'ect_result': 'N', 'ect_taps': 8,
     'ect_failure_depth_cm': 25.0, 'ect_failure_grain': 'FC', 'ect_hardness_above': '1F',
     'ect_hardness_below': '4F', 'ect_fracture_character': 'SC', **TEST_8_1},
    {'site': 8, 'column': 1, 'test_index': 1, 'fracture_index': 2, 'ect_result': 'P', 'ect_taps': 22,
     'ect_failure_depth_cm': 70.0, 'ect_failure_grain': 'DH', 'ect_hardness_above': 'P',
     'ect_hardness_below': 'F', 'ect_fracture_character': 'SP', **TEST_8_1},
    # A repeat test on the same column
    {'site': 8, 'column': 1, 'test_index': 2, 'fracture_index': 1, 'ect_result': 'N', 'ect_taps': 15,
     'ect_failure_depth_cm': 25.0, 'ect_failure_grain': 'FC', 'ect_hardness_above': '1F',
     'ect_hardness_below': '4F', 'ect_fracture_character': 'RP', **TEST_8_1},
    # ECTX: tested, no fracture in 30 taps -- but only to 90 cm, with a suspected PWL below
    {'site': 8, 'column': 2, 'test_index': 1, 'fracture_index': 1, 'ect_result': 'X', 'ect_taps': None,
     'ect_failure_depth_cm': None, 'ect_failure_grain': None, 'ect_hardness_above': None,
     'ect_hardness_below': None, 'ect_fracture_character': None, 'ect_tap_convention': None,
     'ect_slope_angle_deg': 30.0, 'ect_column_depth_cm': 90.0, 'ect_suspected_pwl_below': True,
     'ect_operator': 'collector-01'},
    # ECTPV: propagated during isolation, before any tap
    {'site': 9, 'column': 1, 'test_index': 1, 'fracture_index': 1, 'ect_result': 'PV', 'ect_taps': None,
     'ect_failure_depth_cm': 48.0, 'ect_failure_grain': 'DH', 'ect_hardness_above': 'P',
     'ect_hardness_below': 'F', 'ect_fracture_character': 'SC', 'ect_tap_convention': None,
     'ect_slope_angle_deg': 24.0, 'ect_column_depth_cm': 100.0, 'ect_suspected_pwl_below': None,
     'ect_operator': 'collector-02'},
    # ECTP12 logged on the propagating tap
    {'site': 9, 'column': 2, 'test_index': 1, 'fracture_index': 1, 'ect_result': 'P', 'ect_taps': 12,
     'ect_failure_depth_cm': 62.3, 'ect_failure_grain': 'FC', 'ect_hardness_above': '1F',
     'ect_hardness_below': 'F', 'ect_fracture_character': 'Q1', 'ect_tap_convention': 'propagating',
     'ect_slope_angle_deg': 24.0, 'ect_column_depth_cm': 100.0, 'ect_suspected_pwl_below': False,
     'ect_operator': 'collector-02'},
    # Site 9 column 3 was not tested: it has no row at all.
]

# Image rows that reference the cores above, in the legacy image-table shape
# (the columns the fixture card declares). Site 8 has a full-precision fix;
# note the image table declares coordinates float32 and cannot change.
IMAGE = [
    {'file_path': 'preprocessed/cores/image_1.png', 'datatype': 'core', 'site': 8, 'column': 1,
     'core': 1, 'segment': -1, 'coordinates': [39.66412, -105.87903], 'snowpack_depth': 145.0,
     'core_depth': 10.0, 'slope_angle': 31.0},
    {'file_path': 'preprocessed/profiles/image_2.png', 'datatype': 'profile', 'site': 8, 'column': 1,
     'core': 2, 'segment': 1, 'coordinates': [39.66412, -105.87903], 'snowpack_depth': 145.0,
     'core_depth': 20.0, 'slope_angle': 31.0},
    {'file_path': 'preprocessed/cores/image_3.png', 'datatype': 'core', 'site': 9, 'column': 1,
     'core': 1, 'segment': -1, 'coordinates': [39.62881, -105.94017], 'snowpack_depth': 98.0,
     'core_depth': 10.0, 'slope_angle': 24.0},
]

TABLES = {'pits': PITS, 'layers': LAYERS, 'cores': CORES, 'ect': ECT}

# The image config in the fixture card is a subset of the real one (no Image
# column, so nothing is downloaded), with the same declared types.
IMAGE_FEATURES_YAML = """\
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
  - name: snowpack_depth
    dtype: float32
  - name: core_depth
    dtype: float32
  - name: slope_angle
    dtype: float32"""


def fixture_card():
    """The real side-table front matter, with a reduced image config in front."""
    info = card.dataset_info_yaml().replace(card.IMAGE_FEATURES_YAML, IMAGE_FEATURES_YAML)
    return f"---\n{info}\n{card.configs_yaml()}\n---\nFixture card for the side-table round-trip test.\n"


@pytest.fixture
def dataset_dir(tmp_path):
    root = tmp_path / "rmsnow"
    (root / "metadata").mkdir(parents = True)
    (root / "README.md").write_text(fixture_card(), encoding = 'utf-8')
    schema.write_jsonl(str(root / "metadata" / "preprocessed.jsonl"), IMAGE)
    schema.write_jsonl(str(root / "metadata" / "raw.jsonl"), IMAGE[:1])
    for spec in SIDE_TABLES:
        schema.write_jsonl(str(root / spec.filename), TABLES[spec.name])
    return root


def read_jsonl(dataset_dir, name):
    """The direct-read path: whatever JSON literals are physically in the file."""
    return schema.read_jsonl(str(dataset_dir / "metadata" / f"{name}.jsonl"))


def read_load_dataset(dataset_dir, tmp_path, config):
    """The load_dataset path: the card's declared features applied to the same file."""
    loaded = datasets.load_dataset(
        str(dataset_dir),
        config,
        split = 'train',
        cache_dir = str(tmp_path / "hf-cache"),
        download_mode = 'force_redownload',
    )
    return loaded, list(loaded)


# --- The fixture itself is valid ---------------------------------------------

def test_fixture_passes_every_validation():
    warnings = []
    for spec in SIDE_TABLES:
        warnings.extend(spec.validate_table(TABLES[spec.name]))
    warnings.extend(schema.validate_dataset(TABLES, image_rows = IMAGE))
    assert warnings == []


# --- What the card claims the two paths agree on ----------------------------

@pytest.mark.parametrize("spec", SIDE_TABLES, ids = lambda s: s.name)
def test_side_table_columns_are_value_not_classlabel(dataset_dir, tmp_path, spec):
    """Every side-table column is Value/List, never ClassLabel, so the paths agree."""
    loaded, _ = read_load_dataset(dataset_dir, tmp_path, spec.name)
    assert list(loaded.features) == spec.columns
    for field in spec.fields:
        feature = loaded.features[field.name]
        if field.is_list:
            assert isinstance(feature, datasets.List), (field.name, feature)
            assert feature.feature.dtype == field.hf_dtype
        else:
            assert isinstance(feature, datasets.Value), (
                f"{field.name} is {feature!r}; a ClassLabel would return int codes "
                f"from load_dataset() while the JSONL holds strings"
            )
            assert feature.dtype == field.hf_dtype


@pytest.mark.parametrize("spec", SIDE_TABLES, ids = lambda s: s.name)
def test_side_table_values_identical_on_both_paths(dataset_dir, tmp_path, spec):
    """Every value reads back the same through load_dataset() and the JSONL."""
    _, loaded_rows = read_load_dataset(dataset_dir, tmp_path, spec.name)
    direct_rows = read_jsonl(dataset_dir, spec.name)
    expected = TABLES[spec.name]

    assert len(loaded_rows) == len(direct_rows) == len(expected)
    for position, (loaded, direct, original) in enumerate(zip(loaded_rows, direct_rows, expected)):
        for column in spec.columns:
            if spec.by_name[column].kind == "date":
                continue  # the one documented difference; pinned separately below
            assert loaded[column] == direct[column], (
                f"{spec.name} row {position}: {column} differs between load_dataset() "
                f"({loaded[column]!r}) and the JSONL ({direct[column]!r})"
            )
            assert loaded[column] == original[column], (
                f"{spec.name} row {position}: {column} round-tripped to "
                f"{loaded[column]!r}, expected {original[column]!r}"
            )
            # The type survives too: 12 stays int, 62.3 stays float, True stays bool
            assert type(loaded[column]) is type(original[column]), (spec.name, column)


def test_documented_date_disagreement_still_holds(dataset_dir, tmp_path):
    """
    ``pits.date`` is declared date32, so load_dataset() returns a datetime.date
    while the JSONL holds the ISO string. Same value, different Python type, and
    the card says so. It is NOT declared string: pyarrow infers "2026-12-20" as a
    timestamp before the cast, and a string declaration silently yields
    "2026-12-20 00:00:00" on one path and "2026-12-20" on the other.
    """
    import datetime as dt
    loaded, loaded_rows = read_load_dataset(dataset_dir, tmp_path, 'pits')
    direct_rows = read_jsonl(dataset_dir, 'pits')
    assert loaded.features['date'] == datasets.Value('date32')
    for loaded_row, direct_row in zip(loaded_rows, direct_rows):
        assert isinstance(loaded_row['date'], dt.date)
        assert isinstance(direct_row['date'], str)
        assert loaded_row['date'].isoformat() == direct_row['date']
    # time_of_day carries no date part, so pyarrow leaves it alone: identical
    assert [row['time_of_day'] for row in loaded_rows] == [row['time_of_day'] for row in direct_rows]


def test_pit_coordinates_keep_full_precision_on_both_paths(dataset_dir, tmp_path):
    """pits.coordinates is float64, so nothing is lost -- unlike the image table."""
    _, loaded_rows = read_load_dataset(dataset_dir, tmp_path, 'pits')
    direct_rows = read_jsonl(dataset_dir, 'pits')
    assert loaded_rows[0]['coordinates'] == direct_rows[0]['coordinates'] == [39.66412, -105.87903]


# --- ECT semantics ----------------------------------------------------------

def test_each_ect_result_class_round_trips(dataset_dir, tmp_path):
    """PV, P, N and X each survive both paths with their tap-count semantics."""
    _, loaded_rows = read_load_dataset(dataset_dir, tmp_path, 'ect')
    direct_rows = read_jsonl(dataset_dir, 'ect')

    for rows, label in ((loaded_rows, 'load_dataset'), (direct_rows, 'jsonl')):
        by_key = {ect.SPEC.key_of(row): row for row in rows}

        # PV: propagated during isolation, before any tap
        pv = by_key[(9, 1, 1, 1)]
        assert pv['ect_result'] == 'PV' and pv['ect_taps'] is None, label
        assert pv['ect_failure_grain'] == 'DH', label

        # P12 on the propagating tap; float64 keeps 62.3 exact
        p = by_key[(9, 2, 1, 1)]
        assert (p['ect_result'], p['ect_taps'], p['ect_tap_convention']) == ('P', 12, 'propagating'), label
        assert p['ect_failure_depth_cm'] == 62.3, label

        # N8 then P22: one test, two fractures, in tap order
        first, second = by_key[(8, 1, 1, 1)], by_key[(8, 1, 1, 2)]
        assert (first['ect_result'], first['ect_taps'], first['ect_failure_depth_cm']) == ('N', 8, 25.0), label
        assert (second['ect_result'], second['ect_taps'], second['ect_failure_depth_cm']) == ('P', 22, 70.0), label

        # A repeat test on the same column is a separate test_index
        assert by_key[(8, 1, 2, 1)]['ect_taps'] == 15, label

        # X: nothing fractured -- but the censoring fields are present
        x = by_key[(8, 2, 1, 1)]
        assert x['ect_result'] == 'X', label
        for column in ect.FRACTURE_FIELDS:
            assert x[column] is None, (label, column)
        assert x['ect_column_depth_cm'] == 90.0, label
        assert x['ect_suspected_pwl_below'] is True, label

        assert {row['ect_result'] for row in rows} == {'PV', 'P', 'N', 'X'}, label


def test_untested_column_has_no_row_and_is_not_X(dataset_dir, tmp_path):
    """Not tested is an absent row. It is never an 'X', a 0 or an empty string."""
    _, loaded_rows = read_load_dataset(dataset_dir, tmp_path, 'ect')
    direct_rows = read_jsonl(dataset_dir, 'ect')
    for rows, label in ((loaded_rows, 'load_dataset'), (direct_rows, 'jsonl')):
        tested_columns = {(row['site'], row['column']) for row in rows}
        assert (9, 3) not in tested_columns, label
        assert (8, 2) in tested_columns, f"{label}: the X column was tested"
        assert not any(row['ect_result'] in ('', 0, None) for row in rows), label


def test_multi_fracture_test_is_readable_by_test_index(dataset_dir, tmp_path):
    _, loaded_rows = read_load_dataset(dataset_dir, tmp_path, 'ect')
    fractures = sorted(
        (row['fracture_index'], row['ect_taps'])
        for row in loaded_rows if (row['site'], row['column'], row['test_index']) == (8, 1, 1)
    )
    assert fractures == [(1, 8), (2, 22)]


# --- Layer and core semantics -------------------------------------------------

def test_layer_profile_round_trips_in_order(dataset_dir, tmp_path):
    _, loaded_rows = read_load_dataset(dataset_dir, tmp_path, 'layers')
    direct_rows = read_jsonl(dataset_dir, 'layers')
    for rows, label in ((loaded_rows, 'load_dataset'), (direct_rows, 'jsonl')):
        profile = sorted(rows, key = lambda row: row['layer_index'])
        assert [row['layer_index'] for row in profile] == [1, 2, 3, 4], label
        # Contiguous top-down
        for above, below in zip(profile, profile[1:]):
            assert above['depth_bottom_cm'] == below['depth_top_cm'], label
        # The slab over depth hoar is readable from hardness alone
        assert [row['hand_hardness'] for row in profile] == ['F', 'P', '4F', 'F'], label
        assert [row['grain_type'] for row in profile] == ['PP', 'RG', 'FC', 'DH'], label
        # Nullable layer fields stay null, not 0
        assert profile[3]['density_kg_m3'] is None, label
        assert profile[2]['lwc'] is None, label


def test_core_states_stay_distinct(dataset_dir, tmp_path):
    """Skipped, disintegrated, not recovered and unmeasured are four states."""
    _, loaded_rows = read_load_dataset(dataset_dir, tmp_path, 'cores')
    direct_rows = read_jsonl(dataset_dir, 'cores')
    for rows, label in ((loaded_rows, 'load_dataset'), (direct_rows, 'jsonl')):
        by_key = {cores.SPEC.key_of(row): row for row in rows}
        skipped = by_key[(8, 1, 3)]
        assert skipped['depth_skipped_reason'] == 'ice_layer', label
        assert skipped['breakability_field_count'] is None and skipped['recovery_quality'] is None, label

        crumbled = by_key[(8, 1, 7)]
        assert crumbled['recovery_quality'] == 'disintegrated', label
        assert crumbled['breakability_field_count'] is None, label

        assert by_key[(9, 1, 2)]['recovery_quality'] == 'not_recovered', label

        legacy = by_key[(9, 1, 1)]
        assert legacy['depth_skipped_reason'] is None, f"{label}: legacy core was sampled"
        assert legacy['recovery_quality'] is None, f"{label}: legacy core has no recovery record"
        assert legacy['breakability_field_count'] is None, label
        assert legacy['layer_ids'] is None, label

        assert by_key[(8, 1, 1)]['layer_ids'] == [1] and by_key[(8, 1, 1)]['breakability_field_count'] == 1, label


# --- What the card claims the two paths disagree on -------------------------

def test_documented_image_table_disagreements_still_hold(dataset_dir, tmp_path):
    """
    The image table's ``datatype`` (ClassLabel) and ``coordinates`` (float32)
    differ between the paths, and the card says so. Pinned so a change shows up
    here rather than as zero matched pairs downstream, which is how the
    2026-08-17 ``datatype`` change presented.
    """
    _, loaded_rows = read_load_dataset(dataset_dir, tmp_path, 'default')
    direct_rows = read_jsonl(dataset_dir, 'preprocessed')

    assert [row['datatype'] for row in loaded_rows] == [0, 1, 0]
    assert [row['datatype'] for row in direct_rows] == ['core', 'profile', 'core']

    loaded_lat = loaded_rows[0]['coordinates'][0]
    direct_lat = direct_rows[0]['coordinates'][0]
    assert direct_lat == 39.66412
    assert loaded_lat != direct_lat
    assert abs(loaded_lat - direct_lat) < 1e-5


def test_image_table_no_longer_carries_ect_columns(dataset_dir, tmp_path):
    loaded, _ = read_load_dataset(dataset_dir, tmp_path, 'default')
    for column in ect.LEGACY_IMAGE_COLUMNS:
        assert column not in loaded.features
    for row in read_jsonl(dataset_dir, 'preprocessed'):
        assert not any(column in row for column in ect.LEGACY_IMAGE_COLUMNS)


# --- Empty side tables ---------------------------------------------------------

def test_empty_side_table_is_documented_but_not_a_config(dataset_dir, tmp_path):
    """
    Until the first layer profile lands, layers.jsonl is empty and N/A -- not
    assessed. A direct read gives zero rows. The table is not declared as a
    config (``datasets`` refuses an empty one), so load_dataset() raises for it
    while every table with rows still loads.
    """
    schema.write_jsonl(str(dataset_dir / "metadata" / "layers.jsonl"), [])
    active = card.tables_with_rows(str(dataset_dir))
    assert active == ['pits', 'cores', 'ect']
    info = card.dataset_info_yaml(active).replace(card.IMAGE_FEATURES_YAML, IMAGE_FEATURES_YAML)
    (dataset_dir / "README.md").write_text(
        f"---\n{info}\n{card.configs_yaml(active)}\n---\nFixture card.\n", encoding = 'utf-8'
    )
    assert read_jsonl(dataset_dir, 'layers') == []
    with pytest.raises(ValueError):
        read_load_dataset(dataset_dir, tmp_path, 'layers')
    _, rows = read_load_dataset(dataset_dir, tmp_path, 'ect')
    assert len(rows) == len(ECT)
