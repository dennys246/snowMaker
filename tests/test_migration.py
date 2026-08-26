"""
Tests for scripts/migrate_side_tables.py against a fixture shaped like the
published dataset on 2026-08-25: image rows carrying seven null ect_* columns.
"""

import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cores
import ect
import intake
import schema

spec = importlib.util.spec_from_file_location(
    "migrate_side_tables", os.path.join(ROOT, "scripts", "migrate_side_tables.py")
)
migrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migrate)


def legacy_row(number, datatype, site, column, core, segment, temperature):
    """An image row exactly as published, ect_* included."""
    per_site = {
        0: ([39.66, -105.88], '1/12/25', '11:40 AM MST', 140.0, 30.0, 11.0, 'high',
            'Pilot using slightly modified coring method that took too large segment for '
            'snow profile picturing, advised not to use in higher level models. Two avalanche '
            'on south face nearby.'),
        5: ([39.66334, -105.88593], '2/11/26', '12:00 PM MST', 82.0, 12.0, 13.0, 'high',
            '1 small avalanche spotted on the north side of the bowl'),
    }[site]
    coordinates, date, time, depth, face, angle, wind, notes = per_site
    folder = {'core': 'cores', 'profile': 'profiles', 'magnified_profile': 'magnified_profiles'}[datatype]
    return {
        'image': f'https://huggingface.co/datasets/RMDig/rocky_mountain_snowpack/resolve/main/preprocessed/{folder}/image_{number}.png',
        'file_path': f'preprocessed/{folder}/image_{number}.png',
        'datatype': datatype, 'site': site, 'column': column, 'core': core, 'segment': segment,
        'core_temperature': temperature, 'air_temperature': 7.0,
        'ascending_mountain': 'Loveland Pass', 'city_state_country': 'Silver Plume, Colorado, USA',
        'collector': 'Denny Schaedig', 'coordinates': coordinates, 'date': date, 'time': time,
        'snowpack_depth': depth, 'core_depth': core * 10.0, 'slope_face': face, 'slope_angle': angle,
        'avalanches_spotted': 2, 'wind_loading': wind, 'notes': notes,
        'ect_result': None, 'ect_taps': None, 'ect_failure_depth_cm': None,
        'ect_failure_grain': None, 'ect_hardness_above': None, 'ect_hardness_below': None,
        'ect_operator': None,
    }


PREPROCESSED = [
    legacy_row(1, 'core', 0, 1, 1, -1, 41.0),
    legacy_row(2, 'profile', 0, 1, 1, 1, 41.0),
    legacy_row(3, 'core', 5, 1, 1, -1, 15.8),
    legacy_row(4, 'core', 5, 1, 2, -1, 17.6),
    legacy_row(5, 'magnified_profile', 5, 2, 1, 1, None),
]
RAW = [
    dict(legacy_row(1, 'core', 0, 1, 1, -1, 41.0), datatype = 'crystal_card',
         file_path = 'raw/crystal_cards/image_1.png'),
    dict(legacy_row(5, 'magnified_profile', 5, 2, 1, 1, None),
         file_path = 'raw/magnified_profiles/image_5.png'),
]


@pytest.fixture
def dataset_dir(tmp_path):
    root = tmp_path / "rmsnow"
    (root / "metadata").mkdir(parents = True)
    schema.write_jsonl(str(root / "metadata" / "preprocessed.jsonl"), PREPROCESSED)
    schema.write_jsonl(str(root / "metadata" / "raw.jsonl"), RAW)
    return root


def read(dataset_dir, name):
    return schema.read_jsonl(str(dataset_dir / "metadata" / f"{name}.jsonl"))


def test_check_mode_writes_nothing(dataset_dir):
    before = {name: read(dataset_dir, name) for name in ('preprocessed', 'raw')}
    report = migrate.migrate(str(dataset_dir), check_only = True, log = lambda *_: None)
    assert report['side']['pits']['state'] == 'absent'
    assert {name: read(dataset_dir, name) for name in before} == before
    for spec in intake.SIDE_TABLES:
        assert not (dataset_dir / spec.filename).exists()


def test_image_tables_lose_only_the_seven_null_columns(dataset_dir):
    before = {name: read(dataset_dir, name) for name in ('preprocessed', 'raw')}
    migrate.migrate(str(dataset_dir), log = lambda *_: None)
    for name, rows in before.items():
        after = read(dataset_dir, name)
        assert len(after) == len(rows)
        for original, stripped in zip(rows, after):
            expected = {k: v for k, v in original.items() if k not in ect.LEGACY_IMAGE_COLUMNS}
            assert stripped == expected
            assert list(stripped) == list(expected), "column order preserved"


def test_pits_are_built_from_the_repeated_pit_values(dataset_dir):
    migrate.migrate(str(dataset_dir), log = lambda *_: None)
    pits_rows = read(dataset_dir, 'pits')
    assert [row['site'] for row in pits_rows] == [0, 5]

    pilot, later = pits_rows
    assert pilot['coordinates'] == [39.66, -105.88], "precision preserved as recorded, low as it is"
    assert later['coordinates'] == [39.66334, -105.88593]
    assert pilot['date'] == '2025-01-12' and pilot['time_of_day'] == '11:40-07:00'
    assert later['date'] == '2026-02-11' and later['time_of_day'] == '12:00-07:00'
    assert pilot['snowpack_depth_cm'] == 140.0 and pilot['slope_angle_deg'] == 11.0
    assert pilot['collector_id'] == later['collector_id'] == 'Denny Schaedig'

    # The structured flag replaces the free-text warning
    assert pilot['method_deviation'] is True
    assert pilot['method_deviation_note'].startswith('Pilot using slightly modified coring method')
    assert later['method_deviation'] is False and later['method_deviation_note'] is None

    # Nothing the protocol adds is invented
    for row in pits_rows:
        assert row['corer_id'] is None and row['aspect_deg_true'] is None and row['ground_cover'] is None


def test_cores_are_built_from_the_image_keys(dataset_dir):
    migrate.migrate(str(dataset_dir), log = lambda *_: None)
    rows = read(dataset_dir, 'cores')
    assert [cores.SPEC.key_of(row) for row in rows] == [(0, 1, 1), (5, 1, 1), (5, 1, 2), (5, 2, 1)]
    by_key = {cores.SPEC.key_of(row): row for row in rows}
    assert by_key[(0, 1, 1)]['core_depth_cm'] == 10.0 and by_key[(5, 1, 2)]['core_depth_cm'] == 20.0
    # 41.0 F -> 5.0 C (the pilot's suspect above-freezing reading, kept as read)
    assert by_key[(0, 1, 1)]['core_temperature_c'] == 5.0
    assert by_key[(5, 1, 1)]['core_temperature_c'] == -9.0
    assert by_key[(5, 1, 2)]['core_temperature_c'] == -8.0
    assert by_key[(5, 2, 1)]['core_temperature_c'] is None, "no reading stays no reading"
    for row in rows:
        assert row['depth_skipped_reason'] is None, "every legacy core was sampled"
        assert row['breakability_field_count'] is None
        assert row['recovery_quality'] is None
        assert row['layer_ids'] is None


def test_layers_and_ect_start_empty(dataset_dir):
    migrate.migrate(str(dataset_dir), log = lambda *_: None)
    assert read(dataset_dir, 'layers') == []
    assert read(dataset_dir, 'ect') == []


def test_nothing_backfilled_is_X_zero_empty_or_false(dataset_dir):
    migrate.migrate(str(dataset_dir), log = lambda *_: None)
    for spec in intake.SIDE_TABLES:
        for row in read(dataset_dir, spec.name):
            for field in spec.fields:
                if not field.nullable:
                    continue
                value = row[field.name]
                if field.name == 'method_deviation_note' and row.get('site') == 0:
                    continue
                if field.name == 'core_temperature_c':
                    continue
                assert value is None, (spec.name, field.name, value)
                assert not migrate.is_forbidden(value)


def test_migrated_dataset_validates(dataset_dir):
    migrate.migrate(str(dataset_dir), log = lambda *_: None)
    tables = intake.validate_metadata_dir(str(dataset_dir))
    assert set(tables) == {'raw', 'preprocessed', 'pits', 'layers', 'cores', 'ect'}


def test_rerun_is_a_no_op(dataset_dir):
    migrate.migrate(str(dataset_dir), log = lambda *_: None)
    snapshot = {name: read(dataset_dir, name) for name in ('preprocessed', 'raw', 'pits', 'layers', 'cores', 'ect')}
    report = migrate.migrate(str(dataset_dir), log = lambda *_: None)
    assert all(r['state'] == 'identical' for r in report['side'].values())
    assert all(r['stripped'] == 0 for r in report['image'].values())
    assert {name: read(dataset_dir, name) for name in snapshot} == snapshot


def test_rows_collected_after_migration_are_never_clobbered(dataset_dir):
    migrate.migrate(str(dataset_dir), log = lambda *_: None)
    collected = read(dataset_dir, 'cores') + [cores.legacy_core_row(5, 2, 2)]
    schema.write_jsonl(str(dataset_dir / "metadata" / "cores.jsonl"), collected)
    report = migrate.migrate(str(dataset_dir), log = lambda *_: None)
    assert report['side']['cores']['state'] == 'superset'
    assert read(dataset_dir, 'cores') == collected


def test_a_hand_edited_side_table_is_refused_not_overwritten(dataset_dir):
    migrate.migrate(str(dataset_dir), log = lambda *_: None)
    edited = read(dataset_dir, 'pits')
    edited[0]['snowpack_depth_cm'] = 141.0
    schema.write_jsonl(str(dataset_dir / "metadata" / "pits.jsonl"), edited)
    with pytest.raises(migrate.MigrationError, match = "will not be overwritten"):
        migrate.migrate(str(dataset_dir), log = lambda *_: None)
    assert read(dataset_dir, 'pits') == edited


def test_a_real_ect_value_on_the_image_table_is_refused(dataset_dir):
    rows = read(dataset_dir, 'preprocessed')
    rows[2]['ect_result'] = 'X'
    schema.write_jsonl(str(dataset_dir / "metadata" / "preprocessed.jsonl"), rows)
    with pytest.raises(schema.SchemaError, match = "live in metadata/ect.jsonl now"):
        migrate.migrate(str(dataset_dir), log = lambda *_: None)
    assert read(dataset_dir, 'preprocessed') == rows, "nothing written"


@pytest.mark.parametrize("text, expected", [
    ("1/12/25", "2025-01-12"), ("12/14/25", "2025-12-14"), ("2/22/26", "2026-02-22"),
])
def test_legacy_dates(text, expected):
    assert migrate.legacy_date(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("11:40 AM MST", "11:40-07:00"), ("12:06 PM MST", "12:06-07:00"),
    ("1:08 PM MST", "13:08-07:00"), ("12:00 PM MST", "12:00-07:00"),
    ("10:11 AM MDT", "10:11-06:00"),
])
def test_legacy_times(text, expected):
    assert migrate.legacy_time(text) == expected


def test_unknown_legacy_zone_is_an_error_not_a_guess():
    with pytest.raises(migrate.MigrationError, match = "zone"):
        migrate.legacy_time("11:40 AM PST")


@pytest.mark.parametrize("fahrenheit, celsius", [
    (41.0, 5.0), (39.2, 4.0), (35.6, 2.0), (32.0, 0.0), (30.2, -1.0), (15.8, -9.0), (50.0, 10.0),
])
def test_legacy_temperatures_recover_the_integer_celsius_reading(fahrenheit, celsius):
    assert cores.fahrenheit_to_celsius(fahrenheit) == celsius
