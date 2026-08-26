"""
Tests for the intake side of the side tables: loading, per-site intake, CSV
round trip, GPS precision and assembling the published tables.

These exercise ``valve`` against a minimal intake folder rather than the image
pipeline, so they cover the bookkeeping without needing photographs.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cores
import ect
import intake
import schema


SITE_LOGS = (
    "site,ascending_mountain,city_state_country,collector,coordinates,date,time,"
    "snowpack_depth,slope_face,slope_gradient,air_temperature,avalanches_spotted,"
    "wind_loading,notes\n"
    "0,Loveland Pass,\"Silver Plume, Colorado, USA\",Denny Schaedig,\"39.66, -105.88\","
    "1/12/25,11:40 AM MST,140,30,11,7,2,high,Pilot site\n"
    "5,Berthoud Pass,\"Empire, Colorado, USA\",Denny Schaedig,\"39.79812, -105.77641\","
    "2/03/25,10:15 AM MST,155,95,28,12,0,medium,Second site\n"
)

SITE_TEMPS = "site,column,core,core_temperature\n0,1,1,41.0\n5,1,1,28.4\n"

MANIFEST = "image,original_filename,image_type\n"

SITE_PITS = (
    "site,collector_id,corer_id,coordinates,aspect_deg_true,slope_angle_deg,date,time_of_day,"
    "ground_cover,snowpack_depth_cm,method_deviation,method_deviation_note\n"
    "5,Denny Schaedig,apple-corer-40mm,\"39.79812, -105.77641\",95,28,2025-02-03,10:15-07:00,"
    "talus,155,false,\n"
)

SITE_CORES = (
    "site,column,core,core_depth_cm,depth_skipped_reason,breakability_field_count,"
    "recovery_quality,core_temperature_c,layer_ids\n"
    "5,1,1,10,,3,partial,-2,1\n"
    "5,1,2,20,bottomed_out,,,,\n"
)

SITE_LAYERS = (
    "site,column,layer_index,depth_top_cm,depth_bottom_cm,hand_hardness,grain_type,"
    "grain_size_mm,density_kg_m3,temperature_c,lwc\n"
    "5,1,1,0,40,4F,RG,0.5,,-4,D\n"
    "5,1,2,40,155,F,DH,2.5,,-1,\n"
)

# Column 1 was tested (two fractures); column 2 gave an X; column 3 was not tested.
SITE_ECT = (
    "site,column,test_index,fracture_index,ect_result,ect_taps,ect_tap_convention,"
    "ect_failure_depth_cm,ect_failure_grain,ect_hardness_above,ect_hardness_below,"
    "ect_fracture_character,ect_slope_angle_deg,ect_column_depth_cm,ect_suspected_pwl_below,"
    "ect_operator\n"
    "5,1,1,1,N,8,initiating,25,FC,1F,4F,SC,28,90,false,Denny Schaedig\n"
    "5,1,1,2,P,22,initiating,70,DH,P,F,SP,28,90,false,Denny Schaedig\n"
    "5,2,1,1,X,,,,,,,,27,90,true,Denny Schaedig\n"
)

FILES = {'pits': SITE_PITS, 'cores': SITE_CORES, 'layers': SITE_LAYERS, 'ect': SITE_ECT}


def build_dataset(tmp_path, master = None, site_folder = None):
    """
    A minimal dataset folder with the intake CSVs ``valve`` reads.

    ``master`` puts side-table CSVs in intake/ (the master tables); ``site_folder``
    puts them in intake/site_5/ (a site waiting to be intaken).
    """
    root = tmp_path / "rmsnow"
    (root / "intake").mkdir(parents = True, exist_ok = True)
    (root / "metadata").mkdir(exist_ok = True)
    (root / "raw" / "magnified_profiles").mkdir(parents = True, exist_ok = True)
    (root / "raw" / "crystal_cards").mkdir(parents = True, exist_ok = True)

    (root / "intake" / "image_manifest.csv").write_text(MANIFEST, encoding = 'utf-8')
    (root / "intake" / "site_logs.csv").write_text(SITE_LOGS, encoding = 'utf-8')
    (root / "intake" / "site_temps.csv").write_text(SITE_TEMPS, encoding = 'utf-8')
    for name, text in (master or {}).items():
        (root / "intake" / f"site_{name}.csv").write_text(text, encoding = 'utf-8')
    if site_folder:
        (root / "intake" / "site_5").mkdir(exist_ok = True)
        for name, text in site_folder.items():
            (root / "intake" / "site_5" / f"site_{name}.csv").write_text(text, encoding = 'utf-8')

    return intake.valve(str(root).replace('\\', '/'))


# --- Loading the master tables ------------------------------------------------

def test_master_side_tables_load_and_validate(tmp_path):
    valve = build_dataset(tmp_path, master = FILES)
    assert [row['site'] for row in valve.side_tables['pits']] == [5]
    assert valve.side_tables['pits'][0]['coordinates'] == [39.79812, -105.77641]
    assert valve.side_tables['pits'][0]['method_deviation'] is False
    assert valve.side_tables['pits'][0]['method_deviation_note'] is None

    by_key = {cores.SPEC.key_of(row): row for row in valve.side_tables['cores']}
    assert by_key[(5, 1, 1)]['breakability_field_count'] == 3
    assert by_key[(5, 1, 1)]['layer_ids'] == [1]
    assert by_key[(5, 1, 2)]['depth_skipped_reason'] == 'bottomed_out'
    assert by_key[(5, 1, 2)]['breakability_field_count'] is None

    tests = {ect.SPEC.key_of(row): row for row in valve.side_tables['ect']}
    assert tests[(5, 1, 1, 1)]['ect_taps'] == 8 and tests[(5, 1, 1, 2)]['ect_taps'] == 22
    assert tests[(5, 2, 1, 1)]['ect_result'] == 'X'
    assert tests[(5, 2, 1, 1)]['ect_suspected_pwl_below'] is True
    for column in ect.FRACTURE_FIELDS:
        assert tests[(5, 2, 1, 1)][column] is None
    # Column 3 was not tested: no row, which is not an 'X'
    assert not any(row['column'] == 3 for row in valve.side_tables['ect'])


def test_missing_master_files_start_empty(tmp_path):
    """Sites collected before the field protocol still load."""
    valve = build_dataset(tmp_path)
    assert all(rows == [] for rows in valve.side_tables.values())


def test_out_of_domain_value_is_rejected_on_load(tmp_path):
    bad = SITE_ECT.replace("5,1,1,1,N,8", "5,1,1,1,ECTN8,8")
    with pytest.raises(schema.SchemaError, match = "ect_result must be one of"):
        build_dataset(tmp_path, master = {'ect': bad})


def test_conflicting_test_rows_raise_rather_than_first_wins(tmp_path):
    conflicting = SITE_ECT.replace(
        "5,1,1,2,P,22,initiating,70,DH,P,F,SP,28,90,false,Denny Schaedig",
        "5,1,1,2,P,22,initiating,70,DH,P,F,SP,28,90,false,Someone Else",
    )
    with pytest.raises(schema.SchemaError, match = "not constant"):
        build_dataset(tmp_path, master = {'ect': conflicting})


def test_layer_gap_is_flagged_on_load(tmp_path, capsys):
    gapped = SITE_LAYERS.replace("5,1,2,40,155", "5,1,2,45,155")
    build_dataset(tmp_path, master = {'layers': gapped})
    assert "gap of 5 cm" in capsys.readouterr().out


# --- Per-site intake -----------------------------------------------------------

def test_site_folder_tables_are_added_to_the_masters(tmp_path):
    valve = build_dataset(tmp_path, site_folder = FILES)
    assert valve.side_tables['pits'] == []
    for spec in intake.SIDE_TABLES:
        valve.intake_side_table(spec, 5, "intake/site_5/")
    assert [row['site'] for row in valve.side_tables['pits']] == [5]
    assert len(valve.side_tables['cores']) == 2
    assert len(valve.side_tables['layers']) == 2
    assert len(valve.side_tables['ect']) == 3


def test_site_column_is_filled_from_the_folder_and_checked_against_it(tmp_path):
    without_site = SITE_PITS.replace("\n5,Denny", "\n,Denny")
    valve = build_dataset(tmp_path, site_folder = {'pits': without_site})
    valve.intake_side_table(intake.SIDE_TABLES[0], 5, "intake/site_5/")
    assert valve.side_tables['pits'][0]['site'] == 5

    wrong_site = SITE_PITS.replace("\n5,Denny", "\n6,Denny")
    valve = build_dataset(tmp_path, site_folder = {'pits': wrong_site})
    with pytest.raises(schema.SchemaError, match = "folder is site 5"):
        valve.intake_side_table(intake.SIDE_TABLES[0], 5, "intake/site_5/")


def test_a_site_already_in_the_master_is_skipped(tmp_path, capsys):
    valve = build_dataset(tmp_path, master = {'pits': SITE_PITS}, site_folder = {'pits': SITE_PITS})
    valve.intake_side_table(intake.SIDE_TABLES[0], 5, "intake/site_5/")
    assert len(valve.side_tables['pits']) == 1
    assert "already in the pits table" in capsys.readouterr().out


def test_missing_site_file_leaves_the_table_alone(tmp_path, capsys):
    valve = build_dataset(tmp_path, site_folder = {'pits': SITE_PITS})
    valve.intake_side_table(ect.SPEC, 5, "intake/site_5/")
    assert valve.side_tables['ect'] == []
    assert "No site_ect.csv for site 5" in capsys.readouterr().out


# --- CSV round trip -------------------------------------------------------------

def test_side_tables_survive_a_save_and_reload(tmp_path):
    """
    The riskiest path: 'X', blanks, booleans and lists all go through CSV cells.

    A blank cell must come back as null (not 0, not ''), 'X' as 'X', false as
    False and '1;2' as [1, 2]. If any of those collapsed the dataset would look
    like it had measurements it does not have.
    """
    valve = build_dataset(tmp_path, master = FILES)
    before = {name: list(rows) for name, rows in valve.side_tables.items()}
    valve.save_state()

    reloaded = intake.valve(str(tmp_path / "rmsnow").replace('\\', '/'))
    assert reloaded.side_tables == before

    x_row = next(row for row in reloaded.side_tables['ect'] if row['ect_result'] == 'X')
    assert x_row['ect_taps'] is None and x_row['ect_suspected_pwl_below'] is True
    skipped = next(row for row in reloaded.side_tables['cores'] if row['core'] == 2)
    assert skipped['depth_skipped_reason'] == 'bottomed_out'
    assert skipped['breakability_field_count'] is None
    assert reloaded.side_tables['pits'][0]['method_deviation'] is False
    assert reloaded.side_tables['pits'][0]['coordinates'] == [39.79812, -105.77641]


# --- GPS precision --------------------------------------------------------------

def test_coordinates_keep_full_recorded_precision(tmp_path, capsys):
    valve = build_dataset(tmp_path)
    coordinates = valve.parse_coordinates("39.79812, -105.77641", 5)
    assert coordinates == [39.79812, -105.77641]
    assert "WARNING" not in capsys.readouterr().out


def test_low_precision_coordinates_warn(tmp_path, capsys):
    """Site 0's two-decimal fix is a ~1.1 km box that straddles an elevation band."""
    valve = build_dataset(tmp_path)
    coordinates = valve.parse_coordinates("39.66, -105.88", 0)
    assert coordinates == [39.66, -105.88]

    printed = capsys.readouterr().out
    assert "WARNING" in printed
    assert "latitude 39.66" in printed
    assert "longitude -105.88" in printed


def test_coordinates_parse_without_a_space_after_the_comma(tmp_path):
    valve = build_dataset(tmp_path)
    assert valve.parse_coordinates("39.79812,-105.77641", 5) == [39.79812, -105.77641]


def test_malformed_coordinates_raise(tmp_path):
    valve = build_dataset(tmp_path)
    with pytest.raises(ValueError, match = "latitude, longitude"):
        valve.parse_coordinates("39.79812", 5)


def test_nothing_in_the_coordinate_path_rounds(tmp_path):
    """Seven places in, seven places out, through parse and CSV encode alike."""
    valve = build_dataset(tmp_path)
    parsed = valve.parse_coordinates("39.7981234, -105.7764187", 5)
    assert parsed == [39.7981234, -105.7764187]
    field = intake.pits.SPEC.by_name['coordinates']
    assert schema.cell_to_text(field, parsed) == "39.7981234, -105.7764187"


# --- Assembling the published tables ------------------------------------------

def image_row(site, column, core, number):
    return {'file_path': f'preprocessed/cores/image_{number}.png', 'datatype': 'core',
            'site': site, 'column': column, 'core': core, 'segment': -1,
            'core_temperature': 28.4 if site == 5 else None,
            'coordinates': [39.79812, -105.77641] if site == 5 else [39.66, -105.88],
            'snowpack_depth': 155.0 if site == 5 else 140.0, 'core_depth': core * 10.0,
            'slope_angle': 28.0 if site == 5 else 11.0}


def test_photographed_core_without_a_record_gets_a_legacy_row(tmp_path, capsys):
    valve = build_dataset(tmp_path, master = FILES)
    images = {'preprocessed': [image_row(5, 1, 1, 1), image_row(5, 1, 3, 2)], 'raw': []}
    tables = valve.build_side_tables(images)
    by_key = {cores.SPEC.key_of(row): row for row in tables['cores']}
    assert by_key[(5, 1, 1)]['breakability_field_count'] == 3, "the recorded core is kept"
    added = by_key[(5, 1, 3)]
    assert added['core_depth_cm'] == 30.0
    assert added['breakability_field_count'] is None and added['recovery_quality'] is None
    assert added['layer_ids'] is None and added['core_temperature_c'] is None
    assert "had no site_cores.csv record" in capsys.readouterr().out


def test_legacy_row_carries_the_site_temps_reading_in_celsius(tmp_path):
    valve = build_dataset(tmp_path, master = {'pits': SITE_PITS})
    tables = valve.build_side_tables({'preprocessed': [image_row(5, 1, 1, 1)], 'raw': []})
    assert tables['cores'][0]['core_temperature_c'] == -2.0   # 28.4 F


def test_a_site_with_images_but_no_pit_record_is_refused(tmp_path):
    valve = build_dataset(tmp_path)
    with pytest.raises(schema.SchemaError, match = "no row in pits.jsonl"):
        valve.build_side_tables({'preprocessed': [image_row(0, 1, 1, 1)], 'raw': []})


def test_published_rows_are_kept_and_conflicts_raise(tmp_path):
    valve = build_dataset(tmp_path, master = FILES)
    published = [dict(valve.side_tables['pits'][0], corer_id = 'different-corer')]
    schema.write_jsonl(str(tmp_path / "rmsnow" / "metadata" / "pits.jsonl"), published)
    with pytest.raises(schema.SchemaError, match = "recorded twice with different values"):
        valve.build_side_tables({'preprocessed': [], 'raw': []})


def test_legacy_ect_columns_are_stripped_only_when_null():
    row = {'file_path': 'a.png', 'ect_result': None, 'ect_taps': None, 'ect_failure_depth_cm': None,
           'ect_failure_grain': None, 'ect_hardness_above': None, 'ect_hardness_below': None,
           'ect_operator': None}
    assert intake.strip_legacy_ect(dict(row)) == {'file_path': 'a.png'}
    with pytest.raises(schema.SchemaError, match = "live in metadata/ect.jsonl now"):
        intake.strip_legacy_ect(dict(row, ect_result = 'X'))
