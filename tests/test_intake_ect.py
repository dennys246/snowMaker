"""
Tests for the intake side of the ECT schema: loading, lookup and GPS precision.

These exercise ``valve`` against a minimal intake folder rather than the image
pipeline, so they cover the ECT bookkeeping without needing photographs.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ect
import intake


SITE_LOGS = (
    "site,ascending_mountain,city_state_country,collector,coordinates,date,time,"
    "snowpack_depth,slope_face,slope_gradient,air_temperature,avalanches_spotted,"
    "wind_loading,notes\n"
    "0,Loveland Pass,\"Silver Plume, Colorado, USA\",Denny Schaedig,\"39.66, -105.88\","
    "1/12/25,11:40 AM MST,140,30,11,7,2,high,Pilot site\n"
    "5,Berthoud Pass,\"Empire, Colorado, USA\",Denny Schaedig,\"39.79812, -105.77641\","
    "2/03/25,10:15 AM MST,155,95,28,12,0,moderate,Second site\n"
)

SITE_TEMPS = "site,column,core,core_temperature\n0,1,1,41.0\n5,1,1,28.0\n"

MANIFEST = "image,original_filename,image_type\n"

# Site 5 column 1 was tested; site 0 predates the protocol and has no row here.
SITE_ECTS = (
    "site,column,ect_result,ect_taps,ect_failure_depth_cm,ect_failure_grain,"
    "ect_hardness_above,ect_hardness_below,ect_operator\n"
    "5,1,P,12,62.3,FC,1F,F,Denny Schaedig\n"
    "5,2,X,,,,,,Denny Schaedig\n"
)


def build_dataset(tmp_path, site_ects = SITE_ECTS):
    """A minimal dataset folder with just the intake CSVs ``valve`` reads."""
    root = tmp_path / "rmsnow"
    (root / "intake").mkdir(parents = True)
    (root / "raw" / "magnified_profiles").mkdir(parents = True)
    (root / "raw" / "crystal_cards").mkdir(parents = True)

    (root / "intake" / "image_manifest.csv").write_text(MANIFEST, encoding = 'utf-8')
    (root / "intake" / "site_logs.csv").write_text(SITE_LOGS, encoding = 'utf-8')
    (root / "intake" / "site_temps.csv").write_text(SITE_TEMPS, encoding = 'utf-8')
    if site_ects is not None:
        (root / "intake" / "site_ects.csv").write_text(site_ects, encoding = 'utf-8')

    return intake.valve(str(root).replace('\\', '/'))


def test_ect_lookup_is_keyed_on_site_and_column(tmp_path):
    valve = build_dataset(tmp_path)

    tested = valve.ect_for(5, 1)
    assert tested['ect_result'] == 'P'
    assert tested['ect_taps'] == 12
    assert tested['ect_failure_depth_cm'] == 62.3
    assert tested['ect_failure_grain'] == 'FC'
    assert tested['ect_hardness_above'] == '1F'
    assert tested['ect_hardness_below'] == 'F'
    assert tested['ect_operator'] == 'Denny Schaedig'

    # A second column of the same pit is a separate test
    no_fracture = valve.ect_for(5, 2)
    assert no_fracture['ect_result'] == 'X'
    for column in ect.ECT_FRACTURE_COLUMNS:
        assert no_fracture[column] is None
    assert no_fracture['ect_operator'] == 'Denny Schaedig'


def test_untested_column_reads_as_null_not_X(tmp_path):
    """A site dug before the protocol is *not tested*, which is not an 'X'."""
    valve = build_dataset(tmp_path)

    untested = valve.ect_for(0, 1)
    assert untested == ect.null_ect()
    assert untested['ect_result'] is None
    assert untested['ect_result'] != 'X'
    assert untested['ect_taps'] != 0


def test_missing_ect_file_is_not_an_error(tmp_path):
    """Sites collected before the ECT table existed still load."""
    valve = build_dataset(tmp_path, site_ects = None)
    assert valve.ect_for(5, 1) == ect.null_ect()
    assert valve.ects.empty


def test_lookup_is_a_copy_not_a_shared_reference(tmp_path):
    """Mutating one row's ECT values must not rewrite the whole snow column."""
    valve = build_dataset(tmp_path)
    first = valve.ect_for(5, 1)
    first['ect_taps'] = 99
    assert valve.ect_for(5, 1)['ect_taps'] == 12


def test_out_of_domain_result_is_rejected_on_load(tmp_path):
    bad = (
        "site,column,ect_result,ect_taps,ect_failure_depth_cm,ect_failure_grain,"
        "ect_hardness_above,ect_hardness_below,ect_operator\n"
        "5,1,ECTP12,12,62.3,FC,1F,F,Denny Schaedig\n"
    )
    with pytest.raises(ect.ECTValidationError, match = "ect_result must be one of"):
        build_dataset(tmp_path, site_ects = bad)


def test_tap_count_without_a_tapped_result_is_rejected_on_load(tmp_path):
    bad = (
        "site,column,ect_result,ect_taps,ect_failure_depth_cm,ect_failure_grain,"
        "ect_hardness_above,ect_hardness_below,ect_operator\n"
        "5,1,PV,4,62.3,FC,1F,F,Denny Schaedig\n"
    )
    with pytest.raises(ect.ECTValidationError, match = "before any tap"):
        build_dataset(tmp_path, site_ects = bad)


def test_conflicting_rows_for_one_column_raise_rather_than_first_wins(tmp_path):
    """Two different tests recorded against one snow column is a loud error."""
    conflicting = (
        "site,column,ect_result,ect_taps,ect_failure_depth_cm,ect_failure_grain,"
        "ect_hardness_above,ect_hardness_below,ect_operator\n"
        "5,1,P,12,62.3,FC,1F,F,Denny Schaedig\n"
        "5,1,N,23,62.3,FC,1F,F,Denny Schaedig\n"
    )
    with pytest.raises(ect.ECTValidationError) as caught:
        build_dataset(tmp_path, site_ects = conflicting)
    message = str(caught.value)
    assert 'not constant' in message
    assert 'ect_result' in message and 'ect_taps' in message


def test_missing_ect_column_in_the_csv_is_rejected(tmp_path):
    incomplete = "site,column,ect_result,ect_taps\n5,1,P,12\n"
    with pytest.raises(ect.ECTValidationError, match = "missing required columns"):
        build_dataset(tmp_path, site_ects = incomplete)


# --- GPS precision ----------------------------------------------------------

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
    assert "site 0 latitude" in printed
    assert "site 0 longitude" in printed


def test_coordinates_parse_without_a_space_after_the_comma(tmp_path):
    valve = build_dataset(tmp_path)
    assert valve.parse_coordinates("39.79812,-105.77641", 5) == [39.79812, -105.77641]


def test_malformed_coordinates_raise(tmp_path):
    valve = build_dataset(tmp_path)
    with pytest.raises(ValueError, match = "latitude, longitude"):
        valve.parse_coordinates("39.79812", 5)


# --- CSV round trip ---------------------------------------------------------

def test_ect_survives_a_save_and_reload(tmp_path):
    """
    The riskiest path: 'X' and blank both go through a CSV cell.

    A blank cell comes back from pandas as NaN and must read as *not tested*,
    while an 'X' must come back as 'X'. If those ever collapsed into each other
    the dataset would look like it had stability measurements it does not have.
    """
    import pandas as pd

    valve = build_dataset(tmp_path, site_ects = None)
    assert valve.ects.empty

    for row in (
        {'site': 8, 'column': 1, 'ect_result': 'P', 'ect_taps': 12,
         'ect_failure_depth_cm': 62.3, 'ect_failure_grain': 'FC',
         'ect_hardness_above': '1F', 'ect_hardness_below': 'F',
         'ect_operator': 'Denny Schaedig'},
        {'site': 8, 'column': 2, 'ect_result': 'X', 'ect_operator': 'Denny Schaedig'},
    ):
        values = ect.normalize_ect(row, context = "fixture: ")
        values.update({key: row[key] for key in ect.ECT_GROUP_KEYS})
        valve.ects = pd.concat([valve.ects, pd.DataFrame([values])], ignore_index = True)
    valve.build_ect_lookup(valve.ects, source = "fixture")

    valve.save_state()

    reloaded = intake.valve(str(tmp_path / "rmsnow").replace('\\', '/'))

    assert reloaded.ect_for(8, 1) == valve.ect_for(8, 1)
    assert reloaded.ect_for(8, 1)['ect_taps'] == 12
    assert reloaded.ect_for(8, 1)['ect_failure_depth_cm'] == 62.3

    # 'X' came back as 'X', and its blank cells came back as null -- not as 0,
    # not as an empty string
    no_fracture = reloaded.ect_for(8, 2)
    assert no_fracture['ect_result'] == 'X'
    for column in ect.ECT_FRACTURE_COLUMNS:
        assert no_fracture[column] is None
    assert no_fracture['ect_operator'] == 'Denny Schaedig'

    # A column that was never in the table is still not tested
    assert reloaded.ect_for(9, 1) == ect.null_ect()
