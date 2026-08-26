"""
Validation rules for the side tables: rejected loudly, never coerced.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cores
import ect
import layers
import pits
import schema


def pit(**overrides):
    row = {
        'site': 8, 'collector_id': 'collector-01', 'corer_id': None,
        'coordinates': [39.66412, -105.87903], 'aspect_deg_true': None,
        'slope_angle_deg': 31.0, 'date': '2026-12-20', 'time_of_day': '09:45-07:00',
        'ground_cover': None, 'snowpack_depth_cm': 145.0,
        'method_deviation': False, 'method_deviation_note': None,
    }
    row.update(overrides)
    return row


def layer(index, top, bottom, **overrides):
    row = {
        'site': 8, 'column': 1, 'layer_index': index, 'depth_top_cm': top,
        'depth_bottom_cm': bottom, 'hand_hardness': '4F', 'grain_type': 'RG',
        'grain_size_mm': None, 'density_kg_m3': None, 'temperature_c': None, 'lwc': None,
    }
    row.update(overrides)
    return row


def core(number, **overrides):
    row = {
        'site': 8, 'column': 1, 'core': number, 'core_depth_cm': number * 10.0,
        'depth_skipped_reason': None, 'breakability_field_count': None,
        'recovery_quality': None, 'core_temperature_c': None, 'layer_ids': None,
    }
    row.update(overrides)
    return row


def fracture(result, taps = None, fracture_index = 1, test_index = 1, **overrides):
    row = {
        'site': 8, 'column': 1, 'test_index': test_index, 'fracture_index': fracture_index,
        'ect_result': result, 'ect_taps': taps,
        'ect_tap_convention': 'initiating' if taps is not None else None,
        'ect_failure_depth_cm': None if result == 'X' else 40.0,
        'ect_failure_grain': None if result == 'X' else 'FC',
        'ect_hardness_above': None, 'ect_hardness_below': None,
        'ect_fracture_character': None, 'ect_slope_angle_deg': 30.0,
        'ect_column_depth_cm': 90.0, 'ect_suspected_pwl_below': None,
        'ect_operator': 'collector-01',
    }
    row.update(overrides)
    return row


# --- Domains: rejected, not coerced ------------------------------------------

@pytest.mark.parametrize("spec, raw, reason", [
    (ect.SPEC, fracture('ECTP', 12), "field notation, not the stored letter"),
    (ect.SPEC, fracture('P1'), "taps belong in ect_taps"),
    (ect.SPEC, fracture('p', 12), "case matters"),
    (ect.SPEC, fracture('P', 12, ect_failure_grain = 'facets'), "not an ICSSG code"),
    (ect.SPEC, fracture('P', 12, ect_hardness_above = 'medium'), "not a hardness class"),
    (ect.SPEC, fracture('P', 'twelve'), "taps must be an integer"),
    (ect.SPEC, fracture('P', 12.5), "taps must be whole"),
    (ect.SPEC, fracture('P', 12, ect_tap_convention = 'first'), "not a tap convention"),
    (ect.SPEC, fracture('P', 12, ect_fracture_character = 'clean'), "not a fracture character"),
    (ect.SPEC, fracture('P', 12, ect_suspected_pwl_below = 1), "a bool is not 1"),
    (ect.SPEC, fracture('P', 12, ect_suspected_pwl_below = 'yes'), "a bool is not yes"),
    (ect.SPEC, fracture('P', 12, ect_failure_depth_cm = -3.0), "depth below surface is positive"),
    (layers.SPEC, layer(1, 0, 10, hand_hardness = 'soft'), "not a hand hardness"),
    (layers.SPEC, layer(1, 0, 10, hand_hardness = 'f'), "case matters"),
    (layers.SPEC, layer(1, 0, 10, grain_type = 'facets'), "not an ICSSG code"),
    (layers.SPEC, layer(1, 0, 10, lwc = 'wet'), "not an LWC class"),
    (cores.SPEC, core(1, recovery_quality = 'whole'), "not a recovery quality"),
    (cores.SPEC, core(1, depth_skipped_reason = 'skipped'), "not a skip reason"),
    (cores.SPEC, core(1, breakability_field_count = 0), "a count starts at 1"),
    (cores.SPEC, core(1, layer_ids = []), "an empty list is not null"),
    (cores.SPEC, core(1, layer_ids = [1, 1]), "duplicate layer ids"),
    (cores.SPEC, core(1, layer_ids = 'one'), "not a list of ints"),
    (pits.SPEC, pit(ground_cover = 'meadow'), "not a ground cover"),
    (pits.SPEC, pit(method_deviation = 'yes'), "a bool is not yes"),
    (pits.SPEC, pit(method_deviation = 0), "a bool is not 0"),
    (pits.SPEC, pit(date = '1/12/25'), "not ISO"),
    (pits.SPEC, pit(time_of_day = '11:40 AM MST'), "not HH:MM+offset"),
    (pits.SPEC, pit(coordinates = [39.66]), "needs two numbers"),
    (pits.SPEC, pit(coordinates = [95.0, -105.0]), "latitude out of range"),
    (pits.SPEC, pit(aspect_deg_true = 400.0), "aspect out of range"),
    (pits.SPEC, pit(slope_angle_deg = -1.0), "slope out of range"),
])
def test_out_of_domain_values_rejected_not_coerced(spec, raw, reason):
    with pytest.raises(schema.SchemaError):
        spec.normalize(raw, context = f"{reason}: ")


def test_unknown_column_is_rejected_so_a_typo_cannot_vanish():
    with pytest.raises(schema.SchemaError, match = "unknown column"):
        pits.SPEC.normalize(pit(colector_id = 'x'))


def test_required_value_missing_is_rejected():
    with pytest.raises(schema.SchemaError, match = "missing required"):
        pits.SPEC.normalize(pit(collector_id = None))
    with pytest.raises(schema.SchemaError, match = "missing required"):
        pits.SPEC.normalize(pit(method_deviation = None))
    with pytest.raises(schema.SchemaError, match = "missing required"):
        layers.SPEC.normalize(layer(1, 0, 10, hand_hardness = None))
    with pytest.raises(schema.SchemaError, match = "missing required"):
        ect.SPEC.normalize(fracture('P', 12, ect_operator = None))


def test_blank_csv_cells_read_as_null_not_as_values():
    """pandas/csv hand back '' and NaN for blank cells; both mean not measured."""
    raw = core(1, breakability_field_count = '', recovery_quality = '   ',
               core_temperature_c = float('nan'), layer_ids = '')
    values = cores.SPEC.normalize(raw)
    assert values['breakability_field_count'] is None
    assert values['recovery_quality'] is None
    assert values['core_temperature_c'] is None
    assert values['layer_ids'] is None


def test_csv_forms_are_parsed_exactly():
    values = cores.SPEC.normalize(core(1, layer_ids = '3;4;5', breakability_field_count = '4'))
    assert values['layer_ids'] == [3, 4, 5]
    assert values['breakability_field_count'] == 4
    values = pits.SPEC.normalize(pit(coordinates = '39.79812, -105.77641', method_deviation = 'true'))
    assert values['coordinates'] == [39.79812, -105.77641]
    assert values['method_deviation'] is True


# --- ECT cross-field rules ----------------------------------------------------

@pytest.mark.parametrize("raw, reason", [
    (fracture('PV', 3), "PV means it went before any tap"),
    (fracture('X', 30), "X means nothing fractured"),
    (fracture('X', ect_failure_grain = 'FC'), "X has no failure interface"),
    (fracture('X', ect_failure_depth_cm = 40.0), "X has no failure depth"),
    (fracture('X', ect_hardness_above = 'P'), "X has no interface to bracket"),
    (fracture('X', ect_fracture_character = 'SP'), "X has no fracture character"),
    (fracture('P'), "P needs a tap count"),
    (fracture('N'), "N needs a tap count"),
    (fracture('P', 0), "taps start at 1"),
    (fracture('P', 31), "taps stop at 30"),
    (fracture('P', 12, ect_tap_convention = None), "a tap count needs its convention"),
    (fracture('P', 12, ect_failure_depth_cm = 95.0, ect_column_depth_cm = 90.0),
     "a fracture cannot be deeper than the cut"),
])
def test_invalid_ect_records_rejected(raw, reason):
    with pytest.raises(schema.SchemaError):
        ect.SPEC.normalize(raw, context = f"{reason}: ")


def test_X_permits_the_censoring_fields():
    values = ect.SPEC.normalize(fracture('X', ect_column_depth_cm = 85.0, ect_suspected_pwl_below = True))
    assert values['ect_column_depth_cm'] == 85.0
    assert values['ect_suspected_pwl_below'] is True


def test_X_must_be_the_only_row_of_its_test():
    rows = [fracture('X'), fracture('N', 8, fracture_index = 2)]
    with pytest.raises(schema.SchemaError, match = "only row"):
        ect.SPEC.validate_table(rows)


def test_fractures_must_be_in_tap_order():
    with pytest.raises(schema.SchemaError, match = "order fractures by tap"):
        ect.SPEC.validate_table([fracture('P', 22), fracture('N', 8, fracture_index = 2)])
    with pytest.raises(schema.SchemaError, match = "order fractures by tap"):
        ect.SPEC.validate_table([fracture('N', 8), fracture('PV', fracture_index = 2)])
    with pytest.raises(schema.SchemaError, match = "fracture_index must run"):
        ect.SPEC.validate_table([fracture('N', 8), fracture('P', 22, fracture_index = 3)])


def test_per_test_fields_must_be_constant_within_a_test():
    rows = [fracture('N', 8), fracture('P', 22, fracture_index = 2, ect_operator = 'someone else')]
    with pytest.raises(schema.SchemaError) as caught:
        ect.SPEC.validate_table(rows)
    assert 'ect_operator' in str(caught.value)
    assert 'not constant' in str(caught.value)


def test_duplicate_keys_are_loud():
    with pytest.raises(schema.SchemaError, match = "duplicate key"):
        ect.SPEC.validate_table([fracture('N', 8), fracture('N', 8)])


# --- Layer rules ---------------------------------------------------------------

def test_layers_must_not_overlap():
    with pytest.raises(schema.SchemaError, match = "overlap"):
        layers.SPEC.validate_table([layer(1, 0, 30), layer(2, 25, 60)])


def test_layer_gaps_are_flagged_not_rejected():
    warnings = layers.SPEC.validate_table([layer(1, 0, 30), layer(2, 35, 60)])
    assert len(warnings) == 1 and 'gap of 5 cm' in warnings[0]
    warnings = layers.SPEC.validate_table([layer(1, 10, 30), layer(2, 30, 60)])
    assert len(warnings) == 1 and 'top 10 cm is unprofiled' in warnings[0]


def test_layer_indices_must_be_contiguous_from_one():
    with pytest.raises(schema.SchemaError, match = "layer_index must run"):
        layers.SPEC.validate_table([layer(1, 0, 30), layer(3, 30, 60)])


def test_layer_bottom_must_be_below_top():
    with pytest.raises(schema.SchemaError, match = "must be below"):
        layers.SPEC.normalize(layer(1, 30, 30))


def test_above_freezing_temperatures_are_stored_and_flagged():
    warnings = layers.SPEC.validate_table([layer(1, 0, 30, temperature_c = 2.0)])
    assert len(warnings) == 1 and 'above freezing' in warnings[0]
    warnings = cores.SPEC.validate_table([core(1, core_temperature_c = 5.0)])
    assert len(warnings) == 1 and 'above freezing' in warnings[0]


# --- Core rules ----------------------------------------------------------------

@pytest.mark.parametrize("raw, reason", [
    (core(3, depth_skipped_reason = 'bottomed_out', breakability_field_count = 2), "a skipped rung has no count"),
    (core(3, depth_skipped_reason = 'time', recovery_quality = 'intact'), "a skipped rung has no recovery"),
    (core(3, depth_skipped_reason = 'ice_layer', core_temperature_c = -2.0), "a skipped rung has no temperature"),
    (core(1, recovery_quality = 'not_recovered', breakability_field_count = 1), "nothing came out to count"),
    (core(1, recovery_quality = 'disintegrated', breakability_field_count = 5), "could not be counted"),
])
def test_invalid_core_records_rejected(raw, reason):
    with pytest.raises(schema.SchemaError):
        cores.SPEC.normalize(raw, context = f"{reason}: ")


def test_skipped_rung_may_still_name_its_layers():
    values = cores.SPEC.normalize(core(3, depth_skipped_reason = 'ice_layer', layer_ids = [2]))
    assert values['layer_ids'] == [2]


# --- Pit rules -----------------------------------------------------------------

def test_deviation_note_without_the_flag_is_rejected():
    """Free text is not a flag: a note must come with method_deviation = true."""
    with pytest.raises(schema.SchemaError, match = "must also set the flag"):
        pits.SPEC.normalize(pit(method_deviation = False, method_deviation_note = 'used a shovel'))


def test_low_precision_coordinates_are_flagged():
    warnings = pits.SPEC.validate_table([pit(site = 0, coordinates = [39.66, -105.88])])
    assert len(warnings) == 2
    assert 'latitude 39.66' in warnings[0] and 'longitude -105.88' in warnings[1]
    assert pits.SPEC.validate_table([pit()]) == []


# --- Cross-table rules ---------------------------------------------------------

def test_layer_ids_must_reference_existing_layers_in_the_same_column():
    tables = {
        'pits': [pit()],
        'layers': [layer(1, 0, 30), layer(2, 30, 60)],
        'cores': [core(1, layer_ids = [1]), core(2, layer_ids = [3])],
        'ect': [],
    }
    with pytest.raises(schema.SchemaError, match = r"layer_ids \[3\]"):
        schema.validate_dataset(tables)
    tables['cores'][1]['layer_ids'] = [2]
    assert schema.validate_dataset(tables) == []
    # Same index, different column: not the same layer
    tables['cores'][1]['column'] = 2
    with pytest.raises(schema.SchemaError, match = "layer_ids"):
        schema.validate_dataset(tables)


def test_every_table_needs_a_pit_record():
    with pytest.raises(schema.SchemaError, match = "no row in pits.jsonl"):
        schema.validate_dataset({'pits': [], 'layers': [], 'cores': [core(1)], 'ect': []})
    with pytest.raises(schema.SchemaError, match = "no row in pits.jsonl"):
        schema.validate_dataset({'pits': [], 'layers': [], 'cores': [], 'ect': [fracture('X')]})


def image(site = 8, column = 1, number = 1, **overrides):
    row = {'file_path': f'preprocessed/cores/image_{number}.png', 'site': site, 'column': column,
           'core': number, 'coordinates': [39.66412, -105.87903], 'snowpack_depth': 145.0,
           'core_depth': number * 10.0, 'slope_angle': 31.0}
    row.update(overrides)
    return row


def test_image_rows_must_reference_pits_and_cores():
    tables = {'pits': [pit()], 'layers': [], 'cores': [core(1)], 'ect': []}
    assert schema.validate_dataset(tables, image_rows = [image()]) == []
    with pytest.raises(schema.SchemaError, match = "no row in cores.jsonl"):
        schema.validate_dataset(tables, image_rows = [image(number = 2)])
    with pytest.raises(schema.SchemaError, match = "no row in pits.jsonl"):
        schema.validate_dataset(tables, image_rows = [image(site = 9)])


def test_a_skipped_rung_cannot_have_photographs():
    tables = {'pits': [pit()], 'layers': [], 'cores': [core(1, depth_skipped_reason = 'time')], 'ect': []}
    with pytest.raises(schema.SchemaError, match = "skipped ladder rungs"):
        schema.validate_dataset(tables, image_rows = [image()])


def test_image_table_pit_values_must_agree_with_pits_jsonl():
    tables = {'pits': [pit()], 'layers': [], 'cores': [core(1)], 'ect': []}
    with pytest.raises(schema.SchemaError, match = "disagrees with pits.jsonl snowpack_depth_cm"):
        schema.validate_dataset(tables, image_rows = [image(snowpack_depth = 140.0)])
    with pytest.raises(schema.SchemaError, match = "disagrees with pits.jsonl coordinates"):
        schema.validate_dataset(tables, image_rows = [image(coordinates = [39.66, -105.88])])
    with pytest.raises(schema.SchemaError, match = "disagrees with cores.jsonl core_depth_cm"):
        schema.validate_dataset(tables, image_rows = [image(core_depth = 15.0)])
