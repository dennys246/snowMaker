"""
``metadata/cores.jsonl`` -- one row per core, or per ladder rung.

The corer enters the pit wall **horizontally**, so a 10 cm core lies largely
within one snow layer and measures within-layer cohesion at one depth. Cores are
taken on a standard depth ladder (10, 20, 30, ... cm). A rung that was *not*
sampled still gets a row, with ``depth_skipped_reason`` set, so "not sampled"
never collapses into "sampled and found nothing".

``breakability_field_count`` is the number of pieces the core came out in,
**counted at extraction**. The legacy image-table ``segment`` numbering is a
photography artefact -- how many pieces were photographed for profile imaging --
and is a different measurement; nothing here is derived from it.

``recovery_quality`` separates "came out whole" from "fell apart and could not
be counted", which the image table conflates.

``layer_ids`` links a core to the layers it intersected in ``layers.jsonl``.
That link is what makes the layer table useful for the core imagery.
"""

from schema import Field, SchemaError, TableSpec

SKIP_REASONS = ("bottomed_out", "ice_layer", "time")
RECOVERY_QUALITIES = ("intact", "partial", "disintegrated", "not_recovered")

FIELDS = [
    Field("site", "int", nullable = False, minimum = 0, description = "Site number"),
    Field("column", "int", nullable = False, minimum = 1,
          description = "Snow column within the pit"),
    Field("core", "int", nullable = False, minimum = 1,
          description = "Core number within the column; the ladder rung, 1 at the top"),
    Field("core_depth_cm", "float", nullable = False, minimum = 0.0,
          description = "Depth of the core centre below the snow surface, cm, from the standard ladder"),
    Field("depth_skipped_reason", "enum", domain = SKIP_REASONS,
          description = "Set when this ladder rung was **not** sampled, and why. Null means the core was taken"),
    Field("breakability_field_count", "int", minimum = 1,
          description = "Pieces the core came out in, counted at extraction, independent of photography"),
    Field("recovery_quality", "enum", domain = RECOVERY_QUALITIES,
          description = "How the core came out of the wall"),
    Field("core_temperature_c", "float",
          description = "Core snow temperature, degrees C. Readings above 0 are stored as read and flagged on ingest"),
    Field("layer_ids", "int_list",
          description = "`layer_index` values in `layers.jsonl` (same site and column) this core intersected"),
]

# Measurements that a rung which was never sampled cannot carry.
SAMPLED_ONLY = ("breakability_field_count", "recovery_quality", "core_temperature_c")


def record_rules(values, context = ""):
    if values["depth_skipped_reason"] is not None:
        populated = [f for f in SAMPLED_ONLY if values[f] is not None]
        if populated:
            raise SchemaError(
                f"{context}depth_skipped_reason={values['depth_skipped_reason']!r} says "
                f"this rung was not sampled, but {populated} are populated"
            )
    quality = values["recovery_quality"]
    count = values["breakability_field_count"]
    if quality == "not_recovered" and (count is not None or values["core_temperature_c"] is not None):
        raise SchemaError(
            f"{context}recovery_quality 'not_recovered' means nothing came out, so "
            f"breakability_field_count and core_temperature_c must be null"
        )
    if quality == "disintegrated" and count is not None:
        raise SchemaError(
            f"{context}recovery_quality 'disintegrated' means the pieces could not be "
            f"counted, so breakability_field_count must be null, got {count!r}"
        )


def group_rules(records, context = ""):
    warnings = []
    for record in records:
        temperature = record["core_temperature_c"]
        if temperature is not None and temperature > 0:
            warnings.append(
                f"{context}cores (site {record['site']}, column {record['column']}, "
                f"core {record['core']}): core_temperature_c={temperature} is above "
                f"freezing; stored as read, but snow cannot be warmer than 0 C -- "
                f"treat as suspect"
            )
    return warnings


SPEC = TableSpec(
    name = "cores",
    key = ("site", "column", "core"),
    fields = FIELDS,
    record_rules = record_rules,
    group_rules = group_rules,
    grain = "one row per core (or per unsampled ladder rung)",
)


def fahrenheit_to_celsius(fahrenheit):
    """Convert a legacy core temperature reading from degrees F to degrees C.

    Every legacy reading is an integer number of degrees C that was written down
    in degrees F (41.0, 39.2, 35.6, ... are exactly 5, 4, 2 C), so rounding to one
    decimal place recovers the value as it was read off the thermometer rather
    than manufacturing digits the instrument never had.
    """
    return round((float(fahrenheit) - 32.0) * 5.0 / 9.0, 1)


def legacy_core_row(site, column, core, core_temperature_f = None):
    """The cores.jsonl row for a core that predates the field protocol.

    Only what the legacy pipeline actually recorded is filled: the ladder rung
    depth it already assigned (``core * 10`` cm) and, when a site_temps reading
    exists, that temperature in degrees C. Everything the protocol adds is null
    -- *not measured* -- never a placeholder.
    """
    return SPEC.validate({
        "site": int(site),
        "column": int(column),
        "core": int(core),
        "core_depth_cm": float(core) * 10.0,
        "depth_skipped_reason": None,
        "breakability_field_count": None,
        "recovery_quality": None,
        "core_temperature_c": None if core_temperature_f is None else fahrenheit_to_celsius(core_temperature_f),
        "layer_ids": None,
    }, context = f"legacy core (site {site}, column {column}, core {core}): ")
