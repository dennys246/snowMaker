"""
``metadata/pits.jsonl`` -- one row per site-day.

A pit is one hole dug on one day at one place. Everything that is true of the
whole pit lives here, once, instead of being repeated across every image row of
that site (the image table keeps its legacy copies unchanged; this table is the
canonical grain, and the two are checked against each other on write).

Two fields exist because free text is not a flag:

``method_deviation`` (required)
    Site 0 was dug with a modified pilot coring method and its ``notes`` said
    "advised not to use in higher level models". Every analysis consumed it
    anyway, and it carried an entire correlation result on its own. A boolean
    can be filtered on; a sentence cannot.

``collector_id`` (required)
    Operator effect was structurally unidentifiable with one collector across
    every row. It stays unidentifiable until a second value appears here.

Elevation deliberately has no column: it is recoverable from ``coordinates``
via a DEM, provided the coordinates carry enough precision -- which is why
they are stored at full source precision and flagged below four decimals.
Aspect is **not** DEM-recoverable, hence ``aspect_deg_true``.
"""

from schema import (MIN_COORDINATE_DECIMALS, TIME_PATTERN, Field,
                    SchemaError, TableSpec, coordinate_decimals)

GROUND_COVERS = ("grass", "talus", "rock", "krummholz")

FIELDS = [
    Field("site", "int", nullable = False, minimum = 0,
          description = "Site number; one site is one pit on one day"),
    Field("collector_id", "string", nullable = False,
          description = "Stable identifier of the person who dug and sampled the pit"),
    Field("corer_id", "string",
          description = "Corer used, type and diameter (e.g. `apple-corer-40mm`); different corers fragment differently"),
    Field("coordinates", "coordinates", nullable = False,
          domain_note = "`[latitude, longitude]`, WGS84 decimal degrees, full receiver precision",
          description = "Pit location. Never rounded; ingest flags fewer than 4 decimal places"),
    Field("aspect_deg_true", "float", minimum = 0.0, maximum = 360.0,
          description = "Slope aspect at the pit, degrees true (not magnetic). Not DEM-recoverable"),
    Field("slope_angle_deg", "float", nullable = False, minimum = 0.0, maximum = 90.0,
          description = "Slope angle at the pit, degrees"),
    Field("date", "date", nullable = False,
          description = "Local calendar date the pit was dug"),
    Field("time_of_day", "string", nullable = False, pattern = TIME_PATTERN,
          description = "Local time sampling started, with UTC offset (near-surface snow at 14:00 is not the snow at 08:00)"),
    Field("ground_cover", "enum", domain = GROUND_COVERS,
          description = "Ground surface under the snowpack"),
    Field("snowpack_depth_cm", "float", nullable = False, minimum = 0.0,
          description = "Total snowpack depth at the pit, cm"),
    Field("method_deviation", "bool", nullable = False,
          description = "True if the sampling method deviated from the standard protocol in any way. Filter on this, not on free text"),
    Field("method_deviation_note", "string",
          description = "What deviated and how. Only in addition to the flag, never instead of it"),
]


def record_rules(values, context = ""):
    if values["method_deviation_note"] is not None and not values["method_deviation"]:
        raise SchemaError(
            f"{context}method_deviation_note is recorded but method_deviation is "
            f"false; a deviation described in free text must also set the flag"
        )
    if values["method_deviation_note"] is not None and values["method_deviation_note"] == "":
        raise SchemaError(f"{context}method_deviation_note is an empty string; use null")


def group_rules(records, context = ""):
    """Flag pits whose stored coordinates carry too little precision for a DEM lookup."""
    warnings = []
    for record in records:
        for label, value in zip(("latitude", "longitude"), record["coordinates"]):
            decimals = coordinate_decimals(repr(float(value)))
            if decimals < MIN_COORDINATE_DECIMALS:
                warnings.append(
                    f"{context}pits site {record['site']}: {label} {value!r} carries "
                    f"only {decimals} decimal place(s); a DEM elevation lookup "
                    f"against it cannot be trusted"
                )
    return warnings


SPEC = TableSpec(
    name = "pits",
    key = ("site",),
    fields = FIELDS,
    record_rules = record_rules,
    group_rules = group_rules,
    grain = "one row per pit (site-day)",
)
