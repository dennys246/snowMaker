"""
``metadata/ect.jsonl`` -- the Extended Column Test, one row per **fracture**.

An ECT commonly produces several fractures (e.g. ECTN8 at 25 cm, then ECTP22 at
70 cm), so the row is keyed ``(site, column, test_index, fracture_index)``: one
test can have many fracture rows, and a column can be tested more than once. A
column that was **not tested has no row at all** -- that is what distinguishes
"not tested" from ``"X"`` (tested, nothing fractured in 30 taps), and the
distinction is safety-relevant: conflating them biases everything toward
"looks stable".

Recording notation (Simenhois & Birkeland, 2006)
------------------------------------------------
``PV``  propagated during isolation, before any tap
``P#``  fractured on tap ``#`` and propagated the full column
``N#``  fractured on tap ``#`` but did not propagate across
``X``   no fracture in 30 taps

``ect_result`` holds the letter, ``ect_taps`` the number, so ``ECTP12`` is
``("P", 12)``. ``ect_tap_convention`` records *which* tap is logged for a
propagating result -- the initiating tap or the next one -- because observers
differ and the two are one tap apart.

ECTX is right-censored, not maximum stability
--------------------------------------------
The ECT is reliable only in roughly the top metre. A deep persistent slab
returns a false-stable ``X`` -- the characteristic Colorado failure mode.
``ect_column_depth_cm`` and ``ect_suspected_pwl_below`` are therefore permitted
(and encouraged) on an ``X`` row, so analysis can censor those observations
instead of scoring them as "safest".

Deliberate non-goals
--------------------
No ``ect_propagated`` boolean, no derived ordinal or stability index. A P/N
binary measured *worse than no data* at realistic sample sizes. Store the raw
fields; downstream derives what it needs.

History
-------
Until 2026-08-25 these values lived as seven ``ect_*`` columns on the image
table at ``(site, column)`` grain (``LEGACY_IMAGE_COLUMNS``). They were null on
every row, and one column per test could not hold multiple fractures, so they
moved here before the first real test was recorded.
"""

from schema import Field, SchemaError, TableSpec
from layers import GRAIN_TYPES, HAND_HARDNESS

ECT_RESULTS = ("PV", "P", "N", "X")
ECT_TAPPED_RESULTS = ("P", "N")
ECT_TAPS_MIN, ECT_TAPS_MAX = 1, 30
TAP_CONVENTIONS = ("initiating", "propagating")

# Fracture character (van Herwijnen & Jamieson): sudden planar, sudden collapse,
# resistant planar, progressive compression, non-planar break -- or the older
# shear quality scale Q1 (clean, fast) to Q3 (rough, resistant). Record whichever
# the observer scored; neither is converted to the other.
FRACTURE_CHARACTERS = ("SP", "SC", "RP", "PC", "BRK", "Q1", "Q2", "Q3")

FIELDS = [
    Field("site", "int", nullable = False, minimum = 0, description = "Site number"),
    Field("column", "int", nullable = False, minimum = 1,
          description = "Snow column within the pit"),
    Field("test_index", "int", nullable = False, minimum = 1,
          description = "Which ECT on this column, 1 for the first"),
    Field("fracture_index", "int", nullable = False, minimum = 1,
          description = "Which fracture of this test, 1 for the first, in tap order. An `X` test has exactly one row"),
    Field("ect_result", "enum", nullable = False, domain = ECT_RESULTS,
          description = "Result class for this fracture; `X` means the whole test produced none"),
    Field("ect_taps", "int", minimum = ECT_TAPS_MIN, maximum = ECT_TAPS_MAX,
          description = "Tap the fracture occurred on; null for `PV` and `X`"),
    Field("ect_tap_convention", "enum", domain = TAP_CONVENTIONS,
          description = "Which tap is logged for a propagating result: the initiating tap or the next one. Required whenever `ect_taps` is set"),
    Field("ect_failure_depth_cm", "float", minimum = 0.0,
          description = "Depth of the failure interface below the snow surface, cm"),
    Field("ect_failure_grain", "enum", domain = GRAIN_TYPES,
          description = "ICSSG grain class at the failure interface"),
    Field("ect_hardness_above", "enum", domain = HAND_HARDNESS,
          description = "Hand hardness immediately above the failure interface"),
    Field("ect_hardness_below", "enum", domain = HAND_HARDNESS,
          description = "Hand hardness immediately below the failure interface"),
    Field("ect_fracture_character", "enum", domain = FRACTURE_CHARACTERS,
          description = "Fracture character (SP/SC/RP/PC/BRK) or shear quality (Q1-Q3)"),
    Field("ect_slope_angle_deg", "float", minimum = 0.0, maximum = 90.0,
          description = "Slope angle at the ECT column (not the pit), degrees"),
    Field("ect_column_depth_cm", "float", minimum = 0.0,
          description = "How deep the column was cut, cm. On an `X` this is the depth to which 'no fracture' was actually observed"),
    Field("ect_suspected_pwl_below", "bool",
          description = "Observer suspects a persistent weak layer below the tested depth. Lets an `X` be censored rather than read as stable"),
    Field("ect_operator", "string", nullable = False,
          description = "Who ran the test; tap force varies by person and by fatigue"),
]

# Describe a fracture; must be null when there was none.
FRACTURE_FIELDS = (
    "ect_taps",
    "ect_failure_depth_cm",
    "ect_failure_grain",
    "ect_hardness_above",
    "ect_hardness_below",
    "ect_fracture_character",
)

# Belong to the test, not the fracture; constant across a test's rows.
TEST_FIELDS = (
    "ect_tap_convention",
    "ect_slope_angle_deg",
    "ect_column_depth_cm",
    "ect_suspected_pwl_below",
    "ect_operator",
)

# The columns the image table carried until 2026-08-25, for the migration.
LEGACY_IMAGE_COLUMNS = (
    "ect_result",
    "ect_taps",
    "ect_failure_depth_cm",
    "ect_failure_grain",
    "ect_hardness_above",
    "ect_hardness_below",
    "ect_operator",
)


class ECTValidationError(SchemaError):
    """Kept as a name for callers written against the 2026-08-19 schema."""


def record_rules(values, context = ""):
    result = values["ect_result"]
    taps = values["ect_taps"]

    if result == "PV" and taps is not None:
        raise ECTValidationError(
            f"{context}ect_result 'PV' means the column propagated during isolation, "
            f"before any tap, so ect_taps must be null, got {taps!r}"
        )
    if result == "X":
        populated = [f for f in FRACTURE_FIELDS if values[f] is not None]
        if populated:
            raise ECTValidationError(
                f"{context}ect_result 'X' means no fracture in 30 taps, so "
                f"{list(FRACTURE_FIELDS)} must all be null, but {populated} are populated"
            )
    if result in ECT_TAPPED_RESULTS and taps is None:
        raise ECTValidationError(
            f"{context}ect_result {result!r} requires a tap count in "
            f"{ECT_TAPS_MIN}..{ECT_TAPS_MAX}, got null"
        )
    if taps is not None and values["ect_tap_convention"] is None:
        raise ECTValidationError(
            f"{context}ect_taps is {taps!r} but ect_tap_convention is null; record "
            f"whether the initiating or the propagating tap is logged"
        )
    depth = values["ect_failure_depth_cm"]
    cut = values["ect_column_depth_cm"]
    if depth is not None and cut is not None and depth > cut:
        raise ECTValidationError(
            f"{context}ect_failure_depth_cm {depth} is below ect_column_depth_cm "
            f"{cut}; a fracture cannot be deeper than the column was cut"
        )


def group_rules(records, context = ""):
    """Rows of one test are coherent: an X stands alone, fractures are in tap order."""
    tests = {}
    for record in records:
        key = (record["site"], record["column"], record["test_index"])
        tests.setdefault(key, []).append(record)

    for (site, column, test_index), rows in sorted(tests.items()):
        rows = sorted(rows, key = lambda row: row["fracture_index"])
        where = f"{context}ect (site {site}, column {column}, test {test_index}): "

        expected = list(range(1, len(rows) + 1))
        actual = [row["fracture_index"] for row in rows]
        if actual != expected:
            raise ECTValidationError(
                f"{where}fracture_index must run 1..{len(rows)}, got {actual}"
            )
        if any(row["ect_result"] == "X" for row in rows) and len(rows) > 1:
            raise ECTValidationError(
                f"{where}an 'X' result means the test produced no fracture, so it "
                f"must be the test's only row, but the test has {len(rows)} rows"
            )
        # PV rows precede tapped rows, and taps do not decrease along the test
        last_taps = 0
        seen_tapped = False
        for row in rows:
            if row["ect_result"] == "PV":
                if seen_tapped:
                    raise ECTValidationError(
                        f"{where}fracture {row['fracture_index']} is 'PV' (before any "
                        f"tap) but follows a tapped fracture; order fractures by tap"
                    )
                continue
            if row["ect_taps"] is not None:
                seen_tapped = True
                if row["ect_taps"] < last_taps:
                    raise ECTValidationError(
                        f"{where}fracture {row['fracture_index']} at tap "
                        f"{row['ect_taps']} follows a fracture at tap {last_taps}; "
                        f"order fractures by tap"
                    )
                last_taps = row["ect_taps"]
    return []


SPEC = TableSpec(
    name = "ect",
    key = ("site", "column", "test_index", "fracture_index"),
    fields = FIELDS,
    record_rules = record_rules,
    group_rules = group_rules,
    constant_within = {("site", "column", "test_index"): TEST_FIELDS},
    grain = "one row per ECT fracture; an untested column has no row",
)
