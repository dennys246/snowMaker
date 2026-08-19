"""
ECT (Extended Column Test) schema for the Rocky Mountain snowpack dataset.

This module is the single owner of the ECT column definitions, their value
domains and their validation rules. ``intake`` and the backfill/verification
scripts import from here so there is exactly one place the schema is written
down.

Grain
-----
One ECT is run per snow **column**, so the seven values live at ``(site,
column)`` grain and repeat across every row sharing that pair -- the same way
``snowpack_depth`` and ``slope_face`` already repeat.

Recording notation (Simenhois & Birkeland, 2006)
------------------------------------------------
``PV``  propagated during isolation, before any tap
``P#``  fractured on tap ``#`` **and** propagated the full column
``N#``  fractured on tap ``#`` but did **not** propagate across
``X``   no fracture in 30 taps

``ect_result`` stores the letter and ``ect_taps`` stores the number, so ``P12``
is recorded as ``("P", 12)``.

Deliberate non-goals
--------------------
There is no ``ect_propagated`` boolean and no derived ordinal / stability-index
column, and neither should be added. Collapsing the tap count to a P/N binary
measured *worse* than the existing CAIC baseline at every sample size (power
0.586 vs 0.639 at n=15) and 0.000 at n=7, because two levels cannot reach
p <= 0.05 there however clean the ordering. Keeping the tap count gives 0.881
at n=15. Store the raw fields; let analysis derive whatever scale it wants, so
the encoding can change without a schema migration. See
``snowGradient/docs/plans/ECT_PROTOCOL.md`` section 5.

``None`` vs ``"X"``
-------------------
These are different measurements and must never be conflated. ``None`` means
*not tested*. ``"X"`` means *tested, no fracture in 30 taps*. Every row
collected before this schema landed is ``None`` across all seven columns --
never ``"X"``, never ``0``, never ``""``.
"""

import math

# --- Value domains ----------------------------------------------------------

# The four ECT result classes. Anything outside this set is rejected, not
# coerced.
ECT_RESULTS = ("PV", "P", "N", "X")

# ICSSG grain codes (Fierz et al. 2009) for the grain type at the failure
# interface.
ECT_GRAIN_CODES = ("PP", "DF", "RG", "FC", "DH", "SH", "MF", "IF", "MM")

# Hand hardness scale, softest to hardest.
ECT_HARDNESS = ("F", "4F", "1F", "P", "K")

# Tap count bounds: 10 from the wrist, 10 from the elbow, 10 from the shoulder.
ECT_TAPS_MIN = 1
ECT_TAPS_MAX = 30

# The results that carry a tap count.
ECT_TAPPED_RESULTS = ("P", "N")

# --- Column definitions -----------------------------------------------------

# The seven columns, in schema order.
ECT_COLUMNS = (
    "ect_result",
    "ect_taps",
    "ect_failure_depth_cm",
    "ect_failure_grain",
    "ect_hardness_above",
    "ect_hardness_below",
    "ect_operator",
)

# The columns that must be null when the test found no fracture in 30 taps.
ECT_FRACTURE_COLUMNS = (
    "ect_taps",
    "ect_failure_depth_cm",
    "ect_failure_grain",
    "ect_hardness_above",
    "ect_hardness_below",
)

# The key an ECT record is grained on.
ECT_GROUP_KEYS = ("site", "column")

# Hugging Face ``Features`` dtypes, as they appear in the dataset card.
#
# ``ect_result``, the grain code and the two hardness columns are
# ``Value("string")`` and deliberately NOT ``ClassLabel``. A ClassLabel makes
# ``load_dataset()`` hand back integer codes while a direct read of
# ``metadata/preprocessed.jsonl`` hands back the strings that are physically in
# the file -- exactly the ``datatype`` discrepancy that caused a false alarm on
# 2026-08-17 -- and nulls under ClassLabel are hazardous besides
# (``encode_example(None)`` raises, and the value historically encoded as -1,
# which is trivially mistaken for a class index). These columns are null for
# every historical row, so that matters here more than anywhere else.
#
# ``ect_failure_depth_cm`` is float64, not the float32 used by the older
# columns, so a JSON literal like ``62.3`` reads back as ``62.3`` rather than
# ``62.29999923706055``. This is a new column, so no migration is involved.
ECT_FEATURE_DTYPES = {
    "ect_result": "string",
    "ect_taps": "int64",
    "ect_failure_depth_cm": "float64",
    "ect_failure_grain": "string",
    "ect_hardness_above": "string",
    "ect_hardness_below": "string",
    "ect_operator": "string",
}


class ECTValidationError(ValueError):
    """Raised when an ECT record violates the schema.

    Always raised rather than swallowed. A silently dropped or silently
    first-value-wins ECT record is worse than a crash: it looks like a
    measurement.
    """


# --- Null handling ----------------------------------------------------------

def is_null(value):
    """Is ``value`` an absent measurement?

    Treats ``None``, an empty/whitespace string and NaN (what pandas hands back
    for a blank CSV cell) as absent. Note that the *string* ``"X"`` and the
    *integer* ``0`` are emphatically not absent -- ``"X"`` is a real result and
    a tap count of 0 is out of domain, and both are caught by validation rather
    than quietly normalized away here.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def null_ect():
    """A fresh ECT record meaning *not tested* -- all seven columns null."""
    return {column: None for column in ECT_COLUMNS}


def backfill_row(entry):
    """Add any missing ECT columns to ``entry`` as null, in place.

    Only fills columns that are *absent*. An ECT value already present is never
    overwritten, so this is safe to run repeatedly over a metadata file that
    mixes backfilled and freshly collected rows.
    """
    for column in ECT_COLUMNS:
        if column not in entry:
            entry[column] = None
    return entry


# --- Coercion and validation ------------------------------------------------

def _coerce_taps(value, context):
    """Parse ``ect_taps`` without silently accepting a non-integer count."""
    if isinstance(value, bool):
        raise ECTValidationError(f"{context}ect_taps must be an integer, got bool {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ECTValidationError(
                f"{context}ect_taps must be a whole number of taps, got {value!r}"
            )
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            raise ECTValidationError(
                f"{context}ect_taps must be an integer, got {value!r}"
            ) from None
    raise ECTValidationError(
        f"{context}ect_taps must be an integer, got {type(value).__name__} {value!r}"
    )


def _coerce_depth(value, context):
    """Parse ``ect_failure_depth_cm`` as a finite, non-negative depth."""
    try:
        depth = float(value)
    except (TypeError, ValueError):
        raise ECTValidationError(
            f"{context}ect_failure_depth_cm must be a number, got {value!r}"
        ) from None
    if math.isnan(depth) or math.isinf(depth):
        raise ECTValidationError(
            f"{context}ect_failure_depth_cm must be finite, got {value!r}"
        )
    if depth < 0:
        raise ECTValidationError(
            f"{context}ect_failure_depth_cm is measured below the snow surface and "
            f"must be non-negative, got {depth!r}"
        )
    return depth


def _coerce_enum(value, column, domain, context):
    """Match ``value`` against a closed domain, rejecting rather than coercing."""
    text = str(value).strip()
    if text not in domain:
        raise ECTValidationError(
            f"{context}{column} must be one of {list(domain)}, got {value!r}"
        )
    return text


def coerce_ect(raw, context=""):
    """Normalize a raw ECT record (CSV row, dict, pandas Series) to schema types.

    Absent values become ``None``. Values outside a closed domain raise rather
    than being coerced to something adjacent. Does not enforce the cross-field
    rules -- call :func:`validate_ect` for those, or :func:`normalize_ect` for
    both in one step.
    """
    values = null_ect()
    for column in ECT_COLUMNS:
        try:
            value = raw[column]
        except (KeyError, IndexError, TypeError):
            continue
        if is_null(value):
            continue
        if column == "ect_result":
            values[column] = _coerce_enum(value, column, ECT_RESULTS, context)
        elif column == "ect_taps":
            values[column] = _coerce_taps(value, context)
        elif column == "ect_failure_depth_cm":
            values[column] = _coerce_depth(value, context)
        elif column == "ect_failure_grain":
            values[column] = _coerce_enum(value, column, ECT_GRAIN_CODES, context)
        elif column in ("ect_hardness_above", "ect_hardness_below"):
            values[column] = _coerce_enum(value, column, ECT_HARDNESS, context)
        else:  # ect_operator -- free text, the person who ran the test
            values[column] = str(value).strip()
    return values


def validate_ect(values, context=""):
    """Enforce the cross-field ECT rules, raising :class:`ECTValidationError`.

    Rules:

    - ``ect_result`` is ``PV``/``P``/``N``/``X`` or null.
    - ``PV`` (propagated during isolation, before any tap) => ``ect_taps`` null.
    - ``X`` (no fracture in 30 taps) => every fracture-describing column null.
    - ``P``/``N`` => ``ect_taps`` present and in 1..30.
    - ``ect_taps`` present => ``ect_result`` is ``P`` or ``N``.
    - ``ect_result`` null (*not tested*) => all seven columns null.

    Returns ``values`` so it can be used inline.
    """
    result = values.get("ect_result")
    taps = values.get("ect_taps")

    if result is not None and result not in ECT_RESULTS:
        raise ECTValidationError(
            f"{context}ect_result must be one of {list(ECT_RESULTS)}, got {result!r}"
        )

    if result is None:
        # Not tested. A depth, grain, hardness or operator without a result is a
        # half-recorded test, which reads downstream as a measurement.
        populated = [c for c in ECT_COLUMNS if values.get(c) is not None]
        if populated:
            raise ECTValidationError(
                f"{context}ect_result is null (not tested) but {sorted(populated)} "
                f"are populated -- a partially recorded ECT cannot be interpreted. "
                f"Either record the result or clear all seven columns."
            )
        return values

    if result == "PV":
        if taps is not None:
            raise ECTValidationError(
                f"{context}ect_result 'PV' means the column propagated during "
                f"isolation, before any tap, so ect_taps must be null, got {taps!r}"
            )

    if result == "X":
        populated = [c for c in ECT_FRACTURE_COLUMNS if values.get(c) is not None]
        if populated:
            raise ECTValidationError(
                f"{context}ect_result 'X' means no fracture in 30 taps, so "
                f"{list(ECT_FRACTURE_COLUMNS)} must all be null, but "
                f"{sorted(populated)} are populated"
            )

    if result in ECT_TAPPED_RESULTS:
        if taps is None:
            raise ECTValidationError(
                f"{context}ect_result {result!r} requires a tap count, got null. "
                f"Record the tap the column fractured on."
            )
        if not ECT_TAPS_MIN <= taps <= ECT_TAPS_MAX:
            raise ECTValidationError(
                f"{context}ect_taps must be in {ECT_TAPS_MIN}..{ECT_TAPS_MAX}, "
                f"got {taps!r}"
            )

    if taps is not None and result not in ECT_TAPPED_RESULTS:
        raise ECTValidationError(
            f"{context}ect_taps is {taps!r} but ect_result is {result!r}; a tap "
            f"count only exists for {list(ECT_TAPPED_RESULTS)}"
        )

    return values


def normalize_ect(raw, context=""):
    """Coerce then validate a raw ECT record. The usual entry point."""
    return validate_ect(coerce_ect(raw, context), context)


def validate_group_constancy(records, context=""):
    """Assert the seven ECT values are constant within each ``(site, column)``.

    One ECT is run per snow column, so every row sharing a ``(site, column)``
    pair must carry identical ECT values. Disagreement means the intake source
    is inconsistent; that is raised loudly rather than resolved by taking the
    first value, which would silently discard a real measurement.
    """
    groups = {}
    for index, record in enumerate(records):
        try:
            key = tuple(record[k] for k in ECT_GROUP_KEYS)
        except (KeyError, TypeError):
            raise ECTValidationError(
                f"{context}record {index} is missing {list(ECT_GROUP_KEYS)}, so its "
                f"ECT grain cannot be checked"
            ) from None
        values = {c: record.get(c) for c in ECT_COLUMNS}
        groups.setdefault(key, []).append((index, values))

    conflicts = []
    for key, members in sorted(groups.items(), key=lambda item: repr(item[0])):
        for column in ECT_COLUMNS:
            distinct = {values[column] for _, values in members}
            if len(distinct) > 1:
                conflicts.append(
                    f"  (site={key[0]}, column={key[1]}) {column}: "
                    f"{sorted(distinct, key=repr)} across {len(members)} rows"
                )

    if conflicts:
        raise ECTValidationError(
            f"{context}ECT values are not constant within their (site, column) "
            f"groups. One ECT is run per column, so these disagreements mean the "
            f"intake source is inconsistent:\n" + "\n".join(conflicts)
        )


# --- Hugging Face features --------------------------------------------------

def hf_features():
    """The seven ECT columns as a ``dict`` of ``datasets.Value``.

    Imported lazily so ``datasets`` stays an optional dependency of intake.
    """
    from datasets import Value

    return {column: Value(dtype) for column, dtype in ECT_FEATURE_DTYPES.items()}


def card_feature_yaml(indent="    "):
    """The ECT block for ``dataset_info.features`` in the dataset card."""
    lines = []
    for column, dtype in ECT_FEATURE_DTYPES.items():
        lines.append(f"{indent}- name: {column}")
        lines.append(f"{indent}  dtype: {dtype}")
    return "\n".join(lines)
