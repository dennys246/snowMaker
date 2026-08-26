"""
Schema machinery for the Rocky Mountain snowpack dataset's metadata tables.

The dataset is published as one **image table** plus four **side tables**, one
per grain::

    metadata/preprocessed.jsonl   image rows, keyed (site, column, core, segment)
    metadata/raw.jsonl            image rows, unprocessed source photographs
    metadata/pits.jsonl           one row per site-day,           key (site,)
    metadata/layers.jsonl         one row per snow layer,         key (site, column, layer_index)
    metadata/cores.jsonl          one row per core (or ladder rung), key (site, column, core)
    metadata/ect.jsonl            one row per ECT fracture,       key (site, column, test_index, fracture_index)

Each side table is declared once, as a :class:`TableSpec`, in its own module
(``pits``, ``layers``, ``cores``, ``ect``). This module owns everything those
declarations share: the null convention, coercion of raw CSV/JSON values into
schema types, per-record and per-table validation, the cross-table integrity
checks, JSONL I/O and the Hugging Face card fragments.

Why side tables
---------------
Every value coarser than an image used to be repeated across image rows, so a
pit-level number appeared on ~80 rows and the grain of a value could only be
known out of band. Counting those rows as observations inflates n by roughly two
orders of magnitude. A row in a side table is one observation at that table's
grain, so the grain is self-evident from the file it is in.

Null convention
---------------
``null`` means *not measured*. It is never a value, never a default and never
produced by coalescing: not-measured, measured-as-absent and measured-as-zero are
three different states. An ECT that was not run is a column with **no row** in
``ect.jsonl``, which is not the same thing as a row with ``ect_result == "X"``.
Enum values outside their domain are rejected, never mapped to a neighbour.
Where the grain implies a value is constant within a key, disagreement is an
error, never first-value-wins.
"""

import json
import math
import os
import re


class SchemaError(ValueError):
    """Raised for any record that violates the schema.

    Always raised rather than swallowed. A silently dropped, coerced or
    first-value-wins record is worse than a crash: it looks like a measurement.
    """


# --- Null handling ----------------------------------------------------------

def is_null(value):
    """Is ``value`` an absent measurement?

    ``None``, NaN (what pandas returns for a blank cell) and an empty or
    whitespace-only string are absent. The string ``"X"``, the integer ``0``,
    ``False`` and ``[]`` are emphatically **not** absent; each is a value, and
    each is checked against its field's rules instead of being normalised away.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


# --- Field types ------------------------------------------------------------

# Field ``kind`` -> Hugging Face ``Value`` dtype written into the dataset card.
# Everything nullable is a ``Value`` (or a list of one), never a ``ClassLabel``:
# a ClassLabel returns integer codes from load_dataset() while the JSONL holds
# strings, and encodes null as -1, which is trivially mistaken for a class.
HF_DTYPES = {
    "int": "int64",
    "float": "float64",
    "bool": "bool",
    "string": "string",
    "enum": "string",
    # A calendar date is declared date32, NOT string: pyarrow's JSON reader infers
    # an ISO "YYYY-MM-DD" string as a timestamp before the card's cast runs, so a
    # string declaration hands load_dataset() users "2026-12-20 00:00:00" while the
    # file says "2026-12-20". date32 gives datetime.date(2026, 12, 20) instead --
    # a different Python type from the JSONL string, but the same value, and the
    # card says so.
    "date": "date32",
    "int_list": "int64",      # list of int64
    "coordinates": "float64", # list of float64, [latitude, longitude]
}

LIST_KINDS = ("int_list", "coordinates")

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME_PATTERN = re.compile(r"^\d{2}:\d{2}[+-]\d{2}:\d{2}$")

# Fewer decimal places than this on a coordinate is a box wide enough to
# straddle an elevation band, which makes a DEM lookup against it unreliable.
MIN_COORDINATE_DECIMALS = 4


class Field:
    """One column of a side table.

    Arguments:
        - name (str) - Column name as written in the JSONL
        - kind (str) - One of the keys of ``HF_DTYPES``
        - nullable (bool) - Whether ``null`` (not measured) is a legal value
        - domain (tuple) - Closed set of legal values, for ``enum`` fields
        - minimum / maximum (number) - Inclusive bounds, for numeric fields
        - pattern (re.Pattern) - Required shape, for string fields
        - description (str) - One line for the dataset card
        - domain_note (str) - Domain column for the dataset card, when the
          domain is not simply the enum tuple
    """

    def __init__(self, name, kind, nullable = True, domain = None, minimum = None,
                 maximum = None, pattern = None, description = "", domain_note = None):
        if kind not in HF_DTYPES:
            raise ValueError(f"unknown field kind {kind!r} for {name}")
        if kind == "enum" and not domain:
            raise ValueError(f"enum field {name} needs a domain")
        self.name = name
        self.kind = kind
        self.nullable = nullable
        self.domain = tuple(domain) if domain else None
        self.minimum = minimum
        self.maximum = maximum
        self.pattern = pattern
        self.description = description
        self.domain_note = domain_note

    @property
    def hf_dtype(self):
        return HF_DTYPES[self.kind]

    @property
    def is_list(self):
        return self.kind in LIST_KINDS

    def card_type(self):
        """The 'type' cell for the card's column table."""
        base = {
            "int": "int64", "float": "float64", "bool": "bool", "string": "string",
            "enum": "string", "date": "date32", "int_list": "list[int64]",
            "coordinates": "list[float64]",
        }[self.kind]
        return base + (", nullable" if self.nullable else "")

    def card_domain(self):
        """The 'domain' cell for the card's column table."""
        if self.domain_note:
            return self.domain_note
        if self.domain:
            return " / ".join(f"`{value}`" for value in self.domain)
        bounds = []
        if self.minimum is not None:
            bounds.append(f">= {self.minimum}")
        if self.maximum is not None:
            bounds.append(f"<= {self.maximum}")
        if bounds:
            return ", ".join(bounds)
        if self.kind == "date":
            return "`YYYY-MM-DD` in the JSONL; `datetime.date` from `load_dataset()`"
        if self.pattern is TIME_PATTERN:
            return "`HH:MM+HH:MM` (local time with UTC offset)"
        return "free text" if self.kind == "string" else ""


# --- Coercion ---------------------------------------------------------------

def _coerce_int(value, field, context):
    if isinstance(value, bool):
        raise SchemaError(f"{context}{field.name} must be an integer, got bool {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise SchemaError(f"{context}{field.name} must be a whole number, got {value!r}")
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        if re.fullmatch(r"[+-]?\d+\.0*", text):
            return int(float(text))
        raise SchemaError(f"{context}{field.name} must be an integer, got {value!r}")
    raise SchemaError(
        f"{context}{field.name} must be an integer, got {type(value).__name__} {value!r}"
    )


def _coerce_float(value, field, context):
    if isinstance(value, bool):
        raise SchemaError(f"{context}{field.name} must be a number, got bool {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise SchemaError(f"{context}{field.name} must be a number, got {value!r}") from None
    if math.isnan(number) or math.isinf(number):
        raise SchemaError(f"{context}{field.name} must be finite, got {value!r}")
    return number


def _coerce_bool(value, field, context):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "true":
            return True
        if text == "false":
            return False
    # 0/1, yes/no and the like are not accepted: a boolean that arrives as a
    # number is usually a count or a code that has been mistaken for a flag.
    raise SchemaError(
        f"{context}{field.name} must be true or false, got {value!r}"
    )


def _coerce_enum(value, field, context):
    text = str(value).strip()
    if text not in field.domain:
        raise SchemaError(
            f"{context}{field.name} must be one of {list(field.domain)}, got {value!r}"
        )
    return text


def _coerce_string(value, field, context):
    text = str(value).strip()
    if field.pattern is not None and not field.pattern.fullmatch(text):
        raise SchemaError(
            f"{context}{field.name} must match {field.pattern.pattern}, got {value!r}"
        )
    return text


def _coerce_int_list(value, field, context):
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                raise SchemaError(f"{context}{field.name} is not a list, got {value!r}") from None
        else:
            # CSV form: semicolon separated, e.g. "3;4;5"
            value = [part for part in text.split(";") if part.strip()]
    if not isinstance(value, (list, tuple)):
        raise SchemaError(f"{context}{field.name} must be a list of integers, got {value!r}")
    if len(value) == 0:
        # An empty list would read as "measured: none", which no field using
        # this kind can mean. Record null for not measured instead.
        raise SchemaError(
            f"{context}{field.name} is an empty list; use null for not measured"
        )
    items = [_coerce_int(item, field, context) for item in value]
    if len(set(items)) != len(items):
        raise SchemaError(f"{context}{field.name} has duplicate entries: {items}")
    return items


def coordinate_decimals(text):
    """Decimal places carried by one coordinate written as text."""
    return len(text.strip().partition(".")[2])


def _coerce_coordinates(value, field, context, warnings):
    """Parse ``[latitude, longitude]`` at the full precision it was recorded at.

    Accepts the CSV form ``"39.66361, -105.88222"`` or a two-element list.
    Nothing here rounds or truncates. A coordinate carrying fewer than
    ``MIN_COORDINATE_DECIMALS`` places is flagged, not rejected, so historical
    sites still load.
    """
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = [repr(part) if isinstance(part, float) else str(part) for part in value]
    else:
        raise SchemaError(f"{context}{field.name} must be [latitude, longitude], got {value!r}")
    if len(parts) != 2:
        raise SchemaError(f"{context}{field.name} must be [latitude, longitude], got {value!r}")

    numbers = []
    for label, part, limit in zip(("latitude", "longitude"), parts, (90.0, 180.0)):
        number = _coerce_float(part, field, context)
        if abs(number) > limit:
            raise SchemaError(f"{context}{label} {number!r} is outside +/-{limit}")
        decimals = coordinate_decimals(part)
        if decimals < MIN_COORDINATE_DECIMALS:
            warnings.append(
                f"{context}{label} {part} carries only {decimals} decimal place(s); "
                f"{MIN_COORDINATE_DECIMALS} are needed to place the pit within an "
                f"elevation band. Log the receiver's full precision."
            )
        numbers.append(number)
    return numbers


def coerce_value(field, value, context, warnings = None):
    """Coerce one raw value into its field's type, or raise :class:`SchemaError`.

    ``None`` is returned for an absent value; whether that is legal is decided by
    :func:`check_nullability`, after every field has been read, so an error can
    name the whole record.
    """
    if warnings is None:
        warnings = []
    if is_null(value):
        return None
    kind = field.kind
    if kind == "int":
        result = _coerce_int(value, field, context)
    elif kind == "float":
        result = _coerce_float(value, field, context)
    elif kind == "bool":
        result = _coerce_bool(value, field, context)
    elif kind == "enum":
        result = _coerce_enum(value, field, context)
    elif kind == "string":
        result = _coerce_string(value, field, context)
    elif kind == "date":
        # Stored and validated as the ISO string; only the card's dtype differs
        if hasattr(value, "isoformat") and not isinstance(value, str):
            value = value.isoformat()
        result = _coerce_string(value, field, context)
        if not DATE_PATTERN.fullmatch(result):
            raise SchemaError(f"{context}{field.name} must be YYYY-MM-DD, got {value!r}")
    elif kind == "int_list":
        result = _coerce_int_list(value, field, context)
    elif kind == "coordinates":
        result = _coerce_coordinates(value, field, context, warnings)
    else:  # pragma: no cover - guarded by Field.__init__
        raise ValueError(kind)

    if kind in ("int", "float"):
        if field.minimum is not None and result < field.minimum:
            raise SchemaError(
                f"{context}{field.name} must be >= {field.minimum}, got {result!r}"
            )
        if field.maximum is not None and result > field.maximum:
            raise SchemaError(
                f"{context}{field.name} must be <= {field.maximum}, got {result!r}"
            )
    return result


# --- Tables -----------------------------------------------------------------

class TableSpec:
    """A side table: its file, key, fields and rules.

    Arguments:
        - name (str) - Table name; also the Hugging Face config name
        - key (tuple of str) - Fields that identify a row; unique per table
        - fields (list of Field) - Every column, in schema order, keys first
        - record_rules (callable) - ``fn(values, context)`` raising SchemaError
          for cross-field violations within one record
        - group_rules (callable) - ``fn(records, context) -> list[str]`` run over
          the whole table, raising SchemaError for violations and returning
          human-readable warnings (e.g. layer gaps) that do not block
        - constant_within (dict) - ``{group_key_tuple: (field, ...)}``: fields
          that must be identical across every record sharing the group key
        - grain (str) - One line for the card: what one row is
    """

    def __init__(self, name, key, fields, record_rules = None, group_rules = None,
                 constant_within = None, grain = ""):
        self.name = name
        self.key = tuple(key)
        self.fields = list(fields)
        self.record_rules = record_rules
        self.group_rules = group_rules
        self.constant_within = dict(constant_within or {})
        self.grain = grain
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"{name}: duplicate field names")
        for key_field in self.key:
            if key_field not in names:
                raise ValueError(f"{name}: key field {key_field} is not a field")
        self.by_name = {field.name: field for field in self.fields}

    @property
    def filename(self):
        return f"metadata/{self.name}.jsonl"

    @property
    def columns(self):
        return [field.name for field in self.fields]

    def key_of(self, record):
        return tuple(record[k] for k in self.key)

    # -- records --

    def coerce(self, raw, context = "", warnings = None):
        """Read one raw record (CSV row, dict, pandas Series) into schema types.

        Unknown columns raise: a misspelt column would otherwise vanish, and the
        value with it. Missing columns read as null and are then checked for
        nullability, so a required column that is simply absent is still an
        error. Cross-field rules are applied by :meth:`validate`.
        """
        if warnings is None:
            warnings = []
        keys = getattr(raw, "keys", None)
        if keys is None:
            raise SchemaError(f"{context}{self.name} record must be a mapping, got {type(raw).__name__}")
        unknown = [column for column in keys() if column not in self.by_name]
        if unknown:
            raise SchemaError(
                f"{context}{self.name} has unknown column(s) {sorted(unknown)}; "
                f"known columns are {self.columns}"
            )
        values = {}
        for field in self.fields:
            try:
                value = raw[field.name]
            except (KeyError, IndexError):
                value = None
            values[field.name] = coerce_value(field, value, context, warnings)
        return values

    def check_nullability(self, values, context = ""):
        missing = [
            field.name for field in self.fields
            if not field.nullable and values.get(field.name) is None
        ]
        if missing:
            raise SchemaError(
                f"{context}{self.name} record is missing required value(s) {missing}"
            )

    def validate(self, values, context = ""):
        """Nullability, then the table's cross-field rules. Returns ``values``."""
        self.check_nullability(values, context)
        if self.record_rules is not None:
            self.record_rules(values, context)
        return values

    def normalize(self, raw, context = "", warnings = None):
        """Coerce then validate one record. The usual entry point."""
        return self.validate(self.coerce(raw, context, warnings), context)

    # -- tables --

    def validate_table(self, records, context = ""):
        """Validate a whole table: every record, key uniqueness, constancy, rules.

        Returns a list of warnings (strings). Raises :class:`SchemaError` for
        anything that must not be written.
        """
        warnings = []
        seen = {}
        for position, record in enumerate(records):
            row_context = f"{context}{self.name} row {position}: "
            for field in self.fields:
                if field.name not in record:
                    raise SchemaError(f"{row_context}missing column {field.name}")
            extra = [column for column in record if column not in self.by_name]
            if extra:
                raise SchemaError(f"{row_context}unknown column(s) {sorted(extra)}")
            self.validate(record, row_context)
            key = self.key_of(record)
            if key in seen:
                raise SchemaError(
                    f"{row_context}duplicate key {dict(zip(self.key, key))} "
                    f"(also at row {seen[key]})"
                )
            seen[key] = position

        # Table rules first: they name the specific violation (an X that is not
        # alone, fractures out of tap order) where a constancy failure on the same
        # rows would only say "not constant".
        if self.group_rules is not None:
            warnings.extend(self.group_rules(records, context) or [])
        self._check_constancy(records, context)
        return warnings

    def _check_constancy(self, records, context):
        for group_key, fields in self.constant_within.items():
            groups = {}
            for record in records:
                groups.setdefault(tuple(record[k] for k in group_key), []).append(record)
            conflicts = []
            for key, members in sorted(groups.items()):
                for field in fields:
                    distinct = {json.dumps(member[field], sort_keys = True) for member in members}
                    if len(distinct) > 1:
                        conflicts.append(
                            f"  {dict(zip(group_key, key))} {field}: "
                            f"{sorted(distinct)} across {len(members)} rows"
                        )
            if conflicts:
                raise SchemaError(
                    f"{context}{self.name}: values are not constant within "
                    f"{group_key}. One record per {group_key} carries these, so a "
                    f"disagreement means the source is inconsistent:\n"
                    + "\n".join(conflicts)
                )

    # -- Hugging Face --

    def hf_features(self):
        """The table's columns as a ``datasets.Features``."""
        from datasets import Features, List, Value

        features = {}
        for field in self.fields:
            if field.is_list:
                features[field.name] = List(Value(field.hf_dtype))
            else:
                features[field.name] = Value(field.hf_dtype)
        return Features(features)

    def card_features_yaml(self, indent = "  "):
        """The ``features:`` block for this config in the card's ``dataset_info``."""
        lines = [f"{indent}features:"]
        for field in self.fields:
            lines.append(f"{indent}- name: {field.name}")
            if field.is_list:
                lines.append(f"{indent}  dtype:")
                lines.append(f"{indent}    list:")
                lines.append(f"{indent}      dtype: {field.hf_dtype}")
            else:
                lines.append(f"{indent}  dtype: {field.hf_dtype}")
        return "\n".join(lines)

    def card_config_yaml(self):
        """The ``configs:`` entry for this table."""
        return (
            f"- config_name: {self.name}\n"
            f"  data_files:\n"
            f"  - split: train\n"
            f"    path: \"{self.filename}\""
        )

    def card_columns_table(self):
        """A Markdown table of the columns, for the card."""
        lines = [
            "| column | type | domain | description |",
            "|---|---|---|---|",
        ]
        for field in self.fields:
            key_note = " **(key)**" if field.name in self.key else ""
            lines.append(
                f"| `{field.name}`{key_note} | {field.card_type()} | "
                f"{field.card_domain()} | {field.description} |"
            )
        return "\n".join(lines)


# --- JSONL I/O --------------------------------------------------------------

def read_jsonl(path):
    """Read a JSONL file into a list of dicts, preserving order. Blank lines skipped."""
    rows = []
    with open(path, "r", encoding = "utf-8") as handle:
        for position, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SchemaError(f"{path} line {position + 1} is not valid JSON: {error}") from None
    return rows


def write_jsonl(path, rows):
    """Write rows as JSONL atomically (temp file + rename)."""
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding = "utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(temporary, path)


def ordered(spec, record):
    """A copy of ``record`` with columns in schema order."""
    return {column: record[column] for column in spec.columns}


# --- Cross-table integrity ---------------------------------------------------

# The legacy pit-level columns on the image table and the pits.jsonl column
# each must agree with, per site. The image table keeps its copies (changing it
# would be a breaking change); pits.jsonl is the canonical grain.
IMAGE_PIT_FIELDS = {
    "coordinates": "coordinates",
    "snowpack_depth": "snowpack_depth_cm",
    "slope_angle": "slope_angle_deg",
}

# Values on the image table that are constant per (site, column, core).
IMAGE_CORE_FIELDS = ("core_depth",)


def validate_dataset(tables, image_rows = None, context = ""):
    """Check the references between tables, and between them and the image rows.

    Arguments:
        - tables (dict) - ``{table_name: records}`` for pits, layers, cores, ect
        - image_rows (list) - image-table rows (preprocessed + raw), or None to
          skip the image-side checks
        - context (str) - prefix for error messages

    Raises :class:`SchemaError` on a dangling reference. Returns a list of
    warnings.
    """
    warnings = []
    pits = tables.get("pits", [])
    layers = tables.get("layers", [])
    cores = tables.get("cores", [])
    ects = tables.get("ect", [])

    pit_sites = {row["site"] for row in pits}
    layer_keys = {(row["site"], row["column"], row["layer_index"]) for row in layers}
    core_keys = {(row["site"], row["column"], row["core"]) for row in cores}

    def require_pit(name, rows):
        missing = sorted({row["site"] for row in rows} - pit_sites)
        if missing:
            raise SchemaError(
                f"{context}{name} references site(s) {missing} that have no row in "
                f"pits.jsonl; every site needs a pit record (collector_id, "
                f"method_deviation, coordinates) before its data can be published"
            )

    require_pit("layers", layers)
    require_pit("cores", cores)
    require_pit("ect", ects)

    for row in cores:
        ids = row.get("layer_ids")
        if ids is None:
            continue
        dangling = [i for i in ids if (row["site"], row["column"], i) not in layer_keys]
        if dangling:
            raise SchemaError(
                f"{context}cores (site {row['site']}, column {row['column']}, core "
                f"{row['core']}) layer_ids {dangling} reference layers that do not "
                f"exist in layers.jsonl for that column"
            )

    if image_rows is None:
        return warnings

    image_sites = {row["site"] for row in image_rows}
    missing_pits = sorted(image_sites - pit_sites)
    if missing_pits:
        raise SchemaError(
            f"{context}image rows reference site(s) {missing_pits} with no row in "
            f"pits.jsonl"
        )

    image_cores = {(row["site"], row["column"], row["core"]) for row in image_rows}
    missing_cores = sorted(image_cores - core_keys)
    if missing_cores:
        raise SchemaError(
            f"{context}image rows reference core(s) {missing_cores[:10]}"
            f"{' ...' if len(missing_cores) > 10 else ''} with no row in cores.jsonl"
        )

    skipped = {
        (row["site"], row["column"], row["core"])
        for row in cores if row.get("depth_skipped_reason") is not None
    }
    photographed_skips = sorted(skipped & image_cores)
    if photographed_skips:
        raise SchemaError(
            f"{context}core(s) {photographed_skips} are recorded as skipped ladder "
            f"rungs in cores.jsonl but have image rows"
        )

    # Pit-level values the image table repeats must agree with pits.jsonl
    pits_by_site = {row["site"]: row for row in pits}
    for image_field, pit_field in IMAGE_PIT_FIELDS.items():
        for row in image_rows:
            if image_field not in row:
                continue
            expected = pits_by_site[row["site"]].get(pit_field)
            if expected is None:
                continue
            if _differs(row[image_field], expected):
                raise SchemaError(
                    f"{context}{row.get('file_path', '?')}: image-table {image_field}="
                    f"{row[image_field]!r} disagrees with pits.jsonl {pit_field}="
                    f"{expected!r} for site {row['site']}"
                )

    cores_by_key = {(row["site"], row["column"], row["core"]): row for row in cores}
    for row in image_rows:
        core = cores_by_key[(row["site"], row["column"], row["core"])]
        if "core_depth" in row and _differs(row["core_depth"], core["core_depth_cm"]):
            raise SchemaError(
                f"{context}{row.get('file_path', '?')}: image-table core_depth="
                f"{row['core_depth']!r} disagrees with cores.jsonl core_depth_cm="
                f"{core['core_depth_cm']!r}"
            )
    return warnings


def _differs(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) != len(b) or any(_differs(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return not math.isclose(float(a), float(b), rel_tol = 0, abs_tol = 1e-9)
    return a != b


# --- CSV I/O (intake side) --------------------------------------------------
#
# Side tables arrive from the field as CSV, one file per table per site. They are
# read with the csv module rather than pandas so that no cell is ever type-guessed:
# a blank is null, "X" is "X", 12 is 12 -- coercion happens once, in coerce_value.

import csv


def cell_to_text(field, value):
    """Encode one schema-typed value as a CSV cell. ``None`` is a blank cell."""
    if value is None:
        return ""
    if field.kind == "bool":
        return "true" if value else "false"
    if field.kind == "int_list":
        return ";".join(str(item) for item in value)
    if field.kind == "coordinates":
        return ", ".join(repr(float(item)) for item in value)
    if field.kind == "float":
        return repr(float(value))
    return str(value)


def read_csv(spec, path, context = None, warnings = None):
    """Read a side-table CSV into validated records. Missing file -> empty list."""
    if not os.path.exists(path):
        return []
    if context is None:
        context = path
    if warnings is None:
        warnings = []
    records = []
    with open(path, "r", encoding = "utf-8-sig", newline = "") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        for position, raw in enumerate(reader):
            row_context = f"{context} row {position + 1}: "
            if None in raw:
                raise SchemaError(f"{row_context}more cells than header columns")
            records.append(spec.normalize(raw, row_context, warnings))
    return records


def write_csv(spec, path, records):
    """Write validated records as a side-table CSV, columns in schema order."""
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding = "utf-8", newline = "") as handle:
        writer = csv.writer(handle)
        writer.writerow(spec.columns)
        for record in records:
            writer.writerow([cell_to_text(spec.by_name[c], record[c]) for c in spec.columns])
    os.replace(temporary, path)


def merge_records(spec, existing, incoming, context = ""):
    """Union two record lists on the table key.

    A key present in both with identical values is kept once. A key present in
    both with different values raises: neither copy is allowed to win.
    """
    merged = {spec.key_of(record): record for record in existing}
    for record in incoming:
        key = spec.key_of(record)
        if key in merged and merged[key] != record:
            raise SchemaError(
                f"{context}{spec.name} {dict(zip(spec.key, key))} is recorded twice "
                f"with different values:\n  existing: {merged[key]}\n  incoming: {record}"
            )
        merged[key] = record
    return [merged[key] for key in sorted(merged)]
