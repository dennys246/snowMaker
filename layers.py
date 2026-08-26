"""
``metadata/layers.jsonl`` -- one row per snow layer, the layer profile.

This is a new grain. A pit's stratigraphy is 8-16 layers per column, ordered
top-down, each spanning ``depth_top_cm`` to ``depth_bottom_cm`` below the snow
surface. Layers within a column must tile the depth axis: contiguous and
non-overlapping. An overlap is an error; a gap is flagged as a warning, because
an unprofiled interval is a real field outcome that should be visible rather
than silently closed.

Why hand hardness is the priority: the core break count is **non-monotonic** in
instability. A cohesive, well-sintered slab breaks into few pieces; cohesionless
depth hoar also gives few pieces, because it crumbles instead of breaking into
countable ones. Opposite physical states, identical counts. Hand hardness
separates them (low count + P/K = slab, low count + F = facets), and a slab over
depth hoar is the archetypal fatal Colorado configuration.
"""

from schema import Field, SchemaError, TableSpec

# ICSSG hand hardness (Fierz et al. 2009), softest to hardest.
HAND_HARDNESS = ("F", "4F", "1F", "P", "K")

# ICSSG main grain classes.
GRAIN_TYPES = ("PP", "DF", "RG", "FC", "DH", "SH", "MF", "IF", "MM")

# ICSSG liquid water content classes: dry, moist, wet, very wet, soaked.
LWC_CLASSES = ("D", "M", "W", "V", "S")

FIELDS = [
    Field("site", "int", nullable = False, minimum = 0, description = "Site number"),
    Field("column", "int", nullable = False, minimum = 1,
          description = "Snow column within the pit"),
    Field("layer_index", "int", nullable = False, minimum = 1,
          description = "Layer number within the column, 1 at the surface, increasing downward"),
    Field("depth_top_cm", "float", nullable = False, minimum = 0.0,
          description = "Top of the layer, cm below the snow surface"),
    Field("depth_bottom_cm", "float", nullable = False, minimum = 0.0,
          description = "Bottom of the layer, cm below the snow surface"),
    Field("hand_hardness", "enum", nullable = False, domain = HAND_HARDNESS,
          description = "ICSSG hand hardness of the layer"),
    Field("grain_type", "enum", nullable = False, domain = GRAIN_TYPES,
          description = "ICSSG main grain class"),
    Field("grain_size_mm", "float", minimum = 0.0,
          description = "Typical grain size, mm"),
    Field("density_kg_m3", "float", minimum = 0.0,
          description = "Layer density, kg/m^3"),
    Field("temperature_c", "float",
          description = "Layer snow temperature, degrees C. Readings above 0 are stored as read and flagged on ingest"),
    Field("lwc", "enum", domain = LWC_CLASSES,
          description = "ICSSG liquid water content class"),
]


def record_rules(values, context = ""):
    if values["depth_bottom_cm"] <= values["depth_top_cm"]:
        raise SchemaError(
            f"{context}depth_bottom_cm ({values['depth_bottom_cm']}) must be below "
            f"depth_top_cm ({values['depth_top_cm']})"
        )
    if values["grain_size_mm"] is not None and values["grain_size_mm"] == 0:
        raise SchemaError(f"{context}grain_size_mm must be positive, got 0")


def group_rules(records, context = ""):
    """Layers within a column tile the depth axis: contiguous, non-overlapping.

    Raises for overlaps and out-of-order indices. Returns warnings for gaps,
    including an unprofiled interval above the first layer, and for temperature
    readings above freezing.
    """
    warnings = []
    columns = {}
    for record in records:
        columns.setdefault((record["site"], record["column"]), []).append(record)

    for (site, column), layers in sorted(columns.items()):
        layers = sorted(layers, key = lambda row: row["layer_index"])
        where = f"{context}layers (site {site}, column {column}): "

        expected = list(range(1, len(layers) + 1))
        actual = [row["layer_index"] for row in layers]
        if actual != expected:
            raise SchemaError(
                f"{where}layer_index must run 1..{len(layers)} with no gaps, got {actual}"
            )

        for above, below in zip(layers, layers[1:]):
            if below["depth_top_cm"] < above["depth_bottom_cm"]:
                raise SchemaError(
                    f"{where}layer {below['layer_index']} starts at "
                    f"{below['depth_top_cm']} cm, above the bottom of layer "
                    f"{above['layer_index']} at {above['depth_bottom_cm']} cm (overlap)"
                )
            if below["depth_top_cm"] > above["depth_bottom_cm"]:
                warnings.append(
                    f"{where}gap of {below['depth_top_cm'] - above['depth_bottom_cm']:g} cm "
                    f"between layer {above['layer_index']} (bottom "
                    f"{above['depth_bottom_cm']}) and layer {below['layer_index']} "
                    f"(top {below['depth_top_cm']}); that interval is unprofiled"
                )
        if layers[0]["depth_top_cm"] > 0:
            warnings.append(
                f"{where}first layer starts at {layers[0]['depth_top_cm']} cm; the top "
                f"{layers[0]['depth_top_cm']:g} cm is unprofiled"
            )
        for row in layers:
            if row["temperature_c"] is not None and row["temperature_c"] > 0:
                warnings.append(
                    f"{where}layer {row['layer_index']} temperature_c="
                    f"{row['temperature_c']} is above freezing; stored as read, "
                    f"but snow cannot be warmer than 0 C -- check the instrument"
                )
    return warnings


SPEC = TableSpec(
    name = "layers",
    key = ("site", "column", "layer_index"),
    fields = FIELDS,
    record_rules = record_rules,
    group_rules = group_rules,
    grain = "one row per snow layer within a column",
)
