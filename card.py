"""
Dataset-card fragments generated from the table specs.

The YAML front matter of the dataset card (README.md in the dataset repo) is
what ``load_dataset()`` applies to the JSONL files, so it must match the schema
modules exactly. Generate it from here rather than editing it by hand::

    python -c "import card; print(card.front_matter_yaml())"

The image table's feature list is kept verbatim: every existing column keeps
its declared type (``datatype`` and ``wind_loading`` stay ``class_label``,
``coordinates`` stays ``float32``), because changing a published column's type
is a breaking change for every consumer.
"""

import ect
import pits, layers, cores

SIDE_TABLES = (pits.SPEC, layers.SPEC, cores.SPEC, ect.SPEC)

# The image table (config 'default'), exactly as declared before 2026-08-25 minus
# the seven ect_* columns that moved to ect.jsonl.
IMAGE_FEATURES_YAML = """\
  features:
  - name: image
    dtype: Image
  - name: file_path
    dtype: string
  - name: datatype
    dtype:
      class_label:
        names:
          0: 'core'
          1: 'profile'
          2: 'magnified_profile'
          3: 'crystal_card'
  - name: site
    dtype: int64
  - name: column
    dtype: int64
  - name: core
    dtype: int64
  - name: segment
    dtype: int64
  - name: core_temperature
    dtype: float32
  - name: air_temperature
    dtype: float32
  - name: ascending_mountain
    dtype: string
  - name: city_state_country
    dtype: string
  - name: collector
    dtype: string
  - name: coordinates
    dtype:
      list:
        dtype: float32
  - name: date
    dtype: string
  - name: time
    dtype: string
  - name: snowpack_depth
    dtype: float32
  - name: core_depth
    dtype: float32
  - name: slope_face
    dtype: float32
  - name: slope_angle
    dtype: float32
  - name: avalanches_spotted
    dtype: int64
  - name: wind_loading
    dtype:
      class_label:
        names:
          0: 'none'
          1: 'low'
          2: 'medium'
          3: 'high'
  - name: notes
    dtype: string"""

IMAGE_CONFIG_YAML = """\
- config_name: default
  data_files:
  - split: train
    path: "metadata/preprocessed.jsonl"
  - split: raw
    path: "metadata/raw.jsonl\""""


def active_tables(active = None):
    """
    The side tables declared as ``load_dataset()`` configs.

    A table with no rows yet is documented on the card but **not** declared as a
    config: ``datasets`` refuses an empty config ("corresponds to no data"), so
    declaring it would only turn "not assessed yet" into an error. Pass the names
    of the tables that hold data; ``None`` means all of them.
    """
    if active is None:
        return list(SIDE_TABLES)
    return [spec for spec in SIDE_TABLES if spec.name in set(active)]


def dataset_info_yaml(active = None):
    """The ``dataset_info:`` block, one entry per config."""
    blocks = ["dataset_info:", "- config_name: default", IMAGE_FEATURES_YAML]
    for spec in active_tables(active):
        blocks.append(f"- config_name: {spec.name}")
        blocks.append(spec.card_features_yaml(indent = "  "))
    return "\n".join(blocks)


def configs_yaml(active = None):
    """The ``configs:`` block, one entry per config."""
    blocks = ["configs:", IMAGE_CONFIG_YAML]
    for spec in active_tables(active):
        blocks.append(spec.card_config_yaml())
    return "\n".join(blocks)


def front_matter_yaml(active = None):
    """``dataset_info`` and ``configs`` together, ready to paste into the card."""
    return dataset_info_yaml(active) + "\n" + configs_yaml(active)


def tables_with_rows(dataset_dir):
    """Names of the side tables whose JSONL holds at least one row."""
    import os
    import schema

    if not dataset_dir.endswith("/"):
        dataset_dir += "/"
    names = []
    for spec in SIDE_TABLES:
        path = f"{dataset_dir}{spec.filename}"
        if os.path.exists(path) and schema.read_jsonl(path):
            names.append(spec.name)
    return names


def columns_markdown():
    """A Markdown section per side table listing every column, for the card body."""
    sections = []
    for spec in SIDE_TABLES:
        sections.append(
            f"#### `metadata/{spec.name}.jsonl` -- {spec.grain}\n\n"
            f"Key: {', '.join(f'`{k}`' for k in spec.key)}\n\n"
            + spec.card_columns_table()
        )
    return "\n\n".join(sections)


def access_paths_markdown():
    """Per column of every side table: what load_dataset() returns vs the JSONL."""
    lines = [
        "| file | column | `load_dataset()` | direct JSONL read |",
        "|---|---|---|---|",
    ]
    for spec in SIDE_TABLES:
        for field in spec.fields:
            if field.kind == "date":
                loaded, direct = "`datetime.date` or `None`", "`\"YYYY-MM-DD\"` or `null` — **differs in type, same value**"
            elif field.is_list:
                loaded = f"`list[{field.hf_dtype}]` or `None`"
                direct = "list or `null` — **identical**"
            else:
                loaded = f"`{field.hf_dtype}` or `None`"
                direct = "same or `null` — **identical**"
            lines.append(f"| `{spec.name}.jsonl` | `{field.name}` | {loaded} | {direct} |")
    return "\n".join(lines)
