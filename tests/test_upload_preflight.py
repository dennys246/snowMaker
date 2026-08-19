"""
Tests for the upload pre-flight check.

The check exists because of a concrete near-miss: on 2026-08-19 the local working
copy of the dataset held sites 0-2 (2345 preprocessed rows) while the Hub held
sites 0-6 (4040). Uploading that copy would have silently deleted 1695 rows.
``diff_metadata`` is the pure part of that guard, so it is tested without touching
the network.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import intake


def rows(*paths):
    return [{'file_path': path} for path in paths]


def test_new_file_counts_as_all_added():
    added, removed = intake.diff_metadata(rows('a.png', 'b.png'), None)
    assert added == {'a.png', 'b.png'}
    assert removed == set()


def test_identical_copies_are_a_no_op():
    added, removed = intake.diff_metadata(rows('a.png', 'b.png'), rows('a.png', 'b.png'))
    assert added == set()
    assert removed == set()


def test_new_rows_are_reported_as_added():
    added, removed = intake.diff_metadata(rows('a.png', 'b.png'), rows('a.png'))
    assert added == {'b.png'}
    assert removed == set()


def test_stale_copy_is_reported_as_removing_rows():
    """The near-miss: local is a subset of what the Hub already holds."""
    added, removed = intake.diff_metadata(rows('a.png'), rows('a.png', 'b.png', 'c.png'))
    assert added == set()
    assert removed == {'b.png', 'c.png'}


def test_divergent_copies_report_both_directions():
    added, removed = intake.diff_metadata(rows('a.png', 'x.png'), rows('a.png', 'b.png'))
    assert added == {'x.png'}
    assert removed == {'b.png'}


def test_row_order_and_duplicates_do_not_affect_the_diff():
    added, removed = intake.diff_metadata(
        rows('b.png', 'a.png', 'a.png'), rows('a.png', 'b.png')
    )
    assert added == set() and removed == set()


def test_read_jsonl_round_trips(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text(
        '{"file_path": "a.png", "ect_result": null}\n'
        '\n'                                            # blank lines are skipped
        '{"file_path": "b.png", "ect_result": "X"}\n',
        encoding = 'utf-8',
    )
    loaded = intake.read_jsonl(str(path))
    assert loaded == [
        {'file_path': 'a.png', 'ect_result': None},
        {'file_path': 'b.png', 'ect_result': 'X'},
    ]
