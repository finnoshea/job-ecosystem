"""Tests for jobecosystem.tailor.layout: the PDF layout spec."""

from __future__ import annotations

import json

import pytest

from jobecosystem.tailor import layout as L
from jobecosystem.tailor.models import TailorError


def test_defaults_are_sane():
    layout = L.Layout()
    assert layout.page == "LETTER"
    assert layout.margin_left > 0
    assert layout.body_size > layout.meta_size


def test_resolve_prefers_the_argument(tmp_path, monkeypatch):
    monkeypatch.setenv("TAILOR_LAYOUT_FILE", str(tmp_path / "env.json"))
    assert L.resolve_layout_path(tmp_path / "given.json") == tmp_path / "given.json"


def test_resolve_uses_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TAILOR_LAYOUT_FILE", str(tmp_path / "env.json"))
    assert L.resolve_layout_path() == tmp_path / "env.json"


def test_resolve_defaults_to_none(monkeypatch):
    monkeypatch.delenv("TAILOR_LAYOUT_FILE", raising=False)
    assert L.resolve_layout_path() is None


def test_load_without_a_path_is_the_defaults(monkeypatch):
    monkeypatch.delenv("TAILOR_LAYOUT_FILE", raising=False)
    assert L.load_layout() == L.Layout()


def test_load_applies_overrides(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(json.dumps({"page": "A4", "body_size": 12}), encoding="utf-8")
    layout = L.load_layout(path)
    assert layout.page == "A4"
    assert layout.body_size == 12
    assert layout.margin_left == L.Layout().margin_left


def test_load_rejects_unknown_keys(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text(json.dumps({"margins_left": 1}), encoding="utf-8")
    with pytest.raises(TailorError, match="unknown key"):
        L.load_layout(path)


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(TailorError, match="not found"):
        L.load_layout(tmp_path / "nope.json")


def test_load_rejects_non_object(tmp_path):
    path = tmp_path / "layout.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(TailorError, match="JSON object"):
        L.load_layout(path)


def test_layout_to_dict_round_trips():
    layout = L.Layout(page="A4")
    assert L.layout_to_dict(layout)["page"] == "A4"
