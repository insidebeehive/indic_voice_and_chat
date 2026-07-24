"""tests/unit/test_ingest_kb_script.py"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ingest_kb_script",
    pathlib.Path(__file__).parent.parent.parent / "scripts" / "ingest_kb.py",
)
ingest_kb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ingest_kb)

_EXTS = {".md", ".txt"}


def test_collect_files_dir_only_sweeps_recursively(tmp_path: pathlib.Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "sub" / "b.md").write_text("b")
    (tmp_path / "ignored.pdf").write_text("x")

    files = ingest_kb._collect_files(str(tmp_path), None, _EXTS)

    assert {f.name for f in files} == {"a.md", "b.md"}


def test_collect_files_file_only_ignores_extension_filter(tmp_path: pathlib.Path) -> None:
    weird = tmp_path / "notes.weird"
    weird.write_text("x")

    files = ingest_kb._collect_files(None, [str(weird)], _EXTS)

    assert files == [weird]


def test_collect_files_missing_explicit_file_raises(tmp_path: pathlib.Path) -> None:
    missing = str(tmp_path / "does-not-exist.md")

    with pytest.raises(FileNotFoundError):
        ingest_kb._collect_files(None, [missing], _EXTS)


def test_collect_files_dir_and_file_combined_deduplicates(tmp_path: pathlib.Path) -> None:
    (tmp_path / "a.md").write_text("a")
    dupe_path = str(tmp_path / "a.md")  # already swept by --dir

    files = ingest_kb._collect_files(str(tmp_path), [dupe_path], _EXTS)

    assert len(files) == 1
    assert files[0].name == "a.md"
