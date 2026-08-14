"""Unit tests for Phase 4 — labeling (#27–#32).

Covers the pure logic: vocabulary hierarchy consistency, tree rendering, term
migration with backup, inter-annotator agreement math, and prompt assembly.
No CLAP, torch, or streamlit are loaded.
"""
from __future__ import annotations

import csv
import json

from musictrain import agreement as agree
from musictrain import labels as labels_mod
from musictrain import vocab as vocab_mod
from musictrain.labels import HIERARCHY, VOCAB, PARENT_OF, VOCAB_VERSION
from musictrain.promptbuilder import apply_override, build_prompt


# --------------------------------------------------------------------------- #
# labels.py — hierarchy (#27)
# --------------------------------------------------------------------------- #


def test_hierarchy_children_are_in_vocab():
    for dim, parents in HIERARCHY.items():
        for children in parents.values():
            for child in children:
                assert child in VOCAB[dim], f"{child!r} missing from VOCAB[{dim}]"


def test_hierarchy_parents_are_in_vocab():
    for dim, parents in HIERARCHY.items():
        for parent in parents:
            assert parent in VOCAB[dim], f"{parent!r} missing from VOCAB[{dim}]"


def test_parent_of_inverse():
    for dim, parents in HIERARCHY.items():
        for parent, children in parents.items():
            for child in children:
                assert PARENT_OF[dim][child] == parent


def test_hierarchy_notes_flags_parent_usage(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text(
        "source_id,genre,mood,instruments,section\n"
        "t1,melodic trap,dark,piano|808 bass,chorus\n"
    )
    notes = labels_mod.hierarchy_notes(p)
    assert len(notes) == 1
    assert "808 bass" in notes[0]
    assert "sub bass" in notes[0]  # one of the suggested children


def test_hierarchy_notes_clean_when_leaves_only(tmp_path):
    p = tmp_path / "labels.csv"
    p.write_text(
        "source_id,genre,mood,instruments,section\n"
        "t1,melodic trap,dark,sub bass|piano melody,chorus\n"
    )
    assert labels_mod.hierarchy_notes(p) == []


# --------------------------------------------------------------------------- #
# vocab.py — tree rendering (#27) + migration (#32)
# --------------------------------------------------------------------------- #


def test_render_tree_contains_dims_and_children():
    tree = vocab_mod.render_tree()
    assert f"vocabulary v{VOCAB_VERSION}" in tree
    assert "genre:" in tree
    assert "instruments:" in tree
    assert "808 bass (parent)" in tree
    assert "sub bass" in tree


def test_migrate_dim_specific_rename(tmp_path):
    lp = tmp_path / "labels.csv"
    lp.write_text(
        "source_id,genre,mood,instruments,section\n"
        "a,trap,dark,piano|808 bass,chorus\n"
        "b,pain music,dark|emotional,autotune vocals,verse\n"
    )
    mp = tmp_path / "rename.json"
    mp.write_text(json.dumps({"genre": {"trap": "melodic trap"}}))

    stamp = vocab_mod.migrate(tmp_path, lp, mp, backup=True)
    assert stamp["renames"] == 1

    rows = list(csv.DictReader(lp.open(newline="")))
    assert rows[0]["genre"] == "melodic trap"
    assert rows[1]["genre"] == "pain music"
    assert rows[0]["instruments"] == "piano|808 bass"  # untouched

    backups = list(tmp_path.glob("labels.csv.bak.*"))
    assert len(backups) == 1
    assert (tmp_path / "metadata" / "vocab_version.json").exists()


def test_migrate_flat_map_and_multi_value(tmp_path):
    lp = tmp_path / "labels.csv"
    lp.write_text(
        "source_id,genre,mood,instruments,section\n"
        "a,melodic trap,dark,piano|808 bass,chorus\n"
    )
    mp = tmp_path / "rename.json"
    mp.write_text(json.dumps({"808 bass": "distorted 808", "piano": "keys"}))

    stamp = vocab_mod.migrate(tmp_path, lp, mp, backup=False)
    assert stamp["renames"] == 2
    rows = list(csv.DictReader(lp.open(newline="")))
    assert rows[0]["instruments"] == "keys|distorted 808"


def test_migrate_no_match_is_noop(tmp_path):
    lp = tmp_path / "labels.csv"
    lp.write_text("source_id,genre,mood,instruments,section\n"
                  "a,melodic trap,dark,piano,chorus\n")
    mp = tmp_path / "rename.json"
    mp.write_text(json.dumps({"genre": {"pain music": "trap"}}))
    stamp = vocab_mod.migrate(tmp_path, lp, mp, backup=True)
    assert stamp["renames"] == 0
    # no backup written when nothing changed
    assert not list(tmp_path.glob("labels.csv.bak.*"))


# --------------------------------------------------------------------------- #
# agreement.py — inter-annotator agreement (#29)
# --------------------------------------------------------------------------- #

_COLS = ["source_id", "genre", "mood", "instruments", "section", "section_type"]


def _write(tmp_path, name, rows):
    p = tmp_path / name
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLS)
        w.writeheader()
        w.writerows(rows)
    return p


def _rows_a():
    return [
        {"source_id": "1", "genre": "melodic trap", "mood": "dark",
         "instruments": "piano|808 bass", "section": "chorus", "section_type": "hook"},
        {"source_id": "2", "genre": "melodic trap", "mood": "dark|emotional",
         "instruments": "piano|808 bass", "section": "verse", "section_type": "verse"},
        {"source_id": "3", "genre": "pain music", "mood": "melancholic",
         "instruments": "guitar loop", "section": "verse", "section_type": "verse"},
    ]


def _rows_b():
    return [
        {"source_id": "1", "genre": "melodic trap", "mood": "dark",
         "instruments": "piano|808 bass", "section": "chorus", "section_type": "hook"},
        {"source_id": "2", "genre": "melodic trap", "mood": "emotional",
         "instruments": "piano|808 bass|synth lead", "section": "verse", "section_type": "verse"},
        {"source_id": "3", "genre": "pain music", "mood": "melancholic",
         "instruments": "guitar loop", "section": "verse", "section_type": "verse"},
    ]


def test_agreement_exactness(tmp_path):
    pa = _write(tmp_path, "A.csv", _rows_a())
    pb = _write(tmp_path, "B.csv", _rows_b())
    rep = agree.agreement(pa, pb, tmp_path)

    assert rep["shared_tracks"] == 3
    f = rep["fields"]
    assert f["genre"]["exact_agreement"] == 1.0
    assert abs(f["mood"]["exact_agreement"] - 2 / 3) < 1e-3
    assert abs(f["instruments"]["exact_agreement"] - 2 / 3) < 1e-3
    assert 0.0 < f["mood"]["kappa"] < 1.0
    assert len(rep["disagreements"]) == 2  # mood + instruments on track 2
    assert (tmp_path / "metadata" / "agreement.json").exists()


def test_agreement_identical_files_kappa_one(tmp_path):
    pa = _write(tmp_path, "A.csv", _rows_a())
    rep = agree.agreement(pa, pa, tmp_path, out_rel="metadata/agree_self.json")
    assert rep["overall"]["kappa"] == 1.0
    assert rep["overall"]["exact_agreement"] == 1.0


def test_agreement_no_shared_ids(tmp_path):
    pa = _write(tmp_path, "A.csv", _rows_a())
    pb = _write(tmp_path, "B.csv", [
        {"source_id": "x", "genre": "trap", "mood": "dark", "instruments": "piano",
         "section": "intro", "section_type": "intro"},
    ])
    assert agree.agreement(pa, pb, tmp_path) == {}


def test_cohen_kappa_guards():
    # perfect agreement -> 1.0
    assert agree._cohen_kappa(10, 0, 0, 10) == 1.0
    # no observations -> 0.0
    assert agree._cohen_kappa(0, 0, 0, 0) == 0.0
    # chance-level -> near 0
    k = agree._cohen_kappa(5, 5, 5, 5)
    assert abs(k) < 0.01


# --------------------------------------------------------------------------- #
# promptbuilder.py — prompt assembly (#30)
# --------------------------------------------------------------------------- #


def test_build_prompt_full():
    p = build_prompt(
        section="chorus", genre="melodic trap",
        mood=["dark", "emotional", "dark"], instruments=["piano", "808 bass", "piano"],
        bpm=140, key="A minor", energy=0.8, role="central hook",
    )
    assert p == "chorus, 140 BPM, A minor, melodic trap, dark, emotional, piano, 808 bass, high energy, central hook"


def test_build_prompt_dedupes_and_order():
    p = build_prompt(section="verse", genre="pain music", mood="dark|emotional",
                     instruments="piano, 808 bass", bpm=72, key="E minor", energy=0.3)
    assert p.startswith("verse, 72 BPM, E minor, pain music, dark, emotional, piano, 808 bass, low energy")


def test_build_prompt_partial():
    assert build_prompt(bpm=90) == "90 BPM"
    assert build_prompt() == ""
    assert build_prompt(energy=0.5) == "medium energy"


def test_build_prompt_scalar_robust():
    # bare scalar (e.g. from a selectbox returning an int) must not crash
    p = build_prompt(section=3)
    assert p == "3"


def test_apply_override():
    p = apply_override("chorus, 78 BPM, A minor, dark piano", bpm=140, key="F minor")
    assert "140 BPM" in p and "F minor" in p and "78 BPM" not in p and "A minor" not in p
