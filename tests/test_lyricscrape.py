"""Tests for lyricscrape: ovh boilerplate cleaning, genius HTML extraction,
tracklist coverage for all artist profiles, and dry-run orchestration."""
from __future__ import annotations

from musictrain import lyricscrape as LS
from musictrain.artists import ARTISTS


def test_clean_ovh_strips_boilerplate():
    sample = (
        "Paroles de la chanson Test par Future\n\n"
        "First line here\nSecond line here\n\n"
        f"{LS._OVH_FOOTER}\n"
    )
    assert LS.clean_ovh(sample) == ["First line here", "Second line here"]


def test_clean_ovh_handles_empty():
    assert LS.clean_ovh("") == []


def test_genius_html_extraction():
    import re

    fixture = (
        '<div data-lyrics-container="true">Line one<br/>Line two</div>'
        '<div data-lyrics-container="true">Line three &amp; more</div>'
    )
    blocks = re.findall(r'<div data-lyrics-container="true"[^>]*>(.*?)</div>', fixture, re.S)
    text = re.sub(r"<br\s*/?>", "\n", "\n".join(blocks))
    text = re.sub(r"<[^>]+>", "", text)
    import html as _html

    text = _html.unescape(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    assert lines == ["Line one", "Line two", "Line three & more"]


def test_tracklists_cover_every_artist():
    missing = [a.id for a in ARTISTS if a.id not in LS.TRACKLISTS]
    assert missing == []
    short = [a.id for a in ARTISTS if len(LS.TRACKLISTS.get(a.id, [])) < 5]
    assert short == []


def test_scrape_dry_run_returns_records():
    records = LS.scrape(["drake", "future"], per_artist=2, source="ovh", sleep=0,
                        dry_run=True)
    assert len(records) == 4
    assert all(r["source"] == "ovh (dry-run)" for r in records)
    assert {r["artist"] for r in records} == {"Drake", "Future"}


def test_scrape_unknown_artist_dry_run():
    records = LS.scrape(["not-a-real-artist-xyz"], per_artist=2, source="ovh",
                        sleep=0, dry_run=True)
    assert len(records) == 0  # no tracklist -> no titles -> nothing
