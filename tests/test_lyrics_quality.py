"""Regression tests for lyric-engine quality fixes.

Covers: curated noun-safe rhyme banks (no more "Countin' up the told"),
topic-anchor lines filled with the topic's own nouns ("This pain inside me"),
signature openers used once per song, and template anti-repetition.
"""
from __future__ import annotations

import random

from musictrain import lyrics


def _pain_ctx(artist: str = "future", seed: int = 1) -> lyrics.BeatContext:
    return lyrics.BeatContext(artist=artist, mood="dark", topic="pain", seed=seed)


def test_no_gibberish_fill_words():
    """The user-reported gibberish words must never reappear in output."""
    banned = (
        "endow", "retold", "somehow", "the told", "the secure", "the endure",
        "the sustain", "the remain", "the appall", "the ascend", "the behold",
        "the sell", "the confide", "the convey", "the suffice", "the precise",
        "the explode", "the betray", "the attain", "the stay alive", "the for sure",
        "eatin'", "believin'", "breathin'", "achievin'",
    )
    for seed in (1, 7, 42, 99):
        r = lyrics.generate(_pain_ctx(artist="dababy", seed=seed))
        low = r.full_text().lower()
        for w in banned:
            assert w not in low, f"{w!r} leaked back into output (seed={seed})"


def test_topic_lead_uses_topic_nouns():
    """Section openers should use the topic's own words, not random bank words."""
    for seed in (1, 7, 42):
        r = lyrics.generate(_pain_ctx(seed=seed))
        first = r.sections[0]["lines"][0].lower()
        assert any(w in first for w in ("pain", "hurt", "ache", "tears")), (
            f"opener {first!r} doesn't reflect the 'pain' topic (seed={seed})"
        )


def test_topic_lead_does_not_use_bank_garbage():
    r = lyrics.generate(_pain_ctx(seed=1))
    first = r.sections[0]["lines"][0].lower()
    assert "somehow" not in first and "endow" not in first


def test_signature_opener_once_per_song():
    """A rapper's signature opener should open the song once, not every section."""
    r = lyrics.generate(_pain_ctx(artist="dababy", seed=42))
    openers = [ln for s in r.sections for ln in s["lines"]
               if ln.startswith(("Let's go,", "Yeah,"))]
    assert len(openers) <= 1, f"signature opener repeated {len(openers)}x"


def test_pick_template_avoids_recent():
    """The same line template must not be picked back-to-back."""
    rng = random.Random(3)
    first = lyrics._pick_template_near(14, rng)
    for _ in range(5):
        nxt = lyrics._pick_template_near(14, rng, recent=[first])
        assert nxt != first


def test_no_adjacent_line_uses_same_rhyme_word():
    """Two-slot lines shouldn't bleed their tail word onto the next line's end."""
    r = lyrics.generate(_pain_ctx(seed=42))
    for s in r.sections:
        ends = [ln.split()[-1].strip("'").lower() for ln in s["lines"]]
        for a, b in zip(ends, ends[1:]):
            assert a != b, f"adjacent lines both end with {a!r} in {s['role']}"


def test_generate_still_deterministic_and_structured():
    a = lyrics.generate(_pain_ctx(seed=5))
    b = lyrics.generate(_pain_ctx(seed=5))
    assert a.full_text() == b.full_text()
    roles = [s["role"] for s in a.sections]
    assert roles[0] == "intro" and roles[-1] == "outro"
    assert all(s["lines"] for s in a.sections)


# --------------------------------------------------------------------------- #
# Expanded content library
# --------------------------------------------------------------------------- #
def test_new_rhyme_banks_present():
    for bank in ("oke", "ell", "ash", "aze", "oom", "ine", "one", "it", "eat", "et", "ove", "ale"):
        assert lyrics._RHYME_BANKS[bank], f"bank {bank!r} missing"


def test_new_templates_present():
    for t in ("I don't chase the {R}, the {R} come to me",
              "They said I changed, I just changed the {R}",
              "Talk is cheap, I let the {R} do the talkin'"):
        assert t in lyrics._LINE_TEMPLATES


def test_new_topics_have_lead_lines_and_nouns():
    for topic in ("money", "grind", "respect", "come-up", "party", "rage", "freedom", "boss"):
        assert lyrics._TOPIC_LINES[topic], f"topic {topic!r} missing lead lines"
        assert lyrics._TOPIC_NOUNS[topic], f"topic {topic!r} missing anchor nouns"


def test_new_topic_generates_coherent_opener():
    for topic, nouns in (("boss", ("boss", "show", "game", "block")),
                         ("money", ("money", "racks", "cash", "bags")),
                         ("party", ("party", "club", "vibe", "motion"))):
        r = lyrics.generate(lyrics.BeatContext(artist="future", mood="dark", topic=topic, seed=7))
        first = r.sections[0]["lines"][0].lower()
        assert any(w in first for w in nouns), f"{topic!r} opener {first!r} off-topic"
