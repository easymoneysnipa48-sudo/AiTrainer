"""Fetch real lyrics for the 22 artist profiles into the fine-tune dataset.

Two backends:

* ``genius`` — official Genius API (set ``GENIUS_ACCESS_TOKEN``): searches the
  artist, takes their top songs, and scrapes each song's lyrics page.
* ``lyrics.ovh`` — keyless public API (``https://api.lyrics.ovh/v1``). Needs
  exact artist + title, so it is paired with the curated tracklists below.

``lyricscrape --source auto`` picks Genius when a token is present, otherwise
falls back to lyrics.ovh. Records land in the same shape ``lyricdataset``
expects (``artist``, ``title``, ``lines``, ...) and are merged into
``lyrics/<artist_id>/songs.jsonl`` without touching existing songs.

Note: song *titles* are factual metadata (not copyrightable); the lyrics text
is fetched for personal model fine-tuning. Only fetch lyrics you have the right
to use for your training data.
"""
from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

from . import console

# --------------------------------------------------------------------------- #
# Curated starter tracklists (public song titles) — one per artist profile.
# Used by the keyless lyrics.ovh backend; Genius search ignores these and uses
# the API's own top-songs ranking.
# --------------------------------------------------------------------------- #
TRACKLISTS: Dict[str, List[str]] = {
    "drake": ["God's Plan", "Started From The Bottom", "In My Feelings", "Hotline Bling",
              "One Dance", "Nonstop", "Toosie Slide", "Headlines", "Hold On We're Going Home",
              "Passionfruit", "Nice For What", "Energy", "Best I Ever Had", "Marvin's Room", "Forever"],
    "omb-peezy": ["Trappin", "When I'm Done", "Lame", "Doin Too Much", "Think Twice", "Cap",
                  "Loyalty", "Lil Baby", "Day Ones", "Robbery Part 2", "I Ain't Gon Lie",
                  "Came From Nothing", "Who Is OMB Peezy", "Realest In The Game", "Letter 2 My Brother"],
    "lil-durk": ["All Love", "Just Cause Y'all Waited", "3 Headed Goat", "Vulture Island",
                 "Back In Blood", "Every Chance I Get", "No Auto Durk", "India", "Street Affection",
                 "My Beyoncé", "Different Meaning", "Green Light", "Like Me", "When We Shoot", "Neighbor"],
    "chief-keef": ["I Don't Like", "Love Sosa", "Faneto", "Hate Being Sober", "Earned It", "Kobe",
                   "Save That Shit", "Macaroni Time", "Almighty So", "Citgo", "3Hunna", "Bang Bang",
                   "Kay Kay", "War", "They Know"],
    "meek-mill": ["Dreamchasers", "Amen", "Ima Boss", "Levels", "Lord Knows", "Going Bad", "Shine",
                  "Intro", "R.I.C.O.", "All Eyes On You", "Championships", "1942 Flows", "Cold Hearted",
                  "Trauma", "On My Soul"],
    "kendrick-lamar": ["HUMBLE.", "DNA.", "Alright", "Money Trees", "Swimming Pools", "Backseat Freestyle",
                       "King Kunta", "Bitch Don't Kill My Vibe", "MAAD City", "The Recipe", "i", "LOVE.",
                       "Element", "The Blacker The Berry", "Sing About Me"],
    "jackboy": ["Workin", "Godzilla", "Alcoholic", "Pressure", "Steppin On Bitches", "Finesse", "Hold On",
                "8 Figures", "Baby Jesus", "Play For Keeps", "Walk Em Down", "Out The Mud", "Pain",
                "Racks Up", "On My Own"],
    "quavo": ["Workin Me", "Huncho Dreams", "Chocolate", "Bubble Gum", "Lamb Talk", "Paper Over Here",
              "Big Boss", "Pass Out", "Quavo Huncho", "Hotel Lobby", "Pizza", "Patty Cake", "Greatness",
              "Lost", "Huncho Jack"],
    "gunna": ["Drip Too Hard", "Skybox", "One Call", "Top Off", "Speed It Up", "2000 Miles", "Out The Hood",
              "Mop", "Surf", "Hoodie", "Pushin P", "Who You Foolin", "Space Cadet", "Turbo", "My Oh My"],
    "offset": ["Ric Flair Drip", "Clout", "Wait", "Hurricane", "How Did I Get Here", "Red Room", "Legacy",
               "Underrated", "Blame It On Set", "Night Vision", "Real Deal", "Danger", "Fan", "Vibe", "Boss Life"],
    "takeoff": ["Last Memory", "Casino", "Martians", "Caps Agenda", "Intruder", "Trap Talk", "Not No",
                "Soul Plane", "Infatuation", "Lead The Way", "Bruce Wayne", "See Ya", "Peak", "News", "Say It"],
    "juice-wrld": ["Lucid Dreams", "All Girls Are The Same", "Robbery", "Legends", "Wishing Well",
                   "Come & Go", "Bandit", "Empty", "Hear Me Calling", "Fast", "Lean Wit Me",
                   "Armed And Dangerous", "Life's A Mess", "Righteous", "Black & White"],
    "future": ["Mask Off", "March Madness", "Life Is Good", "Wait For U", "Low Life", "Jumpman",
               "King's Dead", "Codeine Crazy", "Draco", "Stick Talk", "Thought It Was A Drought",
               "F**k Up Some Commas", "Where Ya At", "New Illuminati", "Monster"],
    "lil-baby": ["Drip Too Hard", "Yes Indeed", "Freestyle", "Woah", "Sum 2 Prove", "The Bigger Picture",
                 "We Paid", "Close Friends", "My Dawg", "Pure Cocaine", "Emotions", "Heyy",
                 "Emotionally Scarred", "All In", "Life Goes On"],
    "jay-z": ["Empire State Of Mind", "99 Problems", "N****s In Paris", "Hard Knock Life", "Big Pimpin",
              "Run This Town", "Dead Presidents", "Song Cry", "Izzo", "Dirt Off Your Shoulder",
              "Heart Of The City", "PSA", "The Story Of OJ", "4:44", "Show Me What You Got"],
    "kanye-west": ["Stronger", "Gold Digger", "Heartless", "Flashing Lights", "Runaway", "POWER",
                   "All Of The Lights", "Bound 2", "Good Life", "Jesus Walks", "Can't Tell Me Nothing",
                   "Black Skinhead", "Father Stretch My Hands", "Famous", "Praise God"],
    "michael-jackson": ["Billie Jean", "Beat It", "Thriller", "Smooth Criminal", "Black Or White", "Bad",
                        "Man In The Mirror", "Remember The Time", "Don't Stop 'Til You Get Enough",
                        "Rock With You", "The Way You Make Me Feel", "Dirty Diana",
                        "Wanna Be Startin' Somethin'", "They Don't Care About Us", "You Rock My World"],
    "nocap": ["Ghetto Angels", "Vaccine", "Time Heals", "He Said She Said", "No Mercy", "Prayer Hands",
              "Gucci On My Feet", "Ballin", "Fresh", "Pop Out", "Demons", "Speed", "Now Or Never",
              "Nobody Else", "Lil Bit"],
    "quando-rondo": ["I Remember", "ABG", "Motown", "Scarred From Love", "PTSD", "You Don't Know Me",
                     "Rocks", "Life Goes On", "Let's Go", "Permanently Scarred", "Karma", "Situations",
                     "Mansion", "Money", "4 Ways"],
    "dababy": ["Suge", "BOP", "Rockstar", "Vibez", "Baby On Baby", "Find My Way", "Goin Baby",
               "BLAME IT ON BABY", "Masterpiece", "Peep Hole", "Intro", "Call It Even", "Thug Life",
               "Sold Out Dates", "iPhone"],
    "young-thug": ["Best Friend", "Stoner", "Check", "Lifestyle", "Digits", "Pick Up The Phone",
                   "Halftime", "The London", "Hot", "So Much Fun", "Bad Bad Bad", "Wyclef Jean",
                   "Relationship", "Future Swag", "Givenchy"],
    "lil-gotit": ["Da Real Hoodbitch", "Beat The Shit", "Racks Today", "Im Da Type", "Bubblegum",
                  "Look Like", "Angel", "Doin The Most", "Flavor", "Migo Type", "Freestyle", "Pushin",
                  "Self Made", "Kilos", "Free Da Guys"],
}

_GENIUS_API = "https://api.genius.com"
_OVH_API = "https://api.lyrics.ovh/v1"


# --------------------------------------------------------------------------- #
# HTTP helpers (stdlib urllib only)
# --------------------------------------------------------------------------- #
def _get(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 30.0) -> Optional[str]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - network is best-effort
        console.warn(f"request failed {url[:80]}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Genius backend
# --------------------------------------------------------------------------- #
def _genius_token() -> str:
    return os.environ.get("GENIUS_ACCESS_TOKEN", "").strip()


def genius_search(token: str, artist: str, limit: int = 15) -> List[Tuple[str, str]]:
    """Return (title, path) for the artist's top songs via the search API."""
    q = urllib.parse.quote(artist)
    body = _get(f"{_GENIUS_API}/search?q={q}", {"Authorization": f"Bearer {token}"})
    if not body:
        return []
    try:
        hits = json.loads(body)["response"]["hits"]
    except (KeyError, ValueError):
        return []
    out: List[Tuple[str, str]] = []
    for h in hits[:limit]:
        r = h.get("result") or {}
        if r.get("primary_artist", {}).get("name", "").lower() != artist.lower():
            continue
        path = r.get("path") or ""
        title = r.get("title") or ""
        if path and title:
            out.append((title, path))
    return out


def genius_lyrics(path: str) -> List[str]:
    """Scrape the lyrics from a Genius song page (``data-lyrics-container``)."""
    body = _get("https://genius.com" + path, headers={"User-Agent": "Mozilla/5.0"})
    if not body:
        return []
    # lyrics live in <div data-lyrics-container="true">…</div> blocks
    blocks = re.findall(r'<div data-lyrics-container="true"[^>]*>(.*?)</div>', body, re.S)
    if not blocks:
        return []
    text = "\n".join(blocks)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


# --------------------------------------------------------------------------- #
# lyrics.ovh backend (keyless)
# --------------------------------------------------------------------------- #
_OVH_FOOTER = "*** This Lyrics is NOT for Commercial Use ***"
_OVH_PREFIX_RE = re.compile(r"^paroles? de la chanson\b", re.I)


def clean_ovh(text: str) -> List[str]:
    """Strip lyrics.ovh boilerplate (header line + commercial-use footer)."""
    lines = [ln.strip() for ln in (text or "").splitlines()]
    while lines and (_OVH_PREFIX_RE.match(lines[0]) or not lines[0]):
        lines.pop(0)
    if lines and lines[-1].strip() == _OVH_FOOTER:
        lines.pop()
    while lines and not lines[-1]:
        lines.pop()
    return [ln for ln in lines if ln]


def ovh_lyrics(artist: str, title: str) -> List[str]:
    a = urllib.parse.quote(artist)
    t = urllib.parse.quote(title)
    body = _get(f"{_OVH_API}/{a}/{t}")
    if not body:
        return []
    try:
        text = json.loads(body).get("lyrics", "")
    except ValueError:
        return []
    return clean_ovh(text)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def scrape(
    artists: List[str],
    per_artist: int = 15,
    source: str = "auto",
    extra_titles: Optional[List[str]] = None,
    sleep: float = 1.0,
    dry_run: bool = False,
    limit: int = 0,
) -> List[dict]:
    """Fetch lyrics for the requested artists and return importable records.

    ``source``: ``auto`` (Genius when a token is set, else lyrics.ovh),
    ``genius`` or ``ovh``. ``extra_titles`` are appended to every artist's
    title list (Genius mode only uses them when the search returns nothing).
    """
    from .artists import get_artist

    if source == "auto":
        source = "genius" if _genius_token() else "ovh"
    token = _genius_token() if source == "genius" else ""
    if source == "genius" and not token:
        console.warn("No GENIUS_ACCESS_TOKEN set — falling back to lyrics.ovh (keyless)")
        source = "ovh"

    records: List[dict] = []
    total = 0
    for raw in artists:
        prof = get_artist(raw.strip().lower())
        aid = prof.id if prof else raw.strip().lower().replace(" ", "-")
        display = prof.name if prof else raw.strip()
        found: List[Tuple[str, str]] = []
        titles: List[str] = []
        if source == "genius":
            found = genius_search(token, display, limit=per_artist * 2)
            titles = [t for t, _ in found][:per_artist]
        else:
            titles = list(TRACKLISTS.get(aid, []))[:per_artist]
        titles += [t for t in (extra_titles or []) if t not in titles]

        console.step(f"Scraping {display} ({source}) — {len(titles)} song(s)")
        got = 0
        for title in titles:
            if limit and total >= limit:
                break
            if dry_run:
                records.append({
                    "artist": display, "title": title, "lines": ["[dry-run]"],
                    "mood": "", "topic": "", "source": f"{source} (dry-run)",
                })
                total += 1
                got += 1
                continue
            lines: List[str] = []
            if source == "genius":
                path = next((p for t, p in found if t == title), "")
                if not path:
                    continue
                lines = genius_lyrics(path)
            else:
                lines = ovh_lyrics(display, title)
            if len(lines) < 4:
                console.info(f"  skip {title}: no lyrics found")
                continue
            records.append({
                "artist": display, "title": title, "lines": lines,
                "mood": "", "topic": "", "source": source,
            })
            total += 1
            got += 1
            if sleep > 0:
                time.sleep(sleep)
        console.ok(f"  {display}: {got} song(s) with lyrics")
        if limit and total >= limit:
            break

    console.ok(f"Scraped {len(records)} song record(s) total")
    return records
