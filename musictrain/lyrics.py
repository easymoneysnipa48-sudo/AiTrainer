"""Offline, seedable lyric generation driven by a beat's analysis.

The engine turns a beat's *measured* properties (tempo, key, swing feel, and
detected section structure) plus an artist style profile into structured rap
lyrics. No network, no API key, no heavy model — it's a template + rhyme-bank
generator, so it runs instantly and deterministically for a given seed.

If you want higher-quality output you can opt into an LLM backend by setting
``MUSICTRAIN_LLM_API_KEY`` (and optionally ``MUSICTRAIN_LLM_BASE_URL`` /
``MUSICTRAIN_LLM_MODEL``). The engine then prompts a chat-completions endpoint
with the artist profile + beat context and parses the reply into sections; on
any failure it transparently falls back to the offline generator.
"""

from __future__ import annotations

import json
import os
import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .artists import Artist, get_artist
from .logging import get_logger
from .lyrictools import ARRANGEMENTS, count_syllables, syllable_target

log = get_logger("lyrics")

# --------------------------------------------------------------------------- #
# Beat context
# --------------------------------------------------------------------------- #


@dataclass
class SectionSpec:
    role: str = "verse"          # intro | verse | hook | chorus | bridge | outro
    bars: int = 16
    topic: str = ""              # per-section override (feature #2)
    mood: str = ""               # per-section override
    flow: str = ""               # per-section flow override
    artist: str = ""             # per-section artist override (feature mode #6)
    artist2: str = ""            # duet partner — alternates bars within the section (#5)
    energy: float = 0.0          # detected section energy 0..1 (intensity mapping #3)


@dataclass
class BeatContext:
    bpm: float = 140.0
    key: str = "A minor"
    swing: str = "straight"      # straight | moderate | swung
    energy: float = 0.5
    artist: str = "future"
    mood: str = "dark"
    topic: str = "struggle"
    structure: List[SectionSpec] = field(default_factory=list)
    negative: List[str] = field(default_factory=list)   # feature #29
    weights: Dict[str, float] = field(default_factory=dict)  # feature #28
    seed: int = 42

    def artist_obj(self) -> Artist:
        return get_artist(self.artist) or get_artist("future") or None  # type: ignore[return-value]


_DEFAULT_STRUCTURE = [
    SectionSpec("intro", 4),
    SectionSpec("verse", 16),
    SectionSpec("hook", 8),
    SectionSpec("verse", 16),
    SectionSpec("hook", 8),
    SectionSpec("outro", 4),
]

_BARS_BY_ROLE = {
    "intro": 4, "verse": 16, "hook": 8, "chorus": 8,
    "bridge": 8, "pre-chorus": 4, "outro": 4, "full-song": 16,
}


def beat_context_from_analysis(
    analysis: dict,
    artist: str = "future",
    mood: str = "",
    topic: str = "",
    seed: int = 42,
) -> BeatContext:
    """Build a :class:`BeatContext` from a ``metadata/analysis.jsonl`` record.

    Uses the detected key, tempo, swing feel, and structure roles so the lyrics
    are shaped by the *actual* beat the user uploaded (feature #22).
    """
    key = (analysis.get("key") or {}).get("key", "A minor")
    beat_grid = analysis.get("beat_grid") or {}
    tempo = float(beat_grid.get("tempo", 140.0))
    swing = (analysis.get("swing") or {}).get("feel", "straight")

    structure: List[SectionSpec] = []
    for seg in (analysis.get("structure") or {}).get("segments", []):
        role = seg.get("role") or seg.get("label") or "verse"
        if role not in _BARS_BY_ROLE:
            role = "verse"
        structure.append(
            SectionSpec(
                role=role,
                bars=_BARS_BY_ROLE[role],
                energy=float(seg.get("energy", 0.0)),
            )
        )

    if not structure:
        structure = list(_DEFAULT_STRUCTURE)

    energy = 0.5
    energies = [s.get("energy", 0.0) for s in (analysis.get("structure") or {}).get("segments", [])]
    if energies:
        energy = float(sum(energies) / len(energies))

    return BeatContext(
        bpm=tempo,
        key=key,
        swing=swing,
        energy=energy,
        artist=artist,
        mood=mood,
        topic=topic,
        structure=structure,
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Rhyme banks — grouped end-words so blocks actually rhyme.
# --------------------------------------------------------------------------- #
_RHYME_BANKS: Dict[str, List[str]] = {
    # Curated to noun-safe words only: every {R} slot in a template is a noun
    # position ("the {R}", "my {R}", "through the {R}"), so verbs/adverbs like
    # "endow", "told", "secure" produced gibberish ("Countin' up the told").
    "ain": ["pain", "rain", "chain", "main", "brain", "campaign", "champagne", "domain",
            "stain", "lane", "train", "grain", "terrain", "refrain"],
    "ight": ["night", "fight", "light", "right", "sight", "flight", "hype", "spotlight",
             "hindsight", "might", "height", "bite", "kite"],
    "old": ["gold", "cold", "hold", "control", "roll", "mold", "toll", "stroll", "fold"],
    "ow": ["now", "vow", "clout", "shout", "doubt", "count", "amount", "route", "drought",
           "crowd", "bout"],
    "ay": ["way", "day", "pay", "stay", "play", "sway", "essay", "delay", "replay",
           "ray", "tray", "spray", "clay", "display"],
    "ive": ["drive", "hive", "jive", "five", "dive"],
    "eep": ["deep", "keep", "sleep", "leap", "creep", "jeep", "sheep", "sweep", "beep"],
    "all": ["fall", "call", "ball", "wall", "stall", "haul", "hall", "mall", "crawl",
            "brawl", "sprawl", "recall"],
    "end": ["end", "bend", "friend", "trend", "blend", "weekend", "girlfriend", "boyfriend"],
    "ame": ["name", "fame", "game", "flame", "shame", "claim", "blame", "frame",
            "acclaim", "dame"],
    "ack": ["back", "track", "stack", "black", "attack", "rack", "lack", "facts", "impact",
            "comeback", "crack", "snack", "sack", "pack", "jack", "slack", "tack"],
    "eel": ["feel", "steel", "deal", "seal", "reveal", "appeal", "wheel", "peel", "heel",
            "meal", "spiel", "steal"],
    "ice": ["ice", "price", "sacrifice", "device", "advice", "slice", "rise", "paradise",
            "mice", "dice", "vice", "spice"],
    "oad": ["road", "load", "code", "mode", "abode", "ode", "node", "goad", "episode",
            "download", "upload"],
    "oney": ["money", "honey", "bunny", "phony"],
    "ock": ["block", "lock", "rock", "stock", "clock", "shock", "knock", "dock", "flock",
            "sock"],
    "ide": ["ride", "slide", "hide", "inside", "pride", "tide", "divide", "suicide",
            "guide", "side", "bride", "glide", "stride"],
    "ure": ["future", "feature", "culture", "nature", "creature", "vulture", "picture",
            "capture", "rapture", "gesture", "fracture", "manicure", "tenure", "cure"],
    "ace": ["face", "race", "place", "chase", "embrace", "grace", "space", "disgrace",
            "maze", "days"],
    "eam": ["dream", "scheme", "team", "cream", "extreme", "beam", "self-esteem", "gleam"],
    # multi-syllable / slant banks (feature #4)
    "ation": ["situation", "motivation", "salvation", "elevation", "celebration", "dedication",
              "destination", "generation", "foundation", "education", "reputation"],
    "able": ["stable", "table", "label", "cable", "fable", "sable", "gable"],
    "ust": ["trust", "must", "bust", "dust", "crust", "disgust", "lust", "rust", "thrust",
            "gust"],
    "in": ["villain", "captain", "mountain", "fountain", "satin", "cousin", "ribbon"],
    "ition": ["condition", "position", "ambition", "tradition", "addition", "mission",
              "vision", "edition", "decision", "revision"],
    "oke": ["broke", "smoke", "joke", "stroke", "choke", "yoke", "coke", "folk", "oak", "bloke"],
    "ell": ["bell", "tell", "fell", "well", "shell", "spell", "smell", "cell", "farewell"],
    "ash": ["cash", "flash", "stash", "dash", "crash", "splash", "trash", "smash"],
    "aze": ["maze", "blaze", "phase", "craze", "haze", "raise", "praise", "gaze"],
    "oom": ["room", "boom", "zoom", "groom", "bloom", "doom", "tomb", "broom", "gloom"],
    "ine": ["shine", "line", "mine", "sign", "time", "crime", "prime", "climb", "grind", "design"],
    "one": ["bone", "stone", "throne", "phone", "zone", "cone", "clone", "tone"],
    "it": ["fit", "hit", "lit", "spit", "split", "kit", "pit", "grit", "wit"],
    "eat": ["heat", "beat", "seat", "treat", "street", "meat", "repeat", "defeat", "retreat"],
    "et": ["jet", "set", "bet", "sweat", "debt", "vet", "threat", "reset"],
    "ove": ["love", "glove", "dove", "shove"],
    "ale": ["scale", "sale", "tale", "jail", "nail", "mail", "trail", "whale", "grail"],
}

_LINE_TEMPLATES: List[str] = [
    "I been movin' through the {R}, ain't no time to explain",
    "They don't love you till you got the {R}, that's the game",
    "Came from nothin' but I found the {R} in my lane",
    "I could never fold, I stay solid through the {R}",
    "Late nights, chasin' every single {R} I could gain",
    "They was doubtin' me, now they all remember the {R}",
    "I put my heart in this, now it's part of the {R}",
    "Countin' up the {R} while they sleepin' on the {R}",
    "I stay loyal to the {R}, never switch for the {R}",
    "Lost some real ones, still I carry all the {R}",
    "I was down bad, had to grind up out the {R}",
    "Now I'm up, and they can't even see the {R}",
    "Talkin' to the {R}, prayin' that I make it through the {R}",
    "Every scar is a lesson, I carry the {R}",
    "I ain't stressin' over {R}, I been through worse than the {R}",
    "Whole city on my back, and I'm carryin' the {R}",
    "They gon' talk, but they never walked a mile in my {R}",
    "I was built for the {R}, I don't know how to quit",
    "Pain in my chest but I still chase the {R}",
    "They left me for dead, now they see me in the {R}",
    "I turned my {R} into a blessing, that's the {R}",
    "No handouts, I took the {R} and I ran with the {R}",
    "Realer than most, I don't fake it for the {R}",
    "Money comin' in, but the {R} still the same",
    "I can't sleep, I keep dreamin' of the {R}",
    "If you ain't family, you ain't part of the {R}",
    "I remember nights with nothin' but the {R}",
    "Now it's all {R}, made it through the {R}",
    "Keep your circle small, watch who get the {R}",
    "I'ma make it out, I swear it on the {R}",
    # simile / imagery lines (feature #2)
    "I'm in the {R} like a king with no throne",
    "Cold as the winter, I been through the {R}",
    "Shinin' through the {R} like a light in the dark",
    "Ridin' on the {R} like it's my last ride",
    "Deep as the {R}, I can't even find the bottom",
    "Float like a ghost, movin' through the {R}",
    "Hard as the {R}, but I still keep my soft side",
    "Lost in the {R} like a ship with no shore",
    # stronger anchors
    "They want the old me, but the {R} made me new",
    "I done seen it all, from the bottom to the {R}",
    "Can't knock the hustle, I was born for the {R}",
    "They sleep on the {R}, I been awake through the {R}",
    # extended set — more shapes, less repetition across a long verse
    "I been grindin' in the {R}, now they see the {R}",
    "I don't chase the {R}, the {R} come to me",
    "They said I changed, I just changed the {R}",
    "I been through the {R}, came out with the {R}",
    "Talk is cheap, I let the {R} do the talkin'",
    "Pressure make diamonds, I been shinin' through the {R}",
    "They sleep on me now, they gon' wake up to the {R}",
    "I ain't gotta prove nothin', the {R} speak for itself",
    "Real recognize real, they can't fake the {R}",
    "They wanted me to break, but I built from the {R}",
    "Every loss made me tougher, that's just the {R}",
    "I been patient with the {R}, now it's my time",
    "I put my {R} on the line every single day",
    "Same {R}, new day, I just level up",
    "Whole squad in the {R}, we don't fold under pressure",
]

# Approximate syllable count per template (with a one-syllable placeholder for
# {R}) — used to pace line density against the beat's tempo (feature #1).
_TEMPLATE_SYLLABLES: List[int] = [count_syllables(t.replace("{R}", "X")) for t in _LINE_TEMPLATES]

# Themed lead-ins keyed by topic keyword — injected as the first lines of a verse
# or the anchor line of a hook so the requested topic actually shows up.
_TOPIC_LINES: Dict[str, List[str]] = {
    "pain": ["This {R} inside me runnin' deep, I can't escape", "I been numb to the {R}, but the {R} still aches"],
    "heartbreak": ["You said forever, now forever turned to {R}", "I gave you my all and you left me in the {R}"],
    "love": ["You the only one that make me feel the {R}", "I'd cross the whole world just to keep you in the {R}"],
    "loyalty": ["Loyalty over everything, that's the {R}", "If you ain't solid, you can't ride with the {R}"],
    "struggle": ["Started from the bottom, I was livin' in the {R}", "I had nothin' but a dream and a pocket full of {R}"],
    "success": ["They said I wouldn't make it, now I'm at the {R}", "From the bottom to the {R}, that's the come-up"],
    "wealth": ["Racks on racks, I'm drownin' in the {R}", "Money on my mind, I can't get enough of the {R}"],
    "loss": ["Lost a brother to the {R}, I still feel the {R}", "Every day I mourn the {R} they took from the {R}"],
    "street": ["These streets don't love nobody, that's the {R}", "Out here in the {R}, it's survival of the {R}"],
    "fame": ["Fame come with the {R}, but it cost me the {R}", "They see the lights, they don't see the {R}"],
    "family": ["Everything I do, I do it for the {R}", "My family my rock, they the reason for the {R}"],
    "hustle": ["Grindin' day and night, that's the only {R}", "I was made for the {R}, it's in my {R}"],
    "faith": ["I been prayin' through the {R}, keep my faith in the {R}", "God got me through the {R}, I can't lose the {R}"],
    "demons": ["I been fightin' demons, they be callin' my {R}", "These {R} in my head, they won't leave me the {R}"],
    "anxiety": ["Anxiety on my chest, I can't catch the {R}", "My mind racin', I can't sleep through the {R}"],
    "confidence": ["I'm the one they doubt, but I already know the {R}", "Ain't nobody stoppin' me, I'm built for the {R}"],
    "violence": ["This {R} in the streets, it'll take your {R}", "Keep your head on a swivel, it's {R} in the {R}"],
    "money": ["I'm married to the {R}, I can't get enough", "They talk behind my back 'cause I'm makin' the {R}"],
    "grind": ["I was built in the {R}, that's where I found my {R}", "No sleep, just the {R}, I been on my {R}"],
    "respect": ["I earned the {R}, they can't take that away", "Give me my {R}, I done paid my dues"],
    "come-up": ["This the {R} I been prayin' for", "I seen the {R} comin' before they did"],
    "party": ["It's a {R} every night, I can't slow down", "We turn the {R} up, can't nobody stop us"],
    "rage": ["I got {R} in my veins, I can't calm down", "This {R} buildin' up, I'ma let it out"],
    "freedom": ["I been chasin' the {R} my whole life", "Can't cage the {R}, I was born to be free"],
    "boss": ["I'm the {R} now, I answer to nobody", "They used to run the {R}, now I run the {R}"],
}

_MOOD_PREFIX: Dict[str, str] = {
    "dark": "In the dark, ",
    "emotional": "I feel it deep, ",
    "aggressive": "I'm comin' hard, ",
    "melancholic": "Lately I been low, ",
    "braggadocious": "Ain't no question, ",
    "confident": "I already know, ",
    "menacing": "Don't test me, ",
    "smooth": "Real smooth, ",
    "seductive": "Come a little closer, ",
    "heartbroken": "You tore me apart, ",
    "anxious": "My mind won't stop, ",
    "reflective": "Lookin' back, ",
    "determined": "I'm locked in, ",
    "hopeful": "I see the light, ",
    "triumphant": "We made it out, ",
    "gritty": "Out the mud, ",
    "lonely": "Nobody callin', ",
    "nostalgic": "Take me back to, ",
    "euphoric": "I'm on a high, ",
    "wounded": "Still bleedin', ",
    "defiant": "I won't break, ",
    "toxic": "Yeah we toxic, ",
    "lavish": "Living lavish, ",
    "hustling": "No days off, ",
}

_MOOD_ADLIB: Dict[str, str] = {
    "dark": "shh", "aggressive": "bah", "energetic": "let's go",
    "melancholic": "damn", "confident": "yeah", "menacing": "grr",
    "heartbroken": "why", "braggadocious": "ha", "euphoric": "woo",
    "lonely": "sigh", "determined": "grr", "smooth": "mhmm",
}


# --------------------------------------------------------------------------- #
# Flow / cadence derivation from the beat + artist.
# --------------------------------------------------------------------------- #
def _flow_for(artist: Artist, ctx: BeatContext) -> str:
    flows = list(artist.flow) or ["melodic"]
    if ctx.swing in ("swung", "moderate") and "triplet" not in flows:
        flows.insert(0, "triplet")
    if ctx.bpm >= 130 and (artist.density >= 4 or any("double" in f for f in flows)):
        if "double-time" not in flows:
            flows.insert(0, "double-time")
    if ctx.bpm <= 90 and "laid-back" not in flows:
        flows.insert(0, "laid-back")
    return flows[0]


def _cadence_for(artist: Artist, ctx: BeatContext) -> str:
    if ctx.bpm >= 130 and artist.density >= 4:
        return "fast"
    if ctx.bpm <= 90:
        return "slow"
    return artist.cadence


def _density_for(artist: Artist, ctx: BeatContext, energy: float | None = None) -> int:
    d = artist.density
    e = ctx.energy if energy is None else energy
    if ctx.bpm >= 130:
        d += 1
    if ctx.bpm <= 90:
        d = max(1, d - 1)
    # beat-energy → intensity mapping (feature #3)
    if e > 0.7:
        d += 1
    elif e < 0.3:
        d = max(1, d - 1)
    return max(1, min(5, d))


# --------------------------------------------------------------------------- #
# Line assembly
# --------------------------------------------------------------------------- #
def _pick_rhyme_bank(rng: random.Random) -> str:
    return rng.choice(list(_RHYME_BANKS.keys()))


def _pick_template_near(
    target: int,
    rng: random.Random,
    spread: int = 2,
    recent: Optional[List[str]] = None,
) -> str:
    """Sample a line template whose syllable count is close to the budget.

    ``recent`` holds templates already used in the last few bars so the same
    line shape isn't picked back-to-back ("I turned my {R} into a blessing"
    three times in a row was a real output bug).
    """
    used = set(recent or [])
    candidates = [
        t for t, s in zip(_LINE_TEMPLATES, _TEMPLATE_SYLLABLES)
        if abs(s - target) <= spread and t not in used
    ]
    if not candidates:
        candidates = [
            t for t, s in zip(_LINE_TEMPLATES, _TEMPLATE_SYLLABLES)
            if abs(s - target) <= spread
        ]
    if not candidates:
        candidates = _LINE_TEMPLATES
    return rng.choice(candidates)


def _render_line(template: str, rhyme_words: List[str]) -> str:
    # fill {R} slots left-to-right, cycling through the rhyme words so a line
    # with two slots uses two different words (e.g. "the pain ... the rain").
    out = template
    k = 0
    while "{R}" in out:
        rw = rhyme_words[k % len(rhyme_words)]
        out = out.replace("{R}", rw, 1)
        k += 1
    return out


def _ucfirst(s: str) -> str:
    """Uppercase only the first char, preserving the rest (unlike str.capitalize)."""
    return s[:1].upper() + s[1:] if s else s


def _join_prefix(prefix: str, line: str) -> str:
    """Join a mood prefix onto a line mid-sentence ("In the dark, they talk…"),
    keeping the pronoun "I" capitalized and avoiding double prefixes."""
    if not prefix or line.startswith(prefix):
        return line
    if line[0] == "I":
        return prefix + line
    return prefix + line[0].lower() + line[1:]


def _topic_leads(topic: str, mood: str, rng: random.Random) -> List[str]:
    """Return themed lead line templates (topic keyword matched), else generic."""
    t = (topic or "").lower()
    leads: List[str] = []
    for key, templates in _TOPIC_LINES.items():
        if key in t:
            leads.extend(templates)
    if not leads:
        leads = list(_TOPIC_LINES.get("struggle", []))
    return leads


# Nouns for the topic-anchor lines — the lead-in {R} is filled with the topic's
# own words so the opener reads coherently ("This pain inside me runnin' deep"
# instead of "This endow inside me"). These are anchor lines: their end word is
# a fixed template word in most cases, so they don't need to rhyme with the
# following bars.
_TOPIC_NOUNS: Dict[str, List[str]] = {
    "pain": ["pain", "hurt", "ache", "tears"],
    "heartbreak": ["heartbreak", "regret", "memories", "ghosts"],
    "love": ["love", "heart", "feelings", "affection"],
    "loyalty": ["loyalty", "code", "respect", "trust"],
    "struggle": ["struggle", "grind", "hustle", "mud"],
    "success": ["success", "top", "shine", "victory"],
    "wealth": ["money", "racks", "bags", "cash"],
    "loss": ["loss", "pain", "tears", "grief"],
    "street": ["streets", "block", "game", "war"],
    "fame": ["fame", "lights", "spotlight", "attention"],
    "family": ["family", "blood", "kin", "people"],
    "hustle": ["hustle", "grind", "work", "mission"],
    "faith": ["faith", "prayers", "light", "grace"],
    "demons": ["demons", "voices", "ghosts", "shadows"],
    "anxiety": ["anxiety", "pressure", "noise", "fear"],
    "confidence": ["confidence", "crown", "word", "throne"],
    "violence": ["violence", "war", "smoke", "danger"],
    "ambition": ["ambition", "vision", "dream", "goal"],
    "doubt": ["doubt", "fear", "noise", "hate"],
    "envy": ["envy", "hate", "jealousy", "snakes"],
    "greed": ["greed", "lust", "envy", "snakes"],
    "money": ["money", "racks", "cash", "bags"],
    "grind": ["grind", "hustle", "work", "mission"],
    "respect": ["respect", "props", "love", "credit"],
    "come-up": ["come-up", "moment", "break", "shot"],
    "party": ["party", "club", "vibe", "motion"],
    "rage": ["rage", "anger", "fire", "fury"],
    "freedom": ["freedom", "light", "peace", "dreams"],
    "boss": ["boss", "show", "game", "block"],
}


def _topic_fill(topic: str, rng: random.Random) -> List[str]:
    """Nouns to drop into a topic lead line's {R} slots (coherent openers)."""
    t = (topic or "").lower()
    words: List[str] = []
    for key, nouns in _TOPIC_NOUNS.items():
        if key in t:
            words.extend(nouns)
    if not words:
        words = ["pain", "grind", "hustle", "rain"]
    rng.shuffle(words)
    return words


def _generate_section(
    artist: Artist,
    artist2: Optional[Artist],
    ctx: BeatContext,
    spec: SectionSpec,
    rng: random.Random,
    seen_openers: set,
) -> dict:
    role = spec.role
    bars = spec.bars or _BARS_BY_ROLE.get(role, 8)
    mood = spec.mood or ctx.mood
    topic = spec.topic or ctx.topic
    flow = spec.flow or _flow_for(artist, ctx)
    cadence = _cadence_for(artist, ctx)
    energy = spec.energy if spec.energy else ctx.energy
    density = _density_for(artist, ctx, energy=energy)
    target = syllable_target(ctx.bpm, cadence, density)  # feature #1 flow budget

    # hook/verse contrast (feature #4): hooks short + punchy, verses denser
    if role in ("hook", "chorus"):
        target = max(6, target - 2)
    elif role == "verse":
        target = min(16, target + 1)

    # rhyme-scheme enforcement (feature #1): map each line to a rhyme group
    scheme = (artist.rhyme_scheme or "AABB").lower()

    def _group(i: int) -> int:
        if scheme == "aabb":
            return i // 2
        if scheme in ("abab", "internal"):
            return i % 2
        return 0  # "aaaa" and anything unrecognized

    group_banks: Dict[int, str] = {}
    group_words: Dict[int, List[str]] = {}
    group_usage: Dict[int, int] = {}

    def _bank_for(g: int) -> tuple:
        if g not in group_banks:
            bname = _pick_rhyme_bank(rng)
            # copy before shuffling — shuffling the module-level bank in place
            # would corrupt it and break seed determinism across calls
            bank = list(_RHYME_BANKS[bname])
            rng.shuffle(bank)
            group_banks[g] = bname
            group_words[g] = bank
        return group_banks[g], group_words[g]

    lines: List[str] = []
    line_rhymes: List[str] = []
    line_artists: List[str] = []
    ad_libs: List[str] = []
    prefix = _MOOD_PREFIX.get(mood.lower(), "")
    topic_leads = _topic_leads(topic, mood, rng)
    rng.shuffle(topic_leads)

    n = max(1, int(round(bars)))
    recent_templates: List[str] = []
    for i in range(n):
        # duet mode (#5): alternate artists per bar
        who = artist if (artist2 is None or i % 2 == 0) else artist2
        g = _group(i)
        bname, words = _bank_for(g)
        gi = group_usage.get(g, 0)
        group_usage[g] = gi + 1
        rw = words[gi % len(words)]
        # skip one so two-slot lines ("the {R} ... the {R}") don't bleed the
        # next line's end word onto this line's tail ("...the keep" twice)
        nxt = words[(gi + 2) % len(words)]
        pair = [rw, nxt]

        if role in ("hook", "chorus") and i == 0 and topic_leads:
            tmpl = topic_leads[0]
            lead = _render_line(tmpl, _topic_fill(topic, rng))
            line = _join_prefix(prefix, lead)
        elif i == 0 and topic_leads:
            tmpl = topic_leads[i % len(topic_leads)]
            line = _join_prefix(prefix, _render_line(tmpl, _topic_fill(topic, rng)))
        else:
            tmpl = _pick_template_near(target, rng, recent=recent_templates)
            line = _render_line(tmpl, pair)
            recent_templates.append(tmpl)
            if len(recent_templates) > 3:
                recent_templates.pop(0)

        # signature opener per rapper (duet-aware)
        if who.name not in seen_openers and who.signature_openers:
            opener = rng.choice(who.signature_openers)
            if not line.startswith(opener):
                line = f"{opener} {line}"
            seen_openers.add(who.name)

        lines.append(_ucfirst(line))
        line_rhymes.append(bname)
        line_artists.append(who.name)

        # ad-lib sprinkling per line (duet-aware)
        ad = _sprinkle_one(who, mood, rng)
        if ad:
            ad_libs.append(ad)

    duet = artist2 is not None
    who_label = f"{artist.name} & {artist2.name}" if duet else artist.name
    return {
        "role": role,
        "bars": n,
        "flow": flow,
        "cadence": cadence,
        "density": density,
        "syllable_target": target,
        "energy": round(energy, 3),
        "artist": who_label,
        "duet": duet,
        "mood": mood,
        "topic": topic,
        "rhyme": scheme,
        "lines": lines,
        "line_rhymes": line_rhymes,
        "line_artists": line_artists,
        "ad_libs": ad_libs,
    }


def _sprinkle_one(artist: Artist, mood: str, rng: random.Random) -> str:
    pool = list(artist.ad_libs or ["yeah"])
    mood_extra = _MOOD_ADLIB.get(mood.lower())
    if mood_extra and mood_extra not in pool:
        pool.append(mood_extra)
    if rng.random() < 0.25:
        return rng.choice(pool)
    return ""


def _banned(line: str, negatives: List[str]) -> bool:
    low = line.lower()
    return any(w.strip().lower() in low for w in negatives if w.strip())


def _cjk_ratio(s: str) -> float:
    if not s:
        return 0.0
    return sum(0x4E00 <= ord(ch) <= 0x9FFF for ch in s) / len(s)


def quality_issues(result: LyricsResult) -> List[str]:
    """Quality-gate report for a generated result.

    Flags the failure modes the fine-tuned local model actually exhibited:
    Chinese/multilingual leakage, header-less single-blob output, too-short or
    empty lines, missing sections, and heavy line duplication. An empty list
    means the result is fit to show; ``generate`` uses this to retry LLM/local
    backends on a different seed before falling back to the offline engine.
    """
    issues: List[str] = []
    if not result.sections:
        return ["no sections parsed — output was unusable"]
    if len(result.sections) < 2:
        issues.append(f"only {len(result.sections)} section(s) parsed (expected several)")

    all_lines: List[str] = []
    total = 0
    for s in result.sections:
        lines = s.get("lines") or []
        role = s.get("role", "section")
        if not lines:
            issues.append(f"[{role}] is empty")
        for ln in lines:
            ln = (ln or "").strip()
            if not ln:
                continue
            total += 1
            all_lines.append(ln)
            if _cjk_ratio(ln) > 0.2 or "理解为" in ln or "翻译" in ln:
                issues.append(f"[{role}] multilingual/translation leak: {ln[:48]}")
            elif len(ln) < 3:
                issues.append(f"[{role}] too-short line: {ln!r}")

    if total < 4:
        issues.append(f"only {total} lyric line(s) generated")
    if len(all_lines) > 4:
        dups = len(all_lines) - len(set(all_lines))
        if dups / len(all_lines) > 0.4:
            issues.append(f"{dups}/{len(all_lines)} lines are exact duplicates")
    return issues


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class LyricsResult:
    artist: str
    bpm: float
    key: str
    swing: str
    mood: str
    topic: str
    seed: int
    sections: List[dict] = field(default_factory=list)
    backend: str = "offline"
    gate_issues: List[str] = field(default_factory=list)   # quality-gate report

    def as_dict(self) -> dict:
        return {
            "artist": self.artist,
            "bpm": self.bpm,
            "key": self.key,
            "swing": self.swing,
            "mood": self.mood,
            "topic": self.topic,
            "seed": self.seed,
            "backend": self.backend,
            "gate_issues": self.gate_issues,
            "sections": self.sections,
        }

    def full_text(self) -> str:
        out: List[str] = []
        for s in self.sections:
            who = s.get("artist") or self.artist
            out.append(f"[{s['role']} — {s['flow']} @ {s['cadence']} cadence — {who}]")
            out.extend(s["lines"])
            if s["ad_libs"]:
                out.append("  " + ", ".join(f"({a})" for a in s["ad_libs"]))
            out.append("")
        return "\n".join(out).strip()

    def annotated(self) -> dict:
        """Per-line syllable + rhyme annotations (feature #2)."""
        from .lyrictools import annotate

        return annotate(self)

    def to_sheet(self) -> str:
        """Markdown studio sheet with bar counts and ad-lib cues (feature #9)."""
        from .lyrictools import sheet

        return sheet(self)


def generate(ctx: BeatContext) -> LyricsResult:
    """Generate full lyrics for a beat context (offline or LLM-backed).

    Quality gate: LLM/local output that fails :func:`quality_issues` is retried
    on a different seed (API once, local up to 3 seeds) before falling back to
    the offline engine. The final result always carries ``gate_issues`` so the
    CLI/dashboard can surface any residual warnings.
    """
    artist = ctx.artist_obj()
    if artist is None:
        artist = get_artist("future")  # type: ignore[assignment]

    if os.environ.get("MUSICTRAIN_LLM_API_KEY"):
        llm = _generate_llm(ctx, artist)
        if llm is not None:
            issues = quality_issues(llm)
            if not issues:
                return llm
            log.warning("API output failed quality gate, retrying once: %s", "; ".join(issues))
            llm = _generate_llm(ctx, artist, seed=ctx.seed + 1)
            if llm is not None:
                issues = quality_issues(llm)
                if not issues:
                    return llm
            log.warning("API output still failing quality gate: %s", "; ".join(issues))

    local_path = os.environ.get("MUSICTRAIN_LLM_MODEL_PATH")
    if local_path:
        for attempt in range(3):
            llm = _generate_local(ctx, artist, local_path, seed=ctx.seed + attempt)
            if llm is None:
                continue
            issues = quality_issues(llm)
            if not issues:
                return llm
            log.warning("local output failed quality gate (attempt %d): %s",
                        attempt + 1, "; ".join(issues))

    result = _generate_offline(ctx, artist)
    result.gate_issues = quality_issues(result)
    return result


def _generate_offline(ctx: BeatContext, artist: Artist) -> LyricsResult:
    """The template + rhyme-bank generator (deterministic for a given seed)."""
    rng = random.Random(ctx.seed)
    structure = ctx.structure or list(_DEFAULT_STRUCTURE)
    sections = []
    seen_openers: set = set()  # signature opener once per song, not per section
    for s in structure:
        # multi-artist feature mode (#6): a section can override the artist
        sec_artist = artist
        if s.artist:
            a = get_artist(s.artist)
            if a is not None:
                sec_artist = a
        # duet mode (#5): a second artist alternates bars within the section
        sec_artist2: Optional[Artist] = None
        if s.artist2:
            a2 = get_artist(s.artist2)
            if a2 is not None:
                sec_artist2 = a2
        sections.append(_generate_section(sec_artist, sec_artist2, ctx, s, rng, seen_openers))

    # filter negative terms (feature #29): drop banned words from lines
    if ctx.negative:
        for sec in sections:
            sec["lines"] = [ln for ln in sec["lines"] if not _banned(ln, ctx.negative)]

    return LyricsResult(
        artist=artist.name,
        bpm=ctx.bpm,
        key=ctx.key,
        swing=ctx.swing,
        mood=ctx.mood,
        topic=ctx.topic,
        seed=ctx.seed,
        sections=sections,
        backend="offline",
    )


def default_structure() -> List[SectionSpec]:
    """A fresh copy of the default section layout (safe to mutate)."""
    return [SectionSpec(role=s.role, bars=s.bars) for s in _DEFAULT_STRUCTURE]


def arrangement_presets() -> Dict[str, List[SectionSpec]]:
    """Named section layouts (feature #5)."""
    return {name: arrangement_specs(name) for name in ARRANGEMENTS}


def arrangement_specs(name: str) -> List[SectionSpec]:
    arr = ARRANGEMENTS.get((name or "").strip().lower())
    if not arr:
        return []
    return [SectionSpec(role=r, bars=b) for r, b in arr]


def regenerate_section(ctx: BeatContext, role: str, seed: Optional[int] = None) -> dict:
    """Re-generate a single section, keeping the rest of the context (feature #40)."""
    artist = ctx.artist_obj()
    if artist is None:
        artist = get_artist("future")  # type: ignore[assignment]
    rng = random.Random(seed if seed is not None else ctx.seed)
    spec = SectionSpec(role=role, bars=_BARS_BY_ROLE.get(role, 8))
    return _generate_section(artist, None, ctx, spec, rng, set())


def restyle(prev: LyricsResult, new_artist: str, seed: Optional[int] = None) -> LyricsResult:
    """Re-render existing lyrics in a different artist's style (feature #41).

    Feature-pinned sections (an explicit per-section artist that differs from the
    base artist) are preserved; everything else is re-rendered in ``new_artist``.
    """
    base_name = prev.artist
    structure = []
    for s in prev.sections:
        feat = (s.get("artist") or "") if (s.get("artist") or "") != base_name else ""
        # a duet section's label is "X & Y" and won't resolve — drop it on restyle
        if "&" in feat:
            feat = ""
        structure.append(SectionSpec(
            role=s["role"], bars=s["bars"], artist=feat,
            energy=float(s.get("energy", 0.0)),
        ))
    ctx = BeatContext(
        bpm=prev.bpm,
        key=prev.key,
        swing=prev.swing,
        mood=prev.mood,
        topic=prev.topic,
        artist=new_artist,
        seed=seed if seed is not None else prev.seed,
        structure=structure,
    )
    return generate(ctx)


# --------------------------------------------------------------------------- #
# Optional LLM backend (OpenAI-compatible chat completions).
# --------------------------------------------------------------------------- #
def _llm_prompt(ctx: BeatContext, artist: Artist) -> str:
    structure = ctx.structure or list(_DEFAULT_STRUCTURE)
    secs = ", ".join(f"{s.role}({s.bars} bars)" for s in structure)
    features = ", ".join(f"{s.role} in the style of {s.artist}" for s in structure if s.artist)
    return (
        f"You are a ghostwriter in the style of {artist.name} ({artist.vibe()}).\n"
        f"Beat analysis: {ctx.bpm} BPM, key {ctx.key}, swing feel {ctx.swing}, "
        f"energy {ctx.energy:.2f}.\n"
        f"Structure: {secs}.\n"
        f"Mood: {ctx.mood}. Topic: {ctx.topic}.\n"
        f"Signature ad-libs: {', '.join(artist.ad_libs) or 'none'}.\n"
        f"Slang: {', '.join(artist.slang) or 'none'}.\n"
        f"Write original lyrics ONLY, in English. Never translate, explain, or comment.\n"
        f"Start each section with exactly [intro]/[verse]/[hook]/[chorus]/[bridge]/[outro]\n"
        f"and put that section's lines below the header.\n"
        f"Match the {artist.cadence} cadence and {artist.rhyme_scheme} rhyme scheme.\n"
        + (f"Feature sections: {features}.\n" if features else "")
        + (f"Avoid these words: {', '.join(ctx.negative)}.\n" if ctx.negative else "")
    )


def _parse_llm(text: str, ctx: BeatContext) -> Optional[LyricsResult]:
    import re

    # matches "[intro]", "intro:", "intro", "verse 2" — bare section headers
    # that local instruct models emit without the brackets
    role_re = re.compile(r"^\[?([A-Za-z-]+)(?:\s+\d+)?\]?:?\s*(.*)$")
    roles = {"intro", "verse", "hook", "chorus", "bridge", "outro", "pre-chorus"}

    sections: List[dict] = []
    current: Optional[dict] = None
    plain_lines: List[str] = []  # header-less lyrics (fallback section)
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # skip multilingual-model notes/translations (e.g. Qwen "可以理解为")
        if _cjk_ratio(line) > 0.2 or "理解为" in line or "翻译" in line:
            continue
        plain_lines.append(line)
        m = role_re.match(line)
        if m:
            role = m.group(1).lower()
            if role in roles:
                if current is not None:
                    sections.append(current)
                current = {"role": role, "bars": 0,
                           "lines": [], "ad_libs": [], "flow": "melodic", "cadence": "medium"}
                rest = m.group(2).strip()
                if rest:
                    current["lines"].append(rest)
                continue
            if role == "section":  # "[SECTION]" placeholder header — skip
                continue
        if current is not None:
            current["lines"].append(line)
    if current is not None:
        sections.append(current)
    if not sections:
        # header-less output still counts — a local model that ignores the
        # format instruction shouldn't silently fall back to the template
        if len(plain_lines) >= 2:
            sections = [{"role": "verse", "bars": len(plain_lines),
                         "lines": plain_lines, "ad_libs": [],
                         "flow": "melodic", "cadence": "medium"}]
        else:
            return None
    for s in sections:
        s["bars"] = len(s["lines"])
    return LyricsResult(
        artist=ctx.artist_obj().name if ctx.artist_obj() else ctx.artist,
        bpm=ctx.bpm, key=ctx.key, swing=ctx.swing,
        mood=ctx.mood, topic=ctx.topic, seed=ctx.seed,
        sections=sections, backend="llm",
    )


_local_model_cache: Dict[str, tuple] = {}  # path -> (model, tokenizer, device)


def _generate_local(ctx: BeatContext, artist: Artist, path: str,
                    seed: Optional[int] = None) -> Optional[LyricsResult]:
    """Generate with a locally fine-tuned model (a ``train-lyrics`` adapter dir).

    Reads ``metadata.json`` inside the adapter dir for the base model name;
    loads it + the LoRA adapter lazily (cached per path), builds the same
    prompt as the hosted backend, and parses the reply with ``_parse_llm``.
    One (seed-controlled) generation attempt; the caller drives the retry
    loop. Any failure returns None so the caller falls back to offline.
    """
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return None

    key = str(path)
    try:
        if key not in _local_model_cache:
            # resolve to an absolute path — huggingface_hub rejects relative
            # repo-style paths like "checkpoints/lyrics/<run>"
            p = Path(path).resolve()
            if not p.exists() or not p.is_dir():
                log.warning("local model path not found, falling back to offline: %s", path)
                return None
            meta: Dict[str, str] = {}
            meta_path = p / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            base_name = meta.get("base_model") or ""
            tok = AutoTokenizer.from_pretrained(str(p))
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            device = ("cuda" if torch.cuda.is_available()
                      else ("mps" if getattr(torch.backends, "mps", None)
                            and torch.backends.mps.is_available() else "cpu"))
            if base_name:
                base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float32)
                model = PeftModel.from_pretrained(base, str(p))
            else:
                model = AutoModelForCausalLM.from_pretrained(str(p), torch_dtype=torch.float32)
            model.eval()
            model.to(device)
            _local_model_cache[key] = (model, tok, device)
        model, tok, device = _local_model_cache[key]

        msgs = [
            {"role": "system", "content": "You write original rap lyrics in the requested style."},
            {"role": "user", "content": _llm_prompt(ctx, artist)},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(device)
        torch.manual_seed(seed if seed is not None else ctx.seed)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=700,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        result = _parse_llm(text, ctx)
        if result is not None:
            result.backend = "local"
        return result
    except Exception:  # noqa: BLE001 - local backend is best-effort
        log.warning("local model generation failed, falling back to offline: %s", path)
        return None


def _generate_llm(ctx: BeatContext, artist: Artist,
                  seed: Optional[int] = None) -> Optional[LyricsResult]:
    key = os.environ.get("MUSICTRAIN_LLM_API_KEY")
    base = os.environ.get("MUSICTRAIN_LLM_BASE_URL") or "https://api.openai.com/v1"
    model = os.environ.get("MUSICTRAIN_LLM_MODEL") or "gpt-4o-mini"
    url = base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You write original rap lyrics in the requested style."},
            {"role": "user", "content": _llm_prompt(ctx, artist)},
        ],
        "temperature": 0.9,
        "seed": seed if seed is not None else ctx.seed,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return _parse_llm(content, ctx)
    except Exception:  # noqa: BLE001
        return None
