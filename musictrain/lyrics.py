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
from typing import Dict, List, Optional

from .artists import Artist, get_artist

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
        structure.append(SectionSpec(role=role, bars=_BARS_BY_ROLE[role]))

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
    "ain": ["pain", "rain", "chain", "main", "brain", "vain", "campaign", "remain"],
    "ight": ["night", "fight", "light", "right", "sight", "flight", "bright", "tight", "hype"],
    "old": ["gold", "cold", "told", "bold", "hold", "sold", "control", "roll"],
    "ow": ["now", "how", "wow", "pow", "allow", "somehow", "vow"],
    "ay": ["away", "today", "pay", "stay", "play", "betray", "grey", "sway"],
    "ive": ["alive", "survive", "drive", "thrive", "arrive", "deprive", "revive"],
    "eep": ["deep", "keep", "sleep", "weep", "leap", "creep", "reap"],
    "all": ["all", "fall", "call", "ball", "wall", "tall", "stall", "haul"],
    "end": ["end", "send", "bend", "friend", "spend", "trend", "defend", "ascend"],
    "ame": ["name", "fame", "game", "flame", "shame", "claim", "blame"],
    "ack": ["back", "track", "stack", "black", "attack", "rack", "lack", "facts"],
    "eel": ["real", "feel", "steel", "deal", "heal", "seal", "reveal"],
    "ice": ["ice", "price", "twice", "sacrifice", "device", "advice", "slice", "rise"],
    "oad": ["road", "load", "code", "mode", "explode", "abode", "gold"],
    "oney": ["money", "funny", "honey", "sunny", "running", "coming"],
    "ock": ["block", "lock", "rock", "stock", "clock", "shock"],
    "ide": ["ride", "slide", "hide", "inside", "pride", "tide", "divide"],
    "ure": ["pure", "sure", "secure", "cure", "endure", "for sure"],
    "ace": ["face", "race", "place", "chase", "embrace", "replace", "grace", "space"],
    "eam": ["dream", "scheme", "team", "cream", "extreme", "beam", "redeem"],
}

_LINE_TEMPLATES: List[str] = [
    "I been movin' through the {R}, ain't no time to explain",
    "They don't love you till you got the {R}, that's the game",
    "Came from nothin' but I found the {R} in my lane",
    "I could never fold, I stay solid through the {R}",
    "Late nights, chasin' every single {R} I could gain",
    "They was doubtin' me, now they all remember the {R}",
    "I put my heart in this, now it's flowin' through my {R}",
    "Countin' up the {R} while they sleepin' on the {R}",
    "I stay loyal to the {R}, never switch for the {R}",
    "Lost some real ones, still I carry all the {R}",
    "I was down bad, had to grind up out the {R}",
    "Now I'm up, and they can't even see the {R}",
    "Talkin' to the {R}, prayin' that I make it through the {R}",
    "Every scar is a lesson, every {R} is a {R}",
    "I ain't stressin' over {R}, I been through worse than the {R}",
    "Whole city on my back, and I'm carryin' the {R}",
    "They gon' talk, but they never walked a mile in my {R}",
    "I was built for the {R}, I don't know how to quit",
    "Pain in my chest but I still gotta {R}",
    "They left me for dead, now they see me in the {R}",
    "I turned my {R} into a blessing, that's the {R}",
    "No handouts, I took the {R} and I ran with the {R}",
    "Realer than most, I don't fake it for the {R}",
    "Money comin' in, but the {R} still the same",
    "I can't sleep, I keep seein' {R} in my {R}",
    "If you ain't family, you ain't ridin' on the {R}",
    "I remember nights with nothin' but the {R}",
    "Now it's all {R}, everything I touch is {R}",
    "Keep your circle small, 'cause the {R} get you {R}",
    "I'ma make it out, I swear it on the {R}",
]

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


def _density_for(artist: Artist, ctx: BeatContext) -> int:
    d = artist.density
    if ctx.bpm >= 130:
        d += 1
    if ctx.bpm <= 90:
        d = max(1, d - 1)
    if ctx.energy > 0.7:
        d += 1
    return max(1, min(5, d))


# --------------------------------------------------------------------------- #
# Line assembly
# --------------------------------------------------------------------------- #
def _pick_rhyme_bank(rng: random.Random) -> str:
    return rng.choice(list(_RHYME_BANKS.keys()))


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


def _generate_section(
    artist: Artist,
    ctx: BeatContext,
    spec: SectionSpec,
    rng: random.Random,
) -> dict:
    role = spec.role
    bars = spec.bars or _BARS_BY_ROLE.get(role, 8)
    mood = spec.mood or ctx.mood
    topic = spec.topic or ctx.topic
    flow = spec.flow or _flow_for(artist, ctx)
    cadence = _cadence_for(artist, ctx)
    density = _density_for(artist, ctx)

    rhyme = _pick_rhyme_bank(rng)
    # copy before shuffling — shuffling the module-level bank in place would
    # corrupt it and break seed determinism across calls
    bank = list(_RHYME_BANKS[rhyme])
    rng.shuffle(bank)
    rhyme_words = bank[:]

    lines: List[str] = []
    prefix = _MOOD_PREFIX.get(mood.lower(), "")
    topic_leads = _topic_leads(topic, mood, rng)
    rng.shuffle(topic_leads)

    n = max(1, int(round(bars)))
    for i in range(n):
        rw = rhyme_words[i % len(rhyme_words)]
        nxt = rhyme_words[(i + 1) % len(rhyme_words)]
        pair = [rw, nxt]
        if role in ("hook", "chorus") and i == 0 and topic_leads:
            tmpl = topic_leads[0]
            lead = _render_line(tmpl, pair)
            line = prefix + lead if prefix else lead
        elif i == 0 and topic_leads:
            tmpl = topic_leads[i % len(topic_leads)]
            line = _render_line(tmpl, pair)
            if prefix and not line.startswith(prefix):
                # lowercase the first char to join mid-sentence, but keep the
                # pronoun "I" capitalized
                if line[0] == "I":
                    line = prefix + line
                else:
                    line = prefix + line[0].lower() + line[1:]
        else:
            tmpl = rng.choice(_LINE_TEMPLATES)
            line = _render_line(tmpl, pair)
        # apply signature opener to the very first line of the section
        if i == 0 and artist.signature_openers:
            opener = rng.choice(artist.signature_openers)
            if not line.startswith(opener):
                line = f"{opener} {line}"
        lines.append(_ucfirst(line))

    # sprinkle ad-libs across the section
    ad_libs = _sprinkle_adlibs(artist, ctx, n, rng)

    return {
        "role": role,
        "bars": n,
        "flow": flow,
        "cadence": cadence,
        "density": density,
        "mood": mood,
        "topic": topic,
        "rhyme": rhyme,
        "lines": lines,
        "ad_libs": ad_libs,
    }


def _sprinkle_adlibs(artist: Artist, ctx: BeatContext, n: int, rng: random.Random) -> List[str]:
    pool = list(artist.ad_libs or ["yeah"])
    mood_extra = _MOOD_ADLIB.get(ctx.mood.lower())
    if mood_extra and mood_extra not in pool:
        pool.append(mood_extra)
    out: List[str] = []
    for i in range(n):
        if rng.random() < 0.25:
            out.append(rng.choice(pool))
    return out


def _banned(line: str, negatives: List[str]) -> bool:
    low = line.lower()
    return any(w.strip().lower() in low for w in negatives if w.strip())


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
            "sections": self.sections,
        }

    def full_text(self) -> str:
        out: List[str] = []
        for s in self.sections:
            out.append(f"[{s['role']} — {s['flow']} @ {s['cadence']} cadence]")
            out.extend(s["lines"])
            if s["ad_libs"]:
                out.append("  " + ", ".join(f"({a})" for a in s["ad_libs"]))
            out.append("")
        return "\n".join(out).strip()


def generate(ctx: BeatContext) -> LyricsResult:
    """Generate full lyrics for a beat context (offline or LLM-backed)."""
    artist = ctx.artist_obj()
    if artist is None:
        artist = get_artist("future")  # type: ignore[assignment]

    if os.environ.get("MUSICTRAIN_LLM_API_KEY"):
        llm = _generate_llm(ctx, artist)
        if llm is not None:
            return llm

    rng = random.Random(ctx.seed)
    structure = ctx.structure or list(_DEFAULT_STRUCTURE)
    sections = [_generate_section(artist, ctx, s, rng) for s in structure]

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


def regenerate_section(ctx: BeatContext, role: str, seed: Optional[int] = None) -> dict:
    """Re-generate a single section, keeping the rest of the context (feature #40)."""
    artist = ctx.artist_obj()
    if artist is None:
        artist = get_artist("future")  # type: ignore[assignment]
    rng = random.Random(seed if seed is not None else ctx.seed)
    spec = SectionSpec(role=role, bars=_BARS_BY_ROLE.get(role, 8))
    return _generate_section(artist, ctx, spec, rng)


def restyle(prev: LyricsResult, new_artist: str, seed: Optional[int] = None) -> LyricsResult:
    """Re-render existing lyrics in a different artist's style (feature #41)."""
    ctx = BeatContext(
        bpm=prev.bpm,
        key=prev.key,
        swing=prev.swing,
        mood=prev.mood,
        topic=prev.topic,
        artist=new_artist,
        seed=seed if seed is not None else prev.seed,
        structure=[SectionSpec(role=s["role"], bars=s["bars"]) for s in prev.sections],
    )
    return generate(ctx)


# --------------------------------------------------------------------------- #
# Optional LLM backend (OpenAI-compatible chat completions).
# --------------------------------------------------------------------------- #
def _llm_prompt(ctx: BeatContext, artist: Artist) -> str:
    structure = ctx.structure or list(_DEFAULT_STRUCTURE)
    secs = ", ".join(f"{s.role}({s.bars} bars)" for s in structure)
    return (
        f"You are a ghostwriter in the style of {artist.name} ({artist.vibe()}).\n"
        f"Beat analysis: {ctx.bpm} BPM, key {ctx.key}, swing feel {ctx.swing}, "
        f"energy {ctx.energy:.2f}.\n"
        f"Structure: {secs}.\n"
        f"Mood: {ctx.mood}. Topic: {ctx.topic}.\n"
        f"Signature ad-libs: {', '.join(artist.ad_libs) or 'none'}.\n"
        f"Slang: {', '.join(artist.slang) or 'none'}.\n"
        f"Write original lyrics ONLY (no explanations). Format each section as:\n"
        f"[SECTION] (one of intro/verse/hook/chorus/bridge/outro) followed by its lines.\n"
        f"Match the {artist.cadence} cadence and {artist.rhyme_scheme} rhyme scheme.\n"
        + (f"Avoid these words: {', '.join(ctx.negative)}.\n" if ctx.negative else "")
    )


def _parse_llm(text: str, ctx: BeatContext) -> Optional[LyricsResult]:
    import re

    sections: List[dict] = []
    current: Optional[dict] = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = re.match(r"^\[([A-Za-z -]+)\]", line)
        if m:
            role = m.group(1).strip().lower()
            if role == "section":
                # "[SECTION]" header only — skip, wait for the real one
                continue
            if current is not None:
                sections.append(current)
            current = {"role": role, "bars": len(current["lines"]) if current else 0,
                       "lines": [], "ad_libs": [], "flow": "melodic", "cadence": "medium"}
            # the header line may also carry the first lyric after a space
            rest = line[m.end():].strip()
            if rest:
                current["lines"].append(rest)
            continue
        if current is not None:
            current["lines"].append(line)
    if current is not None:
        sections.append(current)
    if not sections:
        return None
    for s in sections:
        s["bars"] = len(s["lines"])
    return LyricsResult(
        artist=ctx.artist_obj().name if ctx.artist_obj() else ctx.artist,
        bpm=ctx.bpm, key=ctx.key, swing=ctx.swing,
        mood=ctx.mood, topic=ctx.topic, seed=ctx.seed,
        sections=sections, backend="llm",
    )


def _generate_llm(ctx: BeatContext, artist: Artist) -> Optional[LyricsResult]:
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
        "seed": ctx.seed,
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
