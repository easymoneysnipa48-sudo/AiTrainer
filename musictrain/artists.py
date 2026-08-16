"""Rapper style profiles, genre templates, and the mood catalog.

Single source of truth for the lyric engine. Each :class:`Artist` captures the
*dimensional* qualities a lyric generator needs — flow, rhyme scheme, cadence,
signature ad-libs, slang, and topic pools — so "write like Future" actually
changes the output instead of just swapping a name.

The 22 artists here are the ones the user listens to. Extend ``ARTISTS`` with a
new ``Artist(...)`` and the engine, CLI, and dashboard pick it up automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Artist:
    """A rapper's lyrical DNA."""

    id: str                          # slug, e.g. "lil-durk"
    name: str                        # display name
    aliases: tuple = ()              # e.g. ("Drizzy",)
    flow: tuple = ("melodic",)       # flow descriptors used by the engine
    rhyme_scheme: str = "AABB"       # "AABB" | "ABAB" | "AAAA" | "internal"
    cadence: str = "medium"          # slow | medium | fast (syllables-per-bar feel)
    density: int = 3                 # 1 (sparse) .. 5 (rapid, dense)
    autotune: bool = False           # heavy pitch-correction / melodic delivery
    energy: int = 3                  # 1 (mellow) .. 5 (aggressive)
    bpm_range: tuple = (70, 160)     # preferred tempo window
    ad_libs: tuple = ()              # signature vocal ad-libs
    slang: tuple = ()                # signature words / phrases
    topics: tuple = ()               # recurring themes
    delivery: str = ""               # prose description of the delivery
    signature_openers: tuple = ()    # tags that often open a verse

    def vibe(self) -> str:
        return self.delivery or f"{self.cadence} cadence, {'autotuned ' if self.autotune else ''}{'/'.join(self.flow)} flow"


# --------------------------------------------------------------------------- #
# The 22 artists the user listens to.
# --------------------------------------------------------------------------- #
ARTISTS: List[Artist] = [
    Artist(
        id="drake", name="Drake", aliases=("Drizzy", "6 God", "Champagne Papi"),
        flow=("melodic", "half-sung", "conversational"), rhyme_scheme="AABB",
        cadence="medium", density=2, autotune=True, energy=2, bpm_range=(70, 110),
        ad_libs=("yeah", "woo", "6", "know that"),
        slang=("started from the bottom", "no new friends", "the 6", "OVO", "runnin' through the 6"),
        topics=("heartbreak", "loyalty", "success", "late nights", "fame", "relationships", "the city"),
        delivery="melancholic, melodic, half-sung confessionals over soft keys",
        signature_openers=("Yeah,", "Look,", "6 side,",),
    ),
    Artist(
        id="omb-peezy", name="OMB Peezy", aliases=("Peezy",),
        flow=("laid-back", "melodic", "street"), rhyme_scheme="AABB",
        cadence="medium", density=2, autotune=True, energy=3, bpm_range=(80, 100),
        ad_libs=("yeah", "uh-huh", "on God"),
        slang=("loyalty over love", "in these streets", "came up", "on my own", "trenches"),
        topics=("struggle", "loyalty", "pain", "street life", "survival", "family"),
        delivery="soulful, strained, melodic street pain — Alabama to the West Coast",
        signature_openers=("Man,", "On God,",),
    ),
    Artist(
        id="lil-durk", name="Lil Durk", aliases=("Smurk", "The Voice"),
        flow=("melodic", "auto-tuned", "storytelling"), rhyme_scheme="AABB",
        cadence="medium", density=3, autotune=True, energy=3, bpm_range=(80, 160),
        ad_libs=("yeah", "no cap", "the voice"),
        slang=("the voice", "no auto", "smurk", "OTF", "back to back"),
        topics=("pain", "loyalty", "loss", "street life", "brotherhood", "survival", "heartbreak"),
        delivery="grief-stricken melody, sung pain over 808s — 'the voice of the streets'",
        signature_openers=("Yeah, yeah,", "Smurk,",),
    ),
    Artist(
        id="chief-keef", name="Chief Keef", aliases=("Sosa",),
        flow=("mumbled", "deadpan", "drill"), rhyme_scheme="AAAA",
        cadence="slow", density=2, autotune=False, energy=4, bpm_range=(130, 160),
        ad_libs=("bang bang", "sosa", "ay", "beep"),
        slang=("bang bang", "sosa", "glory boys", "3Hunna", "GBE", "finally rich"),
        topics=("drill", "street life", "wealth", "loyalty", "violence", "the block"),
        delivery="cold, deadpan drill delivery over sparse, hard 808s",
        signature_openers=("Sosa,", "Bang,",),
    ),
    Artist(
        id="meek-mill", name="Meek Mill", aliases=("Meek",),
        flow=("double-time", "aggressive", "battle"), rhyme_scheme="internal",
        cadence="fast", density=5, autotune=False, energy=5, bpm_range=(120, 160),
        ad_libs=("woo", "yeah", "huh", "get it"),
        slang=("dreamchasers", "philly", "jump out", "chase the money", "level up"),
        topics=("struggle", "hustle", "justice", "ambition", "street life", "overcoming", "wealth"),
        delivery="ferocious, breathless double-time bars with internal rhymes",
        signature_openers=("Woo,", "Yeah,",),
    ),
    Artist(
        id="kendrick-lamar", name="Kendrick Lamar", aliases=("K.Dot", "Kung Fu Kenny"),
        flow=("multi-syllabic", "storytelling", "shape-shifting"), rhyme_scheme="internal",
        cadence="medium", density=4, autotune=False, energy=4, bpm_range=(70, 140),
        ad_libs=("huh", "yeah", "top of the morning"),
        slang=("compton", "top dawg", "hiii power", "good kid", "humble"),
        topics=("identity", "faith", "community", "trauma", "systemic injustice", "self-worth", "legacy"),
        delivery="dense, theatrical, multi-syllabic storytelling with shifting voices",
        signature_openers=("I remember,", "Huh,",),
    ),
    Artist(
        id="jackboy", name="Jackboy", aliases=(),
        flow=("melodic", "street", "confessional"), rhyme_scheme="AABB",
        cadence="medium", density=2, autotune=True, energy=3, bpm_range=(80, 100),
        ad_libs=("yeah", "huh"),
        slang=("in these streets", "no love", "on my own", "pain"),
        topics=("pain", "street life", "loyalty", "betrayal", "survival"),
        delivery="soulful, wounded street crooning",
        signature_openers=("Yeah,",),
    ),
    Artist(
        id="quavo", name="Quavo", aliases=("Huncho",),
        flow=("triplet", "melodic", "ad-lib-heavy"), rhyme_scheme="AABB",
        cadence="medium", density=3, autotune=True, energy=3, bpm_range=(120, 160),
        ad_libs=("huncho", "mama", "yeah", "ooh", "woo"),
        slang=("huncho", "migo", "culture", "taking off", "the nawf"),
        topics=("wealth", "success", "the culture", "loyalty", "the trap", "family"),
        delivery="bouncy triplet flow, catchy melodic hooks, call-and-response ad-libs",
        signature_openers=("Huncho,", "Mama,",),
    ),
    Artist(
        id="gunna", name="Gunna", aliases=("Wunna", "Gunna Wunna"),
        flow=("melodic", "laid-back", "luxurious"), rhyme_scheme="AABB",
        cadence="medium", density=2, autotune=True, energy=2, bpm_range=(120, 160),
        ad_libs=("slatt", "yeah", "hmm"),
        slang=("slatt", "wunna", "drip", "no pressure", "at the top"),
        topics=("luxury", "drip", "success", "wealth", "lifestyle", "fashion"),
        delivery="smooth, unhurried melodic flow over lush, luxurious production",
        signature_openers=("Yeah,", "Slatt,",),
    ),
    Artist(
        id="offset", name="Offset", aliases=("Set",),
        flow=("triplet", "aggressive", "melodic"), rhyme_scheme="internal",
        cadence="fast", density=4, autotune=True, energy=4, bpm_range=(120, 160),
        ad_libs=("offset", "hey", "woo", "yeah"),
        slang=("migo", "taking off", "set", "the nawf", "no cap"),
        topics=("wealth", "success", "fashion", "family", "the trap", "loyalty"),
        delivery="rapid triplet staccato, aggressive yet melodic",
        signature_openers=("Hey,", "Offset,",),
    ),
    Artist(
        id="takeoff", name="Takeoff", aliases=("Take",),
        flow=("rapid", "double-time", "technical"), rhyme_scheme="internal",
        cadence="fast", density=5, autotune=False, energy=4, bpm_range=(130, 160),
        ad_libs=("yeah", "hey"),
        slang=("taking off", "the nawf", "migo", "no cap"),
        topics=("wealth", "success", "the trap", "family", "loyalty"),
        delivery="relentless, technical double-time — the Migos' quiet technician",
        signature_openers=("Yeah,",),
    ),
    Artist(
        id="juice-wrld", name="Juice WRLD", aliases=("Juice",),
        flow=("melodic", "emo", "freestyled"), rhyme_scheme="AABB",
        cadence="medium", density=3, autotune=True, energy=3, bpm_range=(140, 170),
        ad_libs=("yeah", "ay", "huh"),
        slang=("all girls are the same", "lucid dreams", "999", "sad boys", "demons"),
        topics=("heartbreak", "anxiety", "demons", "addiction", "love", "loss", "inner demons"),
        delivery="wounded emo melody, stream-of-consciousness confessions",
        signature_openers=("Yeah,", "I been,",),
    ),
    Artist(
        id="future", name="Future", aliases=("Pluto", "Future Hendrix", "Fewtch"),
        flow=("mumbled", "melodic", "drug-inflected"), rhyme_scheme="AABB",
        cadence="medium", density=3, autotune=True, energy=3, bpm_range=(120, 160),
        ad_libs=("yeah", "ay", "if young metro don't trust you", "woo"),
        slang=("pluto", "percocet", "codeine", "astronaut", "toxic", "wizard", "beast mode"),
        topics=("wealth", "drugs", "toxic love", "success", "the trap", "excess"),
        delivery="slurred, auto-tuned melodic mumble — druggy, toxic, hypnotic",
        signature_openers=("Yeah, yeah,", "Pluto,",),
    ),
    Artist(
        id="lil-baby", name="Lil Baby", aliases=("Baby",),
        flow=("melodic", "rapid", "street"), rhyme_scheme="AABB",
        cadence="fast", density=4, autotune=True, energy=4, bpm_range=(80, 160),
        ad_libs=("yeah", "ay", "uh"),
        slang=("4PF", "drip too hard", "the bigger picture", "street cred", "no cap"),
        topics=("hustle", "struggle", "success", "street life", "wealth", "loyalty", "growth"),
        delivery="urgent, rapid-fire melodic street bars with relentless momentum",
        signature_openers=("Yeah,", "Ay,",),
    ),
    Artist(
        id="jay-z", name="Jay-Z", aliases=("Hov", "Hova", "Jigga"),
        flow=("laid-back", "double-entendre", "luxurious"), rhyme_scheme="internal",
        cadence="medium", density=4, autotune=False, energy=3, bpm_range=(70, 110),
        ad_libs=("uh", "ha", "can't knock the hustle"),
        slang=("hova", "roc", "marcy", "blueprint", "reasonable doubt", "empire state"),
        topics=("wealth", "legacy", "the hustle", "business", "power", "success", "wisdom"),
        delivery="cool, conversational, double-entendre-laden — business-class bars",
        signature_openers=("Hov,", "Uh,",),
    ),
    Artist(
        id="kanye-west", name="Kanye West", aliases=("Ye", "Yeezy"),
        flow=("chopped", "confessional", "experimental"), rhyme_scheme="AABB",
        cadence="medium", density=3, autotune=False, energy=3, bpm_range=(70, 140),
        ad_libs=("huh", "yeah", "we in this"),
        slang=("yeezy", "the life of pablo", "ultralight beam", "college dropout", "genius"),
        topics=("faith", "fame", "ego", "family", "mental health", "art", "ambition"),
        delivery="raw, confessional, unpredictable — bars between vulnerability and bravado",
        signature_openers=("Uh,", "I been thinkin',",),
    ),
    Artist(
        id="michael-jackson", name="Michael Jackson", aliases=("MJ", "King of Pop"),
        flow=("sung", "rhythmic", "percussive"), rhyme_scheme="AABB",
        cadence="medium", density=2, autotune=False, energy=3, bpm_range=(90, 130),
        ad_libs=("hee hee", "shamone", "ow", "hoo"),
        slang=("moonwalk", "neverland", "heal the world", "billie jean", "king of pop"),
        topics=("love", "unity", "healing", "humanity", "fame", "isolation", "hope"),
        delivery="percussive, rhythmic sung delivery with iconic vocal stabs",
        signature_openers=("Ooh,", "Hee hee,",),
    ),
    Artist(
        id="nocap", name="NoCap", aliases=("Cap",),
        flow=("melodic", "storytelling", "pain"), rhyme_scheme="AABB",
        cadence="medium", density=3, autotune=True, energy=3, bpm_range=(80, 100),
        ad_libs=("yeah", "uh"),
        slang=("in these streets", "no cap", "pain", "the trenches"),
        topics=("pain", "street life", "loss", "loyalty", "struggle", "survival"),
        delivery="melodic, introspective street storytelling — real pain, no cap",
        signature_openers=("Yeah,",),
    ),
    Artist(
        id="quando-rondo", name="Quando Rondo", aliases=("Quando",),
        flow=("melodic", "street", "emotional"), rhyme_scheme="AABB",
        cadence="medium", density=3, autotune=True, energy=3, bpm_range=(80, 100),
        ad_libs=("yeah", "uh-huh"),
        slang=("in these streets", "no love", "pain", "on my own"),
        topics=("pain", "street life", "heartbreak", "loss", "survival", "loyalty"),
        delivery="wounded, melodic street pain — smooth and sorrowful",
        signature_openers=("Yeah,",),
    ),
    Artist(
        id="dababy", name="DaBaby", aliases=("Baby",),
        flow=("staccato", "bouncy", "charismatic"), rhyme_scheme="AABB",
        cadence="fast", density=4, autotune=False, energy=5, bpm_range=(130, 160),
        ad_libs=("let's go", "yeah", "ha", "okay"),
        slang=("let's go", "suge", "billion dollar baby", "haha", "turn up"),
        topics=("success", "wealth", "confidence", "street life", "hustle", "the block"),
        delivery="bouncy, staccato, high-charisma bars built for the turn-up",
        signature_openers=("Let's go,", "Yeah,",),
    ),
    Artist(
        id="young-thug", name="Young Thug", aliases=("Thugger", "Slime"),
        flow=("melodic", "experimental", "freeform"), rhyme_scheme="internal",
        cadence="fast", density=4, autotune=True, energy=4, bpm_range=(120, 160),
        ad_libs=("slatt", "skrrt", "yeah", "huh"),
        slang=("slime", "slatt", "yessirski", "spider", "racks", "the jungle"),
        topics=("wealth", "fashion", "loyalty", "the trap", "success", "excess"),
        delivery="freeform, shape-shifting melody — unpredictable cadences and ad-libs",
        signature_openers=("Slatt,", "Yessir,",),
    ),
    Artist(
        id="lil-gotit", name="Lil Gotit", aliases=("Gotit",),
        flow=("melodic", "mumbled", "slime"), rhyme_scheme="AABB",
        cadence="medium", density=3, autotune=True, energy=3, bpm_range=(120, 160),
        ad_libs=("slatt", "yeah", "skrrt"),
        slang=("slatt", "slime", "hood baby", "racks", "the city"),
        topics=("wealth", "street life", "success", "loyalty", "the trap"),
        delivery="smooth slime-trap melody, mumble-adjacent but melodic",
        signature_openers=("Slatt,", "Yeah,",),
    ),
]

_ARTIST_BY_ID: Dict[str, Artist] = {a.id: a for a in ARTISTS}
_ARTIST_BY_NAME: Dict[str, Artist] = {}
for _a in ARTISTS:
    _ARTIST_BY_NAME[_a.name.lower()] = _a
    for _alias in _a.aliases:
        _ARTIST_BY_NAME[_alias.lower()] = _a


# --------------------------------------------------------------------------- #
# Genre templates (feature 23) — preset mood/flow/topic defaults per genre.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GenreTemplate:
    name: str
    moods: tuple = ()
    flows: tuple = ()
    topics: tuple = ()
    default_energy: int = 3
    default_density: int = 3
    autotune: bool = False
    bpm_hint: tuple = (70, 160)
    description: str = ""


GENRES: List[GenreTemplate] = [
    GenreTemplate(
        name="melodic trap", moods=("dark", "emotional", "melancholic"),
        flows=("melodic", "triplet"), topics=("pain", "heartbreak", "success"),
        default_energy=3, default_density=3, autotune=True, bpm_hint=(130, 160),
        description="Auto-tuned melody over rolling 808s — Future, Lil Durk, Gunna territory.",
    ),
    GenreTemplate(
        name="pain music", moods=("grief", "melancholic", "reflective"),
        flows=("melodic", "storytelling"), topics=("loss", "struggle", "loyalty"),
        default_energy=2, default_density=2, autotune=True, bpm_hint=(80, 100),
        description="Wounded street confessions — OMB Peezy, NoCap, Quando Rondo.",
    ),
    GenreTemplate(
        name="drill", moods=("menacing", "gritty", "aggressive"),
        flows=("deadpan", "drill"), topics=("street life", "the block", "violence"),
        default_energy=5, default_density=2, autotune=False, bpm_hint=(130, 160),
        description="Cold, deadpan drill — Chief Keef energy over sparse hard 808s.",
    ),
    GenreTemplate(
        name="trap", moods=("braggadocious", "energetic", "confident"),
        flows=("triplet", "staccato"), topics=("wealth", "success", "the trap"),
        default_energy=4, default_density=4, autotune=True, bpm_hint=(120, 160),
        description="Bouncy trap triplets — Migos, DaBaby, Lil Baby turn-up.",
    ),
    GenreTemplate(
        name="southern trap", moods=("smooth", "luxurious", "confident"),
        flows=("melodic", "laid-back"), topics=("wealth", "drip", "success"),
        default_energy=2, default_density=2, autotune=True, bpm_hint=(120, 160),
        description="Laid-back luxury trap — Gunna, Young Thug slime vibes.",
    ),
    GenreTemplate(
        name="emo rap", moods=("heartbroken", "anxious", "melancholic"),
        flows=("melodic", "emo"), topics=("heartbreak", "demons", "anxiety"),
        default_energy=3, default_density=3, autotune=True, bpm_hint=(140, 170),
        description="Sad, stream-of-consciousness melody — Juice WRLD energy.",
    ),
    GenreTemplate(
        name="boom bap", moods=("reflective", "gritty", "determined"),
        flows=("double-time", "battle"), topics=("hustle", "legacy", "struggle"),
        default_energy=4, default_density=5, autotune=False, bpm_hint=(80, 100),
        description="Lyrical, drum-heavy — Meek Mill, Jay-Z, Kendrick bars.",
    ),
    GenreTemplate(
        name="rnb", moods=("smooth", "seductive", "nostalgic"),
        flows=("melodic", "half-sung"), topics=("love", "relationships", "longing"),
        default_energy=2, default_density=2, autotune=True, bpm_hint=(70, 110),
        description="Smooth melodic R&B — Drake and Michael Jackson-flavored crooning.",
    ),
]


# --------------------------------------------------------------------------- #
# Expanded mood catalog (feature 25).
# --------------------------------------------------------------------------- #
MOODS: List[str] = [
    "dark", "emotional", "determined", "energetic", "calm", "melancholic",
    "uplifting", "aggressive", "reflective", "tense", "epic", "mysterious",
    "nostalgic", "hopeful", "somber", "dreamy", "atmospheric",
    # rap / beat-specific additions
    "braggadocious", "gritty", "confident", "menacing", "heartbroken",
    "anxious", "smooth", "seductive", "introspective", "triumphant",
    "paranoid", "celebratory", "lonely", "grief", "lavish", "hustling",
    "vengeful", "spiritual", "streetwise", "euphoric", "wounded",
    "defiant", "cold", "toxic", "loyal", "motivated", "carefree",
]


def artist_ids() -> List[str]:
    return [a.id for a in ARTISTS]


def artist_names() -> List[str]:
    return [a.name for a in ARTISTS]


def get_artist(key: str) -> Optional[Artist]:
    """Resolve an artist by id, name, or alias (case-insensitive)."""
    k = (key or "").strip().lower()
    return _ARTIST_BY_ID.get(k) or _ARTIST_BY_NAME.get(k)


def genre_names() -> List[str]:
    return [g.name for g in GENRES]


def get_genre(name: str) -> Optional[GenreTemplate]:
    n = (name or "").strip().lower()
    for g in GENRES:
        if g.name.lower() == n:
            return g
    return None
