"""Pretrained model + prompt templates for the musictrain toolkit.

Single source of truth for the model dropdown in the Generate page and the
Settings page's "Pretrained templates" picker. Also re-homes the curated
prompt templates so the dashboard and the CLI share one catalog.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelTemplate:
    """A selectable pretrained MusicGen checkpoint."""

    name: str                       # human-friendly label
    model_id: str                   # HuggingFace repo id
    description: str = ""
    size: str = ""                  # e.g. "~300M params"
    default_guidance: float = 3.0
    melody_capable: bool = False    # supports melody/chroma conditioning
    stereo: bool = False
    presets: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptTemplate:
    """A curated prompt covering a section / energy / BPM angle."""

    name: str
    section: str
    energy: str
    prompt: str


# --------------------------------------------------------------------------- #
# Pretrained models — the "pretrained templates" the user selects from.
# --------------------------------------------------------------------------- #
MODELS: List[ModelTemplate] = [
    ModelTemplate(
        name="MusicGen Small (300M)",
        model_id="facebook/musicgen-small",
        description="Fast default — best for trap/hip-hop sketches on Apple Silicon.",
        size="~300M",
        default_guidance=3.0,
    ),
    ModelTemplate(
        name="MusicGen Medium (1.5B)",
        model_id="facebook/musicgen-medium",
        description="Higher fidelity than small; slower on MPS.",
        size="~1.5B",
        default_guidance=3.0,
    ),
    ModelTemplate(
        name="MusicGen Large (3.3B)",
        model_id="facebook/musicgen-large",
        description="Best quality; needs substantial RAM, not ideal for MPS.",
        size="~3.3B",
        default_guidance=3.0,
    ),
    ModelTemplate(
        name="MusicGen Melody",
        model_id="facebook/musicgen-melody",
        description="Condition generation on a melody / chroma input.",
        size="~1.5B",
        default_guidance=3.0,
        melody_capable=True,
    ),
    ModelTemplate(
        name="MusicGen Stereo Small",
        model_id="facebook/musicgen-stereo-small",
        description="Stereo output at the small-model speed tier.",
        size="~300M",
        default_guidance=3.0,
        stereo=True,
    ),
]

_MODEL_BY_ID: Dict[str, ModelTemplate] = {m.model_id: m for m in MODELS}


def model_ids() -> List[str]:
    """HF repo ids in catalog order (back-compat with the old hardcoded list)."""
    return [m.model_id for m in MODELS]


def get_model(model_id: str) -> Optional[ModelTemplate]:
    return _MODEL_BY_ID.get(model_id)


def find_model_by_name(name: str) -> Optional[ModelTemplate]:
    for m in MODELS:
        if m.name == name:
            return m
    return None


# --------------------------------------------------------------------------- #
# Prompt templates — curated (section/energy/BPM angle).
# --------------------------------------------------------------------------- #
PROMPT_TEMPLATES: List[PromptTemplate] = [
    PromptTemplate("Sparse intro", "intro", "low",
                   "sparse cinematic intro, 70 BPM, A minor, dark piano loop, airy pads, low energy, wide reverb"),
    PromptTemplate("Trap verse", "verse", "mid",
                   "melodic trap verse, 140 BPM, B minor, rolling 808 bass, trap hi-hats, pluck melody, aggressive"),
    PromptTemplate("Melodic chorus", "chorus", "high",
                   "melodic trap chorus, 96 BPM, A minor, dark piano, deep 808 bass, wide strings, powerful drums, emotional"),
    PromptTemplate("Emotional pre-chorus", "pre-chorus", "mid",
                   "emotional pre-chorus, 84 BPM, F minor, warm pads, soft piano, build tension, atmospheric"),
    PromptTemplate("Bridge breakdown", "bridge", "low",
                   "bridge breakdown, 90 BPM, C minor, stripped drums, ambient pads, melancholic, reflective"),
    PromptTemplate("Outro fade", "outro", "low",
                   "outro fade, 72 BPM, A minor, piano and pads fading out, dark, spacious, low energy"),
    PromptTemplate("Orchestral intro", "intro", "low",
                   "orchestral intro, 60 BPM, D minor, strings swells, timpani roll, cinematic, epic"),
    PromptTemplate("Full-song demo", "full-song", "high",
                   "full trap song demo, 140 BPM, E minor, intro verse chorus structure, 808 bass, trap hi-hats, dark piano, emotional"),
]


def prompt_template_names() -> List[str]:
    return [t.name for t in PROMPT_TEMPLATES]


def get_prompt_template(name: str) -> Optional[PromptTemplate]:
    for t in PROMPT_TEMPLATES:
        if t.name == name:
            return t
    return None
