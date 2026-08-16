"""Typed configuration loaded from YAML with sane defaults."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, get_origin, get_type_hints

import yaml


@dataclass
class NormalizeCfg:
    sample_rate: int = 32000
    channels: int = 1
    codec: str = "pcm_s16le"
    strip_metadata: bool = True
    target_lufs: Optional[float] = None  # e.g. -14.0 -> loudnorm
    extensions: List[str] = field(
        default_factory=lambda: [".wav", ".mp3", ".flac", ".m4a", ".aiff", ".aif"]
    )


@dataclass
class SegmentCfg:
    segment_seconds: float = 30.0
    bar_aligned: bool = True
    beats_per_bar: int = 4
    min_segment_seconds: float = 8.0
    downbeat_aligned: bool = False   # cut on detected downbeats (#21)
    overlap_seconds: float = 0.0     # overlap between consecutive segments (#24)
    fade_seconds: float = 0.0        # fade in/out at cut boundaries (#25)


@dataclass
class FeaturesCfg:
    sr: int = 32000
    hop_length: int = 512
    silence_threshold_db: float = -60.0
    clip_threshold: float = 0.999


@dataclass
class SplitCfg:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42
    mode: str = "copy"  # "copy" | "link"
    stratify: str = ""   # "" | "key" | "bpm" | "genre" | "mood" (#23)
    k_folds: int = 0     # 0 = train/val/test; N = N-fold CV (#22)


@dataclass
class InferenceCfg:
    model_name: str = "facebook/musicgen-small"
    device: str = "mps"
    torch_dtype: str = "float32"
    do_sample: bool = True
    guidance_scale: float = 3.0
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 1.0
    seed: Optional[int] = None
    # -- Phase 5 (#33-#39) --
    preset: str = ""                 # name of a sampling preset (#37)
    presets: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "standard": {"temperature": 1.0, "top_k": 250, "top_p": 1.0, "guidance_scale": 3.0},
            "creative": {"temperature": 1.2, "top_k": 400, "top_p": 0.98, "guidance_scale": 2.0},
            "precise": {"temperature": 0.7, "top_k": 120, "top_p": 0.95, "guidance_scale": 4.5},
        }
    )
    target_seconds: Optional[float] = None  # override max_new_tokens via duration (#39)
    negative_prompt: str = ""              # CLAP-checked "no X" constraints (#33)
    negative_threshold: float = 0.25        # CLAP sim above this -> violation
    negative_retries: int = 0               # auto-regenerate until not violating


@dataclass
class CheckCfg:
    bpm_tolerance: float = 0.05       # 5% deviation tolerated
    max_time_stretch: float = 0.10    # max stretch before rejecting
    beats_per_bar: int = 4
    sr: int = 32000


@dataclass
class MlflowCfg:
    enabled: bool = True
    tracking_uri: str = ""             # empty -> <project_root>/mlruns
    experiment_name: str = "musicgen-style"


@dataclass
class ClapCfg:
    enabled: bool = True
    model_name: str = "laion/clap-htsat-unfused"
    device: str = "auto"               # auto | mps | cpu | cuda


@dataclass
class QualityCfg:
    min_bitrate_kbps: float = 128.0
    min_sample_rate: int = 32000
    max_clipping_ratio: float = 0.001
    max_silence_ratio: float = 0.5
    max_dc_offset: float = 0.01
    min_rolloff_hz: float = 4000.0


@dataclass
class DedupCfg:
    threshold: float = 0.97            # perceptual similarity threshold
    exact_only: bool = False           # content-hash only
    action: str = "report"             # report | move


@dataclass
class AutolabelCfg:
    enabled: bool = True
    top_k: int = 3
    min_confidence: float = 0.15
    device: str = "auto"               # reuse CLAP device


@dataclass
class OodCfg:
    bpm_range: List[float] = field(default_factory=lambda: [70.0, 160.0])
    action: str = "report"             # report | move
    tag_exclude: List[str] = field(default_factory=lambda: ["ambient", "orchestral"])


@dataclass
class StemsCfg:
    model: str = "htdemucs"            # htdemucs | htdemucs_ft | htdemucs_6s
    device: str = "auto"               # auto | mps | cpu | cuda
    two_stems: bool = False            # True -> vocals + accompaniment
    segment_seconds: Optional[float] = None  # split long tracks before separating


@dataclass
class AnalysisCfg:
    sr: int = 32000
    hop_length: int = 512
    chord_frame: float = 0.5           # seconds per chord label
    beats_per_bar: int = 4
    structure_min_segments: int = 2
    structure_max_segments: int = 8
    structure_segment_seconds: float = 10.0
    vocal_enabled: bool = True         # reuse CLAP for vocal/instrumental
    key_top_k: int = 3                 # key candidates to report


@dataclass
class EvalCfg:
    # Phase 6 (#43): auto-reject thresholds applied to eval verdicts.
    min_clap_score: float = 0.0        # mean CLAP below this -> reject
    max_abs_deviation: float = 0.20    # |deviation| above this -> reject
    per_tag_clap: bool = True          # score each tag phrase separately (#46)
    significance_alpha: float = 0.05   # p-value cutoff for #44 verdicts
    # advanced eval #10: per-genre CLAP/deviation gates (fallback: "default").
    genre_gates: dict = field(
        default_factory=lambda: {
            "default": {"min_clap": 0.30, "max_abs_deviation": 0.20},
            "melodic trap": {"min_clap": 0.32, "max_abs_deviation": 0.15},
            "ambient": {"min_clap": 0.22, "max_abs_deviation": 0.30},
            "orchestral": {"min_clap": 0.28, "max_abs_deviation": 0.25},
        }
    )


@dataclass
class MetricsCfg:
    sr: int = 32000
    n_mels: int = 64          # mel bins for spectral KL (#41)
    hop_length: int = 512
    n_fft: int = 1024
    fad_eps: float = 1e-6     # covariance regularization for FAD (#41)
    fad_threshold: float = 10.0  # FAD gate: score above this blocks promotion


@dataclass
class ExportCfg:
    format: str = "arrow"      # arrow | jsonl | csv (#26)
    which: str = "all"         # train | val | test | all
    audio_column: bool = True  # include an audio column (HF Audio feature)
    max_shard_size: str = "500MB"


@dataclass
class Config:
    project_root: Path = field(default_factory=Path.cwd)
    normalize: NormalizeCfg = field(default_factory=NormalizeCfg)
    segment: SegmentCfg = field(default_factory=SegmentCfg)
    features: FeaturesCfg = field(default_factory=FeaturesCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    inference: InferenceCfg = field(default_factory=InferenceCfg)
    check: CheckCfg = field(default_factory=CheckCfg)
    mlflow: MlflowCfg = field(default_factory=MlflowCfg)
    clap: ClapCfg = field(default_factory=ClapCfg)
    quality: QualityCfg = field(default_factory=QualityCfg)
    dedup: DedupCfg = field(default_factory=DedupCfg)
    autolabel: AutolabelCfg = field(default_factory=AutolabelCfg)
    ood: OodCfg = field(default_factory=OodCfg)
    stems: StemsCfg = field(default_factory=StemsCfg)
    analysis: AnalysisCfg = field(default_factory=AnalysisCfg)
    export: ExportCfg = field(default_factory=ExportCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    metrics: MetricsCfg = field(default_factory=MetricsCfg)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        from dataclasses import asdict

        data = asdict(self)
        data["project_root"] = str(self.project_root)
        return data

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Config":
        cfg = cls()
        if not data:
            return cfg
        hints = get_type_hints(cls)
        for f in fields(cls):
            if f.name not in data or data[f.name] is None:
                continue
            cfg_value = _coerce(hints.get(f.name, f.type), data[f.name])
            if f.name == "project_root":
                cfg_value = Path(cfg_value)
            setattr(cfg, f.name, cfg_value)
        return cfg

    @classmethod
    def load(cls, path: Path) -> "Config":
        path = Path(path)
        if not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text()) or {}
        return cls.from_dict(data)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _coerce(ftype: Any, value: Any) -> Any:
    """Coerce a raw dict value into the dataclass type (recursively)."""
    origin = get_origin(ftype)
    if origin is list:
        return list(value) if isinstance(value, (list, tuple)) else []
    if hasattr(ftype, "__dataclass_fields__"):
        if not isinstance(value, dict):
            return ftype()
        known = {f.name for f in fields(ftype)}
        return ftype(**{k: v for k, v in value.items() if k in known})
    return value
