"""Typed configuration loaded from YAML with sane defaults."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, List, Optional, get_origin, get_type_hints

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
    seed: Optional[int] = None


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
