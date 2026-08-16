"""Model-ops helpers (advanced model-ops batch #41-#50).

* **Alias migration (#41)** — map deprecated stages to champion/challenger
  aliases (MLflow 2.9+).
* **A/B routing harness (#42)** — paired win-rate + routing decision between
  a champion and a challenger.
* **Auto-promotion (#43)** — run the eval gate and, on pass, promote to
  Production (CI-friendly).
* **Model lineage (#44)** — record + graph parent/child checkpoint relations.
* **Artifact checksums (#45)** — SHA-256 manifest for registry integrity.
* **Rollback (#46)** — one-click revert to the previous champion.
* **Nightly eval (#47)** — `.github/workflows/nightly-eval.yml` (see repo).
* **Alert webhook (#48)** — Slack/email (``alerts``) + Discord (see alerts.py).
* **Per-run cost attribution (#49)** — per-seed / per-prompt cost breakdown.
* **Config schema linter (#50)** — validate config ranges before any run.

Pure helpers are unit-testable; MLflow-touching helpers degrade gracefully.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from . import console
from .config import Config

# --------------------------------------------------------------------------- #
# #41 alias migration
# --------------------------------------------------------------------------- #
def alias_mapping(versions: Sequence[dict]) -> Dict[int, str]:
    """Map a registered-model version list (stage field) -> alias.

    Production -> "champion", Staging -> "challenger", else no alias.
    """
    out: Dict[int, str] = {}
    for v in versions:
        stage = (v.get("stage") or v.get("current_stage") or "").lower()
        if stage == "production":
            out[int(v["version"])] = "champion"
        elif stage == "staging":
            out[int(v["version"])] = "challenger"
    return out


def migrate_stages_to_aliases(cfg: Config, registry_name: str) -> Dict[int, str]:
    """Set champion/challenger aliases from the current stage assignments."""
    _configure_mlflow(cfg)
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        versions = [
            {"version": v.version, "stage": v.current_stage}
            for v in client.get_latest_versions(registry_name)
        ]
        mapping = alias_mapping(versions)
        for version, alias in mapping.items():
            client.set_registered_model_alias(registry_name, alias, str(version))
            console.ok(f"{registry_name}: v{version} -> alias {alias!r}")
        return mapping
    except Exception as exc:  # noqa: BLE001
        console.warn(f"Alias migration failed (is MLflow available?): {exc}")
        return {}


def _configure_mlflow(cfg: Config):
    from .experiments import _configure

    return _configure(cfg)


# --------------------------------------------------------------------------- #
# #42 A/B routing harness
# --------------------------------------------------------------------------- #
def ab_win_rate(a_scores: Sequence[float], b_scores: Sequence[float],
                higher_is_better: bool = True) -> dict:
    """Paired win-rate of B vs A (challenger vs champion).

    Matches sample i in both sets; ties count separately. Returns the win
    rate, a Wilson-like lower bound for safety, and a Wilcoxon p-value.
    """
    a = np.asarray([v for v in a_scores if v is not None], dtype=float)
    b = np.asarray([v for v in b_scores if v is not None], dtype=float)
    n = min(len(a), len(b))
    if n == 0:
        return {"n": 0, "win_rate": None, "ties": 0, "wins": 0, "losses": 0, "p_value": None}

    a, b = a[:n], b[:n]
    better = b > a if higher_is_better else b < a
    worse = b < a if higher_is_better else b > a
    wins = int(np.sum(better))
    losses = int(np.sum(worse))
    ties = int(n - wins - losses)
    win_rate = wins / n if n else 0.0

    p_value = None
    if n >= 5 and wins != losses:
        from scipy.stats import wilcoxon

        try:
            _, p_value = wilcoxon(b - a)
            p_value = float(p_value)
        except ValueError:
            p_value = None

    # Wilson lower bound (95%) for a conservative "is it actually better?"
    z = 1.96
    if n:
        p = win_rate
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        lower = max(0.0, center - margin)
    else:
        lower = 0.0

    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": round(win_rate, 4),
        "wilson_lower": round(lower, 4),
        "p_value": round(p_value, 4) if p_value is not None else None,
        "decision": "promote" if lower > 0.5 else "hold",
    }


def route(cfg: Config, champion: str, challenger: str,
          champion_share: float = 0.5) -> dict:
    """A routing decision: split traffic between champion/challenger.

    A pragmatic harness stub: it records the decision and returns the split.
    The actual serving integration lives in ``musictrain serve``.
    """
    share = max(0.0, min(1.0, champion_share))
    decision = {
        "champion": champion,
        "challenger": challenger,
        "champion_share": round(share, 4),
        "challenger_share": round(1.0 - share, 4),
        "at": datetime.now(timezone.utc).isoformat(),
    }
    path = cfg.project_root / "metadata" / "ab_route.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decision, indent=2))
    console.ok(f"A/B route: {champion} {share:.0%} / {challenger} {1 - share:.0%}")
    return decision


# --------------------------------------------------------------------------- #
# #43 auto-promotion on gate pass
# --------------------------------------------------------------------------- #
def auto_promote(cfg: Config, candidate: str, baseline: str) -> dict:
    """Run the eval gate; promote the candidate to Production on pass."""
    from .gates import eval_gate
    from .registry_ml import transition

    gate = eval_gate(cfg.project_root, cfg, baseline=baseline, candidate=candidate)
    passed = bool(gate.get("passed"))
    result = {"gate": gate, "promoted": False}
    if passed:
        promoted = transition(cfg, f"{cfg.mlflow.experiment_name.replace('/', '_')}-models",
                              candidate, "Production")
        result["promoted"] = promoted
        if promoted:
            console.ok(f"Gate passed — {candidate} promoted to Production.")
    else:
        console.warn(f"Gate blocked — {candidate} NOT promoted.")
    return result


# --------------------------------------------------------------------------- #
# #44 model lineage
# --------------------------------------------------------------------------- #
def record_lineage(cfg: Config, parent: str, child: str,
                   note: str = "") -> dict:
    """Append a parent->child lineage edge (e.g. fine-tune derived from base)."""
    record = {
        "parent": parent,
        "child": child,
        "note": note,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    path = cfg.project_root / "metadata" / "lineage.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    console.ok(f"Lineage: {parent} -> {child}")
    return record


def lineage_graph(cfg: Config) -> dict:
    """Read metadata/lineage.jsonl into nodes + edges."""
    path = cfg.project_root / "metadata" / "lineage.jsonl"
    if not path.exists():
        return {"nodes": [], "edges": []}
    edges: List[dict] = []
    nodes = set()
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        edges.append({"parent": r["parent"], "child": r["child"], "note": r.get("note", "")})
        nodes.add(r["parent"])
        nodes.add(r["child"])
    return {"nodes": sorted(nodes), "edges": edges}


# --------------------------------------------------------------------------- #
# #45 artifact checksums
# --------------------------------------------------------------------------- #
def checksum_dir(model_dir: Path, extensions=(".safetensors", ".bin", ".json")) -> dict:
    """SHA-256 manifest over a model directory's weight/config files."""
    import hashlib

    files = [p for p in sorted(model_dir.rglob("*")) if p.is_file()
             and p.suffix.lower() in extensions]
    manifest = {}
    for p in files:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        manifest[str(p.relative_to(model_dir))] = h.hexdigest()
    return {
        "dir": str(model_dir),
        "n_files": len(manifest),
        "files": manifest,
        "at": datetime.now(timezone.utc).isoformat(),
    }


def verify_checksum(model_dir: Path, manifest: dict) -> bool:
    """Recompute hashes and compare against a stored manifest."""
    current = checksum_dir(model_dir)
    if current["n_files"] != manifest.get("n_files", -1):
        return False
    return current["files"] == manifest.get("files", {})


# --------------------------------------------------------------------------- #
# #46 rollback
# --------------------------------------------------------------------------- #
def rollback(cfg: Config, registry_name: str) -> dict:
    """Revert the champion alias to the previous-highest version."""
    from mlflow.tracking import MlflowClient

    _configure_mlflow(cfg)
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(registry_name)
        ordered = sorted(versions, key=lambda v: int(v.version), reverse=True)
        if len(ordered) < 2:
            console.warn(f"Need >= 2 versions of {registry_name} to roll back.")
            return {"rolled_back": False, "reason": "not enough versions"}
        previous = ordered[1]
        client.set_registered_model_alias(registry_name, "champion", str(previous.version))
        console.ok(f"Rollback: {registry_name} champion -> v{previous.version}")
        return {"rolled_back": True, "champion_version": int(previous.version)}
    except Exception as exc:  # noqa: BLE001
        console.warn(f"Rollback failed: {exc}")
        return {"rolled_back": False, "reason": str(exc)}


# --------------------------------------------------------------------------- #
# #49 per-run cost attribution
# --------------------------------------------------------------------------- #
def cost_breakdown(model_name: str, n_prompts: int, n_seeds: int,
                   tokens_per_clip: int = 256, device: str = "mps") -> dict:
    """Per-prompt / per-seed cost attribution from the aggregate estimate."""
    from .cost import estimate

    n_clips = n_prompts * n_seeds
    est = estimate(model_name, n_clips, tokens_per_clip)
    kwh = est["estimated_kwh"]
    # derive per-prompt from per-seed so the attribution stays internally consistent
    per_seed = (kwh / n_clips) if n_clips else 0.0
    return {
        "model": model_name,
        "device": device,
        "n_prompts": n_prompts,
        "n_seeds": n_seeds,
        "n_clips": n_clips,
        "total_kwh": round(kwh, 8),
        "per_prompt_kwh": round(per_seed * n_seeds, 8) if n_prompts else None,
        "per_seed_kwh": round(per_seed, 8) if n_clips else None,
        "per_clip_kwh": round(per_seed, 8) if n_clips else None,
    }


# --------------------------------------------------------------------------- #
# #50 config schema linter
# --------------------------------------------------------------------------- #
def lint_config(cfg: Config) -> List[dict]:
    """Validate config ranges; returns a list of {field, issue} problems."""
    issues: List[dict] = []

    def _add(field: str, issue: str) -> None:
        issues.append({"field": field, "issue": issue})

    if not (0 < cfg.check.bpm_tolerance < 1):
        _add("check.bpm_tolerance", "must be in (0, 1)")
    if cfg.check.max_time_stretch <= 0:
        _add("check.max_time_stretch", "must be > 0")
    if cfg.check.beats_per_bar <= 0:
        _add("check.beats_per_bar", "must be > 0")

    if cfg.split.k_folds > 0 and cfg.split.k_folds < 2:
        _add("split.k_folds", "must be 0 (off) or >= 2")
    total = cfg.split.train + cfg.split.val + cfg.split.test
    if not (0.999 <= total <= 1.001):
        _add("split.train+val+test", f"must sum to 1 (got {total:.3f})")

    if not (0 < cfg.eval.max_abs_deviation < 1):
        _add("eval.max_abs_deviation", "must be in (0, 1)")
    if not (0 <= cfg.eval.min_clap_score < 1):
        _add("eval.min_clap_score", "must be in [0, 1)")

    if not (0 < cfg.dedup.threshold <= 1):
        _add("dedup.threshold", "must be in (0, 1]")

    if not (0 < cfg.metrics.n_mels <= 256):
        _add("metrics.n_mels", "must be in (0, 256]")

    if cfg.inference.temperature <= 0:
        _add("inference.temperature", "must be > 0")
    if not (0 <= cfg.inference.top_p <= 1):
        _add("inference.top_p", "must be in [0, 1]")
    if cfg.inference.max_new_tokens <= 0 and cfg.inference.target_seconds is None:
        _add("inference.max_new_tokens", "must be > 0 (or set target_seconds)")

    return issues


def lint(cfg: Config) -> dict:
    issues = lint_config(cfg)
    report = {
        "valid": not issues,
        "n_issues": len(issues),
        "issues": issues,
        "at": datetime.now(timezone.utc).isoformat(),
    }
    path = cfg.project_root / "metadata" / "config_lint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    if issues:
        for i in issues:
            console.warn(f"{i['field']}: {i['issue']}")
        console.warn(f"Config lint: {len(issues)} issue(s) -> metadata/config_lint.json")
    else:
        console.ok("Config lint: clean -> metadata/config_lint.json")
    return report
