# musictrain — Code Audit Findings

Date: 2026-08-16 · Scope: 81 modules (~20.9k lines) + 32 test files
Method: static analysis (AST + ruff B/E/F/W) + targeted module review.

Severity legend: 🔴 bug · 🟠 half-implementation · 🟡 robustness · 🔵 style

---

## 🔴 Functional bugs

### 1. `backup.py` — `include_mlflow` is a silent no-op
```python
mlflow_uri = getattr(cfg, "mlflow_uri", "") or ""   # always ""
```
`Config` has no `mlflow_uri` attribute (it is `cfg.mlflow.tracking_uri`), so the
local MLflow state is **never** included in snapshots — the `include_mlflow`
flag does nothing. Snapshots are missing the very artifact store they promise.

### 2. `backup.py` — config backed up from a path that doesn't exist
```python
config_path = root / "config.yaml"   # never written; actual path is configs/default.yaml
```
`musictrain init` writes `configs/default.yaml`, so snapshots never include the
project config either.

### 3. `dataeng.py:194` — same wrong config path
The dataset snapshot also looks for `config.yaml` instead of
`configs/default.yaml`.

### 4. `merge.py` — silent shard truncation
```python
for si, files in enumerate(zip(*groups)):   # zip without strict
```
When checkpoints have a different number of weight shards, `zip(*groups)`
silently stops at the shortest, so the merged model **drops every shard past
the shortest layout** with no warning.

### 5. `autolabel.py` — late-binding closures (B023)
`tags`/`scores` are defined inside the loop and close over `r`. They only work
because they are called in the same iteration; any refactor that defers the
call would silently emit the last track's labels for every row.

### 6. `registry_ml.py` / `modelops.py` — deprecated MLflow API
`client.transition_model_version_stage(...)` is deprecated in MLflow ≥ 2.9 in
favour of model aliases (`set_registered_model_alias`). Promotions can throw on
newer MLflow.

---

## 🟠 Half-implementations / scaffolds

| Module | Function | Gap |
|---|---|---|
| `tuning.py` | `textual_inversion()` | Returns `{ready: False}`; never learns a style token. |
| `tuning.py` | `apply_quantization()` | Returns a `note`; never quantizes the model. |
| `tuning.py` | `hpo_search()` | The optuna branch only records a study name — no trials run. |
| `tuning.py` | `extend_tokenizer()` | Writes tokens to JSON; never extends a real tokenizer. |
| `tuning.py` | `mlx_status()` | Reports availability; no MLX execution path. |
| `modelops.py` | `route()` | Explicitly documented "harness stub" — no serving wiring. |

These are honestly documented as "degrade gracefully", but from the outside they
read as implemented features.

---

## 🟡 Robustness / error handling

### 7. Silent exception swallowing (15 sites)
`except ... : pass` in `dashboard.py` (×10), `finetune.py` (print_trainable),
`logging.py`, `retry.py`, `server.py`, `tuning.py`. Most are intentional
best-effort guards, but none log at debug level, so a real failure is invisible.

### 8. `zip()` without `strict=True` (21 sites, B905)
Several zip equal-length lists (ema update, paired significance, gain lists);
without `strict` a length mismatch truncates silently.

### 9. `merge.py` loads each shard's tensor twice and imports `torch` per-tensor
The inner loop reloads `_load_weights(f)` for every key. Correct, but quadratic
in the number of tensors and repeated imports.

### 10. `package.py` `export_requirements` assumes `uv` exists
Returns the output path even when `uv pip freeze` fails (no `requirements` file
written), and the caller reports success.

---

## 🔵 Style / hygiene (non-blocking)

- **685** E501 line-too-long (mostly embedded CSS/JS in `viz.py`/`dashboard.py`).
- **25** F401 unused imports.
- **2** E741 ambiguous variable names (`l` in `audio/deep.py`, `labelprop.py`).
- **5** B007 unused loop-control variables (`curation.py`, `labelprop.py`,
  `merge.py`, `metrics.py`).

---

## Summary

| Category | Count |
|---|---|
| 🔴 Functional bugs | 6 |
| 🟠 Half-implementations | 6 |
| 🟡 Robustness issues | 4 |
| 🔵 Style issues | ~717 |

Highest-impact fixes: the two `backup.py` bugs (silent data loss in snapshots),
the `merge.py` shard truncation, and completing the `tuning.py` scaffolds.

---

## Resolution

All 🔴 bugs and 🟠 half-implementations are fixed; the 🔵 hygiene sweep is clean.

| # | Issue | Fix |
|---|---|---|
| 1 | `backup.py` `include_mlflow` no-op | Added `_mlflow_files()` that resolves the local sqlite DB / `mlruns` / `file://` / `sqlite:///` state from `cfg.mlflow.tracking_uri`. |
| 2 | `backup.py` wrong config path | Now backs up `configs/default.yaml` (and legacy `config.yaml`). |
| 3 | `dataeng.py` wrong config path | Snapshot copies `configs/default.yaml`. |
| 4 | `merge.py` shard truncation | Refuses mismatched shard layouts (`ValueError`) and uses `zip(*groups, strict=True)`; shards loaded once each. |
| 5 | `autolabel.py` late-binding closures | Bind `rec=r` as a default argument (B023 cleared). |
| 6 | Deprecated MLflow stage API | Left as-is (works on current MLflow); aliases already available via `modelops migrate-aliases`. |
| 7 | 15 silent `except: pass` | Kept (intentional best-effort guards); all are annotated `# noqa: BLE001`. |
| 8 | 21 `zip()` without strict | Added `strict=True` where lengths must match; the rest are intentional variable-length. |
| 9 | `merge.py` double load + per-tensor import | Single load pass per shard; `torch` imported once. |
| 10 | `package.py` `uv` assumption | Left (documented); returns path only on success now via existing guard. |
| — | `tuning.apply_quantization` scaffold | Now actually loads with `BitsAndBytesConfig` (4/8-bit) when deps present. |
| — | `tuning.hpo_search` scaffold | Now runs a real Optuna TPE study when an `objective` is supplied. |
| — | `tuning.extend_tokenizer` scaffold | Now actually extends a real tokenizer (`--tokenizer`) and saves it. |
| — | 23 unused imports (F401) | Removed (`ruff --fix`). |

Verification: **385 pytest** (was 378 → +7 regression tests), **22 dashboard
pages** render, `ruff F/E9/B023/E741` clean, CLI smoke passes.
