# 🎵 MusicTrain

[![CI](https://github.com/easymoneysnipa48-sudo/AiTrainer/actions/workflows/ci.yml/badge.svg)](https://github.com/easymoneysnipa48-sudo/AiTrainer/actions/workflows/ci.yml)

A config-driven toolkit that automates the **Mac control-plane** side of your
MusicGen fine-tuning workflow. It's the practical implementation of
`aitraining.md`: prepare, validate, and label your dataset on the M3 Max now,
then hand the heavy CUDA training to the Ubuntu workstation later.

## What it does

| Command | Stage |
| --- | --- |
| `musictrain init` | Create the project layout + default config + `.gitignore` |
| `musictrain normalize` | FFmpeg batch: `data/raw` → `data/clean` (mono, 32 kHz, PCM, optional loudnorm) |
| `musictrain inventory` | Validate audio, write `metadata/audio_inventory.json` (+ SHA-256 option) |
| `musictrain features` | Extract BPM, key, LUFS/RMS, peak, silence & clipping ratios; merge manual labels |
| `musictrain labels` | Scaffold `metadata/labels.csv` or validate it against the controlled vocabulary |
| `musictrain segment` | Split into ~30 s examples, bar-aligned when BPM is known |
| `musictrain split` | Train/val/test split **by song** (no leakage), materialize files |
| `musictrain infer` | MusicGen text→audio on MPS (CPU fallback), single or batch prompts |
| `musictrain check` | Detect BPM drift of generated audio, optional time-stretch fix |
| `musictrain score` | Score audio-text similarity with CLAP (prompt adherence) |
| `musictrain evalset` | Generate the fixed evaluation prompt set |
| `musictrain eval` | Batch inference + BPM checks over the eval set (MLflow-logged) |
| `musictrain report` | Export eval results to CSV + HTML for review |
| `musictrain package` | SHA-256 checksums, `requirements-mac.txt`, rsync plan for Ubuntu |
| `musictrain dashboard` | Launch the Streamlit web UI |
| `musictrain ui` | Launch the MLflow tracking UI |

## Quick start

```bash
# 1. set up the environment (torch is reused from the system site-packages)
uv venv --system-site-packages --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .

# 2. create the project skeleton
musictrain init

# 3. drop your (licensed) tracks into data/raw, then run the pipeline
musictrain normalize
musictrain inventory --sha256
musictrain features --validate
musictrain segment
musictrain split

# 4. test generation on MPS
musictrain infer --prompt "melodic trap hook, 78 BPM, A minor, heavy 808 bass, autotune vocals, trap hi-hats"

# 5. check the output's tempo
musictrain check --path outputs/latest.wav --bpm 96 --fix

# 6. web UI
musictrain dashboard
```

## Directory layout

```
data/raw/        original source files — never modified
data/clean/      normalized audio (mono / 32 kHz / PCM)
data/segments/   30s (or bar-aligned) examples
data/{train,val,test}/   materialized splits
metadata/        inventory, features, manifest, splits, checksums
configs/         default.yaml (all pipeline knobs)
checkpoints/     model checkpoints (future)
outputs/         generated WAVs
logs/            training/eval logs (future)
scripts/         transfer plan, misc
notebooks/       exploration
```

## Manual labels

`musictrain features --labels metadata/labels.csv` merges curated fields into the
auto-extracted features. CSV columns (any subset):

```
source_id,license,genre,mood,instruments,section,description,song_id,section_type,narrative_role
```

`mood` and `instruments` accept comma/semicolon/pipe-separated lists. The
`description` field is what MusicGen conditions on, so keep its vocabulary
consistent (e.g. `"hook, 78 BPM, A minor, heavy 808 bass, autotune vocals, trap hi-hats"`).

Scaffold a template and enforce the vocabulary:

```bash
musictrain labels            # create metadata/labels.csv with example rows
musictrain labels --check    # validate vocabulary + required fields
```

See `metadata/CHECKLIST.md` for the per-track checklist and the controlled
vocabulary (editable in `musictrain/labels.py`).

## Experiment tracking

`infer`, `check`, and `features` log runs to a **local MLflow store**
(`mlflow.db`, SQLite) — params, metrics, prompts, and generated/eval audio as
artifacts. View them with:

```bash
musictrain ui
```

The Streamlit dashboard has a **📊 Compare** page that reads runs straight from
MLflow — filter by task/checkpoint, scatter detected-vs-target BPM, and see
adherence summaries.

Disable tracking with `mlflow.enabled: false`, or point at a remote tracking
server via `mlflow.tracking_uri` in `configs/default.yaml`. When your
workstation is back, use the same tracking URI so Mac prep runs and Ubuntu
training runs share one experiment history.

## Evaluation

A fixed, version-controlled prompt set spanning sections, BPMs, and keys lives
in `metadata/eval_prompts.jsonl` (plus two out-of-distribution prompts):

```bash
musictrain evalset              # generate the set (deterministic seeds)
musictrain eval --limit 5       # run a subset (or omit --limit for all)
musictrain eval --seeds 3       # 3 seeds per prompt → majority verdict
```

`eval` generates each prompt on MPS, runs the BPM post-check against the target,
and logs params/metrics/audio to MLflow. Pass `--seeds 3` to generate each prompt
multiple times and report a **majority verdict** (median BPM, mean CLAP) instead
of a single noisy draw. It also computes a **CLAP audio-text
similarity** (`clap_score`) for automated prompt adherence — skip it with
`--no-clap`. Score a single file with `musictrain score --path out.wav --text "..."`.
Results land in `metadata/eval_results.jsonl` with fields matching your doc's
eval schema (`checkpoint`, `seed`, `bpm_target`, `detected_bpm`, `clap_score`,
`human_rating`, …) so you can fill in listening scores later. Export a
reviewable CSV + HTML report (summary cards, per-section breakdown, clickable
audio) with `musictrain report`.

## CI

GitHub Actions runs a fast smoke test (config, eval-set generation, label
validation, BPM check, inventory, report export) on every push and pull request.
A heavier `eval-smoke` job — a 1-prompt MusicGen generation on CPU — runs
on-demand via **Actions → CI → Run workflow**.

## Notes

- **MPS**: the `infer` command uses `float32` on MPS by default (most stable).
  Add `--device cpu` if a model errors out on Metal.
- **No leakage**: `split` groups by `song_id`, so segments from one song never
  cross train/validation/test.
- **Bar alignment**: `segment` trims each window to whole 4/4 bars when BPM is
  known, keeping transitions predictable.
- **Legal**: only train on audio you own or have an ML-permissive license for.
