# 🎵 MusicTrain

[![CI](https://github.com/easymoneysnipa48-sudo/AiTrainer/actions/workflows/ci.yml/badge.svg)](https://github.com/easymoneysnipa48-sudo/AiTrainer/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/easymoneysnipa48-sudo/AiTrainer/blob/main/LICENSE)

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
| `musictrain quality` | Score audio quality (bitrate, clipping, silence, DC offset, lowpass) → `metadata/quality_report.json` |
| `musictrain loudnorm` | Normalize loudness to a target LUFS (e.g. −14) in place |
| `musictrain stems` | Separate stems with Demucs (vocals/drums/bass/other) → `data/stems/` |
| `musictrain dedup` | Find exact + near-duplicate audio (pitch/tempo-robust) → `metadata/duplicates.json` |
| `musictrain similar` | CLAP nearest-neighbour search: "find tracks like this one" |
| `musictrain autolabel` | Suggest genre/mood/instrument tags via CLAP → `metadata/autolabels.csv` |
| `musictrain corpus` | BPM/key/tag coverage statistics → `metadata/corpus_stats.json` |
| `musictrain ood` | Flag off-distribution tracks (tempo/tag outliers) → `metadata/ood_tracks.json` |
| `musictrain analyze` | Deep audio analysis: chords, beat/downbeat grid, key confidence, onset density, tempo curve, swing, structure, vocal/instrumental → `metadata/analysis.json(l)` |
| `musictrain labels` | Scaffold `metadata/labels.csv` or validate it against the controlled vocabulary |
| `musictrain segment` | Split into ~30 s examples — bar-aligned, or downbeat-aligned with overlap/fades (`--downbeat --overlap --fade`) |
| `musictrain split` | Train/val/test split **by song** (no leakage), stratified (`--stratify`) or k-fold (`--k-folds`) |
| `musictrain export` | Export the split corpus to HF `datasets` (arrow/jsonl/csv) → `data/dataset/` |
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

## Labeling workflow

Phase 4 keeps the controlled vocabulary consistent and makes labeling faster:

```bash
musictrain vocab                      # print the hierarchical vocabulary tree (#27)
musictrain vocab --version            # current vocabulary version
musictrain vocab --migrate rename.json            # rename terms across labels.csv atomically (#32)
musictrain agree --a A.csv --b B.csv             # inter-annotator agreement -> agreement.json (#29)
musictrain suggest --query track.wav --top 5     # CLAP tag proposals + labeled neighbors (#31)
musictrain prompt --section chorus --genre "melodic trap" --mood dark --bpm 140 --key "A minor"  # (#30)
musictrain labels --notes             # flag broad parent terms ("808 bass" when "sub bass" fits)
```

* **Hierarchical vocab** — `labels.py` gains `HIERARCHY` (parent → children,
e.g. `808 bass → sub bass`) and `VOCAB_VERSION`. `musictrain vocab` renders the
tree; `labels --check` still enforces the flat term list, and `--notes` adds
soft suggestions to prefer specific children.
* **Versioned migration (#32)** — `--migrate` takes a JSON map (`{"genre":
{"trap": "melodic trap"}}` or a flat map applied to all fields), rewrites the
CSV with a timestamped `.bak` backup, and stamps
`metadata/vocab_version.json` so old results stay attributable.
* **Annotator agreement (#29)** — percent agreement + Cohen's kappa per field
(genre/mood/instruments/section/section_type) over shared `source_id`s, plus a
sample of disagreements.
* **Auto-suggest (#31)** — CLAP tag proposals per dimension plus the nearest
labeled neighbors from the embedding cache, so you can copy labels from a
known track instead of guessing.
* **Prompt builder (#30)** — assemble a generation prompt from vocabulary
selections in the same shape as the training labels; the 🪄 *Prompt builder*
dashboard page makes this clickable.

The 🏷️ *Labels* dashboard page surfaces all of it: vocab tree + version, per-
dimension coverage (used vs. unused terms, #28), suggestions, and agreement.

## Dataset hygiene

Before training, sweep the corpus for junk and leakage:

```bash
musictrain quality            # bitrate/clipping/silence/DC/lowpass -> quality_report.json
musictrain dedup              # exact + near-duplicate detection -> duplicates.json
musictrain dedup --move       # …and move copies to data/dupes/
musictrain loudnorm --target -14.0 --force   # consistent perceived loudness
musictrain corpus             # BPM/key/tag coverage -> corpus_stats.json
musictrain ood                # flag off-distribution tracks -> ood_tracks.json
musictrain ood --move         # …and move them to data/ood/
musictrain autolabel          # CLAP-suggested genre/mood/instruments -> autolabels.csv
musictrain similar --query track.wav --top 10   # "find tracks like this one"
musictrain stems              # Demucs stem separation -> data/stems/
musictrain analyze            # deep analysis: chords, beat grid, key confidence, structure
musictrain analyze --path track.wav   # …or a single file
```

`similar` and `autolabel` reuse the cached CLAP model and store audio embeddings
in `metadata/audio_embeddings.json`, so repeat runs are instant. `quality`
thresholds live under the `quality:` config section, `dedup:`/`ood:`/`autolabel:`
tune the other sweeps.

## Deep audio analysis

`musictrain analyze` produces a per-track record with nine dimensions (all
librosa-based, so no extra model downloads beyond the CLAP already used for
prompt adherence):

| Field | What it reports |
| --- | --- |
| `key` | Krumhansl-Schmuckler key + softmax confidence + top-3 candidates |
| `chords` | Time-stamped chord labels (24 major/minor triads) with confidence |
| `beat_grid` | Beat + downbeat timestamps, downbeat phase, BPM |
| `onsets` | Onset density, mean inter-onset interval, rhythmic complexity (CV of IOI) |
| `tempo_curve` | Tempo over time (mean/median/std + per-bin curve) |
| `swing` | Off-beat vs on-beat energy ratio → straight/moderate/swung |
| `structure` | Segment boundaries + coarse role labels (intro/verse/chorus/…, energy-based) |
| `vocal` | CLAP vocal vs instrumental verdict |
| `timbre` | CLAP audio-embedding dim/norm (reuses the embedding index) |

Thresholds and behaviour live under the `analysis:` config section. Structure
roles are heuristics (energy + position), not ground truth — treat them as a
starting point for the manual labels.

## Segmentation & splits

```bash
musictrain segment --downbeat            # cut on detected downbeats (#21)
musictrain segment --overlap 2.0         # 2 s overlap between consecutive segments (#24)
musictrain segment --fade 0.05           # fade in/out to de-click boundaries (#25)
musictrain split --stratify genre        # balanced key/bpm/genre/mood per split (#23)
musictrain split --k-folds 5             # 5-fold CV -> metadata/folds.json (#22)
musictrain export --format arrow         # HF datasets DatasetDict -> data/dataset/ (#26)
musictrain export --format jsonl --which val
```

`split` groups by song (no train/val leakage). `--stratify` splits each
attribute bucket proportionally so rare genres/keys don't collapse into one
fold; `--k-folds` writes rotating train/val folds instead of a single split.
`export` joins each segment with its feature manifest and emits an Arrow
DatasetDict (Audio column) or flat JSONL/CSV. Install the optional dep with
`uv pip install -e '.[export]'`.

## Experiment tracking

`infer`, `check`, and `features` log runs to a **local MLflow store**
(`mlflow.db`, SQLite) — params, metrics, prompts, and generated/eval audio as
artifacts. View them with:

```bash
musictrain ui
```

The Streamlit dashboard has a **📊 Compare** page that reads runs straight from
MLflow — filter by task/checkpoint, scatter detected-vs-target BPM, and see
adherence summaries. It also shows a **🧪 Stable verdicts** summary (majority
vote over repeated seeds) read from `metadata/eval_results.jsonl`: per-section
in-tolerance counts, mean |deviation|, and mean CLAP — the reliable baseline to
diff against future checkpoints, distinct from the raw per-seed MLflow runs.

A **🧹 Hygiene** page surfaces the four dataset sweeps — audio-quality scores
and grades, duplicate groups, BPM/key/tag corpus coverage, and OOD flags — with
one-click buttons to re-run each sweep from the UI.

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
musictrain eval --section chorus           # single section
musictrain eval --section "chorus,outro"   # multiple sections
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

### Eval & metrics (Phase 6)

```bash
# #43 auto-reject thresholds — tune in configs/default.yaml under `eval:`
#   min_clap_score: 0.0      reject prompts whose mean CLAP is too low
#   max_abs_deviation: 0.20  reject prompts whose |BPM deviation| is too high
#   per_tag_clap: true       also score each tag phrase separately (#46)
musictrain eval --seeds 3

# #44 significance — paired Wilcoxon between two result sets / checkpoints
musictrain significance --a results_old.jsonl --b results_new.jsonl
musictrain significance --checkpoint-a ckpt-a --checkpoint-b ckpt-b

# #45 leaderboard — rank checkpoints by CLAP + deviation + verdict + human rating
musictrain leaderboard

# #41 distribution metrics — FAD (on CLAP embeddings) + spectral KL
musictrain metrics --ref data/clean --gen outputs/eval
```

With `per_tag_clap` on, each generated clip is scored against its own section,
genre, key, mood, instruments, and BPM phrases, stored as `clap_per_tag` in the
results. The HTML report gains a per-tag CLAP column, and the leaderboard shows
the per-tag breakdown per checkpoint (#46). Human ratings from the 🎧
**Listening** dashboard page (#42) merge into the leaderboard and report
(`metadata/human_ratings.jsonl`). The 🏆 **Leaderboard** page renders the
rankings with a rebuild button.

## Generation controls

Phase 5 adds per-run knobs to `musictrain infer`:

```bash
# #37 sampling presets — one flag instead of raw knobs
musictrain presets                        # list presets
musictrain infer --preset creative --prompt "…"

# #39 target duration — derive max_new_tokens from seconds
musictrain infer --target-seconds 15 --prompt "…"

# #35 continuation — keep going from an existing clip (saves full track)
musictrain infer --continue-from data/clean/intro.wav --prompt "build into the verse"

# #36 melody conditioning — follow a clip's melody (use facebook/musicgen-melody)
musictrain infer --melody-from data/clean/hook.wav --model facebook/musicgen-melody --prompt "…"

# #33 negative prompting — CLAP-scored "no X" constraints with auto-retry
musictrain infer --negative "vocals, riser" --negative-retries 2 --prompt "…"

# #34 batch prompt files — plain lines or JSONL with per-item options
musictrain infer --prompts-file prompts.jsonl
#   {"prompt": "…", "seed": 7, "negative_prompt": "heavy drums", "target_seconds": 1.5}
#   {"prompt": "…", "preset": "creative", "target_seconds": 1.5}

# #38 reproducibility manifest — pin config/vocab/git per run
musictrain manifest                      # last 5 runs
musictrain manifest --diff 1 2           # diff the two most recent
```

Every generation (and each eval run) appends a record to
`metadata/repro_manifest.jsonl` snapshotting the config, vocabulary version,
git commit (+dirty flag), model, prompt, and the generation parameters used —
so an old result can be reproduced or traced exactly.

**How the pieces work**

- **Presets** (`standard`/`creative`/`precise`) override temperature / top-k /
top-p / guidance as a group; `creative` loosens sampling, `precise` tightens it.
- **Target seconds** converts at 50 tokens/sec (MusicGen's 32 kHz codec rate).
- **Continuation & melody** both condition on a clip via transformers'
`input_values` path (the audio encoder encodes it into codes/chroma);
continuation writes the full track (conditioning audio + new audio) so you can
chain sections end-to-end.
- **Negative prompting** has no native MusicGen support, so it's enforced
post-hoc: the generated audio is CLAP-scored against the negative text and
flagged/re-generated when similarity ≥ `inference.negative_threshold` (0.25).
`negative_retries` auto-regenerates with a new seed until clean.

## Advanced commands (50)

Every command below runs through the same CLI — `musictrain <cmd> --help` for
full options. Results land under `metadata/`; the 🧹 **Hygiene** page surfaces
drift / curation / leakage, and the 🪵 **Logs** page tails the structured runlog.

### Batch 1 — Evaluation analytics (#1-10)

```bash
# #1-2 distribution metrics — FAD (CLAP or VGGish) + spectral KL
musictrain metrics --ref data/clean --gen outputs/eval            # clap FAD
musictrain metrics --ref data/clean --gen outputs/eval --fad vggish  # fadtk

# #3-5 bootstrap CIs, Bayesian A/B (P(B > A)), two-sample KLD/MMD/1-NN
musictrain significance --a old.jsonl --b new.jsonl
musictrain significance --checkpoint-a ckpt-a --checkpoint-b ckpt-b

# #6 meta-analysis — pool studies by their {delta, se}
musictrain significance --meta study1.json study2.json

# #7-9 prompt difficulty, section x BPM interaction, CLAP z-scores,
#       auto-reject threshold calibration, negative mining
musictrain difficulty

# #10 adversarial (tricky) prompts appended to the eval set
musictrain evalset --adversarial 10
```

### Batch 2 — Inference & training (#11-20)

```bash
# #11 larger models + bf16 dtype for memory headroom
musictrain infer --model facebook/musicgen-large --dtype bf16 --prompt "…"

# #12 LoRA fine-tune of the decoder on segments + labels (0 = dry run)
musictrain finetune --steps 500

# #13 merge 2+ checkpoints (weighted average of shards)
musictrain merge --models ckpt-a ckpt-b --weights 0.5 0.5 --out checkpoints/merged

# #14 prompt ensembling — n variants, pick the best CLAP
musictrain sweep --kind ensemble --prompt "melodic trap chorus, 96 BPM" --n 3

# #15 conditioning chaining — each step melody-conditions on the previous output
musictrain sweep --kind chain --prompt "dark piano intro" --steps 3

# #16 melody-from-audio — upload a reference clip on the dashboard Generate page

# #17 negative mining — hardest prompts surface in `musictrain difficulty` output

# #18 guidance-scale sweep          # #19 seed search
musictrain sweep --kind sweep --guidance 1.5,3,4.5 --prompt "…"
musictrain sweep --kind seeds --seeds 0,1,2,3 --prompt "…"

# #20 deterministic caching — identical (model,prompt,seed) reuse the saved clip
musictrain infer --cache --prompt "…" --seed 7
```

### Batch 3 — Data engineering (#21-30)

```bash
# #21 stems re-synthesis — re-mix Demucs stems with per-stem gain
musictrain resynth --gain vocals=1.2 --gain drums=0.5
musictrain resynth --instrumental          # drop vocals entirely

# #22 audio-to-prompt inversion — features template + CLAP prompt retrieval
musictrain invert --path data/clean/ref.wav

# #23 active learning — rank unlabeled tracks by uncertainty + diversity
musictrain active --top 20

# #24 augmentation — pitch/stretch/noise/EQ/quiet training variants
musictrain augment --ops pitch_up,stretch,noise,eq,quiet

# #25 post-segment dedup — near-identical segments out of the fine-tune set
musictrain dedup --segments

# #26 auto-section labeling — map segments onto detected structure roles
musictrain sections          # requires `musictrain analyze` + `musictrain segment`

# #27 feature drift monitoring (KS test + PSI)
musictrain drift --reference clean --current train

# #28 curation score — 0-100 quality/novelty/coverage/dup ranking
musictrain curation --top 20

# #29 embedding cache refresh — prune stale, re-embed changed audio
musictrain embed-refresh

# #30 semi-supervised pseudo-labels + train/val/test leakage check
musictrain labelprop --min-confidence 0.55
musictrain labelprop --check-leakage
```

### Batch 4 — Model ops (#31-40)

```bash
# #31 checkpoint registry — index checkpoints/ with config sig + size
musictrain registry

# #32 eval gate — block promotion on regression (exit 1 = block; CI-friendly)
musictrain gate --baseline ckpt-a --candidate ckpt-b

# #33 early stopping on the CLAP series (patience + min-delta)
musictrain early-stop --series 0.30,0.32,0.31,0.31 --patience 3

# #34 weight diff — per-tensor max-abs delta between two checkpoints
musictrain diff --a ckpt-a --b ckpt-b

# #35 experiment matrix — flatten MLflow runs into rows x metrics
musictrain matrix

# #36 drift detector — CI gate that fails when features drift
musictrain drift-check --reference clean --current train

# #37 promotion report — markdown bundle of rank, significance, coverage
musictrain promote --checkpoint facebook/musicgen-small --baseline ckpt-a

# #38 eval snapshot archive — zip checkpoint + config + reports
musictrain archive --checkpoint facebook/musicgen-small

# #39 training monitor — CLAP trend over MLflow runs per checkpoint
musictrain monitor

# #40 model card — markdown adherence + section coverage for a checkpoint
musictrain modelcard --checkpoint facebook/musicgen-small
```

### Batch 5 — Automation (#41-50)

```bash
# #41-42 FastAPI backend + job queue — POST /eval returns a job_id, poll /jobs/{id}
musictrain serve --port 8000
#   GET  /health  /inventory  /metrics  /leaderboard  /jobs
#   POST /eval (section/seeds/limit)  /generate (prompt/seed)  /jobs/{id}/cancel

# #43 multi-machine sync — Mac preps data + evals, Ubuntu trains (rsync plan)
musictrain package --host user@ubuntu

# #44 MLflow model registry — register checkpoints, move versions through stages
musictrain register --checkpoint facebook/musicgen-small --stage Staging
musictrain models                                    # list models + stages
musictrain stage --name musicgen-style-models --version 1 --stage Production

# #45 W&B export of eval aggregates (CSV fallback when not logged in)
musictrain export-eval --project musictrain

# #46 structured JSON runlog — every job/run appended to metadata/runlog.jsonl
musictrain runlog --event job

# #47 alerting — Slack webhook / SMTP / file when metrics cross thresholds
musictrain alert --min-clap 0.30 --max-dev 0.20 --slack-webhook https://hooks.slack.com/…

# #48 cost tracking — kWh estimates from clip count x model size
musictrain cost --task eval --model musicgen-medium --clips 44

# #49 incremental eval — keep passing rows, re-run only failed/new prompts
musictrain eval --incremental --seeds 3

# #50 CI regression gate — test suite + fixture gate on every push/PR
#     (.github/workflows/eval-gate.yml, .github/scripts/run_gate.py)
```

## CI

GitHub Actions runs a fast smoke test (config, eval-set generation, label
validation, BPM check, inventory, report export) on every push and pull request,
plus an **eval regression gate** (`.github/workflows/eval-gate.yml`) that blocks
a candidate when its BPM adherence regresses past tolerance. A heavier
`eval-smoke` job — a 1-prompt MusicGen generation on CPU — runs on-demand via
**Actions → CI → Run workflow**.

## Notes

- **MPS**: the `infer` command uses `float32` on MPS by default (most stable).
  Add `--device cpu` if a model errors out on Metal.
- **No leakage**: `split` groups by `song_id`, so segments from one song never
  cross train/validation/test.
- **Bar alignment**: `segment` trims each window to whole 4/4 bars when BPM is
  known, keeping transitions predictable.
- **Legal**: only train on audio you own or have an ML-permissive license for.
