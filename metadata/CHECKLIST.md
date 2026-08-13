# Metadata Curation Checklist

Work through this for **every track** before it enters training. The goal is a
clean, *consistent* mapping between audio and the natural-language prompts that
MusicGen conditions on. Inconsistent descriptions are the fastest way to get
muddy generations.

## Per-track checklist

- [ ] **Rights & license** — you own it or have an ML-permissive license.
      Record the exact license in `license` (e.g. `owned`,
      `licensed-commercial`, `cc0`, `cc-by`). No streaming rips.
- [ ] **Source recorded** — `source_id` (and `song_id` when sectioned) filled in.
- [ ] **Audio valid** — passes `musictrain inventory` (32 kHz, mono, sane duration).
- [ ] **BPM & key spot-checked** — `musictrain features` values look right; fix
      obvious octave errors (e.g. 187 detected for a 94 BPM track).
- [ ] **Section labelled** — `section` + `section_type` use the vocabulary below.
- [ ] **Genre, mood, instruments** — picked strictly from the vocabulary below.
- [ ] **Energy** — a number in `[0, 1]`, consistent across tracks.
- [ ] **Description written** — follow the prompt template (see below).
- [ ] **No leakage** — segments from one song are never split across train/val/test.

## Prompt template

Keep descriptions identical in structure so the model learns the slot meanings:

```
<SECTION>, <BPM> BPM, <KEY>, <instruments comma-separated>, <mood>, <energy-level> energy
```

Examples:

```
intro, 75 BPM, A minor, dark piano loop, atmospheric pads, low energy
verse, 72 BPM, E minor, melancholic guitar loop, deep 808, restrained snare, medium energy
hook, 78 BPM, A minor, heavy 808 bass, autotune vocals, trap hi-hats, high energy
outro, 75 BPM, A minor, fading piano, reduced drums, reflective mood
```

## Controlled vocabulary

These exact strings are enforced by `musictrain labels --check`. Edit
`musictrain/labels.py` (`VOCAB`) if you need new terms — then keep both in sync.

**Section / section_type**

```
intro | verse | pre-chorus | chorus | hook | bridge | outro | instrumental-transition | full-song
```

**Genre**

```
melodic trap | pain music | emo rap | trap | drill | southern trap | hip hop
| cinematic hip hop | lo-fi | ambient | electronic | orchestral | film score
| pop | rnb | jazz | synthwave
```

**Mood** (combine with `|`)

```
dark | emotional | determined | energetic | calm | melancholic | uplifting
| aggressive | reflective | tense | epic | mysterious | nostalgic | hopeful
| somber | dreamy | atmospheric
```

**Instruments** (combine with `|`)

```
piano | keys | guitar loop | acoustic guitar | electric guitar | 808 bass
| sub bass | bass | 808 slides | strings | pads | synths | synth lead | drums
| percussion | trap hi-hats | snare | clap | snare roll | brass | choir
| vocals | autotune vocals | ad-libs | organ | flute | harp | bells
| vinyl crackle | riser
```

## Workflow

```bash
# 1. scaffold the template (once)
musictrain labels

# 2. fill in metadata/labels.csv (delete the example rows)

# 3. validate vocabulary + required fields
musictrain labels --check

# 4. merge labels with auto-extracted features
musictrain features --labels metadata/labels.csv --validate
```
