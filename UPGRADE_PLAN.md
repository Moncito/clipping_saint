# Clipper upgrade plan ("our own Opus Clip")

## What Opus Clip actually does (research, Nov 2026)
- **Virality Score 0-99** per clip, built from 4 AI judgements:
  **Hook** (do the first seconds grab you?), **Flow** (does it move logically to a satisfying end?),
  **Value** (is it worth watching / relatable?), **Trend** (does it match what audiences want now?).
  Clips are sorted by score, best first.
- Reviewers' warning: the score is a *sorting signal*, not a view prediction. It only gets
  trustworthy when you compare it to your own real results over time.
- Other features people pay for: auto-reframe that follows faces, animated captions with keyword
  highlight + emoji, removal of dead air ("jump cuts"), AI titles/captions/hashtags per platform.

## Where our clipper is today
- Finds moments by loudness spikes + hype words, an AI (Claude CLI / Ollama) or loudness picks clips.
- Renders 9:16 with word captions + 3s title hook, or campaign mode (one approved line + cover box).
- Desktop app (Clipper.bat). No scoring, no ranking, no feedback loop.

## Phases

### Phase 1 — Virality Score + ranked report  ✅ building now
- New `score` stage after render (also `python clip.py score <video> --name <job>` for old jobs).
- Each clip gets **Hook / Flow / Value / Trend (0-25 each) → total 0-99**, from two sources:
  - **Signals (free, offline, always on):** speech in the first 3s, loudness of the opening,
    hype words, dead air %, words/second, does it end on a finished sentence, does it start
    mid-word, length sweet spot.
  - **AI judge (Claude CLI → Ollama fallback):** reads each clip's transcript + the signals,
    scores the 4 categories, says *why*, suggests a better title and one concrete fix.
  - Final = 65% AI + 35% signals (signals only if no AI is available).
- `jobs/<name>/report.html`: clips ranked best→worst, playable, score badge, 4 bars,
  why/fix notes. App gets an **Open report** button and opens it after each run.

### Phase 2 — Better clips, not just better ranking  ✅ built
- Smart pick: the picker finds 2× `--count` clips (`--pool`), they're pre-scored on transcript +
  audio (no render needed), only the best `--count` get rendered. Overlapping picks are dropped.
- Sentence snapping: clips start at the start of a sentence and end after one finishes.
- Jump cuts: pauses > 1s with no speech AND quiet audio are cut (loud gameplay is kept);
  captions are re-timed to the cut. `--no-jump-cuts` to turn off. Not used in campaign mode.
- Captions: chunks break on punctuation, active word pops (scale) and turns green on hype words.
- Still open: hook boost (flash the strongest line as the opener).

### Phase 3 — Reframe + polish
- Face tracking reframe (MediaPipe on CPU, fits the 4 GB GPU limit) for non-facecam videos.
- Per-platform caption + hashtag suggestions written next to each clip (never auto-posted).

### Phase 4 — Learn from real results (the part Opus can't do for *you*)
- `performance.csv`: after posting, you add views / likes / watch time per clip.
- Compare predicted score vs real views; reweight Hook/Flow/Value/Trend for *your* audience
  (Tier 1-2 SEA, gaming + music campaigns). After ~30 posted clips the score becomes yours.

## Housekeeping found along the way
- NVIDIA driver too old for NVENC (needs 610+): CPU encode works, GPU would be ~5× faster.
- Ollama default `qwen2.5:3b` isn't installed; installed: qwen3:1.7b, gemma3:1b.
- Claude CLI picker errors were hidden (only stderr was shown) — now shows stdout too.
