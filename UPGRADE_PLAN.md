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

### Phase 3 — Reframe + polish  ✅ built
- `track` preset: face-tracking reframe for 16:9 videos. OpenCV YuNet face detector on the CPU
  (keeps the 4 GB GPU free), ~4 samples/s, follows the biggest face, "camera operator" smoothing
  (dead zone + 0.5s hold, 0.4s glides, instant cuts on shot changes). Falls back to blur when
  there's a face in < 25% of the clip. Works together with jump cuts.
- Post captions: the AI judge also writes TikTok / Reels / Shorts captions with hashtags
  (`--hashtags` adds required tags, e.g. a campaign's `#lany`), brand-safe rules in the prompt.
  Weak/duplicate AI captions fall back to templates. Saved as `clips/clip_NN_post.txt` and shown
  in the report with Copy buttons. Never auto-posted.

### Phase 4 — Learn from real results (the part Opus can't do for *you*)  ✅ built
- Log results: app **Results (learn)** tab, or `python clip.py result <job> <clip> <platform> --views ...`
  → `performance.csv` (gitignored, personal).
- `python clip.py learn --open` / **Learn + open dashboard**: views are log-scaled and compared
  within each platform; ridge regression re-weights Hook/Flow/Value/Trend, blended with the default
  equal weights (starts at 10 posted clips, full trust at 50). Saved to `calibration.json`; every
  new score uses it (report says "tuned to your real views").
- `results.html`: does the score predict views (rank correlation before/after), current weights,
  patterns in the measured signals (e.g. "less dead air got more views"), and every posted clip
  with its engagement rate vs the campaign minimum for its view tier.
- Verified on 40 synthetic results where views follow Hook: learned Hook 1.95×, Flow 1.05×,
  Value 0.43×, Trend 0.56×; correlation 0.49 → 0.75; flagged dead air as hurting.

### Phase 5 — Beat Opus Clip  ✅ built
Why scores sat at ≤60: every score came from the tiny local model (constant numbers), the judge
prompt told it to score harshly, dead air was measured before jump cuts removed it, openings were
quieter than average and clips ran 39-50s.
- **Diagnostics:** every run writes `jobs/<name>/run.log`; `clip.py test-ai` / app **Test AI** says
  exactly why Claude or Ollama can't judge.
- **Fair scoring:** Opus-style rubric anchors (17-20 per category = normal postable clip), signal
  scores mapped onto the same scale, dead air = only *quiet* gaps, measured on the clip as cut.
- **Hook-first starts:** the start slides (up to 8s) to the sentence with the strongest opening
  (loudness + hype words + talking in the first 2-3s). Picker prompt asks for it too.
- **Shorter clips:** `--target-len 35`; longer clips end at a sentence before target + 5s.
- **Zoom punch-ins:** up to 4 quick zooms per clip on loud peaks / hype words (`--no-zoom`).
- **Platform titles:** 3 options each for TikTok, Reels (hook line ≤60 chars) and Shorts
  (searchable title ≤90 chars), best first, Copy buttons in the report, saved in `clip_NN_post.txt`.
- **Claude sees the clip:** 3 frames per clip are given to Claude (Read tool) with the transcript.
- **A/B versions:** the 2 best clips also get `clip_NN_b.mp4`, a "payoff first" cold open. Log both
  (clip `3` and `3b`); the dashboard reports whether payoff-first wins for your audience.

## Housekeeping found along the way
- NVIDIA driver too old for NVENC (needs 610+): CPU encode works, GPU would be ~5× faster.
- Ollama default `qwen2.5:3b` isn't installed; installed: qwen3:1.7b, gemma3:1b.
- Claude CLI picker errors were hidden (only stderr was shown) — now shows stdout too.
