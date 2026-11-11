# clipper: notes for Claude Code

This is Sainty's clipping pipeline for paid clipping campaigns (Whop Content Rewards, etc.). It runs on Windows with an RTX 3050 Ti laptop GPU (4 GB VRAM) and 24 GB RAM. Use the venv: `.\.venv\Scripts\python.exe`.

## What lives where
- `clip.py`: the whole pipeline (stages: download, transcribe, candidates, pick, render). Read its docstring first.
- `presets.json`: vertical layout presets. `stack` layouts take `cam: [x, y, w, h]` as fractions of the source frame.
- `jobs/<name>/`: one folder per video: `job.json`, `audio.wav`, `transcript.json`, `candidates.json`, `pick_prompt.txt`, `highlights.json`, `clips/`.

## Common requests
- "Clip this link": `python clip.py run <url> --count 8 --preset <preset>`. Ask which streamer/layout if there's no preset for them yet.
- "Set up a preset for this streamer": run `python clip.py frame <url> --at 900`, look at the PNG, estimate the facecam box as fractions, and add a new `stack` preset to presets.json. Render one clip to check it.
- "Pick better clips": read `jobs/<name>/pick_prompt.txt` and `transcript.json`, then write `highlights.json` yourself (format: `{"picker": "claude-code", "clips": [{"id", "start", "end", "title", "reason"}]}`, times in absolute seconds, 20-55s each). Then `python clip.py run <src> --from render`.
- Changing a clip's timing or title: edit highlights.json, delete that clip's mp4 (or use `--from render`), rerender.

- Campaign briefs with required on-screen text: use `--text "<approved line, exact>" --clips "m:ss-m:ss" --preset center` plus `--cover x,y,w,h` if the footage has its own text. Look at `clips/clip_NN_preview.png` and confirm NO original text is visible; adjust the cover and rerender if it is.

## Constraints
- Only one GPU model at a time (4 GB VRAM). Whisper runs alone; the clip pickers run after it's unloaded.
- Never auto-post anything. Sainty reviews and uploads by hand.
- Only clip content from campaigns that allow it.
