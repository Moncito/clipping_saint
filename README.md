# clipper

Long stream or video in, vertical captioned clips out. Runs on your laptop (RTX 3050 Ti, 4 GB VRAM).

```
download -> transcribe -> candidates -> pick -> render
yt-dlp      faster-whisper  loudness +     Claude /   ffmpeg 9:16 + captions
            (GPU)           hype words     Ollama     (NVENC on the GPU)
```

Every stage saves its output in `jobs/<name>/`. If a step fails or you change your mind, rerun from that step. Nothing earlier gets redone.

## Setup (once)

```powershell
cd $HOME\Documents\clipper
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

This installs ffmpeg and Python packages and checks your GPU. Then use `.\.venv\Scripts\python.exe` (or activate the venv with `.\.venv\Scripts\activate`, then plain `python`).

## Daily use

```powershell
# whole pipeline from a link (YouTube / Twitch VOD / Kick) or a local file
python clip.py run "https://www.youtube.com/watch?v=XXXX" --count 8 --preset blur

# facecam streamer: stack the cam on top and the gameplay below
python clip.py run <url> --preset cam-bottom-right

# didn't like the picks? edit jobs/<name>/highlights.json (start/end/title), then:
python clip.py run <url> --from render

# try a different picker without re-transcribing
python clip.py run <url> --from pick --picker ollama
```

Finished clips are in `jobs/<name>/clips/clip_01.mp4`, and so on.

### Pickers (`--picker`)
| picker | quality | cost | needs |
|---|---|---|---|
| `claude` | best | your Claude plan | `claude` CLI (Claude Code) on PATH |
| `ollama` | ok | free, local | Ollama running + `ollama pull qwen2.5:3b` |
| `heuristic` | rough | free | nothing (just the loudest moments) |
| `auto` (default) | tries claude → ollama → heuristic | | |

### Layout presets (`presets.json`)
- `blur`: the whole frame, with a blurred copy behind it. Safe for anything.
- `center`: center crop of the frame. Good for gameplay with no facecam.
- `cam-bottom-right` / `cam-bottom-left` / `cam-top-right` / `cam-top-left`: facecam on top, gameplay below.

**New streamer?** Grab a still frame and measure their cam:
```powershell
python clip.py frame <url> --at 900
```
Copy one of the `cam-*` presets and set `"cam": [x, y, w, h]`. These are fractions of the frame: `[0.75, 0.70, 0.25, 0.30]` means the cam starts 75% across and 70% down, and is 25% wide and 30% tall. Or open Claude Code in this folder and ask it to look at the frame and make the preset for you.

## Campaign mode (briefs with required on-screen text, like LANY "Stuck")

Many music campaigns require **one approved line of text, word for word**, and nothing else on screen. Use `--text` to turn off auto captions and the hook. Use `--clips` to pick the moments yourself, which skips transcription and AI picking:

```powershell
python clip.py run "C:\path\to\lany_video.mp4" --preset center --clips "0:42-0:58,1:30-1:45" `
  --text "some part of me will always be yours" --cover 0,0.06,1,0.16
```

- `--cover x,y,w,h` draws a solid box (fractions of the 9:16 output) and puts your line inside it. Use it whenever the source has its own text: **if any original text is visible, the clip isn't paid.**
- Check `clips\clip_01_preview.png` after rendering. If the old text peeks out, make the box bigger or move it, then rerun.
- Avoid the `blur` preset when the source has text, because the blurred background shows a second copy of it. Use `center`.
- `--box-color white` gives black text on a white box.

## If something breaks
- **Out of GPU memory** during transcribe: add `--whisper-model small`.
- **"cublas64_12.dll not found"**: rerun `setup.ps1` (it installs the CUDA libs). The script adds them to the DLL path automatically.
- **NVENC error** on render: the script falls back to the CPU automatically. Updating your NVIDIA driver usually fixes it.
- **YouTube download fails**: `python -m pip install -U yt-dlp`, and install Deno (`winget install -e --id DenoLand.Deno`).
- **Tagalog/Taglish streams**: auto-detect usually works. Force it with `--lang tl` or `--lang en`.

## Rules that keep you paid
- Only clip content from campaigns that pay for it (Whop Content Rewards, and so on), and follow each campaign's brief.
- Watch every clip before posting. Upload by hand; don't auto-post.
