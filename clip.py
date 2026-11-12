#!/usr/bin/env python3
"""
clipper - turn a long stream / video into vertical, captioned short clips.

Pipeline (each stage saves its output in jobs/<name>/, so you can rerun from any stage):
  download   -> source.mp4          (yt-dlp, or a local file you pass in)
  transcribe -> transcript.json     (faster-whisper on the GPU, word timestamps)
  candidates -> candidates.json     (loudness spikes + hype keywords -> moments worth a look)
  pick       -> highlights.json     (picks ~2x the clips asked for, snaps them to sentences, pre-scores them,
                                     keeps the best --count)
  render     -> clips/*.mp4         (9:16 by preset - 'track' follows faces -, jump cuts, animated captions)
  score      -> report.html         (virality score 0-99: hook/flow/value/trend, clips ranked best first,
                                     plus ready-to-paste TikTok / Reels / Shorts captions per clip)

Usage:
  python clip.py run <url-or-file> [--preset blur] [--count 8] [--picker auto]
  python clip.py run <url-or-file> --from render        # re-render after editing highlights.json
  python clip.py frame <url-or-file> --at 600            # save a still to measure a facecam box
  python clip.py result <job> 3 tiktok --views 12000 --likes 900   # log how a posted clip did
  python clip.py learn --open                             # retune the score to your audience
"""
import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / "jobs"
PRESETS_FILE = ROOT / "presets.json"
STAGES = ["download", "transcribe", "candidates", "pick", "render", "score"]

# Make venv console scripts (yt-dlp.exe etc.) findable even when the venv isn't activated.
_SCRIPTS = Path(sys.executable).resolve().parent
os.environ["PATH"] = str(_SCRIPTS) + os.pathsep + os.environ.get("PATH", "")

# Extra yt-dlp auth args (cookies), filled in by main() from --cookies / --cookies-from-browser
# or from ./cookies.txt if it exists. Needed for age-restricted / members-only videos.
YTDLP_AUTH = []


def set_ytdlp_auth(args):
    cookies = getattr(args, "cookies", None)
    browser = getattr(args, "cookies_from_browser", None)
    if browser:
        YTDLP_AUTH[:] = ["--cookies-from-browser", browser]
    elif cookies:
        YTDLP_AUTH[:] = ["--cookies", str(Path(cookies).resolve())]
    elif (ROOT / "cookies.txt").exists():
        YTDLP_AUTH[:] = ["--cookies", str(ROOT / "cookies.txt")]

HYPE_WORDS = [
    "no way", "let's go", "lets go", "oh my god", "omg", "what the", "bro", "dude",
    "hahaha", "haha", "insane", "crazy", "clutch", "run", "wait wait", "oh no",
    "grabe", "tangina", "putang", "hala", "ayun", "gago", "lods", "idol",
]


FILLER_OPENERS = ["shout out", "shoutout", "subscribe", "sponsor", "link in", "link below", "welcome back",
                  "what's up guys", "whats up guys", "today we", "in this video", "before we start",
                  "like and subscribe", "hit the bell", "patreon"]


# ----------------------------------------------------------------- helpers
LOG_FILE = None  # set per job by cmd_run: every line also goes to jobs/<name>/run.log


def log(msg):
    print(f"[clipper] {msg}", flush=True)
    if LOG_FILE:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{msg}\n")
        except OSError:
            pass


def run(cmd, **kw):
    cmd = [str(c) for c in cmd]
    log(" ".join(cmd)[:240])
    return subprocess.run(cmd, check=True, **kw)


def add_cuda_dlls():
    """On Windows, pip's nvidia-cublas / nvidia-cudnn wheels put DLLs in site-packages/nvidia/*/bin.
    faster-whisper (CTranslate2) needs them on the DLL search path."""
    if os.name != "nt":
        return
    import site
    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass
    for sp in roots:
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def is_url(s):
    return bool(re.match(r"^https?://", s))


def slug(s):
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", s).strip("-")
    return s[:60] or "job"


def job_name_for(src, name=None):
    if name:
        return slug(name)
    if is_url(src):
        try:
            out = subprocess.run(["yt-dlp", *YTDLP_AUTH, "--print", "id", "--skip-download", "--no-warnings", src],
                                 capture_output=True, text=True, timeout=60)
            vid = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
            if vid:
                return slug(vid)
        except Exception:
            pass
        return "url-" + hashlib.md5(src.encode()).hexdigest()[:10]
    return slug(Path(src).stem)


def load_json(p, default=None):
    p = Path(p)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p, data):
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def fmt_t(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def source_path(job):
    meta = load_json(job / "job.json", {})
    return Path(meta.get("source", job / "source.mp4"))


# ----------------------------------------------------------------- stage 1: download
def stage_download(job, src, force):
    meta = load_json(job / "job.json", {})
    if not force and meta.get("source") and Path(meta["source"]).exists():
        log("download: already have source, skipping")
        return
    if is_url(src):
        run(["yt-dlp", *YTDLP_AUTH, "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
             "--merge-output-format", "mp4", "-o", str(job / "source.%(ext)s"), src])
        found = sorted(job.glob("source.*"))
        found = [f for f in found if f.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")]
        if not found:
            sys.exit("download failed: no video file appeared")
        save_json(job / "job.json", {"source": str(found[0]), "url": src})
    else:
        p = Path(src).resolve()
        if not p.exists():
            sys.exit(f"file not found: {src}")
        save_json(job / "job.json", {"source": str(p)})
    log(f"download: source = {load_json(job / 'job.json')['source']}")


# ----------------------------------------------------------------- stage 2: transcribe
def extract_audio(job, force=False):
    wav = job / "audio.wav"
    if force or not wav.exists():
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", source_path(job),
             "-vn", "-ac", "1", "-ar", "16000", wav])
    return wav


def stage_transcribe(job, args, force):
    out = job / "transcript.json"
    if out.exists() and not force:
        log("transcribe: already done, skipping")
        return
    wav = extract_audio(job, force)
    add_cuda_dlls()
    from faster_whisper import WhisperModel  # imported late so other stages work without it

    try:
        model = WhisperModel(args.whisper_model, device="cuda", compute_type="int8_float16")
        log(f"transcribe: {args.whisper_model} on GPU")
    except Exception as e:  # no CUDA / missing DLLs -> CPU (slow but works)
        log(f"transcribe: GPU unavailable ({e}); falling back to CPU int8 - this will be slow")
        model = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")

    segments, info = model.transcribe(str(wav), word_timestamps=True, vad_filter=True,
                                      language=args.lang or None, beam_size=1)
    segs = []
    for s in segments:
        words = [{"w": w.word.strip(), "s": round(w.start, 2), "e": round(w.end, 2)}
                 for w in (s.words or []) if w.word.strip()]
        segs.append({"s": round(s.start, 2), "e": round(s.end, 2), "text": s.text.strip(), "words": words})
        if len(segs) % 50 == 0:
            log(f"transcribe: {fmt_t(s.end)} done")
    save_json(out, {"language": info.language, "segments": segs})
    log(f"transcribe: {len(segs)} segments, language={info.language}")
    del model  # free VRAM before anything else touches the GPU


# ----------------------------------------------------------------- stage 3: candidates
def loudness_curve(wav_path, hop=0.5):
    import numpy as np
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    n = int(sr * hop)
    frames = len(data) // n
    if frames == 0:
        return np.zeros(1), hop
    rms = np.sqrt(np.mean(data[: frames * n].reshape(frames, n) ** 2, axis=1) + 1e-10)
    db = 20 * np.log10(rms)
    k = 5  # smooth over ~2.5s
    db = np.convolve(db, np.ones(k) / k, mode="same")
    return db, hop


def text_between(segs, a, b):
    return " ".join(s["text"] for s in segs if s["e"] > a and s["s"] < b)


def stage_candidates(job, args, force):
    out = job / "candidates.json"
    if out.exists() and not force:
        log("candidates: already done, skipping")
        return
    import numpy as np
    segs = load_json(job / "transcript.json", {"segments": []})["segments"]
    wav = extract_audio(job)
    db, hop = loudness_curve(wav)
    dur = len(db) * hop
    z = (db - np.median(db)) / (np.std(db) + 1e-6)

    # hype-word bonus per time bucket
    bonus = np.zeros_like(z)
    for s in segs:
        t = s["text"].lower()
        hits = sum(t.count(w) for w in HYPE_WORDS)
        if hits:
            i0, i1 = int(s["s"] / hop), max(int(s["e"] / hop), int(s["s"] / hop) + 1)
            bonus[i0:i1] += 0.6 * hits
    score = z + bonus

    # greedy peak picking with minimum spacing
    min_gap = int(40 / hop)
    order = np.argsort(-score)
    taken = []
    for i in order:
        if len(taken) >= args.max_candidates:
            break
        if all(abs(i - j) >= min_gap for j in taken):
            taken.append(int(i))

    cands = []
    for i in sorted(taken):
        t = i * hop
        a, b = max(0.0, t - 35), min(dur, t + 25)
        cands.append({
            "id": len(cands) + 1,
            "peak": round(t, 1),
            "window": [round(a, 1), round(b, 1)],
            "score": round(float(score[i]), 2),
            "text": text_between(segs, a, b)[:900],
        })
    save_json(out, {"duration": round(dur, 1), "candidates": cands})
    log(f"candidates: {len(cands)} moments from {fmt_t(dur)} of video")


# ----------------------------------------------------------------- smart editing (phase 2)
def sentence_spans(words, gap=0.8):
    """Group words into sentences: break on . ! ? or a pause longer than `gap`."""
    spans, cur = [], []
    for w in words:
        if cur and w["s"] - cur[-1]["e"] > gap:
            spans.append((cur[0]["s"], cur[-1]["e"]))
            cur = []
        cur.append(w)
        if re.search(r"[.!?\u2026]$", w["w"].strip()):
            spans.append((cur[0]["s"], cur[-1]["e"]))
            cur = []
    if cur:
        spans.append((cur[0]["s"], cur[-1]["e"]))
    return spans


def snap_to_sentences(s, e, spans, min_len, max_len, dur):
    """Move start to the start of a sentence and end to the end of one, so clips never open or close mid-line."""
    if not spans:
        return s, e
    inside = [sp for sp in spans if sp[0] < s < sp[1]]
    if inside and s - inside[0][0] <= 4:
        s = inside[0][0]                     # back up to the start of the sentence we cut into
    else:
        nxt = [sp[0] for sp in spans if sp[0] >= s - 0.01]
        if nxt and nxt[0] - s <= 3:
            s = nxt[0]                       # skip the dead lead-in
    s = max(0.0, s - 0.25)
    inside = [sp for sp in spans if sp[0] < e < sp[1]]
    if inside:
        if inside[0][1] - s <= max_len and inside[0][1] - e <= 4:
            e = inside[0][1]                 # let the sentence finish
        else:
            ends = [sp[1] for sp in spans if sp[1] <= e and sp[1] - s >= min_len]
            if ends:
                e = ends[-1]                 # or stop after the previous one
    e = e + 0.35
    if dur:
        e = min(dur, e)
    return round(s, 2), round(e, 2)


def quiet_cuts(s, e, words, z=None, hop=0.25, gap=1.0, pad=0.25):
    """Pauses inside [s, e] worth cutting: no words for > gap seconds AND not loud (loud = gameplay action).
    Returns keep-intervals relative to s, and total seconds removed."""
    ws = [w for w in words if w["s"] >= s and w["e"] <= e]
    cuts = []
    for a, b in zip(ws, ws[1:]):
        c0, c1 = a["e"] + pad, b["s"] - pad
        if b["s"] - a["e"] <= gap or c1 - c0 < 0.3:
            continue
        if z is not None:
            seg = z[int(c0 / hop): max(int(c1 / hop), int(c0 / hop) + 1)]
            if len(seg) and float(seg.mean()) > 0.3:
                continue                     # something loud is happening, keep it
        cuts.append((c0 - s, c1 - s))
    keeps, t0 = [], 0.0
    for c0, c1 in cuts:
        if c0 - t0 > 0.15:
            keeps.append((t0, c0))
        t0 = c1
    if (e - s) - t0 > 0.15:
        keeps.append((t0, e - s))
    removed = sum(c1 - c0 for c0, c1 in cuts)
    return keeps, removed


def hook_start(s, e, spans, words, z, hop, min_len):
    """Slide the start to the sentence (within the first 8s) with the strongest opening:
    loud audio + hype words + someone talking in its first 2-3 seconds."""
    def strength(t0):
        seg = z[int(t0 / hop): int((t0 + 2) / hop) + 1] if z is not None else []
        loud = float(seg.mean()) if len(seg) else 0.0
        early = [w for w in words if t0 <= w["s"] < t0 + 3]
        hype = sum(" ".join(w["w"] for w in early).lower().count(h) for h in HYPE_WORDS)
        talk = min(len([w for w in early if w["s"] < t0 + 2]), 5) / 5
        text = " ".join(w["w"] for w in early).lower()
        filler = any(f in text for f in FILLER_OPENERS)   # intros / sponsor lines kill the hook
        return loud + 0.8 * hype + 0.25 * talk - (1.5 if filler else 0.0)
    best_t, best = s, strength(s)
    for a, _ in spans:
        if s < a <= s + 8 and e - a >= min_len:
            v = strength(a)
            if v > best + 0.25:
                best_t, best = a, v
    return round(max(0.0, best_t - 0.2), 2) if best_t != s else s


def trim_to_target(s, e, spans, min_len, target):
    """If a clip runs long, end it after the last sentence that finishes before target + 5s."""
    if e - s <= target + 5:
        return e
    ends = [b for _, b in spans if s + min_len <= b <= s + target + 5]
    return round(ends[-1] + 0.35, 2) if ends else e


def punch_times(s, e, words, z, hop, keeps=None):
    """Up to 4 moments in the clip worth a quick zoom: loud peaks and hype words, at least 4s apart."""
    cands = []
    if z is not None:
        seg = z[int(s / hop): int(e / hop)]
        if len(seg):
            med = float(sorted(seg)[len(seg) // 2])
            for i in range(1, len(seg) - 1):
                if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] - med > 1.0:
                    cands.append((float(seg[i] - med), i * hop))
    for w in words:
        if s <= w["s"] <= e and any(h in w["w"].lower() for h in HYPE_WORDS):
            cands.append((1.5, w["s"] - s))
    out = []
    for _, t0 in sorted(cands, reverse=True):
        if t0 < 1.0 or t0 > e - s - 0.8 or any(abs(t0 - x) < 4 for x in out):
            continue
        out.append(t0)
        if len(out) == 4:
            break
    if keeps:
        out = [remap_time(x, keeps) for x in out]
    return sorted(out)


def zoom_filter(times, label_in, label_out, amount=0.10):
    """Quick 'punch-in' zoom around each time (in output seconds), centred."""
    pulse = "+".join(f"clip((t-{p:.2f})/0.08\\,0\\,1)*clip(({p + 0.55:.2f}-t)/0.2\\,0\\,1)" for p in times)
    f = f"(1+{amount}*({pulse}))"
    return (f";[{label_in}]scale=w={W}*{f}:h={H}*{f}:eval=frame,"
            f"crop={W}:{H}:(iw-{W})/2:(ih-{H})/2[{label_out}]")


def remap_time(t, keeps):
    """Time in the original clip -> time in the jump-cut clip."""
    out = 0.0
    for a, b in keeps:
        if t <= a:
            return out
        if t <= b:
            return out + (t - a)
        out += b - a
    return out


# ----------------------------------------------------------------- stage 4: pick
PICK_PROMPT = """You are an expert short-form video editor picking viral clips (TikTok / Reels / Shorts) from a long gaming stream.

Below are {n} candidate moments. Each has a time window in seconds (absolute, from the start of the video), a loudness/hype score, and the transcript of that window.

Pick the {k} best moments. Prefer: a clear setup and payoff, funny or shocking reactions, clutch plays, strong emotion, something a stranger understands with no context. Avoid: dead air, menu screens, inside jokes, moments that need prior context.

For each pick, choose a precise start and end INSIDE its window:
- start 1-2 seconds before the setup begins, so the clip hooks immediately
- end right after the payoff/reaction lands (no trailing dead time)
- length between {min_len} and {max_len} seconds, ideally 20-35 seconds (short clips get rewatched)
- the first 2 seconds must already be interesting: start ON the strongest line or reaction, not on build-up

Reply with ONLY this JSON, no other text:
{{"clips": [{{"id": <candidate id>, "start": <seconds>, "end": <seconds>, "title": "<on-screen hook, max 7 words>", "reason": "<one short line>"}}]}}

CANDIDATES:
{body}
"""


def build_prompt(cands, args, k=None):
    body = "\n\n".join(
        f"[id {c['id']}] window {c['window'][0]}-{c['window'][1]}s ({fmt_t(c['window'][0])}-{fmt_t(c['window'][1])}), "
        f"score {c['score']}\n{c['text'] or '(no speech - loud gameplay moment)'}"
        for c in cands)
    return PICK_PROMPT.format(n=len(cands), k=k or args.count, min_len=args.min_len, max_len=args.max_len, body=body)


def parse_clips(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in model output")
    data = json.loads(m.group(0))
    return data["clips"] if isinstance(data, dict) else data


def pick_with_claude(prompt):
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("claude CLI not found on PATH")
    out = subprocess.run([exe, "-p", "Follow the instructions in the input exactly and reply with JSON only."],
                         input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=600)
    if out.returncode != 0:
        raise RuntimeError((out.stderr.strip() or out.stdout.strip())[:300])
    return parse_clips(out.stdout)


def pick_with_ollama(prompt, model):
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json",
                         "options": {"temperature": 0.3, "num_ctx": 8192}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.loads(r.read().decode())
    return parse_clips(resp.get("response", ""))


def pick_heuristic(cands, args, k=None):
    best = sorted(cands, key=lambda c: -c["score"])[: k or args.count]
    clips = []
    for c in best:
        L = (args.min_len + args.max_len) / 2
        start = max(c["window"][0], c["peak"] - L * 0.65)
        clips.append({"id": c["id"], "start": start, "end": start + L,
                      "title": "", "reason": f"loudness score {c['score']}"})
    return clips


def clean_clips(clips, cands, args, dur, limit=None):
    by_id = {c["id"]: c for c in cands}
    good = []
    for c in clips:
        try:
            s, e = float(c["start"]), float(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e - s < args.min_len:
            e = s + args.min_len
        if e - s > args.max_len:
            e = s + args.max_len
        s, e = max(0.0, s), min(dur, e) if dur else e
        if e - s < 5:
            continue
        good.append({"id": c.get("id"), "start": round(s, 2), "end": round(e, 2),
                     "title": str(c.get("title", ""))[:60], "reason": str(c.get("reason", ""))[:160],
                     "window": by_id.get(c.get("id"), {}).get("window")})
    good.sort(key=lambda c: c["start"])
    # drop clips that overlap an earlier one by more than half
    out = []
    for c in good:
        if out and c["start"] < out[-1]["end"] - 0.5 * (c["end"] - c["start"]):
            continue
        out.append(c)
    return out[: limit or args.count]


def stage_pick(job, args, force):
    out = job / "highlights.json"
    if out.exists() and not force:
        log("pick: highlights.json exists, skipping (edit it by hand or use --from pick)")
        return
    cdata = load_json(job / "candidates.json")
    cands, dur = cdata["candidates"], cdata["duration"]
    if not cands:
        sys.exit("no candidates found")
    pool = max(args.count, int(round(args.count * args.pool)))
    prompt = build_prompt(cands, args, pool)
    (job / "pick_prompt.txt").write_text(prompt, encoding="utf-8")

    order = {"auto": ["claude", "ollama", "heuristic"], "claude": ["claude"],
             "ollama": ["ollama"], "heuristic": ["heuristic"]}[args.picker]
    clips, used = None, None
    for p in order:
        try:
            if p == "claude":
                clips = pick_with_claude(prompt)
            elif p == "ollama":
                clips = pick_with_ollama(prompt, args.ollama_model)
            else:
                clips = pick_heuristic(cands, args, pool)
            used = p
            break
        except Exception as e:
            log(f"pick: {p} failed ({e})")
    if clips is None:
        sys.exit("pick: every picker failed")
    clips = clean_clips(clips, cands, args, dur, limit=pool)
    words = [w for s in load_json(job / "transcript.json", {"segments": []})["segments"] for w in s.get("words", [])]
    spans = sentence_spans(words)
    z = None
    if (job / "audio.wav").exists():
        import scorer
        z, _ = scorer.loudness(job / "audio.wav", 0.25)
    for c in clips:
        c["start"], c["end"] = snap_to_sentences(c["start"], c["end"], spans, args.min_len, args.max_len, dur)
        s0 = c["start"]
        c["start"] = hook_start(c["start"], c["end"], spans, words, z, 0.25, args.min_len)
        c["end"] = trim_to_target(c["start"], c["end"], spans, args.min_len, args.target_len)
        if c["start"] != s0:
            log(f"pick: hook-first start {fmt_t(s0)} -> {fmt_t(c['start'])}")
    clips.sort(key=lambda c: c["start"])
    clips = [c for i, c in enumerate(clips) if i == 0 or c["start"] >= clips[i - 1]["end"] - 1]
    scored_by = None
    if len(clips) > args.count and not args.no_score:
        import scorer
        log(f"pick: pre-scoring {len(clips)} picks to keep the best {args.count}")
        try:
            scored_by = scorer.score_clips(job, clips, used, args.judge, args.ollama_model, log=log,
                                           hashtags=args.hashtags, jump_cuts=not args.no_jump_cuts)
            ranked = sorted(clips, key=lambda c: -c["score"]["total"])
            for c in ranked[args.count:]:
                log(f"pick: dropped {fmt_t(c['start'])}-{fmt_t(c['end'])} (score {c['score']['total']})")
            clips = sorted(ranked[: args.count], key=lambda c: c["start"])
        except Exception as e:
            log(f"pick: pre-score failed ({e}), keeping the first {args.count}")
    clips = clips[: args.count]
    data = {"picker": used, "clips": clips}
    if scored_by and all("score" in c for c in clips):
        data["scored_by"] = scored_by
    save_json(out, data)
    log(f"pick: {len(clips)} clips chosen by {used}")
    for i, c in enumerate(clips, 1):
        log(f"  {i:02d}. {fmt_t(c['start'])}-{fmt_t(c['end'])}  {c['title']}  ({c['reason']})")


# ----------------------------------------------------------------- stage 5: render
W, H = 1080, 1920


def video_filter(preset, track=None):
    layout = preset.get("layout", "blur")
    if layout == "track" and track:  # follow the main face (see reframe.py)
        import reframe
        return (f"[0:v]crop=ih*9/16:ih:{reframe.crop_x_expr(track)}:0,"
                f"scale={W}:{H},setsar=1[base]")
    if layout == "center":
        x = preset.get("x", 0.5)  # horizontal focus 0..1
        return f"[0:v]crop=ih*9/16:ih:(iw-ih*9/16)*{x}:0,scale={W}:{H},setsar=1[base]"
    if layout == "stack":
        cx, cy, cw, ch = preset["cam"]  # facecam box as fractions of the source frame
        top = int(preset.get("top_height", 640))
        bot = H - top
        gx = preset.get("game_x", 0.5)
        return (f"[0:v]split[c][g];"
                f"[c]crop=iw*{cw}:ih*{ch}:iw*{cx}:ih*{cy},scale={W}:{top}:force_original_aspect_ratio=increase,"
                f"crop={W}:{top}[cam];"
                f"[g]crop=ih*{W}/{bot}:ih:(iw-ih*{W}/{bot})*{gx}:0,scale={W}:{bot}[game];"
                f"[cam][game]vstack,setsar=1[base]")
    # blur: whole frame fit to width, blurred zoomed copy behind it
    return (f"[0:v]split[a][b];"
            f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=20:2[bg];"
            f"[b]scale={W}:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]")


def ass_time(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def ass_escape(s):
    return s.replace("\\", "").replace("{", "(").replace("}", ")")


def parse_cover(spec):
    """'x,y,w,h' as fractions of the 1080x1920 output -> pixel box."""
    x, y, w, h = [float(v) for v in spec.split(",")]
    return int(x * W), int(y * H), int(w * W), int(h * H)


def write_overlay_ass(path, text, clip_len, cover, box_color):
    """Campaign mode: ONE fixed line of text for the whole clip.
    With a cover box, the text sits centered inside the solid box; without one, it gets its own opaque band near the top."""
    dark_box = box_color == "black"
    fg = "&H00FFFFFF" if dark_box else "&H00000000"
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: InBox,Arial,58,{fg},{fg},&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,5,60,60,0,1
Style: Band,Arial,60,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,3,22,0,8,80,80,200,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    t = ass_escape(text)
    if cover:
        x, y, w, h = cover
        ml, mr = x + 30, W - (x + w) + 30
        line = f"Dialogue: 0,{ass_time(0)},{ass_time(clip_len + 1)},InBox,,{ml},{mr},0,,{{\\pos({x + w // 2},{y + h // 2})}}{t}"
    else:
        line = f"Dialogue: 0,{ass_time(0)},{ass_time(clip_len + 1)},Band,,0,0,0,,{t}"
    Path(path).write_text(head + line + "\n", encoding="utf-8")


CAPTION_POP_WORDS = ["no way", "what", "omg", "insane", "crazy", "clutch", "holy", "bro", "wow", "grabe", "hala"]


def write_ass(path, words, title, clip_len, preset):
    cap_y = int(preset.get("caption_margin_v", 620))  # distance from bottom
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial,92,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,3,2,60,60,{cap_y},1
Style: Hook,Arial,68,&H00FFFFFF,&H00FFFFFF,&H00000000,&HC0000000,-1,0,0,0,100,100,0,0,3,18,0,8,70,70,170,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    if title:
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(min(3.0, clip_len))},Hook,,0,0,0,,{ass_escape(title.upper())}")
    # chunks of up to 3 words; current word highlighted yellow
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) == 3 or w["s"] - cur[-1]["e"] > 0.6 or re.search(r"[.!?,\u2026]$", cur[-1]["w"].strip())):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    for k, ch in enumerate(chunks):
        next_start = chunks[k + 1][0]["s"] if k + 1 < len(chunks) else clip_len
        for i, w in enumerate(ch):
            start = w["s"]
            # never overlap the next chunk, or two caption lines draw on top of each other
            end = ch[i + 1]["s"] if i + 1 < len(ch) else min(w["e"] + 0.15, next_start)
            if end <= start:
                continue
            parts = []
            for j, x in enumerate(ch):
                txt = ass_escape(x["w"].upper())
                if j == i:   # active word: yellow (green for hype words) with a quick pop
                    col = "&H50FF50&" if any(h in x["w"].lower() for h in CAPTION_POP_WORDS) else "&H00FFFF&"
                    parts.append("{\\c" + col + "\\fscx112\\fscy112\\t(0,90,\\fscx100\\fscy100)}" + txt
                                 + "{\\c&HFFFFFF&\\fscx100\\fscy100}")
                else:
                    parts.append(txt)
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(min(end, clip_len))},Cap,,0,0,0,,{' '.join(parts)}")
    Path(path).write_text(head + "\n".join(lines) + "\n", encoding="utf-8")


ENCODERS = {
    "nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0"],
    "x264": ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"],
}


def stage_render(job, args, force):
    hl = load_json(job / "highlights.json")
    if not hl or not hl["clips"]:
        sys.exit("render: no highlights.json - run the pick stage first")
    segs = load_json(job / "transcript.json", {"segments": []})["segments"]
    presets = load_json(PRESETS_FILE, {})
    if args.preset not in presets:
        sys.exit(f"unknown preset '{args.preset}'. Available: {', '.join(presets)}")
    preset = presets[args.preset]
    src = source_path(job)
    outdir = job / "clips"
    outdir.mkdir(exist_ok=True)
    encoder = "nvenc" if not args.cpu_encode else "x264"

    all_words = [w for s in segs for w in s.get("words", [])]
    z = None
    use_cuts = bool(not args.text and not args.no_jump_cuts and all_words)
    use_zoom = bool(not args.text and not args.no_zoom)
    if (use_cuts or use_zoom) and (job / "audio.wav").exists():
        import scorer
        z, _ = scorer.loudness(job / "audio.wav", 0.25)
    for n, c in enumerate(hl["clips"], 1):
        name = f"clip_{n:02d}"
        mp4 = outdir / f"{name}.mp4"
        if mp4.exists() and not force:
            log(f"render: {mp4.name} exists, skipping")
            continue
        s, e = c["start"], c["end"]
        words = [{"w": w["w"], "s": w["s"] - s, "e": w["e"] - s} for w in all_words if w["s"] >= s and w["e"] <= e]
        ass = outdir / f"{name}.ass"
        amap, out_len = "0:a?", e - s
        keeps, removed = (quiet_cuts(s, e, all_words, z) if use_cuts else ([], 0.0))
        cutting = removed >= 0.5 and len(keeps) > 1
        track = None
        if preset.get("layout") == "track":
            import reframe
            track = reframe.face_track(src, s, e, log=log)
            if track and cutting:
                track = [(round(remap_time(tk, keeps), 2), x) for tk, x in track]
            c["reframe"] = "face" if track else "blur (no face)"
        graph = video_filter(preset, track)
        if cutting:
            expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in keeps)
            graph = (f"[0:v]select='{expr}',setpts=N/FRAME_RATE/TB[vcut];"
                     f"[0:a]aselect='{expr}',asetpts=N/SR/TB[acut];" + graph.replace("[0:v]", "[vcut]", 1))
            amap = "[acut]"
            words = [{"w": w["w"], "s": remap_time(w["s"], keeps), "e": remap_time(w["e"], keeps)} for w in words]
            out_len = sum(b - a for a, b in keeps)
        c["cut_seconds"] = round(removed if amap == "[acut]" else 0.0, 1)
        c["keeps"] = [[round(a, 3), round(b, 3)] for a, b in keeps] if cutting else []
        bl = "base"
        if use_zoom:
            punches = punch_times(s, e, all_words, z, 0.25, keeps if cutting else None)
            c["zooms"] = [round(x, 2) for x in punches]
            if punches:
                graph += zoom_filter(punches, "base", "basez")
                bl = "basez"
        if args.text:  # campaign mode: one approved line, no auto captions, no hook
            cover = parse_cover(args.cover) if args.cover else None
            write_overlay_ass(ass, args.text, e - s, cover, args.box_color)
            if cover:
                x, y, w, h = cover
                graph += f";[{bl}]drawbox=x={x}:y={y}:w={w}:h={h}:color={args.box_color}@1:t=fill[base2]"
                graph += f";[base2]subtitles={ass.name}[v]"
            else:
                graph += f";[{bl}]subtitles={ass.name}[v]"
        else:
            write_ass(ass, words if not args.no_captions else [], "" if args.no_hook else c.get("title", ""), out_len, preset)
            graph += f";[{bl}]subtitles={ass.name}[v]"
        base = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{s:.2f}", "-t", f"{e - s:.2f}", "-i", str(src),
                "-filter_complex", graph, "-map", "[v]", "-map", amap, "-r", "30"]
        tail = ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", mp4.name]
        try:
            run(base + ENCODERS[encoder] + tail, cwd=outdir)  # cwd so the .ass path needs no escaping
        except subprocess.CalledProcessError:
            if encoder == "nvenc":
                log("render: NVENC failed, switching to CPU x264 for the rest")
                encoder = "x264"
                run(base + ENCODERS[encoder] + tail, cwd=outdir)
            else:
                raise
        # still frame so you (or Claude Code) can check the text fully covers any original text
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{min(1.0, (e - s) / 2):.2f}", "-i", mp4.name,
                        "-frames:v", "1", "-vf", "scale=540:-1", f"{name}_preview.png"], cwd=outdir)
        cut = f", {c['cut_seconds']:.0f}s of pauses cut" if c.get("cut_seconds") else ""
        zm = f", {len(c.get('zooms', []))} zooms" if c.get("zooms") else ""
        log(f"render: {mp4.name}  ({out_len:.0f}s{cut}{zm})  {c.get('title', '')}")
        c["out_len"] = round(out_len, 2)
        c["_words_rendered"] = words  # for version B's teaser (dropped before saving)
    if not args.text and args.variants > 0:
        render_variants(job, hl, z, args, encoder)
    for c in hl["clips"]:
        c.pop("_words_rendered", None)
    save_json(job / "highlights.json", hl)
    log(f"done -> {outdir}")


def render_variants(job, hl, z, args, encoder):
    """Version B of the best clips: the payoff moment plays first (a 2s 'cold open'), then the clip.
    Post both; the learning loop (learn.py) finds out which style works for your audience."""
    clips = list(enumerate(hl["clips"], 1))
    ranked = sorted(clips, key=lambda t: -t[1].get("score", {}).get("total", 0))[: args.variants]
    for n, c in ranked:
        name = f"clip_{n:02d}"
        src, out = job / "clips" / f"{name}.mp4", job / "clips" / f"{name}_b.mp4"
        if not src.exists():
            continue
        L = c.get("out_len") or (c["end"] - c["start"])
        # payoff = loudest 2s in the back 60% of the clip (on the rendered timeline)
        best, peak = -9.0, None
        if z is not None:
            keeps = c.get("keeps") or [[0.0, c["end"] - c["start"]]]
            for k in range(int(L * 0.4 / 0.25), int((L - 1.5) / 0.25)):
                tt = k * 0.25
                # map rendered time back to source time to read the loudness
                acc, srcs = 0.0, None
                for a, b in keeps:
                    if tt <= acc + (b - a):
                        srcs = c["start"] + a + (tt - acc)
                        break
                    acc += b - a
                if srcs is None:
                    continue
                seg = z[int(srcs / 0.25): int((srcs + 2) / 0.25)]
                if len(seg) and float(seg.mean()) > best:
                    best, peak = float(seg.mean()), tt
        if peak is None:
            peak = L * 0.7
        # better: the payoff line the AI judge pointed at, found in the transcript
        words = c.get("_words_rendered") or []
        pl = (c.get("score", {}).get("payoff") or "").lower().split()
        if len(pl) >= 2 and words:
            ws = [w["w"].lower().strip(".,!?\"'") for w in words]
            for i in range(len(ws) - 1):
                if ws[i] == pl[0].strip(".,!?\"'") and ws[i + 1] == pl[1].strip(".,!?\"'"):
                    peak = words[i]["s"] + 0.3
                    break
        a, b = max(0.0, peak - 0.3), min(L, peak + 1.9)
        # snap the teaser to whole words: start at a word start, end after a word ends (max ~3s)
        if words:
            starts = [w["s"] for w in words if a - 0.6 <= w["s"] <= a + 0.6]
            if starts:
                a = max(0.0, min(starts, key=lambda x: abs(x - a)) - 0.1)
            ends = [w["e"] for w in words if a + 1.2 <= w["e"] <= a + 3.0]
            if ends:
                b = min(L, ends[-1] + 0.15)
        txt = " ".join(w["w"] for w in words if a <= w["s"] < b).lower()
        if any(f in txt for f in FILLER_OPENERS):
            log(f"render: skipped version B of {name} (its best moment is an intro / sponsor line)")
            continue
        graph = (f"[0:v]split[v1][v2];[0:a]asplit[a1][a2];"
                 f"[v1]trim={a:.2f}:{b:.2f},setpts=PTS-STARTPTS[tv];[a1]atrim={a:.2f}:{b:.2f},asetpts=PTS-STARTPTS[ta];"
                 f"[v2]setpts=PTS-STARTPTS[mv];[a2]asetpts=PTS-STARTPTS[ma];"
                 f"[tv][ta][mv][ma]concat=n=2:v=1:a=1[v][a]")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src.name, "-filter_complex", graph,
               "-map", "[v]", "-map", "[a]", "-r", "30"] + ENCODERS[encoder] + \
              ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out.name]
        try:
            run(cmd, cwd=job / "clips")
        except subprocess.CalledProcessError:
            run(cmd[:cmd.index("-r") + 2] + ENCODERS["x264"] + cmd[cmd.index("-c:a"):], cwd=job / "clips")
        c["variant_b"] = {"file": out.name, "teaser": [round(a, 2), round(b, 2)]}
        log(f"render: {out.name}  (version B: payoff first, {b - a:.1f}s teaser)")


# ----------------------------------------------------------------- stage 6: score
def stage_score(job, args, force):
    if getattr(args, "no_score", False):
        return
    import scorer
    try:
        scorer.score_job(job, ai=args.judge, ollama_model=args.ollama_model, log=log, hashtags=args.hashtags,
                         jump_cuts=not (args.text or args.no_jump_cuts),
                         rescore=force and args.from_stage == "score")  # reuse pick-stage scores
    except Exception as e:  # scoring is a bonus; never lose the rendered clips over it
        log(f"score: failed ({e})")


# ----------------------------------------------------------------- commands
def to_sec(t):
    parts = [float(p) for p in t.strip().split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return sec


def parse_manual_clips(spec):
    """'0:45-1:02, 2:10-2:25' -> clip list."""
    clips = []
    for i, rng in enumerate(spec.split(","), 1):
        a, b = rng.split("-")
        s, e = to_sec(a), to_sec(b)
        if e <= s:
            sys.exit(f"bad clip range: {rng}")
        clips.append({"id": i, "start": s, "end": e, "title": "", "reason": "manual"})
    return clips


def cmd_run(args):
    JOBS.mkdir(exist_ok=True)
    job = JOBS / job_name_for(args.source, args.name)
    job.mkdir(parents=True, exist_ok=True)
    global LOG_FILE
    LOG_FILE = job / "run.log"
    import datetime
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M} clip.py {' '.join(sys.argv[1:])}\n")
    log(f"job folder: {job}")
    start = STAGES.index(args.from_stage) if args.from_stage else len(STAGES)
    fns = {"download": lambda f: stage_download(job, args.source, f),
           "transcribe": lambda f: stage_transcribe(job, args, f),
           "candidates": lambda f: stage_candidates(job, args, f),
           "pick": lambda f: stage_pick(job, args, f),
           "render": lambda f: stage_render(job, args, f),
           "score": lambda f: stage_score(job, args, f)}
    if args.clips:  # manual times given: no transcription or AI picking needed
        stage_download(job, args.source, start == 0)
        manual = parse_manual_clips(args.clips)
        if args.text:  # campaign: the approved line doubles as the clip's title / caption base
            for c in manual:
                c["title"] = args.text
        save_json(job / "highlights.json", {"picker": "manual", "clips": manual})
        stage_render(job, args, True)
        stage_score(job, args, True)
        return
    for i, st in enumerate(STAGES):
        fns[st](i >= start)
        if args.until and st == args.until:
            log(f"stopped after {st} (--until)")
            break


def cmd_test_ai(args):
    """Check that the AI picker/judge can actually run, and say why not if it can't."""
    import scorer
    log("test-ai: asking Claude (claude -p) for a tiny JSON reply...")
    try:
        out = scorer.ask_claude('Reply with exactly this JSON and nothing else: {"ok": true}', timeout=180)
        log(f"test-ai: Claude OK -> {out.strip()[:80]}")
    except Exception as e:
        log(f"test-ai: Claude FAILED -> {e}")
        log("test-ai: fixes: run 'claude' once in a normal terminal and log in; don't start the app from inside "
            "Claude Code; make sure 'claude --version' works in PowerShell")
    try:
        models = scorer.ollama_models()
        log(f"test-ai: Ollama OK, models: {', '.join(models) or '(none installed)'}")
    except Exception as e:
        log(f"test-ai: Ollama not reachable ({e}). Start the Ollama app if you want the local fallback.")


def cmd_result(args):
    import learn
    n = learn.add_result(args.job, args.clip, args.platform, args.views, args.likes, args.comments, args.shares,
                         args.url or "")
    log(f"result saved ({n} logged in total). Run 'python clip.py learn' to update the score weights.")


def cmd_learn(args):
    import learn
    learn.learn(log=log)
    if args.open:
        os.startfile(str(learn.RESULTS)) if os.name == "nt" else None


def cmd_frame(args):
    JOBS.mkdir(exist_ok=True)
    job = JOBS / job_name_for(args.source, args.name)
    job.mkdir(parents=True, exist_ok=True)
    stage_download(job, args.source, False)
    out = job / f"frame_{int(args.at)}.png"
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(args.at), "-i", source_path(job), "-frames:v", "1", out])
    log(f"saved {out}  - measure the facecam box (x, y, w, h as fractions of width/height) for presets.json")


def main():
    ap = argparse.ArgumentParser(description="Long video -> vertical captioned clips")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument("source", help="YouTube/Twitch/Kick URL or a local video file")
    r.add_argument("--name", help="job folder name (default: video id / file name)")
    r.add_argument("--preset", default="blur", help="layout preset from presets.json")
    r.add_argument("--count", type=int, default=8, help="how many clips to make")
    r.add_argument("--min-len", type=float, default=15)
    r.add_argument("--max-len", type=float, default=55)
    r.add_argument("--target-len", type=float, default=35,
                   help="clips longer than this (+5s) get trimmed at a sentence end; short clips get rewatched")
    r.add_argument("--max-candidates", type=int, default=30)
    r.add_argument("--picker", choices=["auto", "claude", "ollama", "heuristic"], default="auto")
    r.add_argument("--ollama-model", default="qwen3:1.7b")
    r.add_argument("--whisper-model", default="large-v3-turbo",
                   help="faster-whisper model; 'small' if you run out of VRAM")
    r.add_argument("--lang", default=None, help="force language, e.g. en or tl (default: auto)")
    r.add_argument("--from", dest="from_stage", choices=STAGES, help="force rerun from this stage onward")
    r.add_argument("--until", choices=STAGES, help="stop after this stage")
    r.add_argument("--cpu-encode", action="store_true", help="skip NVENC, encode with x264")
    r.add_argument("--no-captions", action="store_true")
    r.add_argument("--no-hook", action="store_true", help="don't show the title hook for the first 3s")
    r.add_argument("--clips", help="manual clip times, e.g. \"0:45-1:02,2:10-2:25\" (skips transcribe + pick)")
    r.add_argument("--text", help="campaign mode: ONE fixed on-screen line for the whole clip (turns off captions + hook)")
    r.add_argument("--cover", help="solid box x,y,w,h as fractions of the 9:16 output, to hide original text; --text goes inside it")
    r.add_argument("--box-color", default="black", choices=["black", "white"])
    r.add_argument("--judge", default="auto", choices=["auto", "claude", "ollama", "none"],
                   help="who rates the clips for the virality score (none = measured signals only)")
    r.add_argument("--no-score", action="store_true", help="skip the virality score + report")
    r.add_argument("--pool", type=float, default=2.0,
                   help="pick this many times --count, pre-score them, keep the best --count (1 = off)")
    r.add_argument("--no-jump-cuts", action="store_true", help="keep quiet pauses instead of cutting them out")
    r.add_argument("--no-zoom", action="store_true", help="no punch-in zooms on loud / hype moments")
    r.add_argument("--variants", type=int, default=2,
                   help="also render a 'payoff first' version B of the N best clips (0 = off)")
    r.add_argument("--hashtags", default="", help="hashtags every post caption must include, e.g. \"#lany\"")
    r.set_defaults(func=cmd_run)

    res = sub.add_parser("result", help="log how a posted clip did (phase 4 learning loop)")
    res.add_argument("job", help="job folder name, e.g. tJbzMqJGH4k")
    res.add_argument("clip", help="clip number (clip_03.mp4 -> 3, version B clip_03_b.mp4 -> 3b)")
    res.add_argument("platform", choices=["tiktok", "reels", "shorts"])
    res.add_argument("--views", type=int, required=True)
    res.add_argument("--likes", type=int, default=0)
    res.add_argument("--comments", type=int, default=0)
    res.add_argument("--shares", type=int, default=0)
    res.add_argument("--url", default="")
    res.set_defaults(func=cmd_result)

    ta = sub.add_parser("test-ai", help="check that Claude / Ollama can run, and why not")
    ta.set_defaults(func=cmd_test_ai)

    lr = sub.add_parser("learn", help="retune the score from logged results, write results.html")
    lr.add_argument("--open", action="store_true", help="open results.html afterwards")
    lr.set_defaults(func=cmd_learn)

    f = sub.add_parser("frame", help="save a still frame (to set up a facecam preset)")
    f.add_argument("source")
    f.add_argument("--name")
    f.add_argument("--at", type=float, default=600, help="seconds into the video")
    f.set_defaults(func=cmd_frame)

    for p in (r, f):
        p.add_argument("--cookies", help="cookies.txt (Netscape format) for yt-dlp; default: ./cookies.txt if present")
        p.add_argument("--cookies-from-browser", metavar="BROWSER",
                       help="let yt-dlp read cookies from a browser, e.g. firefox, edge, chrome")

    args = ap.parse_args()
    set_ytdlp_auth(args)
    args.func(args)


if __name__ == "__main__":
    main()
