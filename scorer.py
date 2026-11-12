"""Virality score for rendered clips (our take on Opus Clip's 0-99 score).

Each clip gets Hook / Flow / Value / Trend, 0-25 each, total 0-99:
  - signals: measured from the audio + word timestamps (free, offline, always available)
  - AI judge: Claude CLI, else a local Ollama model, reads the transcript + signals
  final = 65% AI + 35% signals, or signals only when no AI answered.

Writes the scores into highlights.json (clip["score"]) and builds jobs/<name>/report.html,
ranked best first. The score is a sorting aid, not a view prediction.
"""
import html
import json
import math
import os
import re
import shutil
import subprocess
import urllib.request
import wave
from pathlib import Path

HYPE = ["no way", "let's go", "lets go", "oh my god", "omg", "what the", "bro", "dude", "haha", "insane",
        "crazy", "clutch", "wait", "oh no", "holy", "shit", "fuck", "grabe", "hala", "tangina", "yo"]
CATS = ["hook", "flow", "value", "trend"]
PLATFORMS = ["tiktok", "reels", "shorts"]
PLATFORM_TAG = {"tiktok": "#fyp", "reels": "#reels", "shorts": "#shorts"}


def with_tags(text, required):
    """Make sure every required hashtag is in the caption."""
    for tag in required.split():
        tag = tag if tag.startswith("#") else "#" + tag
        if tag.lower() not in text.lower():
            text = f"{text} {tag}"
    return text.strip()


def post_captions(ai_post, title, hashtags):
    """Final captions per platform: the AI's, or a simple template when no AI answered."""
    ai_post = {k: v for k, v in (ai_post or {}).items() if len(v) >= 12 and " " in v}
    if len(set(ai_post.values())) < len(PLATFORMS):   # weak model repeated itself: use templates
        ai_post = {}
    base = (title or "wait for it").strip().rstrip(".")
    templates = {"tiktok": f"{base} \U0001F62D", "reels": f"{base}... watch till the end \U0001F440",
                 "shorts": f"{base}"}
    out = {}
    for pf in PLATFORMS:
        text = ai_post.get(pf) or templates[pf]
        out[pf] = with_tags(text, f"{hashtags} {PLATFORM_TAG[pf] if pf == 'shorts' else ''}")
    return out
AI_WEIGHT = 0.65


def _clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def _sig(x):
    return 1 / (1 + math.exp(-x))


# ------------------------------------------------------------------ signals
def loudness(wav_path, hop=0.25):
    import numpy as np
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    n = int(sr * hop)
    frames = max(1, len(data) // n)
    rms = np.sqrt(np.mean(data[: frames * n].reshape(frames, n) ** 2, axis=1) + 1e-10)
    db = 20 * np.log10(rms)
    z = (db - np.median(db)) / (np.std(db) + 1e-6)
    return z, hop


def clip_signals(s, e, words, segs, z, hop, has_transcript):
    import numpy as np
    L = e - s
    zc = z[int(s / hop): max(int(e / hop), int(s / hop) + 1)]
    z3 = z[int(s / hop): int((s + 3) / hop) + 1]
    w_in = [w for w in words if w["s"] >= s - 0.05 and w["e"] <= e + 0.05]
    sig = {"length": round(L, 1),
           "energy": round(float(np.mean(zc)) if len(zc) else 0.0, 2),
           "peak": round(float(np.max(zc)) if len(zc) else 0.0, 2),
           "hook_energy": round(float(np.mean(z3)) if len(z3) else 0.0, 2)}
    if has_transcript:
        first3 = [w for w in w_in if w["s"] < s + 3]
        text5 = " ".join(w["w"] for w in w_in if w["s"] < s + 5).lower()
        # dead air = sum of gaps between words longer than 1.2s (plus lead-in / tail)
        times = [s] + [t for w in w_in for t in (w["s"], w["e"])] + [e]
        gaps = sum(max(0.0, times[i + 1] - times[i]) for i in range(0, len(times) - 1, 2)
                   if times[i + 1] - times[i] > 1.2)
        last = w_in[-1] if w_in else None
        sig.update({
            "words_first_3s": len(first3),
            "hype_first_5s": sum(text5.count(h) for h in HYPE),
            "dead_air_pct": round(100 * _clamp(gaps / L if L else 1), 0),
            "words_per_sec": round(len(w_in) / L, 2) if L else 0,
            "ends_clean": bool(last and re.search(r"[.!?…]$", last["w"].strip()) and e - last["e"] < 2.0),
            "starts_mid_word": any(w["s"] < s < w["e"] for w in words),
        })
    return sig


def signal_scores(sig, has_transcript):
    """Map raw signals to 0-25 per category."""
    L = sig["length"]
    length_fit = 1.0 if 15 <= L <= 45 else _clamp(1 - (abs(L - 30) - 15) / 30)
    loud_open = _sig(sig["hook_energy"] * 1.5)
    energy = _sig(sig["energy"] * 1.5)
    if has_transcript:
        talk_open = _clamp(sig["words_first_3s"] / 6)
        hype = _clamp(sig["hype_first_5s"] / 2)
        flow_air = 1 - sig["dead_air_pct"] / 100
        wps = sig["words_per_sec"]
        pace = 1.0 if 1.8 <= wps <= 3.8 else _clamp(1 - abs(wps - 2.8) / 2.8)
        hook = 0.45 * max(talk_open, loud_open) + 0.35 * loud_open + 0.20 * hype
        flow = 0.45 * flow_air + 0.25 * float(sig["ends_clean"]) + 0.15 * float(not sig["starts_mid_word"]) \
            + 0.15 * length_fit
        value = 0.5 * energy + 0.5 * pace
    else:  # music / no speech: judge on sound energy only
        hook = loud_open
        flow = 0.6 * length_fit + 0.4 * energy
        value = energy
    trend = 0.5  # signals can't see trends; AI judge fills this in
    return {k: round(25 * v, 1) for k, v in zip(CATS, [hook, flow, value, trend])}


# ------------------------------------------------------------------ AI judge
JUDGE_PROMPT = """You are a strict short-form video editor rating clips for TikTok / Reels / Shorts, like Opus Clip's virality score.
Context: {context}

Rate EACH clip on 4 categories, 0-25 each (be harsh, most clips are 10-18, only great ones get 22+):
- hook: do the first 3 seconds grab a stranger scrolling past?
- flow: does it move logically and end on a satisfying beat (not mid-sentence)?
- value: is it funny, emotional, surprising or useful enough to watch to the end / share?
- trend: does it fit what this audience is into right now?

Measured signals are given per clip (dead air %, words/sec, loudness of the opening, etc.). Use them.

Reply with ONLY this JSON:
{{"clips": [{{"n": <clip number>, "hook": 0-25, "flow": 0-25, "value": 0-25, "trend": 0-25,
"why": "<one short line: the main reason for the score>",
"title": "<better on-screen hook, max 7 words>",
"fix": "<one concrete edit that would raise the score, e.g. start 3s later at 'X'>",
"post": {{"tiktok": "<post caption>", "reels": "<post caption>", "shorts": "<post caption>"}}}}]}}

Post captions (what goes under the video when posting, one per platform):
- short (max ~120 characters before hashtags), same mood as the clip, may use 1-2 emoji
- end with 3-5 relevant hashtags; ALWAYS include these required hashtags: {hashtags}
- shorts caption must include #shorts; make the three captions different from each other
- nothing negative about anyone in the video, no gossip, no misleading claims, no clickbait lies

CLIPS:
{body}
"""


def judge_prompt(clips, context, hashtags=""):
    parts = []
    for c in clips:
        sg = c["_sig"]
        meta = ", ".join(f"{k}={v}" for k, v in sg.items())
        parts.append(f"[clip {c['_n']}] {c['end'] - c['start']:.0f}s, current title: {c.get('title') or '(none)'}\n"
                     f"signals: {meta}\ntranscript: {c['_text'] or '(no speech: music / gameplay audio)'}")
    return JUDGE_PROMPT.format(context=context, hashtags=hashtags or "(none)", body="\n\n".join(parts))


def ask_claude(prompt):
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("claude CLI not found")
    out = subprocess.run([exe, "-p", "Follow the instructions in the input exactly and reply with JSON only."],
                         input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace",
                         timeout=600)
    if out.returncode != 0:
        raise RuntimeError((out.stderr.strip() or out.stdout.strip())[:300])
    return out.stdout


def ollama_models():
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
        return [m["name"] for m in json.loads(r.read().decode()).get("models", [])]


def ask_ollama(prompt, model):
    have = ollama_models()
    if not have:
        raise RuntimeError("no Ollama models installed")
    if model not in have:
        model = next((m for m in have if m.startswith("qwen")), have[0])
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False, "format": "json", "think": False,
                         "options": {"temperature": 0.2, "num_ctx": 8192}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read().decode()).get("response", ""), model


def parse_judge(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in reply")
    data = json.loads(m.group(0))
    out = {}
    for c in data.get("clips", data if isinstance(data, list) else []):
        try:
            n = int(c["n"])
            out[n] = {k: _clamp(float(c[k]), 0, 25) for k in CATS}
            out[n].update({k: str(c.get(k, ""))[:160] for k in ("why", "title", "fix")})
            post = c.get("post") if isinstance(c.get("post"), dict) else {}
            out[n]["post"] = {k: str(post.get(k, "")).strip()[:400] for k in PLATFORMS if post.get(k)}
        except (KeyError, TypeError, ValueError):
            continue
    if not out:
        raise ValueError("reply had no usable scores")
    return out


# ------------------------------------------------------------------ main entry
def score_clips(job, clips, picker="", ai="auto", ollama_model="qwen3:1.7b", context=None, log=print,
                hashtags=""):
    """Adds clip["score"] to every clip (in place). Works before rendering: only needs audio + transcript.
    Returns who judged ('claude', 'ollama:<model>' or 'signals')."""
    job = Path(job)
    tr = job / "transcript.json"
    segs = json.loads(tr.read_text(encoding="utf-8"))["segments"] if tr.exists() else []
    words = [w for s in segs for w in s.get("words", [])]
    has_tr = bool(words)
    wav = job / "audio.wav"
    if not wav.exists():
        src = json.loads((job / "job.json").read_text(encoding="utf-8"))["source"]
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                       check=True)
    z, hop = loudness(wav)

    for n, c in enumerate(clips, 1):
        c["_n"] = n
        c["_sig"] = clip_signals(c["start"], c["end"], words, segs, z, hop, has_tr)
        c["_text"] = " ".join(w["w"] for w in words if c["start"] <= w["s"] <= c["end"])[:1200]
        c["_sigscore"] = signal_scores(c["_sig"], has_tr)

    if context is None:
        context = ("clips cut from a long stream / video; audience: Tier 1-2 countries incl. South East Asia"
                   if picker != "manual" else
                   "campaign clips of provided footage (often live music) with one fixed on-screen text line")
    judged, by = {}, "signals"
    if ai != "none":
        prompt = judge_prompt(clips, context, hashtags)
        (job / "score_prompt.txt").write_text(prompt, encoding="utf-8")
        tries = {"auto": ["claude", "ollama"], "claude": ["claude"], "ollama": ["ollama"]}.get(ai, [])
        for t in tries:
            try:
                if t == "claude":
                    judged, by = parse_judge(ask_claude(prompt)), "claude"
                else:
                    reply, used = ask_ollama(prompt, ollama_model)
                    judged, by = parse_judge(reply), f"ollama:{used}"
                break
            except Exception as e:
                log(f"score: {t} judge failed ({str(e)[:200]})")

    import learn
    weights = learn.load_weights()
    for c in clips:
        sg, aj = c.pop("_sigscore"), judged.get(c["_n"])
        if aj:
            parts = {k: round(AI_WEIGHT * aj[k] + (1 - AI_WEIGHT) * sg[k], 1) for k in CATS}
        else:
            parts = dict(sg)
        total = int(round(_clamp(sum(parts.values()), 1, 99)))
        raw = total
        if weights:  # phase 4: weights learned from your real views (learn.py)
            total = learn.calibrated_total(parts, weights)
        c["score"] = {"total": total, "raw_total": raw, "tuned": bool(weights), **parts, "by": by if aj else "signals", "signals": c.pop("_sig"),
                      "why": (aj or {}).get("why", ""), "fix": (aj or {}).get("fix", ""),
                      "ai_title": (aj or {}).get("title", ""),
                      "post": post_captions((aj or {}).get("post"), c.get("title") or (aj or {}).get("title", ""),
                                            hashtags)}
        c.pop("_n"), c.pop("_text")
    return by


def score_job(job, ai="auto", ollama_model="qwen3:1.7b", context=None, log=print, rescore=True, hashtags=""):
    """Score the clips in highlights.json (unless already scored and rescore=False), save, write report."""
    job = Path(job)
    hl = json.loads((job / "highlights.json").read_text(encoding="utf-8"))
    clips = hl.get("clips", [])
    if not clips:
        log("score: no clips")
        return
    if rescore or not all("score" in c for c in clips):
        hl["scored_by"] = score_clips(job, clips, hl.get("picker", ""), ai, ollama_model, context, log, hashtags)
        (job / "highlights.json").write_text(json.dumps(hl, indent=2, ensure_ascii=False), encoding="utf-8")
    by = hl.get("scored_by", "signals")
    for n, c in enumerate(clips, 1):  # ready-to-paste post captions next to each clip
        post = c["score"].get("post") or post_captions(None, c.get("title"), hashtags)
        c["score"]["post"] = post
        if (job / "clips").exists():
            (job / "clips" / f"clip_{n:02d}_post.txt").write_text(
                "\n\n".join(f"[{pf.upper()}]\n{post[pf]}" for pf in PLATFORMS) + "\n", encoding="utf-8")
    (job / "highlights.json").write_text(json.dumps(hl, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(job, hl)
    ranked = sorted(enumerate(clips, 1), key=lambda t: -t[1]["score"]["total"])
    log(f"score: ranked by {by} (best first)")
    for n, c in ranked:
        sc = c["score"]
        log(f"  {sc['total']:2d}/99  clip_{n:02d}  hook {sc['hook']:.0f} flow {sc['flow']:.0f} "
            f"value {sc['value']:.0f} trend {sc['trend']:.0f}  {c.get('title', '')}")
    log(f"score: report -> {job / 'report.html'}")


# ------------------------------------------------------------------ report
def _color(t):
    return "#22c55e" if t >= 80 else "#eab308" if t >= 60 else "#ef4444"


def write_report(job, hl):
    clips = list(enumerate(hl["clips"], 1))
    clips.sort(key=lambda t: -t[1].get("score", {}).get("total", 0))
    cards = []
    for rank, (n, c) in enumerate(clips, 1):
        sc = c.get("score", {})
        name = f"clip_{n:02d}"
        bars = "".join(
            f'<div class="bar"><span>{k.title()}</span><i><b style="width:{sc.get(k, 0) * 4:.0f}%"></b></i>'
            f'<em>{sc.get(k, 0):.0f}/25</em></div>' for k in CATS)
        sg = sc.get("signals", {})
        chips = []
        if "dead_air_pct" in sg:
            chips += [f"dead air {sg['dead_air_pct']:.0f}%", f"{sg['words_per_sec']} words/s",
                      "ends clean" if sg["ends_clean"] else "ends mid-sentence"]
            if sg["starts_mid_word"]:
                chips.append("starts mid-word")
        chips += [f"{sg.get('length', 0):.0f}s", f"opening loudness {sg.get('hook_energy', 0):+.1f}"]
        m, s = divmod(int(c["start"]), 60)
        notes = ""
        if sc.get("why"):
            notes += f'<p><strong>Why:</strong> {html.escape(sc["why"])}</p>'
        if sc.get("fix"):
            notes += f'<p><strong>Fix:</strong> {html.escape(sc["fix"])}</p>'
        if sc.get("ai_title") and sc["ai_title"] != c.get("title"):
            notes += f'<p><strong>Title idea:</strong> {html.escape(sc["ai_title"])}</p>'
        if sc.get("post"):
            notes += '<div class="post">' + "".join(
                f'<div><b>{pf.title()}</b><button onclick="cp(this)">Copy</button>'
                f'<span>{html.escape(sc["post"].get(pf, ""))}</span></div>' for pf in PLATFORMS) + "</div>"
        cards.append(f"""
<article class="card">
  <video src="clips/{name}.mp4" poster="clips/{name}_preview.png" controls preload="none" playsinline></video>
  <div class="body">
    <div class="top"><div class="badge" style="--c:{_color(sc.get('total', 0))}">{sc.get('total', '–')}</div>
      <div><div class="rank">#{rank} · {name} · {m}:{s:02d}</div>
      <h2>{html.escape(c.get('title') or '(no title)')}</h2></div></div>
    {bars}
    <div class="chips">{''.join(f'<span>{html.escape(x)}</span>' for x in chips)}</div>
    {notes}
  </div>
</article>""")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Clip Scores</title>
<style>
:root{{--bg:#0f1115;--card:#181b22;--fg:#e8eaf0;--mut:#9aa1ad;--line:#2a2f3a}}
@media (prefers-color-scheme: light){{:root{{--bg:#f4f5f7;--card:#fff;--fg:#14161a;--mut:#5b6270;--line:#e1e4ea}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,Segoe UI,sans-serif}}
header{{padding:24px 16px 8px;max-width:1200px;margin:auto}}h1{{margin:0;font-size:22px}}
header p{{margin:4px 0 0;color:var(--mut)}}
main{{max-width:1200px;margin:auto;padding:16px;display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(min(340px,100%),1fr))}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;display:flex}}
video{{width:42%;aspect-ratio:9/16;background:#000;object-fit:cover}}
.body{{padding:12px;flex:1;min-width:0}}.top{{display:flex;gap:10px;align-items:center}}
.badge{{width:52px;height:52px;flex:none;border-radius:50%;display:grid;place-items:center;font:700 20px system-ui;
 border:3px solid var(--c);color:var(--c)}}
.rank{{color:var(--mut);font-size:12px}}h2{{margin:2px 0 0;font-size:15px}}
.bar{{display:grid;grid-template-columns:44px 1fr 40px;gap:6px;align-items:center;font-size:12px;margin-top:6px}}
.bar i{{height:6px;background:var(--line);border-radius:3px;overflow:hidden}}.bar b{{display:block;height:100%;background:#6d8bff}}
.bar em{{font-style:normal;color:var(--mut);text-align:right}}
.chips{{display:flex;flex-wrap:wrap;gap:4px;margin:10px 0 4px}}.chips span{{font-size:11px;padding:2px 7px;border:1px solid var(--line);border-radius:10px;color:var(--mut)}}
.body p{{margin:6px 0 0;font-size:13px}}
.post{{margin-top:8px;display:grid;gap:6px}}.post div{{border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-size:12px}}
.post b{{font-size:11px;color:var(--mut);text-transform:uppercase}}.post span{{display:block;margin-top:2px}}
.post button{{float:right;font-size:11px;padding:1px 8px;border-radius:6px;border:1px solid var(--line);background:none;color:var(--fg);cursor:pointer}}
@media (max-width:560px){{main{{grid-template-columns:1fr}}.card{{flex-direction:column}}
 video{{width:100%;max-height:70vh}}}}
</style></head><body>
<header><h1>{html.escape(job.name)} — clips ranked</h1>
<p>Score 0-99 = Hook + Flow + Value + Trend. Scored by {html.escape(hl.get('scored_by', 'signals'))}.
{'Weights tuned to your real views (see results.html). ' if any(c.get('score', {}).get('tuned') for c in hl['clips']) else ''}A sorting aid, not a view prediction. Watch before posting.
After posting, log each clip's views/likes in the app's Results tab so the score learns your audience.</p></header>
<main>{''.join(cards)}</main>
<script>
function cp(b){{const t=b.nextElementSibling.textContent;const done=()=>{{b.textContent='Copied';setTimeout(()=>b.textContent='Copy',1200)}};
try{{navigator.clipboard.writeText(t).then(done,()=>fb(t,done))}}catch(e){{fb(t,done)}}}}
function fb(t,done){{const a=document.createElement('textarea');a.value=t;document.body.appendChild(a);a.select();
try{{document.execCommand('copy');done()}}catch(e){{}}a.remove()}}
</script></body></html>"""
    (job / "report.html").write_text(page, encoding="utf-8")
