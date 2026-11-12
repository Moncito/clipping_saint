"""Phase 4: learn from real results.

After you post clips, log how each one did (views, likes, comments, shares). Then `learn`:
  - compares the predicted score with real views (per platform, since TikTok and Shorts views differ)
  - re-weights Hook / Flow / Value / Trend for YOUR audience (ridge regression, blended with the
    default equal weights until there's enough data; ~50 posted clips = fully learned)
  - finds which measured signals (dead air, pace, length, loud opening...) go with more views
  - flags clips under the campaign engagement minimum
  - writes results.html (dashboard) and calibration.json (used by scorer.py from then on)

Files (gitignored, personal): performance.csv, calibration.json, results.html in the clipper folder.
"""
import csv
import html
import json
import math
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / "jobs"
PERF = ROOT / "performance.csv"
CALIB = ROOT / "calibration.json"
RESULTS = ROOT / "results.html"
FIELDS = ["date", "job", "clip", "platform", "views", "likes", "comments", "shares", "url"]
CATS = ["hook", "flow", "value", "trend"]
SIGNALS = ["length", "hook_energy", "energy", "words_first_3s", "dead_air_pct", "words_per_sec"]
MIN_TO_CALIBRATE = 10        # posted clips needed before the weights start moving
FULL_TRUST = 50              # posted clips at which the learned weights fully replace the defaults
# minimum engagement rate by views tier, from the LANY brief (a sensible default for most campaigns)
ENGAGEMENT_MIN = {"tiktok": [(25_000, .009), (100_000, .008), (math.inf, .005)],
                  "reels": [(25_000, .009), (100_000, .008), (math.inf, .005)],
                  "shorts": [(25_000, .008), (100_000, .006), (math.inf, .004)]}


# ------------------------------------------------------------------ log
def read_rows(path=PERF):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clip_id(clip):
    """3 / '3' / 'clip_03' -> '3';  '3b' / 'clip_03_b' -> '3b' (version B, payoff first)."""
    s = str(clip).lower().replace("clip_", "").replace("_", "").replace(".mp4", "").strip()
    num = "".join(ch for ch in s if ch.isdigit())
    if not num:
        raise ValueError(f"bad clip number: {clip}")
    return f"{int(num)}{'b' if s.endswith('b') else ''}"


def add_result(job, clip, platform, views, likes=0, comments=0, shares=0, url="", path=PERF):
    """Add or update (same job + clip + platform) one posted clip's numbers."""
    clip = clip_id(clip)
    platform = platform.lower().replace("instagram", "reels").replace("youtube", "shorts")
    if platform not in ENGAGEMENT_MIN:
        raise ValueError("platform must be tiktok, reels or shorts")
    rows = [r for r in read_rows(path)
            if not (r["job"] == job and clip_id(r["clip"]) == clip and r["platform"] == platform)]
    rows.append({"date": date.today().isoformat(), "job": job, "clip": clip, "platform": platform,
                 "views": int(views), "likes": int(likes), "comments": int(comments), "shares": int(shares),
                 "url": url})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


# ------------------------------------------------------------------ join with predictions
def joined(rows, jobs_dir=JOBS):
    """Attach each logged result to the clip's score from jobs/<job>/highlights.json."""
    cache, out = {}, []
    for r in rows:
        if r["job"] not in cache:
            p = Path(jobs_dir) / r["job"] / "highlights.json"
            cache[r["job"]] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        hl = cache[r["job"]]
        cid = clip_id(r["clip"])
        n = int(cid.rstrip("b"))
        if not hl or not (1 <= n <= len(hl["clips"])) or "score" not in hl["clips"][n - 1]:
            continue
        c = hl["clips"][n - 1]
        views = int(r["views"])
        eng = (int(r["likes"]) + int(r["comments"]) + int(r["shares"])) / views if views else 0.0
        out.append({**r, "clip": cid, "variant": "B" if cid.endswith("b") else "A", "views": views, "eng": eng,
                    "title": c.get("title", ""), "score": c["score"]})
    return out


def _z(vals):
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) or 1.0
    return [(v - m) / sd for v in vals]


def _rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    za, zb = _z(ra), _z(rb)
    return sum(x * y for x, y in zip(za, zb)) / len(za)


def performance_target(data):
    """Views on a log scale, standardised within each platform (so platforms are comparable)."""
    by_pf = {}
    for i, d in enumerate(data):
        by_pf.setdefault(d["platform"], []).append(i)
    y = [0.0] * len(data)
    for idx in by_pf.values():
        zs = _z([math.log10(data[i]["views"] + 1) for i in idx]) if len(idx) > 1 else [0.0]
        for i, z in zip(idx, zs):
            y[i] = z
    return y


def ridge(X, y, lam=1.0):
    """Tiny ridge regression without numpy (k=4 features)."""
    k = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) + (lam if i == j else 0) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * yy for r, yy in zip(X, y)) for i in range(k)]
    for col in range(k):                         # Gauss-Jordan
        piv = max(range(col, k), key=lambda r: abs(A[r][col]))
        A[col], A[piv], b[col], b[piv] = A[piv], A[col], b[piv], b[col]
        d = A[col][col] or 1e-9
        A[col] = [v / d for v in A[col]]
        b[col] /= d
        for r in range(k):
            if r != col:
                f = A[r][col]
                A[r] = [v - f * w for v, w in zip(A[r], A[col])]
                b[r] -= f * b[col]
    return b


def calibrated_total(score, weights):
    return int(round(max(1, min(99, sum(weights[k] * score[k] for k in CATS)))))


# ------------------------------------------------------------------ learn
def learn(perf=PERF, jobs_dir=JOBS, calib=CALIB, results=RESULTS, log=print):
    data = joined(read_rows(perf), jobs_dir)
    n = len(data)
    weights = {k: 1.0 for k in CATS}
    info = {"n": n, "weights": weights, "updated": date.today().isoformat()}
    if n == 0:
        log("learn: no logged results that match a scored clip yet")
        write_results([], info, results)
        return info
    y = performance_target(data)
    pred = [d["score"]["total"] for d in data]
    info["r_before"] = spearman(pred, y)

    if n >= MIN_TO_CALIBRATE:
        cols = [_z([d["score"][k] for d in data]) for k in CATS]
        X = [[cols[j][i] for j in range(len(CATS))] for i in range(n)]
        beta = ridge(X, y, lam=max(1.0, 20 / n * 5))
        imp = [max(b, 0) + 0.05 for b in beta]
        mean = sum(imp) / len(imp)
        alpha = min(1.0, (n - MIN_TO_CALIBRATE + 1) / (FULL_TRUST - MIN_TO_CALIBRATE + 1))
        weights = {k: round(min(1.8, max(0.4, (1 - alpha) + alpha * v / mean)), 2) for k, v in zip(CATS, imp)}
        s = sum(weights.values())
        weights = {k: round(v * len(CATS) / s, 2) for k, v in weights.items()}   # keep the 0-99 scale
        info.update(weights=weights, beta={k: round(b, 3) for k, b in zip(CATS, beta)}, trust=round(alpha, 2),
                    r_after=spearman([calibrated_total(d["score"], weights) for d in data], y))
        log(f"learn: {n} clips -> weights " + ", ".join(f"{k} {v}" for k, v in weights.items())
            + f" (trust {alpha:.0%})")
    else:
        log(f"learn: {n} clip(s) logged; weights start adapting at {MIN_TO_CALIBRATE}")

    # which measured signals go with more views
    insights = []
    for s in SIGNALS:
        pairs = [(d["score"].get("signals", {}).get(s), yy) for d, yy in zip(data, y)]
        pairs = [(a, b) for a, b in pairs if isinstance(a, (int, float))]
        if len(pairs) >= 5:
            r = spearman([a for a, _ in pairs], [b for _, b in pairs])
            if r is not None:
                insights.append((s, round(r, 2), len(pairs)))
    info["signals"] = insights

    # A/B: same clip + platform posted as normal (A) and payoff-first (B)
    pairs = {}
    for d in data:
        pairs.setdefault((d["job"], d["clip"].rstrip("b"), d["platform"]), {})[d["variant"]] = d["views"]
    ratios = [math.log10((v["B"] + 1) / (v["A"] + 1)) for v in pairs.values() if "A" in v and "B" in v]
    if ratios:
        info["ab"] = {"pairs": len(ratios), "b_vs_a_pct": round((10 ** (sum(ratios) / len(ratios)) - 1) * 100)}
    calib = Path(calib)
    calib.write_text(json.dumps(info, indent=2), encoding="utf-8")
    write_results(data, info, results)
    log(f"learn: dashboard -> {results}")
    return info


def load_weights(calib=CALIB):
    """Weights for scorer.py, or None when not calibrated yet."""
    try:
        info = json.loads(Path(calib).read_text(encoding="utf-8"))
        return info["weights"] if info.get("n", 0) >= MIN_TO_CALIBRATE else None
    except Exception:
        return None


# ------------------------------------------------------------------ dashboard
SIGNAL_NAMES = {"length": "clip length", "hook_energy": "loud opening", "energy": "overall loudness",
                "words_first_3s": "talking in the first 3s", "dead_air_pct": "dead air",
                "words_per_sec": "talking speed"}


def eng_min(platform, views):
    for cap, m in ENGAGEMENT_MIN.get(platform, []):
        if views < cap:
            return m
    return 0


def write_results(data, info, path=RESULTS):
    rows = []
    for d in sorted(data, key=lambda d: -d["views"]):
        need = eng_min(d["platform"], d["views"])
        ok = d["eng"] >= need
        rows.append(f"<tr><td>{html.escape(d['job'])} #{html.escape(str(d['clip']))}<br><small>{html.escape(d['title'])}</small></td>"
                    f"<td>{d['platform']}</td><td class=n>{d['score']['total']}</td><td class=n>{d['views']:,}</td>"
                    f"<td class='n {'ok' if ok else 'bad'}'>{d['eng'] * 100:.1f}%"
                    f"<br><small>min {need * 100:.1f}%</small></td></tr>")
    w = info.get("weights", {})
    wbars = "".join(f"<div class=bar><span>{k.title()}</span><i><b style='width:{min(100, v / 2 * 100):.0f}%'></b></i>"
                    f"<em>{v:.2f}×</em></div>" for k, v in w.items())

    def r_txt(r):
        return "–" if r is None else f"{r:+.2f}"
    tips = []
    for s, r, k in sorted(info.get("signals", []), key=lambda t: -abs(t[1])):
        if abs(r) >= 0.2:
            more = "more" if r > 0 else "less"
            tips.append(f"<li>Clips with <b>{more} {SIGNAL_NAMES.get(s, s)}</b> got more views "
                        f"(r = {r:+.2f}, {k} clips)</li>")
    ab = info.get("ab")
    if ab:
        better = "more" if ab["b_vs_a_pct"] >= 0 else "fewer"
        tips.insert(0, f"<li><b>Payoff-first (version B)</b> got {abs(ab['b_vs_a_pct'])}% {better} views than the "
                       f"normal cut ({ab['pairs']} A/B pair{'s' if ab['pairs'] != 1 else ''})</li>")
    n = info.get("n", 0)
    status = (f"Learned from <b>{n}</b> posted clips (trust {info.get('trust', 0):.0%})." if n >= MIN_TO_CALIBRATE
              else f"<b>{n}</b> posted clip(s) logged. Scores start adapting at {MIN_TO_CALIBRATE}; "
                   f"fully tuned around {FULL_TRUST}.")
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>Clip Results</title><style>
:root{{--bg:#0f1115;--card:#181b22;--fg:#e8eaf0;--mut:#9aa1ad;--line:#2a2f3a;--ok:#22c55e;--bad:#ef4444}}
@media (prefers-color-scheme: light){{:root{{--bg:#f4f5f7;--card:#fff;--fg:#14161a;--mut:#5b6270;--line:#e1e4ea}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,Segoe UI,sans-serif}}
.wrap{{max-width:1000px;margin:auto;padding:24px 16px}}h1{{margin:0 0 4px;font-size:22px}}.mut{{color:var(--mut)}}
.grid{{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(min(280px,100%),1fr));margin:16px 0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px}}h2{{font-size:15px;margin:0 0 8px}}
.big{{font:700 28px system-ui}}.bar{{display:grid;grid-template-columns:50px 1fr 48px;gap:6px;align-items:center;font-size:13px;margin-top:6px}}
.bar i{{height:6px;background:var(--line);border-radius:3px;overflow:hidden}}.bar b{{display:block;height:100%;background:#6d8bff}}
.bar em{{font-style:normal;color:var(--mut);text-align:right}}
.tbl{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}
td,th{{padding:8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}.n{{text-align:right}}
.ok{{color:var(--ok)}}.bad{{color:var(--bad)}}small{{color:var(--mut)}}ul{{margin:0;padding-left:18px}}
</style></head><body><div class=wrap>
<h1>What works for your audience</h1><p class=mut>{status}</p>
<div class=grid>
 <div class=card><h2>Does the score predict views?</h2>
  <div class=big>{r_txt(info.get('r_before'))}</div>
  <p class=mut>Rank correlation between score and views (1 = perfect, 0 = no link).
  {'After learning: <b>' + r_txt(info.get('r_after')) + '</b> (measured on the same clips, so optimistic).' if info.get('r_after') is not None else ''}</p></div>
 <div class=card><h2>How much each part counts now</h2>{wbars}
  <p class=mut>1.00× = default. Higher = this part matters more for your views.</p></div>
 <div class=card><h2>Patterns</h2>{'<ul>' + ''.join(tips) + '</ul>' if tips else '<p class=mut>Needs at least 5 logged clips with a clear pattern.</p>'}</div>
</div>
<div class="card tbl"><h2>Posted clips</h2><table><tr><th>Clip</th><th>Platform</th><th class=n>Score</th>
<th class=n>Views</th><th class=n>Engagement</th></tr>{''.join(rows) or '<tr><td colspan=5 class=mut>Nothing logged yet.</td></tr>'}</table>
<p class=mut>Engagement = (likes + comments + shares) / views. Red = under the campaign minimum for that view tier
(TikTok/Reels 0.9% under 25K, 0.8% to 100K, 0.5% above; Shorts 0.8 / 0.6 / 0.4%).</p></div>
</div></body></html>"""
    Path(path).write_text(page, encoding="utf-8")
