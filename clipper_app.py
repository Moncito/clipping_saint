"""Clipper desktop app: a simple window on top of clip.py.

Two tabs:
  Stream clips   - long stream/VOD -> N auto-picked vertical clips with captions
  Campaign clip  - campaign footage -> clips at times you choose, with ONE approved line of text,
                   plus an optional solid box (drawn on a real frame) to hide the video's own text

Start it with Clipper.bat (or: .venv\\Scripts\\pythonw.exe clipper_app.py).
Nothing here posts anything; it only makes files in jobs\\<name>\\clips.
"""
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / "jobs"
SETTINGS = ROOT / "app_settings.json"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)
STAGES = ["(normal)", "download", "transcribe", "candidates", "pick", "render"]
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def load_settings():
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(data):
    try:
        SETTINGS.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def presets():
    try:
        return list(json.loads((ROOT / "presets.json").read_text(encoding="utf-8")).keys())
    except Exception:
        return ["blur", "center"]


def open_path(p):
    p = Path(p)
    if p.exists():
        os.startfile(str(p))
    else:
        messagebox.showinfo("Clipper", f"Not there yet:\n{p}")


def parse_time(s):
    """'1:05' / '65' / '0:01:05' -> seconds"""
    parts = [float(x) for x in s.strip().split(":")]
    sec = 0.0
    for x in parts:
        sec = sec * 60 + x
    return sec


class CoverPicker(tk.Toplevel):
    """Shows one frame of the video (as it will look in the 9:16 output) and lets you drag a box."""
    W, H = 360, 640

    def __init__(self, master, png, on_done, current=None):
        super().__init__(master)
        self.title("Drag a box over the original text, then Use this box")
        self.resizable(False, False)
        self.on_done = on_done
        self.img = tk.PhotoImage(file=png)
        self.cv = tk.Canvas(self, width=self.W, height=self.H, cursor="crosshair", highlightthickness=0)
        self.cv.pack()
        self.cv.create_image(0, 0, anchor="nw", image=self.img)
        self.rect = None
        self.start = None
        if current:
            x, y, w, h = current
            self.rect = self.cv.create_rectangle(x * self.W, y * self.H, (x + w) * self.W, (y + h) * self.H,
                                                 outline="#00ff66", width=2, fill="black", stipple="gray50")
        self.cv.bind("<ButtonPress-1>", self.press)
        self.cv.bind("<B1-Motion>", self.drag)
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=4)
        ttk.Label(bar, text="Make the box a bit bigger than the text.").pack(side="left", padx=6)
        ttk.Button(bar, text="Use this box", command=self.done).pack(side="right", padx=6)

    def press(self, e):
        self.start = (e.x, e.y)
        if self.rect:
            self.cv.delete(self.rect)
        self.rect = self.cv.create_rectangle(e.x, e.y, e.x, e.y, outline="#00ff66", width=2,
                                             fill="black", stipple="gray50")

    def drag(self, e):
        if self.start and self.rect:
            x = min(max(e.x, 0), self.W)
            y = min(max(e.y, 0), self.H)
            self.cv.coords(self.rect, self.start[0], self.start[1], x, y)

    def done(self):
        if not self.rect:
            self.destroy()
            return
        x1, y1, x2, y2 = self.cv.coords(self.rect)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 5 or y2 - y1 < 5:
            messagebox.showwarning("Clipper", "Box is too small. Drag a bigger one.", parent=self)
            return
        self.on_done(round(x1 / self.W, 3), round(y1 / self.H, 3),
                     round((x2 - x1) / self.W, 3), round((y2 - y1) / self.H, 3))
        self.destroy()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clipper")
        self.geometry("860x760")
        self.minsize(760, 640)
        self.s = load_settings()
        self.proc = None
        self.q = queue.Queue()
        self.last_job = None

        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

        nb = ttk.Notebook(self)
        nb.pack(fill="x", padx=10, pady=(10, 4))
        self.stream_tab = ttk.Frame(nb, padding=10)
        self.camp_tab = ttk.Frame(nb, padding=10)
        nb.add(self.stream_tab, text="  Stream clips  ")
        nb.add(self.camp_tab, text="  Campaign clip  ")
        self.nb = nb
        self.build_stream()
        self.build_campaign()
        nb.select(self.s.get("tab", 0))

        bar = ttk.Frame(self, padding=(10, 0))
        bar.pack(fill="x")
        self.run_btn = ttk.Button(bar, text="▶  Make clips", command=self.start_run)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="■  Stop", command=self.stop_run, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(bar, text="Open clips folder", command=self.open_clips).pack(side="left", padx=(20, 0))
        ttk.Button(bar, text="★ Open scores report", command=self.open_report).pack(side="left", padx=6)
        ttk.Button(bar, text="Open jobs folder", command=lambda: open_path(JOBS)).pack(side="left", padx=6)
        self.status = ttk.Label(bar, text="Ready", foreground="#555")
        self.status.pack(side="right")

        self.log = ScrolledText(self, height=18, font=("Consolas", 9), bg="#111", fg="#ddd",
                                insertbackground="#ddd", wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=10)
        self.log.tag_config("err", foreground="#ff7070")
        self.log.tag_config("ok", foreground="#70ff90")
        self.say("Pick a tab, fill it in, press Make clips. Clips land in jobs\\<name>\\clips. Nothing is posted.\n")

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self.pump)

    # ------------------------------------------------------------ layout helpers
    def row(self, parent, r, label, widget, hint=None):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 8))
        widget.grid(row=r, column=1, sticky="we", pady=3)
        if hint:
            ttk.Label(parent, text=hint, foreground="#777").grid(row=r, column=2, sticky="w", padx=8)
        parent.columnconfigure(1, weight=1)

    def source_row(self, parent, r, var):
        f = ttk.Frame(parent)
        ttk.Entry(f, textvariable=var).pack(side="left", fill="x", expand=True)
        ttk.Button(f, text="Browse…", command=lambda: self.browse(var)).pack(side="left", padx=(6, 0))
        self.row(parent, r, "Video (link or file)", f)

    def browse(self, var):
        p = filedialog.askopenfilename(title="Pick a video",
                                       filetypes=[("Video", "*.mp4 *.mkv *.mov *.webm"), ("All files", "*.*")])
        if p:
            var.set(p)

    # ------------------------------------------------------------ stream tab
    def build_stream(self):
        t, s = self.stream_tab, self.s.get("stream", {})
        self.st_src = tk.StringVar(value=s.get("src", ""))
        self.st_name = tk.StringVar(value=s.get("name", ""))
        self.st_preset = tk.StringVar(value=s.get("preset", "blur"))
        self.st_count = tk.IntVar(value=s.get("count", 8))
        self.st_picker = tk.StringVar(value=s.get("picker", "auto"))
        self.st_whisper = tk.StringVar(value=s.get("whisper", "large-v3-turbo"))
        self.st_lang = tk.StringVar(value=s.get("lang", ""))
        self.st_from = tk.StringVar(value="(normal)")
        self.st_cpu = tk.BooleanVar(value=s.get("cpu", True))
        self.st_nocap = tk.BooleanVar(value=s.get("nocap", False))

        self.source_row(t, 0, self.st_src)
        self.row(t, 1, "Job name", ttk.Entry(t, textvariable=self.st_name), "optional, reuse it to rerun the same video")
        self.row(t, 2, "Layout preset", ttk.Combobox(t, textvariable=self.st_preset, values=presets(), state="readonly"))
        self.row(t, 3, "How many clips", ttk.Spinbox(t, from_=1, to=30, textvariable=self.st_count, width=6))
        self.row(t, 4, "Clip picker", ttk.Combobox(t, textvariable=self.st_picker,
                                                    values=["auto", "claude", "ollama", "heuristic"], state="readonly"))
        self.row(t, 5, "Whisper model", ttk.Combobox(t, textvariable=self.st_whisper,
                                                      values=["large-v3-turbo", "medium", "small"]),
                 "use small if GPU runs out of memory")
        self.row(t, 6, "Language", ttk.Entry(t, textvariable=self.st_lang, width=8), "blank = auto, or en / tl")
        self.row(t, 7, "Redo from step", ttk.Combobox(t, textvariable=self.st_from, values=STAGES, state="readonly"),
                 "render = rerender after editing highlights.json")
        opts = ttk.Frame(t)
        ttk.Checkbutton(opts, text="CPU encode (your GPU driver is too old for NVENC)", variable=self.st_cpu).pack(side="left")
        ttk.Checkbutton(opts, text="No captions", variable=self.st_nocap).pack(side="left", padx=12)
        self.row(t, 8, "", opts)

    # ------------------------------------------------------------ campaign tab
    def build_campaign(self):
        t, s = self.camp_tab, self.s.get("campaign", {})
        self.cp_src = tk.StringVar(value=s.get("src", ""))
        self.cp_name = tk.StringVar(value=s.get("name", ""))
        self.cp_times = tk.StringVar(value=s.get("times", ""))
        self.cp_text = tk.StringVar(value=s.get("text", ""))
        self.cp_preset = tk.StringVar(value=s.get("preset", "center"))
        self.cp_box = tk.StringVar(value=s.get("box", "black"))
        self.cp_use_cover = tk.BooleanVar(value=s.get("use_cover", False))
        self.cp_cover = tk.StringVar(value=s.get("cover", ""))
        self.cp_cpu = tk.BooleanVar(value=s.get("cpu", True))

        self.source_row(t, 0, self.cp_src)
        self.row(t, 1, "Job name", ttk.Entry(t, textvariable=self.cp_name), "e.g. lany-vid2")
        self.row(t, 2, "Clip times", ttk.Entry(t, textvariable=self.cp_times),
                 "0:20-0:38  (several: 0:20-0:38, 1:05-1:20)")

        lines = ttk.Frame(t)
        self.cp_lines = tk.Text(lines, height=4, wrap="word", font=("Segoe UI", 9))
        self.cp_lines.insert("1.0", s.get("lines", ""))
        self.cp_lines.pack(side="left", fill="x", expand=True)
        ttk.Button(lines, text="Use selected\nline ↓", command=self.use_line).pack(side="left", padx=(6, 0))
        self.row(t, 3, "Approved lines\n(paste from brief,\none per line)", lines)
        self.row(t, 4, "On-screen text", ttk.Entry(t, textvariable=self.cp_text),
                 "must match the brief exactly")

        cov = ttk.Frame(t)
        ttk.Checkbutton(cov, text="Cover original text with a box", variable=self.cp_use_cover).pack(side="left")
        ttk.Entry(cov, textvariable=self.cp_cover, width=24).pack(side="left", padx=6)
        ttk.Button(cov, text="Draw box on frame…", command=self.pick_cover).pack(side="left")
        self.row(t, 5, "Cover", cov)
        self.row(t, 6, "Box color", ttk.Combobox(t, textvariable=self.cp_box, values=["black", "white"],
                                                  state="readonly", width=8))
        self.row(t, 7, "Layout preset", ttk.Combobox(t, textvariable=self.cp_preset, values=presets(), state="readonly"),
                 "center = full frame for vertical videos")
        self.row(t, 8, "", ttk.Checkbutton(t, text="CPU encode", variable=self.cp_cpu))
        ttk.Label(t, foreground="#a05000",
                  text="After it renders, open clips\\clip_NN_preview.png and the video itself: "
                       "NO original text may show anywhere, or the clip is not paid.").grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def use_line(self):
        try:
            line = self.cp_lines.get("insert linestart", "insert lineend").strip()
            if self.cp_lines.tag_ranges("sel"):
                line = self.cp_lines.get("sel.first", "sel.last").strip()
        except tk.TclError:
            line = ""
        if line:
            self.cp_text.set(line)
        else:
            messagebox.showinfo("Clipper", "Click on one of the approved lines first.")

    def pick_cover(self):
        src = self.cp_src.get().strip()
        if not src or not Path(src).exists():
            messagebox.showinfo("Clipper", "Pick a video FILE first (download it from the campaign folder).")
            return
        t = 1.0
        m = re.match(r"\s*([\d:.]+)\s*-", self.cp_times.get())
        if m:
            try:
                t = parse_time(m.group(1)) + 1
            except ValueError:
                pass
        png = Path(tempfile.gettempdir()) / "clipper_cover_frame.png"
        vf = f"crop='min(iw,ih*9/16)':ih,scale={CoverPicker.W}:{CoverPicker.H}"
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(t), "-i", src, "-frames:v", "1",
                            "-vf", vf, str(png)], check=True, creationflags=NO_WINDOW)
        except Exception as e:
            messagebox.showerror("Clipper", f"Couldn't grab a frame:\n{e}")
            return
        cur = None
        try:
            cur = [float(v) for v in self.cp_cover.get().split(",")]
            if len(cur) != 4:
                cur = None
        except ValueError:
            pass

        def done(x, y, w, h):
            self.cp_cover.set(f"{x},{y},{w},{h}")
            self.cp_use_cover.set(True)
        CoverPicker(self, str(png), done, cur)

    # ------------------------------------------------------------ running
    def build_cmd(self):
        tab = self.nb.index(self.nb.select())
        if tab == 0:
            src = self.st_src.get().strip()
            if not src:
                raise ValueError("Paste a link or pick a video file.")
            cmd = ["run", src, "--count", str(self.st_count.get()), "--preset", self.st_preset.get(),
                   "--picker", self.st_picker.get(), "--whisper-model", self.st_whisper.get()]
            if self.st_name.get().strip():
                cmd += ["--name", self.st_name.get().strip()]
            if self.st_lang.get().strip():
                cmd += ["--lang", self.st_lang.get().strip()]
            if self.st_from.get() != "(normal)":
                cmd += ["--from", self.st_from.get()]
            if self.st_cpu.get():
                cmd.append("--cpu-encode")
            if self.st_nocap.get():
                cmd.append("--no-captions")
            return cmd
        src = self.cp_src.get().strip()
        times = self.cp_times.get().strip()
        text = self.cp_text.get().strip()
        if not src:
            raise ValueError("Pick the campaign video file.")
        if not times:
            raise ValueError("Enter clip times, e.g. 0:20-0:38")
        if not text:
            raise ValueError("Enter the approved on-screen text.")
        approved = [l.strip() for l in self.cp_lines.get("1.0", "end").splitlines() if l.strip()]
        if approved and text not in approved:
            if not messagebox.askyesno("Clipper", "The on-screen text doesn't exactly match any approved line "
                                                  "you pasted. Clips with unapproved text don't get paid.\n\n"
                                                  "Render anyway?"):
                raise ValueError("Cancelled.")
        for part in times.split(","):
            a, b = part.split("-")
            if parse_time(b) - parse_time(a) < 8:
                if not messagebox.askyesno("Clipper", f"{part.strip()} is under 8 seconds (most briefs need 8s+). "
                                                      "Render anyway?"):
                    raise ValueError("Cancelled.")
        cmd = ["run", src, "--clips", times, "--text", text, "--preset", self.cp_preset.get(),
               "--box-color", self.cp_box.get(), "--from", "render"]
        if self.cp_name.get().strip():
            cmd += ["--name", self.cp_name.get().strip()]
        if self.cp_use_cover.get():
            if not re.fullmatch(r"\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*", self.cp_cover.get()):
                raise ValueError("Cover must be x,y,w,h (use Draw box on frame…).")
            cmd += ["--cover", self.cp_cover.get().replace(" ", "")]
        if self.cp_cpu.get():
            cmd.append("--cpu-encode")
        return cmd

    def start_run(self):
        if self.proc:
            return
        try:
            args = self.build_cmd()
        except ValueError as e:
            if str(e) != "Cancelled.":
                messagebox.showwarning("Clipper", str(e))
            return
        self.remember()
        self.last_job = None
        self.say("\n$ clip.py " + " ".join(f'"{a}"' if " " in a else a for a in args) + "\n", "ok")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        try:
            self.proc = subprocess.Popen([str(PY), str(ROOT / "clip.py"), *args], cwd=ROOT, env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
        except Exception as e:
            self.say(f"Couldn't start: {e}\n", "err")
            self.proc = None
            return
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status.config(text="Working…")
        threading.Thread(target=self.reader, args=(self.proc,), daemon=True).start()

    def reader(self, proc):
        for line in proc.stdout:
            self.q.put(("line", line))
        self.q.put(("exit", proc.wait()))

    def pump(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "line":
                    m = re.search(r"job folder: (.+)$", val.strip())
                    if m:
                        self.last_job = Path(m.group(1).strip())
                    if re.search(r"\] ffmpeg |\] yt-dlp ", val):
                        continue  # long command lines, just noise
                    bad = "Error" in val or "ERROR" in val or "Traceback" in val
                    self.say(val, "err" if bad else None)
                else:
                    self.proc = None
                    self.run_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    if val == 0:
                        self.status.config(text="Done ✓")
                        self.say("Done. The scores report opens now (clips ranked best first). "
                                 "Watch every clip before posting.\n", "ok")
                        job = self.current_job()
                        if job and (job / "report.html").exists():
                            open_path(job / "report.html")
                    else:
                        self.status.config(text=f"Failed (code {val})")
                        self.say(self.hint_for_failure(), "err")
        except queue.Empty:
            pass
        self.after(100, self.pump)

    def hint_for_failure(self):
        text = self.log.get("end-60l", "end")
        if "Sign in to confirm" in text or "age-restricted" in text:
            return ("Failed: YouTube wants a signed-in, age-verified account. Export a fresh cookies.txt "
                    "(private window, youtube.com) into the clipper folder and try again.\n")
        if "out of memory" in text.lower() or "CUDA" in text:
            return "Failed: GPU ran out of memory. Set Whisper model to 'small' and try again.\n"
        return "Failed. Read the red lines above, or paste the log to Claude.\n"

    def stop_run(self):
        if self.proc:
            self.proc.kill()
            self.say("Stopped.\n", "err")

    def current_job(self):
        if self.last_job:
            return self.last_job
        tab = self.nb.index(self.nb.select())
        name = (self.st_name if tab == 0 else self.cp_name).get().strip()
        return JOBS / name if name else None

    def open_clips(self):
        job = self.current_job()
        open_path((job / "clips") if job and (job / "clips").exists() else JOBS)

    def open_report(self):
        job = self.current_job()
        if job and (job / "report.html").exists():
            open_path(job / "report.html")
        else:
            messagebox.showinfo("Clipper", "No scores report for this job yet. Make clips first.")

    def say(self, text, tag=None):
        self.log.insert("end", text, tag)
        self.log.see("end")

    def remember(self):
        self.s["tab"] = self.nb.index(self.nb.select())
        self.s["stream"] = dict(src=self.st_src.get(), name=self.st_name.get(), preset=self.st_preset.get(),
                                count=self.st_count.get(), picker=self.st_picker.get(),
                                whisper=self.st_whisper.get(), lang=self.st_lang.get(),
                                cpu=self.st_cpu.get(), nocap=self.st_nocap.get())
        self.s["campaign"] = dict(src=self.cp_src.get(), name=self.cp_name.get(), times=self.cp_times.get(),
                                  text=self.cp_text.get(), preset=self.cp_preset.get(), box=self.cp_box.get(),
                                  use_cover=self.cp_use_cover.get(), cover=self.cp_cover.get(),
                                  cpu=self.cp_cpu.get(), lines=self.cp_lines.get("1.0", "end").strip())
        save_settings(self.s)

    def on_close(self):
        if self.proc and not messagebox.askyesno("Clipper", "Clips are still being made. Stop and quit?"):
            return
        if self.proc:
            self.proc.kill()
        self.remember()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
