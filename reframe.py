"""Face-tracking reframe for the `track` layout preset.

Turns a wide (16:9) clip into 9:16 by following the main face, like Opus Clip's auto-reframe:
  1. sample the clip at ~4 fps on the CPU (keeps the 4 GB GPU free for Whisper)
  2. detect faces with OpenCV's YuNet model (models/face_detection_yunet_2023mar.onnx),
     falling back to OpenCV's bundled Haar cascade if the model is missing
  3. follow the biggest face, preferring the one closest to where the camera already is
  4. smooth it like a camera operator: hold still while the face stays near the middle,
     glide over ~0.4s when it moves away (no jittery panning)
Returns keyframes [(seconds_from_clip_start, center_x_as_fraction_of_width)] or None when
there is no usable face (then the caller falls back to the blur layout).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YUNET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")

SAMPLE_FPS = 4
DETECT_W = 480          # detection runs on a downscaled frame
MIN_COVERAGE = 0.25     # need a face in at least 25% of samples, else give up
DEADZONE = 0.09         # camera holds until the face drifts this far (fraction of width)
HOLD = 0.5              # ...and stays there this long (ignores glances / passing faces)
GLIDE = 0.4             # seconds to pan to a new position
JUMP = 0.25             # moves bigger than this are a hard cut instead of a pan (shot change)


def _detector(w, h, log):
    import cv2
    if not YUNET.exists():
        try:
            import urllib.request
            YUNET.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(YUNET_URL, YUNET)
        except Exception as e:
            log(f"reframe: couldn't download YuNet ({e}), using the Haar cascade")
    if YUNET.exists():
        det = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.7, 0.3, 50)

        def find(img):
            _, faces = det.detect(img)
            return [] if faces is None else [(f[0], f[1], f[2], f[3], float(f[-1])) for f in faces]
        return find
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def find(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return [(x, y, fw, fh, 1.0) for x, y, fw, fh in casc.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))]
    return find


def face_track(src, start, end, log=print):
    """Keyframes [(t_rel, cx_frac)] following the main face in src between start and end seconds."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        log("reframe: couldn't open the video")
        return None
    W0, H0 = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if not W0 or W0 / H0 <= 9 / 16 + 0.01:
        cap.release()
        return [(0.0, 0.5)]  # already vertical: nothing to track
    dw, dh = DETECT_W, int(round(DETECT_W * H0 / W0))
    find = _detector(dw, dh, log)
    half = (H0 * 9 / 16) / W0 / 2           # half the crop width, as a fraction of the frame
    lo, hi = half, 1 - half

    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    step = max(1, int(round(fps / SAMPLE_FPS)))
    n_frames = int((end - start) * fps)
    samples, prev = [], None
    for i in range(n_frames):
        if not cap.grab():
            break
        if i % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        faces = find(cv2.resize(frame, (dw, dh)))
        cx = None
        if faces:
            def rank(f):
                area = f[2] * f[3]
                c = (f[0] + f[2] / 2) / dw
                near = 1.0 if prev is None else 1.0 / (1.0 + 4 * abs(c - prev))
                return area * f[4] * near
            best = max(faces, key=rank)
            cx = (best[0] + best[2] / 2) / dw
            prev = cx
        samples.append((i / fps, cx))
    cap.release()

    if not samples:
        return None
    found = [x for _, x in samples if x is not None]
    if len(found) / len(samples) < MIN_COVERAGE:
        log(f"reframe: face in only {100 * len(found) / len(samples):.0f}% of the clip, using blur layout")
        return None

    # fill gaps (no face) with the nearest known position, then median-smooth
    xs = [x for _, x in samples]
    last = next(x for x in xs if x is not None)
    for k, x in enumerate(xs):
        if x is None:
            xs[k] = last
        else:
            last = x
    xs = np.array(xs, dtype=float)
    pad = np.pad(xs, 2, mode="edge")
    xs = np.array([np.median(pad[k:k + 5]) for k in range(len(xs))])
    xs = np.clip(xs, lo, hi)
    ts = [t for t, _ in samples]

    # camera-operator smoothing: hold still; move only when the face has stayed outside the dead zone
    # for HOLD seconds. Small moves glide; big jumps (usually a shot change) cut instantly.
    cam = float(xs[0])
    keys = [(0.0, cam)]
    away_since = None
    busy_until = 0.0
    for t, x in zip(ts, xs):
        if t < busy_until:
            continue
        if abs(x - cam) <= DEADZONE:
            away_since = None
            continue
        if away_since is None:
            away_since = t
        if t - away_since < HOLD:
            continue
        t0 = away_since                      # move from when the face actually left
        target = float(np.median(xs[ts.index(t0): ts.index(t) + 1]))
        glide = GLIDE if abs(target - cam) < JUMP else 0.04
        keys.append((round(t0, 2), cam))
        cam = target
        keys.append((round(t0 + glide, 2), cam))
        busy_until = t0 + glide
        away_since = None
    return keys


def crop_x_expr(keys):
    """ffmpeg expression for the crop's x (pixels), piecewise-linear through the keyframes."""
    expr = f"{keys[0][1]:.4f}"
    for (t0, x0), (t1, x1) in zip(keys, keys[1:]):
        if t1 <= t0 or abs(x1 - x0) < 1e-4:
            continue
        expr += f"+({x1 - x0:.4f})*clip((t-{t0:.2f})/{t1 - t0:.2f}\\,0\\,1)"
    return f"clip(iw*({expr})-ow/2\\,0\\,iw-ow)"
