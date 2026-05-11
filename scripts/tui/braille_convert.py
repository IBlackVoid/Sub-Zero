#!/usr/bin/env python3
"""High-fidelity GIF/video → braille JSONL converter.

Produces the cells JSONL the Sub-Zero TUI consumes, but using real
Unicode braille (U+2800..U+28FF) for 2×4 subpixel resolution per cell.
A single 80×40 grid resolves 160×160 effective pixels — matching the
high-density "Drippy Coo" reference style.

Usage:
  python scripts/tui/braille_convert.py input.gif --width 80 --out idle.jsonl
  python scripts/tui/braille_convert.py input.gif --width 100 --palette sub-zero
  python scripts/tui/braille_convert.py inputs/*.gif --batch state_mapping.json

Wire format matches Pixel-Ripper's --output-format cells:
  {"frame":N,"delay_ms":D,"width":W,"height":H,
   "rows":[[[ch,r,g,b], ...], ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Force stdout/stderr to UTF-8 so braille progress messages survive on
# Windows consoles whose default code page is cp1252.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import numpy as np
from PIL import Image, ImageSequence


# ---------------------------------------------------------------------------
# Palette: SUB_ZERO matching the HTML mockup AC table.
# ---------------------------------------------------------------------------
SUB_ZERO_PALETTE: list[tuple[int, int, int]] = [
    (0xf0, 0xc0, 0x20),  # Y - bow yellow
    (0xc8, 0xa0, 0x10),  # y - bow shadow
    (0x7a, 0x30, 0x10),  # H - hair dark
    (0x9a, 0x48, 0x28),  # h - hair mid
    (0xc0, 0x88, 0x48),  # F - face border
    (0xc8, 0x90, 0x4a),  # S - skin
    (0xa0, 0x70, 0x30),  # s - skin shadow
    (0x18, 0x0c, 0x06),  # E - eye dark
    (0x7a, 0x58, 0x20),  # i - iris
    (0xf8, 0xec, 0xd8),  # W - shine
    (0xc8, 0x30, 0x50),  # L - lip
    (0x98, 0x18, 0x38),  # l - lip shadow
    (0xd0, 0x68, 0x20),  # O - orange (TUI accent)
    (0x40, 0xa0, 0xa8),  # C - cyan
    (0x50, 0xc8, 0x70),  # G - status green
    (0xc0, 0x50, 0x50),  # R - status red
    (0x0c, 0x0c, 0x0c),  # background
    (0xc0, 0xc0, 0xc0),  # body grey
]


# Braille dots are arranged in this 2×4 layout (column-major within each
# 2-wide block). Bit numbers are the official Unicode braille bit map.
#   col0 col1
#    1    4   row0
#    2    5   row1
#    3    6   row2
#    7    8   row3
BRAILLE_BITS = [
    [0, 3],
    [1, 4],
    [2, 5],
    [6, 7],
]


def _floyd_steinberg(luma: np.ndarray, bisect: float) -> np.ndarray:
    """Floyd-Steinberg error-diffusion dithering in luma units."""
    h, w = luma.shape
    out = np.zeros_like(luma, dtype=bool)
    err = luma.astype(np.float64).copy()
    for y in range(h):
        for x in range(w):
            old = err[y, x]
            new = 255.0 if old >= bisect else 0.0
            out[y, x] = new > 0.0
            quant_err = old - new
            if x + 1 < w:
                err[y, x + 1] += quant_err * (7.0 / 16.0)
            if y + 1 < h:
                if x > 0:
                    err[y + 1, x - 1] += quant_err * (3.0 / 16.0)
                err[y + 1, x] += quant_err * (5.0 / 16.0)
                if x + 1 < w:
                    err[y + 1, x + 1] += quant_err * (1.0 / 16.0)
    return out


def quantize_to_palette(rgb: tuple[float, float, float],
                        palette: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """Snap an RGB triple to the nearest palette entry by Euclidean distance."""
    r, g, b = rgb
    pal = np.asarray(palette, dtype=np.float32)
    target = np.asarray([r, g, b], dtype=np.float32)
    diffs = pal - target
    dists = np.einsum("ij,ij->i", diffs, diffs)
    return tuple(palette[int(np.argmin(dists))])


def frame_to_braille_cells(
    img: Image.Image,
    cell_w: int,
    cell_h: int,
    *,
    threshold_pct: int = 50,
    use_palette: bool = False,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    gamma: float = 1.0,
    dither: bool = True,
) -> list[list[tuple[str, int, int, int]]]:
    """Convert a single PIL image to a 2D grid of (char, r, g, b) cells.

    Each cell consumes a 2-wide × 4-tall block of pixels; the input is
    resampled to (cell_w*2, cell_h*4) before sampling.

    brightness / contrast / saturation let callers stylise per-state
    (e.g. error = 0.55 / 1.4 / 0.4 for a darker, harsher look).

    When `dither` is True (default), Floyd-Steinberg error diffusion
    decides which sub-pixels are lit. The threshold then represents
    the bisection point of the binary decision rather than a percentile
    cut — bright regions retain texture instead of going solid ⣿,
    dark regions retain detail instead of going blank. Disable for
    line-art or stylised illustrations where the threshold pass
    produces cleaner edges.
    """
    target_w = cell_w * 2
    target_h = cell_h * 4
    rgb_img = img.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    arr = np.asarray(rgb_img, dtype=np.float32)

    if (brightness != 1.0 or contrast != 1.0
            or saturation != 1.0 or gamma != 1.0):
        # Apply image-wide tone adjustments before braille conversion so
        # the threshold/dither pass works against the *adjusted* luma.
        if gamma != 1.0:
            # Gamma compresses (γ < 1) or expands (γ > 1) the dynamic
            # range. Bright TV-broadcast content blows out the highlights
            # because everything sits in the top decile of luma; γ ≈ 0.85
            # pulls the bright end down and lifts midtones, recovering
            # detail in faces and stage objects.
            normalized = arr / 255.0
            arr = np.power(np.clip(normalized, 0, 1), gamma) * 255.0
        if contrast != 1.0:
            arr = (arr - 128.0) * contrast + 128.0
        if brightness != 1.0:
            arr = arr * brightness
        if saturation != 1.0:
            luma3 = (0.2126 * arr[..., 0:1]
                     + 0.7152 * arr[..., 1:2]
                     + 0.0722 * arr[..., 2:3])
            arr = luma3 + (arr - luma3) * saturation
        arr = np.clip(arr, 0, 255)
    arr = arr.astype(np.uint8)

    # Brightness in [0, 255]. Rec. 709 luma weights.
    luma = (0.2126 * arr[:, :, 0]
            + 0.7152 * arr[:, :, 1]
            + 0.0722 * arr[:, :, 2]).astype(np.float32)

    # Compute the lit-mask either by Floyd-Steinberg error diffusion
    # (default — preserves gradients in bright/dark regions) or by a
    # simple percentile threshold (kept for line-art).
    if dither:
        # Bisect at the percentile so the user-supplied --threshold
        # still controls average density. Using 128 alone would lose
        # the brightness/contrast knobs.
        bisect = float(np.percentile(luma, threshold_pct))
        lit_mask = _floyd_steinberg(luma.copy(), bisect)
    else:
        threshold = float(np.percentile(luma, threshold_pct))
        lit_mask = luma > threshold

    rows: list[list[tuple[str, int, int, int]]] = []
    for cy in range(cell_h):
        row: list[tuple[str, int, int, int]] = []
        for cx in range(cell_w):
            bits = 0
            lit_pixels: list[tuple[int, int, int]] = []
            for dy in range(4):
                for dx in range(2):
                    px = cx * 2 + dx
                    py = cy * 4 + dy
                    if lit_mask[py, px]:
                        bits |= 1 << BRAILLE_BITS[dy][dx]
                        r, g, b = arr[py, px]
                        lit_pixels.append((int(r), int(g), int(b)))
            ch = chr(0x2800 + bits)
            if lit_pixels:
                # Average the colours of the lit subpixels — gives the cell
                # its character colour while staying anchored to the source.
                rs, gs, bs = zip(*lit_pixels)
                avg = (sum(rs) / len(rs), sum(gs) / len(gs), sum(bs) / len(bs))
            else:
                # Empty cell: pick the local block's average colour anyway
                # so the empty space still carries faint colour cues if a
                # later renderer wants to use the bg channel.
                block = arr[cy * 4:(cy + 1) * 4, cx * 2:(cx + 1) * 2]
                avg = tuple(block.reshape(-1, 3).mean(axis=0))
            if use_palette:
                r, g, b = quantize_to_palette(avg, SUB_ZERO_PALETTE)
            else:
                r, g, b = int(round(avg[0])), int(round(avg[1])), int(round(avg[2]))
            row.append((ch, r, g, b))
        rows.append(row)
    return rows


def gif_frames(img: Image.Image):
    """Iterate (frame_image, delay_ms) over an animated PIL image. Falls back
    to a single still frame for non-animated inputs."""
    try:
        for frame in ImageSequence.Iterator(img):
            delay = int(frame.info.get("duration") or 80)
            yield frame.copy(), max(delay, 20)
    except Exception:
        yield img, 0


# Extensions Pillow can decode directly. Anything else is treated as a
# video and routed through ffmpeg.
PILLOW_EXTENSIONS = {".gif", ".png", ".webp", ".apng",
                     ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def extract_video_audio(path: Path,
                        out_wav: Path,
                        *,
                        start_sec: float = 0.0,
                        duration_sec: float | None = None) -> bool:
    """Extract the audio track of a video to a WAV sidecar.

    Returns True on success. Returns False (and removes any partial
    output) if ffmpeg fails or the video has no audio stream — both
    cases produce no warning, since silent video is a normal case.

    The sidecar is mono / 44.1 kHz / 16-bit PCM, optimised for the
    `rodio` decoder used by the TUI runtime.
    """
    import subprocess

    cmd = ["ffmpeg", "-loglevel", "error", "-y",
           "-ss", f"{max(start_sec, 0.0):.3f}"]
    if duration_sec is not None:
        cmd += ["-t", f"{max(duration_sec, 0.001):.3f}"]
    cmd += ["-i", str(path),
            "-vn",                 # no video
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            str(out_wav)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=600)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        out_wav.unlink(missing_ok=True)
        return False
    if proc.returncode != 0:
        out_wav.unlink(missing_ok=True)
        return False
    if not out_wav.is_file() or out_wav.stat().st_size == 0:
        out_wav.unlink(missing_ok=True)
        return False
    return True


def video_frames(path: Path,
                 *,
                 start_sec: float = 0.0,
                 duration_sec: float | None = None,
                 fps: int = 12):
    """Iterate (frame_image, delay_ms) over a video file via ffmpeg.

    Spawns ``ffmpeg -i <path> -ss <start> -t <duration> -r <fps>
    -f image2pipe -vcodec ppm -`` and parses each PPM frame off the
    stdout stream. Works for every codec ffmpeg knows (h264, av1,
    vp9, hevc, ...) and adds zero Python dependencies.

    The frame delay is uniform and derived from ``fps`` so the produced
    JSONL animates at exactly the intended rate, regardless of the
    source video's variable frame timing.
    """
    import subprocess

    cmd = ["ffmpeg", "-loglevel", "error",
           "-ss", f"{max(start_sec, 0.0):.3f}"]
    if duration_sec is not None:
        cmd += ["-t", f"{max(duration_sec, 0.001):.3f}"]
    cmd += ["-i", str(path),
            "-r", str(int(fps)),
            "-f", "image2pipe",
            "-vcodec", "ppm",
            "-"]

    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            bufsize=10 * 1024 * 1024)
    delay_ms = max(int(round(1000.0 / max(fps, 1))), 20)
    try:
        while True:
            frame = _read_ppm_frame(proc.stdout)
            if frame is None:
                break
            yield frame, delay_ms
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        rc = proc.wait()
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        if rc not in (0, None) and err.strip():
            print(f"warning: ffmpeg exited {rc}: {err.strip()[:200]}",
                  file=sys.stderr)


def _read_ppm_frame(stream) -> Image.Image | None:
    """Read one binary PPM frame off a pipe and return it as a PIL Image.

    PPM (P6) layout:
        line 1: "P6\\n"
        line 2: "<width> <height>\\n"   (may be preceded by '# comment\\n' lines)
        line 3: "<maxval>\\n"
        bytes : width * height * 3 raw RGB
    """
    if stream is None:
        return None
    magic = stream.readline()
    if not magic:
        return None
    magic = magic.strip()
    if magic != b"P6":
        # End of stream or non-PPM data; bail.
        return None
    # Header lines may include comments starting with '#'.
    def _next_token_line():
        while True:
            ln = stream.readline()
            if not ln:
                return b""
            if not ln.lstrip().startswith(b"#"):
                return ln
    dims = _next_token_line().strip().split()
    if len(dims) != 2:
        return None
    w, h = int(dims[0]), int(dims[1])
    maxval_line = _next_token_line().strip()
    if not maxval_line:
        return None
    maxval = int(maxval_line)
    if maxval != 255:
        # Pillow expects 8-bit data; ffmpeg's default is 255 so this
        # branch should be rare. Bail rather than silently corrupting.
        return None
    raw = stream.read(w * h * 3)
    if len(raw) != w * h * 3:
        return None
    return Image.frombytes("RGB", (w, h), raw)


def _probe_video_dims(path: Path) -> tuple[int, int]:
    """Best-effort (width, height) probe of a video via ffprobe."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", str(path)],
            stderr=subprocess.DEVNULL,
            timeout=20,
        ).decode().strip()
        w, h = out.split("x")[:2]
        return int(w), int(h)
    except Exception:
        return 1920, 1080  # safe fallback aspect


def convert(
    in_path: Path,
    out_path: Path,
    *,
    cell_w: int,
    cell_h: int | None,
    threshold_pct: int,
    use_palette: bool,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    gamma: float = 1.0,
    # Video-only knobs. Ignored for GIFs/images.
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    fps: int = 12,
    extract_audio: bool = True,
    dither: bool = True,
) -> int:
    suffix = in_path.suffix.lower()
    is_video = suffix not in PILLOW_EXTENSIONS

    if is_video:
        src_w, src_h = _probe_video_dims(in_path)
        frame_iter = video_frames(in_path,
                                  start_sec=start_sec,
                                  duration_sec=duration_sec,
                                  fps=fps)
        if extract_audio:
            audio_out = out_path.with_suffix(".wav")
            ok = extract_video_audio(in_path, audio_out,
                                     start_sec=start_sec,
                                     duration_sec=duration_sec)
            if ok:
                print(f"  audio sidecar: {audio_out.name}", file=sys.stderr)
    else:
        img = Image.open(in_path)
        src_w, src_h = img.size
        frame_iter = gif_frames(img)

    if cell_h is None:
        # Default keeps source aspect ratio. Each cell is 2×4 pixels and
        # terminal cells are roughly 1×2 pixels tall, so the *visual*
        # aspect of one braille cell is ≈ 1.0 wide : 2.0 tall.
        cell_h = max(1, int(round(cell_w * (src_h / src_w) * (2.0 / 4.0))))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, (frame, delay) in enumerate(frame_iter):
            cells = frame_to_braille_cells(
                frame,
                cell_w,
                cell_h,
                threshold_pct=threshold_pct,
                use_palette=use_palette,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                gamma=gamma,
                dither=dither,
            )
            payload = {
                "frame": idx,
                "delay_ms": delay,
                "width": cell_w,
                "height": cell_h,
                "rows": [
                    [[ch, r, g, b] for (ch, r, g, b) in row]
                    for row in cells
                ],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            n += 1
    return n


def batch(
    mapping_path: Path,
    out_dir: Path,
    *,
    cell_w: int,
    threshold_pct: int,
    use_palette: bool,
) -> int:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    states = mapping["states"]
    search_dirs = [Path(d) for d in mapping.get("search_dirs") or []]
    extensions = mapping.get("extensions") or [".gif", ".mp4", ".png", ".jpg"]

    def find_source(stem: str) -> Path | None:
        for d in search_dirs:
            for ext in extensions:
                p = d / (stem + ext)
                if p.is_file():
                    return p
        return None

    total = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for state, entry in states.items():
        if state.startswith("_"):
            continue
        src = find_source(entry["source"])
        if not src:
            print(f"warning: source for state '{state}' not found "
                  f"(stem '{entry['source']}')", file=sys.stderr)
            continue
        out_path = out_dir / f"{state}.jsonl"
        # Per-state tone knobs from state_mapping.json. Existing entries
        # use the names from Pixel-Ripper's CLI (brightness / saturate /
        # edge); we accept either spelling and ignore unknowns.
        brightness = float(entry.get("brightness", 1.0))
        contrast   = float(entry.get("contrast",   1.0 + (entry.get("edge", 0.4) - 0.4) * 0.5))
        saturation = float(entry.get("saturation",
                                      entry.get("saturate", 1.0)))
        # Optional per-state video trim knobs.
        start_sec    = float(entry.get("start", 0.0))
        duration_sec = entry.get("duration")
        if duration_sec is not None:
            duration_sec = float(duration_sec)
        fps          = int(entry.get("fps", 12))

        n = convert(
            src,
            out_path,
            cell_w=cell_w,
            cell_h=None,
            threshold_pct=threshold_pct,
            use_palette=use_palette,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            start_sec=start_sec,
            duration_sec=duration_sec,
            fps=fps,
        )
        size = out_path.stat().st_size
        print(f"  {state:10s} {src.name:36s} → {out_path.name:18s} "
              f"{n:3d} frames  {size:>10,} bytes")
        total += n
    return total


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", nargs="?", help="GIF / image / video file")
    p.add_argument("--out", help="Output JSONL path")
    p.add_argument("--width", type=int, default=80, help="Cell width (default: 80)")
    p.add_argument("--height", type=int, help="Cell height (default: aspect-derived)")
    p.add_argument("--threshold", type=int, default=50,
                   help="Brightness percentile that turns a subpixel on (default: 50)")
    p.add_argument("--palette", choices=["sub-zero", "source"], default="source",
                   help="Quantize colours to SUB_ZERO palette or keep source colours")
    p.add_argument("--batch", help="Path to a state_mapping.json — converts every state")
    p.add_argument("--out-dir", default="assets/ascii", help="Batch output directory")
    # Video-only knobs.
    p.add_argument("--start", type=float, default=0.0,
                   help="(video) start time in seconds")
    p.add_argument("--duration", type=float, default=None,
                   help="(video) duration in seconds (default: full video)")
    p.add_argument("--fps", type=int, default=12,
                   help="(video) target frame rate (default: 12)")
    p.add_argument("--no-audio", action="store_true",
                   help="(video) skip audio sidecar extraction")
    p.add_argument("--no-dither", action="store_true",
                   help="disable Floyd-Steinberg dithering (use plain percentile threshold)")
    p.add_argument("--gamma", type=float, default=None,
                   help="tone curve (default: 0.85 for video, 1.0 for images/gif)")
    args = p.parse_args(argv)

    use_palette = (args.palette == "sub-zero")

    if args.batch:
        total = batch(
            Path(args.batch),
            Path(args.out_dir),
            cell_w=args.width,
            threshold_pct=args.threshold,
            use_palette=use_palette,
        )
        print(f"\ntotal: {total} frames")
        return 0

    if not args.input or not args.out:
        p.error("either --batch or both INPUT and --out are required")

    # Default gamma differs by input type: videos blow out the bright end
    # (γ < 1 lifts midtones), still images / GIFs are usually already
    # tone-mapped (γ = 1).
    is_video_input = Path(args.input).suffix.lower() not in PILLOW_EXTENSIONS
    gamma = args.gamma if args.gamma is not None else (0.85 if is_video_input else 1.0)

    n = convert(
        Path(args.input),
        Path(args.out),
        cell_w=args.width,
        cell_h=args.height,
        threshold_pct=args.threshold,
        use_palette=use_palette,
        start_sec=args.start,
        duration_sec=args.duration,
        fps=args.fps,
        extract_audio=not args.no_audio,
        dither=not args.no_dither,
        gamma=gamma,
    )
    print(f"wrote {n} frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
