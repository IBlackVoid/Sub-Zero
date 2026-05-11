#!/usr/bin/env python3
"""Compute a 50ms-resolution amplitude envelope for a WAV file.

Produces `<wav>.env.bin` — a flat little-endian f32 array, one sample
per 50 ms of audio, normalised to [0, 1]. The TUI reads this sidecar
to draw a live waveform display while the audio plays, without
tapping into the playback sink directly.

Usage:
    python scripts/tui/build_envelope.py path/to/file.wav
    python scripts/tui/build_envelope.py *.wav

Pure stdlib (uses the `wave` module) + numpy — no heavy DSP deps.
"""

from __future__ import annotations

import argparse
import struct
import sys
import wave
from pathlib import Path

import numpy as np

WINDOW_MS = 50  # must match Envelope::INTERVAL_MS in tui/src/waveform.rs


def envelope_for_wav(wav_path: Path) -> bytes:
    with wave.open(str(wav_path), "rb") as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)

    if sampwidth == 2:
        dtype = np.int16
        peak = 32_768.0
    elif sampwidth == 1:
        dtype = np.uint8
        peak = 128.0
    elif sampwidth == 4:
        dtype = np.int32
        peak = float(2 ** 31)
    else:
        raise SystemExit(f"unsupported sample width: {sampwidth} bytes")

    arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if sampwidth == 1:
        arr -= peak  # center unsigned 8-bit around 0
    arr /= peak  # → roughly [-1, 1]
    if channels > 1:
        arr = arr.reshape(-1, channels).mean(axis=1)

    frames_per_window = max(1, int(framerate * WINDOW_MS / 1000.0))
    total_windows = max(1, len(arr) // frames_per_window)
    arr = arr[: total_windows * frames_per_window]
    framed = arr.reshape(total_windows, frames_per_window)
    # RMS gives a perceptually steadier envelope than peak.
    rms = np.sqrt((framed ** 2).mean(axis=1))
    # Normalise to [0, 1] by the loudest window, with a small floor so
    # silent files don't divide by zero.
    peak_rms = max(float(rms.max()), 1e-6)
    rms = rms / peak_rms
    return b"".join(struct.pack("<f", float(v)) for v in rms.astype(np.float32))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+", help="WAV file(s) to process")
    args = p.parse_args(argv)
    for inp in args.inputs:
        wav_path = Path(inp)
        if not wav_path.is_file():
            print(f"skip: not a file: {wav_path}", file=sys.stderr)
            continue
        try:
            data = envelope_for_wav(wav_path)
        except Exception as e:
            print(f"error: {wav_path}: {e}", file=sys.stderr)
            continue
        out = wav_path.with_suffix(".env.bin")
        out.write_bytes(data)
        print(f"  {wav_path.name}  →  {out.name}  ({len(data) // 4} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
