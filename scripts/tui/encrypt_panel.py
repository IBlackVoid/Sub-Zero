#!/usr/bin/env python3
"""Offline encryption tool for the TUI's hidden-content slots.

Mirrors the SHA-256-CTR scheme from `tui/src/secret.rs` byte-for-byte
so the Rust runtime can decrypt what this tool produces.

Usage:

    python scripts/tui/encrypt_panel.py PHRASE OUT_DIR \\
        --add jsonl=assets/secret/raw/terry_davis.jsonl \\
              label="Terry Davis — smartest programmer" kind=video \\
        --add jsonl=assets/secret/raw/haruhi_op.jsonl \\
              label="Haruhi - Bouken Desho Desho?" kind=video \\
        --add jsonl=assets/secret/raw/twgok_cover.jsonl \\
              wav=assets/secret/raw/twgok_theme.wav \\
              label="Koi - Kuchizuke Made no Kyori" kind=music

Each `--add` accepts `key=value` tokens (whitespace-separated) and
becomes one HiddenItem in the manifest. Recognised keys:
  label  : human-readable name (required)
  kind   : "video" | "music" (required)
  jsonl  : path to a JSONL animation to encrypt + bundle (optional)
  wav    : path to a WAV audio file to encrypt + bundle (optional)

Outputs the encrypted manifest at `OUT_DIR/manifest.bin` and one
encrypted blob per included asset at `OUT_DIR/{N}.bin`. Also prints
the phrase's SHA-256 digest to stdout as a Rust `[u8; 32]` literal
so you can paste it into `tui/src/digests.rs`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

KEY_SALT = b"\x00sub-zero-easter-v1"


def digest_of(phrase: str) -> bytes:
    return hashlib.sha256(phrase.encode("utf-8")).digest()


def derive_key(phrase: str) -> bytes:
    h = hashlib.sha256()
    h.update(phrase.encode("utf-8"))
    h.update(KEY_SALT)
    return h.digest()


def xor_stream(data: bytes, key: bytes) -> bytes:
    out = bytearray(data)
    counter = 0
    offset = 0
    while offset < len(out):
        h = hashlib.sha256()
        h.update(key)
        h.update(counter.to_bytes(8, "little"))
        block = h.digest()
        take = min(32, len(out) - offset)
        for i in range(take):
            out[offset + i] ^= block[i]
        offset += take
        counter = (counter + 1) & 0xFFFFFFFFFFFFFFFF
    return bytes(out)


def parse_add(tokens: list[str]) -> dict:
    spec: dict = {}
    # Re-glue tokens so a quoted value with spaces is allowed.
    joined = " ".join(tokens)
    # naive key=value parser supporting one quoted span per key
    pos = 0
    while pos < len(joined):
        eq = joined.find("=", pos)
        if eq < 0:
            break
        key = joined[pos:eq].strip()
        pos = eq + 1
        if pos < len(joined) and joined[pos] in ("'", '"'):
            q = joined[pos]
            end = joined.find(q, pos + 1)
            if end < 0:
                raise ValueError(f"unterminated quoted value for key {key!r}")
            value = joined[pos + 1 : end]
            pos = end + 1
        else:
            # value runs until next ' key=' or end
            next_eq = joined.find("=", pos)
            if next_eq < 0:
                value = joined[pos:].strip()
                pos = len(joined)
            else:
                # last space before next_eq belongs to next key
                last_space = joined.rfind(" ", pos, next_eq)
                if last_space < 0:
                    raise ValueError(f"cannot split key {key!r} from next pair")
                value = joined[pos:last_space].strip()
                pos = last_space + 1
        spec[key] = value
    return spec


def encode_digest_literal(d: bytes) -> str:
    rows = []
    for i in range(0, 32, 16):
        rows.append(", ".join(f"0x{b:02x}" for b in d[i : i + 16]))
    return "[\n    " + ",\n    ".join(rows) + ",\n]"


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phrase", help="secret phrase (used as the encryption key)")
    p.add_argument("out_dir", type=Path, help="output directory")
    p.add_argument("--add", action="append", nargs="+", default=[],
                   help="one entry: key=value tokens")
    args = p.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    key = derive_key(args.phrase)
    items = []
    next_blob = 0
    for tokens in args.add:
        spec = parse_add(tokens)
        if "label" not in spec or "kind" not in spec:
            print(f"error: each --add needs label= and kind=: got {spec}",
                  file=sys.stderr)
            return 2
        item = {
            "label": spec["label"],
            "kind": spec["kind"],
            "jsonl_index": None,
            "wav_index": None,
        }
        for assetkey, fieldname in (("jsonl", "jsonl_index"), ("wav", "wav_index")):
            path = spec.get(assetkey)
            if not path:
                continue
            src = Path(path)
            if not src.is_file():
                print(f"error: asset not found: {src}", file=sys.stderr)
                return 2
            blob = src.read_bytes()
            encrypted = xor_stream(blob, key)
            dest = out_dir / f"{next_blob}.bin"
            dest.write_bytes(encrypted)
            item[fieldname] = next_blob
            next_blob += 1
            print(f"  encrypted {src}  ({len(blob):,} → {len(encrypted):,} bytes) → {dest.name}")
        items.append(item)

    manifest = {"items": items}
    manifest_bytes = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    encrypted_manifest = xor_stream(manifest_bytes, key)
    manifest_path = out_dir / "manifest.bin"
    manifest_path.write_bytes(encrypted_manifest)
    print(f"\nwrote {len(items)} item(s) to {manifest_path}")
    print()
    print(f"verification digest for this phrase (paste into tui/src/digests.rs):")
    print(encode_digest_literal(digest_of(args.phrase)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
