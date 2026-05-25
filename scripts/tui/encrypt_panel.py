#!/usr/bin/env python3
"""Build encrypted TUI hidden-content slots.

Runtime format v2:
  manifest.bin = "SZEE2M\\0\\0" || salt16 || nonce12 || ChaCha20Poly1305(ciphertext+tag)
  N.bin        = "SZEE2A\\0\\0" || nonce12 || ChaCha20Poly1305(ciphertext+tag)

The manifest key is Argon2id(phrase, salt). Asset blobs reuse that slot
key with unique nonces. There is no phrase digest to paste into Rust;
unlock succeeds only when authenticated manifest decryption succeeds.

Example:
    python scripts/tui/encrypt_panel.py PHRASE assets/secret/a \\
        --add jsonl=assets/secret/raw/intro.jsonl \\
              wav=assets/secret/raw/intro.wav \\
              label="Intro sequence" kind=video

Requires:
    pip install argon2-cffi cryptography
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MANIFEST_MAGIC = b"SZEE2M\0\0"
ASSET_MAGIC = b"SZEE2A\0\0"
SALT_LEN = 16
NONCE_LEN = 12
KEY_LEN = 32
ARGON2_MEMORY_KIB = 64 * 1024
ARGON2_PASSES = 3
ARGON2_LANES = 1


def ensure_crypto_deps():
    try:
        from argon2.low_level import Type, hash_secret_raw
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
    except ImportError as error:
        print(f"error: missing crypto dependency: {error}", file=sys.stderr)
        print("install with: pip install argon2-cffi cryptography", file=sys.stderr)
        raise SystemExit(2) from error
    return Type, hash_secret_raw, ChaCha20Poly1305


def parse_add(tokens: list[str]) -> dict[str, str]:
    spec: dict[str, str] = {}
    joined = " ".join(tokens)
    pos = 0
    while pos < len(joined):
        eq = joined.find("=", pos)
        if eq < 0:
            break
        key = joined[pos:eq].strip()
        pos = eq + 1
        if not key:
            raise ValueError("empty key in --add")
        if pos < len(joined) and joined[pos] in ("'", '"'):
            quote = joined[pos]
            end = joined.find(quote, pos + 1)
            if end < 0:
                raise ValueError(f"unterminated quoted value for key {key!r}")
            value = joined[pos + 1 : end]
            pos = end + 1
        else:
            next_eq = joined.find("=", pos)
            if next_eq < 0:
                value = joined[pos:].strip()
                pos = len(joined)
            else:
                last_space = joined.rfind(" ", pos, next_eq)
                if last_space < 0:
                    raise ValueError(f"cannot split key {key!r} from next pair")
                value = joined[pos:last_space].strip()
                pos = last_space + 1
        spec[key] = value
    return spec


def derive_key(
    phrase: str,
    salt: bytes,
    *,
    memory_kib: int,
    passes: int,
    hash_secret_raw,
    argon2_type,
) -> bytes:
    return hash_secret_raw(
        phrase.encode("utf-8"),
        salt,
        time_cost=passes,
        memory_cost=memory_kib,
        parallelism=ARGON2_LANES,
        hash_len=KEY_LEN,
        type=argon2_type.ID,
    )


def encrypt_manifest(phrase: str, plaintext: bytes, memory_kib: int, passes: int) -> tuple[bytes, bytes]:
    argon2_type, hash_secret_raw, chacha = ensure_crypto_deps()
    salt = os.urandom(SALT_LEN)
    nonce = os.urandom(NONCE_LEN)
    key = derive_key(
        phrase,
        salt,
        memory_kib=memory_kib,
        passes=passes,
        hash_secret_raw=hash_secret_raw,
        argon2_type=argon2_type,
    )
    ciphertext = chacha(key).encrypt(nonce, plaintext, None)
    return MANIFEST_MAGIC + salt + nonce + ciphertext, key


def encrypt_asset(key: bytes, plaintext: bytes) -> bytes:
    _, _, chacha = ensure_crypto_deps()
    nonce = os.urandom(NONCE_LEN)
    ciphertext = chacha(key).encrypt(nonce, plaintext, None)
    return ASSET_MAGIC + nonce + ciphertext


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("phrase", help="secret phrase used to derive the slot key")
    parser.add_argument("out_dir", type=Path, help="output slot directory")
    parser.add_argument("--add", action="append", nargs="+", default=[], help="one entry: key=value tokens")
    parser.add_argument("--argon2-memory-kib", type=int, default=ARGON2_MEMORY_KIB)
    parser.add_argument("--argon2-passes", type=int, default=ARGON2_PASSES)
    args = parser.parse_args(argv)

    if args.argon2_memory_kib < 1024:
        print("error: --argon2-memory-kib must be at least 1024", file=sys.stderr)
        return 2
    if args.argon2_passes < 1:
        print("error: --argon2-passes must be at least 1", file=sys.stderr)
        return 2

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    assets: list[tuple[Path, int]] = []
    next_blob = 0
    for tokens in args.add:
        spec = parse_add(tokens)
        if "label" not in spec or "kind" not in spec:
            print(f"error: each --add needs label= and kind=: got {spec}", file=sys.stderr)
            return 2
        item = {
            "label": spec["label"],
            "kind": spec["kind"],
            "jsonl_index": None,
            "wav_index": None,
        }
        for asset_key, field_name in (("jsonl", "jsonl_index"), ("wav", "wav_index")):
            raw_path = spec.get(asset_key)
            if not raw_path:
                continue
            src = Path(raw_path)
            if not src.is_file():
                print(f"error: asset not found: {src}", file=sys.stderr)
                return 2
            item[field_name] = next_blob
            assets.append((src, next_blob))
            next_blob += 1
        items.append(item)

    manifest = {"items": items}
    manifest_bytes = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    encrypted_manifest, key = encrypt_manifest(
        args.phrase,
        manifest_bytes,
        memory_kib=args.argon2_memory_kib,
        passes=args.argon2_passes,
    )
    manifest_path = out_dir / "manifest.bin"
    manifest_path.write_bytes(encrypted_manifest)

    for src, index in assets:
        blob = src.read_bytes()
        encrypted = encrypt_asset(key, blob)
        dest = out_dir / f"{index}.bin"
        dest.write_bytes(encrypted)
        print(f"  encrypted {src} ({len(blob):,} -> {len(encrypted):,} bytes) -> {dest.name}")

    print(f"\nwrote {len(items)} item(s) to {manifest_path}")
    print("no Rust digest update is required; the TUI authenticates the encrypted manifest directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
