#!/usr/bin/env python3
"""Sub-Zero Neural MT Backend — CTranslate2 + NLLB-200.

Reads a JSON array of translation requests from stdin,
writes a JSON array of responses to stdout.

Request format:
[
  {
    "index": 1,
    "text": "こんにちは世界",
    "prev_context": ["前の行"],
    "next_context": ["次の行"],
    "context_tags": ["scene_hard", "cue_fast"]
  },
  ...
]

Response format:
[
  {"index": 1, "translation": "Hello world"},
  ...
]
"""

import argparse
import ctypes
import glob
import json
import os
import site
import sys


def main():
    parser = argparse.ArgumentParser(description="Sub-Zero Neural MT Backend")
    parser.add_argument("--model", default="nllb-200-distilled-600M",
                        help="NLLB model name or path")
    parser.add_argument("--model-dir", default=None,
                        help="Directory containing the CTranslate2 model")
    parser.add_argument("--source-lang", default="jpn_Jpan",
                        help="Source language (NLLB code)")
    parser.add_argument("--target-lang", default="eng_Latn",
                        help="Target language (NLLB code)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="Inference device")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for translation")
    parser.add_argument("--max-batch-tokens", type=int, default=8192,
                        help="Maximum token budget per translation batch")
    parser.add_argument("--beam-size", type=int, default=4,
                        help="Beam size for translation")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3,
                        help="N-gram size for repeat blocking (0 disables)")
    parser.add_argument("--oom-retries", type=int, default=2,
                        help="Number of CUDA OOM retries before failure")
    parser.add_argument("--allow-cpu-fallback", action="store_true",
                        help="Allow fallback to CPU when CUDA OOM persists")
    parser.add_argument("--prepend-prev-context", action="store_true",
                        help="Prepend the previous subtitle line as a hint")
    args = parser.parse_args()

    # Read requests from stdin.
    raw_input = sys.stdin.read()
    requests = json.loads(raw_input)

    if not requests:
        print("[]")
        return

    if args.device == "cuda":
        _configure_cuda_runtime()

    try:
        import ctranslate2
        import sentencepiece as spm
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print(f"Install with: pip install ctranslate2 sentencepiece", file=sys.stderr)
        sys.exit(1)

    # Resolve model path.
    model_path = resolve_model_path(args.model, args.model_dir)
    if model_path is None:
        expected_dir = os.path.join("models", os.path.basename(args.model.rstrip("/")))
        print(f"Model not found: {args.model}", file=sys.stderr)
        print(
            f"Download with: ct2-nllb200-converter --model {args.model} --output_dir {expected_dir}",
            file=sys.stderr,
        )
        print(
            "Or pass --model-dir <path-to-converted-ctranslate2-model>",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"sub-zero: mt_model_path={model_path}", file=sys.stderr)

    # Load model and tokenizer.
    sp_model_path = find_sentencepiece_model(model_path)
    if sp_model_path is None:
        print(f"SentencePiece model not found in {model_path}", file=sys.stderr)
        sys.exit(1)

    device = args.device

    # CTranslate2 may lazily load CUDA libs, so construction can succeed even
    # when libcublas is missing.  We always try the requested device first and
    # fall back to CPU on any RuntimeError (e.g. missing CUDA libraries).
    translator = _load_translator(model_path, device, args.allow_cpu_fallback)
    if translator is None:
        print("[]")
        return
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)

    source_prefix = [args.source_lang]
    target_prefix = [args.target_lang]

    # Prepare texts — include context for better translation quality.
    texts_to_translate = []
    for req in requests:
        # Build contextual input: just send the main text for NLLB
        # (context-window concatenation is experimental, we send the primary
        # text and rely on the batch ordering for implicit context.)
        text = req.get("text", "").strip()
        if not text:
            text = " "

        # For NLLB, we can prepend a brief context hint without confusing the
        # model too much — just the previous line as a gentle signal.
        prev = req.get("prev_context", [])
        if args.prepend_prev_context and prev:
            # Use the last previous line as a context hint (separated by newline).
            context_hint = prev[-1].strip()
            if context_hint:
                text = context_hint + " " + text

        # Scrub invalid characters/surrogates that crash sentencepiece's PyBind11 C++ parser
        text = text.encode('utf-8', 'replace').decode('utf-8')

        texts_to_translate.append(text)

    # Tokenize.
    tokenized = [
        source_prefix + sp.Encode(txt, out_type=str) + ["</s>"]
        for txt in texts_to_translate
    ]

    # Translate with adaptive policy:
    # - hard-tagged cues get slightly stronger decode (higher beam, smaller token batch)
    # - base cues keep default decode settings
    # both paths still use the OOM ladder.
    try:
        results = _translate_with_adaptive_policy(
            translator=translator,
            tokenized=tokenized,
            requests=requests,
            target_prefix=target_prefix,
            args=args,
            device=device,
        )
    except RuntimeError as e:
        if (
            device == "cuda"
            and args.allow_cpu_fallback
            and _is_cuda_oom_error(e)
        ):
            print("sub-zero: mt_device=cpu (oom fallback)", file=sys.stderr)
            translator = _load_translator(model_path, "cpu", allow_cpu_fallback=False)
            results = _translate_with_adaptive_policy(
                translator=translator,
                tokenized=tokenized,
                requests=requests,
                target_prefix=target_prefix,
                args=args,
                device="cpu",
            )
        else:
            raise

    # Detokenize and build response.
    responses = []
    for req, result in zip(requests, results):
        tokens = result.hypotheses[0]
        # Remove the target language token if present.
        if tokens and tokens[0] == args.target_lang:
            tokens = tokens[1:]
        translation = sp.Decode(tokens)
        responses.append({
            "index": req["index"],
            "translation": translation,
        })

    # Write as UTF-8 directly to the underlying byte buffer to handle Windows cp1252 issues
    sys.stdout.buffer.write(json.dumps(responses, ensure_ascii=False).encode('utf-8'))
    sys.stdout.buffer.write(b'\n')
    sys.stdout.flush()


def _configure_cuda_runtime():
    """Prepare CUDA shared library resolution for CTranslate2."""
    cuda_dirs = _discover_cuda_library_dirs()
    if not cuda_dirs:
        print("sub-zero: cuda_libs=none-discovered", file=sys.stderr)
        return

    cublas12 = _find_library_in_dirs("libcublas.so.12", cuda_dirs)
    cublaslt12 = _find_library_in_dirs("libcublasLt.so.12", cuda_dirs)

    if (not cublas12 or not cublaslt12) and _env_flag("SUB_ZERO_ENABLE_CUDA_COMPAT_SHIM", True):
        cublas13 = _find_library_in_dirs("libcublas.so.13", cuda_dirs)
        cublaslt13 = _find_library_in_dirs("libcublasLt.so.13", cuda_dirs)
        if cublas13 and cublaslt13:
            compat_dir, compat_error = _ensure_cuda12_compat_shim(cublas13, cublaslt13)
            if compat_dir:
                cuda_dirs.insert(0, compat_dir)
                cublas12 = os.path.join(compat_dir, "libcublas.so.12")
                cublaslt12 = os.path.join(compat_dir, "libcublasLt.so.12")
                print(f"sub-zero: cuda_compat_shim=enabled dir={compat_dir}", file=sys.stderr)
            elif compat_error:
                print(f"sub-zero: cuda_compat_shim=failed error={compat_error}", file=sys.stderr)

    _prepend_ld_library_path(cuda_dirs)

    loaded = []
    if _preload_shared(cublas12):
        loaded.append("libcublas.so.12")
    if _preload_shared(cublaslt12):
        loaded.append("libcublasLt.so.12")

    if loaded:
        print(f"sub-zero: cuda_libs=preloaded libs={','.join(loaded)}", file=sys.stderr)
    else:
        print(
            "sub-zero: cuda_libs=missing required=libcublas.so.12,libcublasLt.so.12 "
            "hint=set SUB_ZERO_CUDA_LIB_DIRS or install CUDA 12 compatible libs",
            file=sys.stderr,
        )


def _discover_cuda_library_dirs():
    default_dirs = [
        "/opt/cuda/lib64",
        "/opt/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/lib64",
        "/usr/lib/wsl/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib",
    ]
    candidates = []

    custom = os.environ.get("SUB_ZERO_CUDA_LIB_DIRS", "")
    if custom:
        candidates.extend(custom.split(os.pathsep))

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if ld_path:
        candidates.extend(ld_path.split(os.pathsep))

    candidates.extend(default_dirs)

    for site_dir in _site_package_dirs():
        candidates.extend(glob.glob(os.path.join(site_dir, "nvidia", "*", "lib")))

    resolved = []
    seen = set()
    for path in candidates:
        if not path:
            continue
        norm = os.path.abspath(path)
        if norm in seen or not os.path.isdir(norm):
            continue
        seen.add(norm)
        resolved.append(norm)
    return resolved


def _site_package_dirs():
    dirs = []
    seen = set()

    for getter in (getattr(site, "getsitepackages", None), getattr(site, "getusersitepackages", None)):
        if getter is None:
            continue
        try:
            value = getter()
        except Exception:
            continue
        if isinstance(value, str):
            value = [value]
        for path in value:
            if path and path not in seen:
                seen.add(path)
                dirs.append(path)
    return dirs


def _find_library_in_dirs(lib_name, dirs):
    for directory in dirs:
        candidate = os.path.join(directory, lib_name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _prepend_ld_library_path(paths):
    existing = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    merged = []
    seen = set()
    for path in [*paths, *existing]:
        if not path:
            continue
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        merged.append(norm)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(merged)


def _preload_shared(path):
    if not path:
        return False
    try:
        mode = getattr(ctypes, "RTLD_GLOBAL", ctypes.DEFAULT_MODE)
        ctypes.CDLL(path, mode=mode)
        return True
    except OSError as error:
        print(f"sub-zero: cuda_preload_failed lib={path} error={error}", file=sys.stderr)
        return False


def _ensure_cuda12_compat_shim(cublas13_path, cublaslt13_path):
    custom_dir = os.environ.get("SUB_ZERO_CUDA_COMPAT_DIR")
    sub_zero_home = os.environ.get("SUB_ZERO_HOME")
    candidate_dirs = []
    if custom_dir:
        candidate_dirs.append(custom_dir)
    if sub_zero_home:
        candidate_dirs.append(os.path.join(sub_zero_home, "cuda-compat"))
    candidate_dirs.extend(
        [
            os.path.join(os.path.expanduser("~"), ".sub-zero", "cuda-compat"),
            "/tmp/sub-zero-cuda-compat",
            os.path.join(os.getcwd(), ".sub-zero-cuda-compat"),
        ]
    )
    links = {
        "libcublas.so.12": cublas13_path,
        "libcublasLt.so.12": cublaslt13_path,
    }
    errors = []
    for compat_dir in candidate_dirs:
        try:
            os.makedirs(compat_dir, exist_ok=True)
            for link_name, target in links.items():
                link_path = os.path.join(compat_dir, link_name)
                if os.path.lexists(link_path):
                    os.remove(link_path)
                os.symlink(target, link_path)
            return compat_dir, None
        except OSError as error:
            errors.append(f"{compat_dir}: {error}")
    return None, "; ".join(errors)


def _env_flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _load_translator(model_path, device, allow_cpu_fallback):
    """Load a CTranslate2 Translator, falling back to CPU on CUDA errors."""
    import ctranslate2

    try:
        translator = ctranslate2.Translator(model_path, device=device, compute_type="auto")
        # Force-init by running a trivial probe — catches lazy CUDA failures.
        translator.translate_batch([["▁test"]], target_prefix=[["eng_Latn"]], max_batch_size=1)
        print(f"sub-zero: mt_device={device}", file=sys.stderr)
        return translator
    except RuntimeError as e:
        if device != "cpu" and allow_cpu_fallback:
            print(f"sub-zero: mt_device=cpu (cuda fallback: {e})", file=sys.stderr)
            translator = ctranslate2.Translator(model_path, device="cpu", compute_type="auto")
            translator.translate_batch([["▁test"]], target_prefix=[["eng_Latn"]], max_batch_size=1)
            return translator
        raise


def _translate_with_oom_ladder(
    translator,
    tokenized,
    target_prefix,
    args,
    device,
    beam_override=None,
    max_batch_tokens_override=None,
):
    beam_size = max(2, beam_override if beam_override is not None else args.beam_size)
    max_batch_tokens = max(
        256,
        max_batch_tokens_override if max_batch_tokens_override is not None else args.max_batch_tokens,
    )
    attempts = max(1, args.oom_retries + 1)

    for attempt in range(1, attempts + 1):
        decode_kwargs = {
            "batch_type": "tokens",
            "max_batch_size": max_batch_tokens,
            "beam_size": beam_size,
            "repetition_penalty": args.repetition_penalty,
        }
        if args.no_repeat_ngram_size > 0:
            decode_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

        try:
            return translator.translate_batch(
                tokenized,
                target_prefix=[target_prefix] * len(tokenized),
                **decode_kwargs,
            )
        except RuntimeError as e:
            if device != "cuda" or not _is_cuda_oom_error(e) or attempt >= attempts:
                raise

            next_beam = max(2, beam_size - 1)
            next_max_batch_tokens = max(256, max_batch_tokens // 2)
            print(
                (
                    "sub-zero: mt_oom_retry "
                    f"attempt={attempt}/{attempts - 1} "
                    f"device={device} "
                    f"beam={beam_size}->{next_beam} "
                    f"max_batch_tokens={max_batch_tokens}->{next_max_batch_tokens}"
                ),
                file=sys.stderr,
            )
            beam_size = next_beam
            max_batch_tokens = next_max_batch_tokens

    raise RuntimeError("unreachable: OOM retry loop exhausted without returning")


def _translate_with_adaptive_policy(translator, tokenized, requests, target_prefix, args, device):
    if not tokenized:
        return []

    hard_indices = []
    base_indices = []
    for idx, req in enumerate(requests):
        tags = req.get("context_tags", []) or []
        if _is_hard_context(tags):
            hard_indices.append(idx)
        else:
            base_indices.append(idx)

    # Avoid tiny hard groups causing overhead/noise.
    if len(hard_indices) < 8:
        base_indices.extend(hard_indices)
        hard_indices = []

    results = [None] * len(tokenized)

    if base_indices:
        base_tokenized = [tokenized[idx] for idx in base_indices]
        base_results = _translate_with_oom_ladder(
            translator=translator,
            tokenized=base_tokenized,
            target_prefix=target_prefix,
            args=args,
            device=device,
        )
        for i, idx in enumerate(base_indices):
            results[idx] = base_results[i]

    if hard_indices:
        hard_tokenized = [tokenized[idx] for idx in hard_indices]
        hard_beam = min(8, max(2, args.beam_size + 1))
        hard_max_batch_tokens = max(256, int(args.max_batch_tokens * 0.75))
        print(
            (
                "sub-zero: mt_adaptive_hard "
                f"count={len(hard_indices)} "
                f"beam={args.beam_size}->{hard_beam} "
                f"max_batch_tokens={args.max_batch_tokens}->{hard_max_batch_tokens}"
            ),
            file=sys.stderr,
        )
        hard_results = _translate_with_oom_ladder(
            translator=translator,
            tokenized=hard_tokenized,
            target_prefix=target_prefix,
            args=args,
            device=device,
            beam_override=hard_beam,
            max_batch_tokens_override=hard_max_batch_tokens,
        )
        for i, idx in enumerate(hard_indices):
            results[idx] = hard_results[i]

    return results


def _is_hard_context(tags):
    lowered = {str(tag).strip().lower() for tag in tags if str(tag).strip()}
    hard_markers = {
        "scene_hard",
        "cue_fast",
        "cue_exclaim",
        "overlap_risk",
        "rapid_dialogue",
    }
    return not lowered.isdisjoint(hard_markers)


def _is_cuda_oom_error(error):
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def resolve_model_path(model_name, model_dir):
    """Find the CTranslate2 model directory."""
    candidates = []

    if model_dir:
        candidates.append(model_dir)

    model_leaf = os.path.basename(model_name.rstrip("/"))
    model_lower = model_leaf.lower()

    # Prefer explicit or model-specific local paths first.
    candidates.extend([
        os.path.join("models", model_name),
        os.path.join("models", model_leaf),
        model_name,
    ])

    # Generic models/nllb fallback is only safe for 600M family,
    # unless explicitly enabled.
    allow_generic_nllb_fallback = _env_flag("SUB_ZERO_ALLOW_GENERIC_NLLB_FALLBACK", False)
    is_600m_family = "600m" in model_lower or model_lower in {
        "nllb",
        "nllb-200-distilled-600m",
    }
    if allow_generic_nllb_fallback or is_600m_family:
        candidates.append(os.path.join("models", "nllb"))

    # Check HuggingFace cache.
    hf_cache = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub",
    )
    if os.path.isdir(hf_cache):
        for entry in os.listdir(hf_cache):
            if model_name.replace("/", "--") in entry:
                snapshot_dir = os.path.join(hf_cache, entry, "snapshots")
                if os.path.isdir(snapshot_dir):
                    for snap in sorted(os.listdir(snapshot_dir), reverse=True):
                        snap_path = os.path.join(snapshot_dir, snap)
                        if is_ct2_model_dir(snap_path):
                            candidates.append(snap_path)

    for path in candidates:
        if path and is_ct2_model_dir(path):
            return path

    if not is_600m_family and not allow_generic_nllb_fallback:
        print(
            (
                "sub-zero: model_resolve_failed "
                f"requested={model_name} "
                "hint=set SUB_ZERO_ALLOW_GENERIC_NLLB_FALLBACK=1 to force generic models/nllb fallback"
            ),
            file=sys.stderr,
        )

    return None


def is_ct2_model_dir(path):
    """Check if a directory looks like a CTranslate2 model."""
    if not os.path.isdir(path):
        return False
    model_bin = os.path.join(path, "model.bin")
    return os.path.isfile(model_bin)


def find_sentencepiece_model(model_dir):
    """Find the .model file for SentencePiece tokenization."""
    candidates = [
        "sentencepiece.bpe.model",
        "spm.model",
        "tokenizer.model",
    ]
    for name in candidates:
        path = os.path.join(model_dir, name)
        if os.path.isfile(path):
            return path

    # Search subdirectories.
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".model"):
                return os.path.join(root, f)

    return None


if __name__ == "__main__":
    main()
