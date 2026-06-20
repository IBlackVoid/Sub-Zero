#!/usr/bin/env python3
"""VoiDex Neural MT Daemon.

Protocol: length-prefixed JSON frames on stdin/stdout.

Request:
  4-byte little-endian length + UTF-8 JSON payload
Response:
  4-byte little-endian length + UTF-8 JSON payload

Commands:
  - {"cmd":"translate","config":{...},"requests":[...]}
  - {"cmd":"shutdown"}
"""

from __future__ import annotations

import json
import sys
import traceback

import translate_batch as backend


class Args:
    def __init__(self, cfg: dict):
        self.model = cfg.get("model", "nllb-200-distilled-600M")
        self.model_dir = cfg.get("model_dir")
        self.source_lang = cfg.get("source_lang", "jpn_Jpan")
        self.target_lang = cfg.get("target_lang", "eng_Latn")
        self.device = cfg.get("device", "cpu")
        self.batch_size = int(cfg.get("batch_size", 32))
        self.max_batch_tokens = int(cfg.get("max_batch_tokens", 8192))
        self.beam_size = int(cfg.get("beam_size", 4))
        self.repetition_penalty = float(cfg.get("repetition_penalty", 1.1))
        self.no_repeat_ngram_size = int(cfg.get("no_repeat_ngram_size", 3))
        self.oom_retries = int(cfg.get("oom_retries", 2))
        self.allow_cpu_fallback = bool(cfg.get("allow_cpu_fallback", False))
        self.prepend_prev_context = bool(cfg.get("prepend_prev_context", False))


def _read_exact(count: int) -> bytes | None:
    data = bytearray()
    while len(data) < count:
        chunk = sys.stdin.buffer.read(count - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def read_frame() -> dict | None:
    header = _read_exact(4)
    if header is None:
        return None
    length = int.from_bytes(header, "little", signed=False)
    payload = _read_exact(length)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def write_frame(obj: dict) -> None:
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sys.stdout.buffer.write(len(payload).to_bytes(4, "little", signed=False))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


class BackendState:
    def __init__(self):
        self.signature = None
        self.args = None
        self.translator = None
        self.sp = None
        self.device = None
        self.source_prefix = None
        self.target_prefix = None
        self.model_path = None

    def ensure_loaded(self, cfg: dict) -> None:
        signature = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
        if signature == self.signature:
            return

        args = Args(cfg)
        if args.device == "cuda":
            backend._configure_cuda_runtime()

        try:
            import sentencepiece as spm
        except ImportError as e:
            raise RuntimeError(f"missing dependency: {e}")

        model_path = backend.resolve_model_path(args.model, args.model_dir)
        if model_path is None:
            raise RuntimeError(f"model not found: {args.model}")

        sp_model_path = backend.find_sentencepiece_model(model_path)
        if sp_model_path is None:
            raise RuntimeError(f"sentencepiece model not found in {model_path}")

        translator = backend._load_translator(
            model_path=model_path,
            device=args.device,
            allow_cpu_fallback=args.allow_cpu_fallback,
        )

        sp = spm.SentencePieceProcessor()
        sp.Load(sp_model_path)

        self.signature = signature
        self.args = args
        self.translator = translator
        self.sp = sp
        self.device = args.device
        self.source_prefix = [args.source_lang]
        self.target_prefix = [args.target_lang]
        self.model_path = model_path

        print(f"voidex: mt_daemon_loaded model={args.model} device={args.device}", file=sys.stderr)

    def translate(self, requests: list[dict]) -> list[dict]:
        if not requests:
            return []
        args = self.args
        sp = self.sp
        translator = self.translator
        device = self.device
        source_prefix = self.source_prefix
        target_prefix = self.target_prefix

        texts_to_translate = []
        for req in requests:
            text = str(req.get("text", "")).strip()
            if not text:
                text = " "
            prev = req.get("prev_context", []) or []
            if args.prepend_prev_context and prev:
                hint = str(prev[-1]).strip()
                if hint:
                    text = hint + " " + text
            text = text.encode("utf-8", "replace").decode("utf-8")
            texts_to_translate.append(text)

        tokenized = [
            source_prefix + sp.Encode(txt, out_type=str) + ["</s>"]
            for txt in texts_to_translate
        ]

        try:
            results = backend._translate_with_adaptive_policy(
                translator=translator,
                tokenized=tokenized,
                requests=requests,
                target_prefix=target_prefix,
                args=args,
                device=device,
            )
        except RuntimeError as e:
            if device == "cuda" and args.allow_cpu_fallback and backend._is_cuda_oom_error(e):
                print("voidex: mt_device=cpu (oom fallback)", file=sys.stderr)
                translator = backend._load_translator(
                    model_path=self.model_path,
                    device="cpu",
                    allow_cpu_fallback=False,
                )
                self.translator = translator
                self.device = "cpu"
                results = backend._translate_with_adaptive_policy(
                    translator=translator,
                    tokenized=tokenized,
                    requests=requests,
                    target_prefix=target_prefix,
                    args=args,
                    device="cpu",
                )
            else:
                raise

        responses = []
        for req, result in zip(requests, results):
            tokens = result.hypotheses[0]
            if tokens and tokens[0] == args.target_lang:
                tokens = tokens[1:]
            translation = sp.Decode(tokens)
            responses.append({"index": req.get("index"), "translation": translation})
        return responses


def main() -> int:
    state = BackendState()
    while True:
        msg = read_frame()
        if msg is None:
            return 0

        cmd = msg.get("cmd")
        if cmd == "shutdown":
            write_frame({"ok": True})
            return 0

        if cmd != "translate":
            write_frame({"ok": False, "error": f"unknown cmd: {cmd}"})
            continue

        try:
            cfg = msg.get("config") or {}
            reqs = msg.get("requests") or []
            state.ensure_loaded(cfg)
            out = state.translate(reqs)
            write_frame({"ok": True, "responses": out})
        except Exception as e:
            print("voidex: mt_daemon_error", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            write_frame({"ok": False, "error": str(e)})


if __name__ == "__main__":
    raise SystemExit(main())

