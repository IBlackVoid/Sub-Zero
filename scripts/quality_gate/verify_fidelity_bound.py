#!/usr/bin/env python3
"""Verify the F.2 fidelity bound (docs/F2_subtitle_information_bottleneck.md, Theorem 1, eq. E10).

Reads paired SRT files for one corpus item:

    --machine    F.2-trained VoiDex output (terms E2/E3/E4 active)
    --baseline   per-cue baseline (no discourse context, no voice prior)
    --reference  human-translated gold cues
    [--manifest  JSON list of {machine, baseline, reference} triples]

Pipeline:
    1. Parse cues, align by index (or by --align-by timing).
    2. Embed each cue with a low-dependency character n-gram TF-IDF
       compressed by truncated SVD to `n_components` dims.
    3. Estimate empirical mutual information I(machine; reference)
       and I(baseline; reference) via the Kraskov-Stoegbauer-Grassberger
       k-NN estimator (KSG, Phys. Rev. E 69, 066138).
    4. Report

           Delta_hat = I(machine; reference) - I(baseline; reference)

       which corresponds to the discourse + voice gap Delta in F.2 (E10).
       Theorem 1 predicts Delta_hat > 0 for any model with terms
       E3 (voice consistency) and E4 (discourse coherence) active.

Exits non-zero in --strict mode if Delta_hat <= 0 on any pair.

Dependencies: numpy only. No scipy, no sklearn, no torch.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# --- SRT parsing -----------------------------------------------------------

_TIMING_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
)


@dataclass(frozen=True)
class Cue:
    index: int
    start_ms: int
    end_ms: int
    text: str


def _to_ms(stamp: str) -> int:
    h, m, rest = stamp.split(":")
    s, ms = rest.split(",")
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(ms)


def parse_srt(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\r?\n\r?\n+", raw.strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        try:
            idx = int(lines[0])
        except ValueError:
            # Some SRT files lack the numeric index line; synthesize one.
            idx = len(cues) + 1
            timing_line = lines[0]
            text_lines = lines[1:]
        else:
            if len(lines) < 3:
                continue
            timing_line = lines[1]
            text_lines = lines[2:]
        m = _TIMING_RE.search(timing_line)
        if not m:
            continue
        start_ms = _to_ms(m.group(1))
        end_ms = _to_ms(m.group(2))
        text = " ".join(text_lines).strip()
        cues.append(Cue(index=idx, start_ms=start_ms, end_ms=end_ms, text=text))
    return cues


# --- alignment -------------------------------------------------------------

def align_by_index(a: list[Cue], b: list[Cue]) -> list[tuple[Cue, Cue]]:
    by_idx_b = {c.index: c for c in b}
    pairs: list[tuple[Cue, Cue]] = []
    for ca in a:
        cb = by_idx_b.get(ca.index)
        if cb is not None:
            pairs.append((ca, cb))
    return pairs


def align_by_timing(a: list[Cue], b: list[Cue], window_ms: int = 1500) -> list[tuple[Cue, Cue]]:
    """Greedy timing alignment: for each `a` cue, take the closest `b` cue
    whose start is within `window_ms` and not already consumed."""
    used = [False] * len(b)
    pairs: list[tuple[Cue, Cue]] = []
    for ca in a:
        best = -1
        best_d = window_ms + 1
        for j, cb in enumerate(b):
            if used[j]:
                continue
            d = abs(ca.start_ms - cb.start_ms)
            if d < best_d:
                best_d = d
                best = j
        if best >= 0:
            pairs.append((ca, b[best]))
            used[best] = True
    return pairs


def align_three(
    machine: list[Cue],
    baseline: list[Cue],
    reference: list[Cue],
    by: str,
) -> tuple[list[Cue], list[Cue], list[Cue]]:
    """Return three same-length lists of aligned cues (machine_i, baseline_i, reference_i)."""
    if by == "index":
        m_r = align_by_index(machine, reference)
        b_r = align_by_index(baseline, reference)
    else:
        m_r = align_by_timing(machine, reference)
        b_r = align_by_timing(baseline, reference)
    # Keep only reference cues that exist in both alignments.
    machine_by_ref = {id(r): m for m, r in m_r}
    baseline_by_ref = {id(r): b for b, r in b_r}
    out_m: list[Cue] = []
    out_b: list[Cue] = []
    out_r: list[Cue] = []
    for r in reference:
        m = machine_by_ref.get(id(r))
        b = baseline_by_ref.get(id(r))
        if m is not None and b is not None:
            out_m.append(m)
            out_b.append(b)
            out_r.append(r)
    return out_m, out_b, out_r


# --- character n-gram TF-IDF embedding -------------------------------------

def char_ngrams(text: str, n_min: int = 2, n_max: int = 3) -> list[str]:
    s = "^" + re.sub(r"\s+", " ", text.lower()) + "$"
    grams: list[str] = []
    for n in range(n_min, n_max + 1):
        if len(s) < n:
            continue
        for i in range(len(s) - n + 1):
            grams.append(s[i : i + n])
    return grams


def build_tfidf(
    docs: list[str],
    n_min: int = 2,
    n_max: int = 3,
    max_vocab: int = 4096,
    min_doc_freq: int = 2,
) -> np.ndarray:
    """Return an (N, V) float matrix of L2-normalised TF-IDF vectors."""
    grams_per_doc = [char_ngrams(d, n_min, n_max) for d in docs]
    df: dict[str, int] = {}
    for grams in grams_per_doc:
        for g in set(grams):
            df[g] = df.get(g, 0) + 1
    N = len(docs)
    # Pick the most discriminative grams: drop too-rare and too-common.
    eligible = [
        (g, c) for g, c in df.items()
        if c >= min_doc_freq and c < N  # keep at least once-doc-paired but never global
    ]
    eligible.sort(key=lambda kv: kv[1], reverse=True)
    vocab = {g: i for i, (g, _) in enumerate(eligible[:max_vocab])}
    V = len(vocab)
    if V == 0:
        # Degenerate corpus; return zero-vectors.
        return np.zeros((N, 1), dtype=np.float64)
    idf = np.zeros(V, dtype=np.float64)
    for g, i in vocab.items():
        idf[i] = math.log((N + 1) / (df[g] + 1)) + 1.0
    X = np.zeros((N, V), dtype=np.float64)
    for row, grams in enumerate(grams_per_doc):
        if not grams:
            continue
        for g in grams:
            j = vocab.get(g)
            if j is not None:
                X[row, j] += 1.0
        # tf-normalise then idf-weight
        row_sum = X[row].sum()
        if row_sum > 0:
            X[row] /= row_sum
        X[row] *= idf
        # L2-normalise
        norm = np.linalg.norm(X[row])
        if norm > 0:
            X[row] /= norm
    return X


def reduce_svd(X: np.ndarray, n_components: int) -> np.ndarray:
    """Truncated SVD via numpy SVD; centred (PCA)."""
    if X.shape[1] <= n_components:
        return X
    Xc = X - X.mean(axis=0, keepdims=True)
    # Economy SVD; for tall matrices np.linalg.svd defaults are fine.
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, S.shape[0])
    return U[:, :k] * S[:k]


# --- KSG mutual-information estimator --------------------------------------

def _digamma(x: np.ndarray) -> np.ndarray:
    """Asymptotic digamma with recurrence push-up to x>=7."""
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    # Push x >= 7 using psi(x+1) = psi(x) + 1/x
    while np.any(x < 7):
        mask = x < 7
        result[mask] = result[mask] - 1.0 / x[mask]
        x = np.where(mask, x + 1.0, x)
    # Asymptotic Stirling-like series.
    return (
        result
        + np.log(x)
        - 1.0 / (2.0 * x)
        - 1.0 / (12.0 * x ** 2)
        + 1.0 / (120.0 * x ** 4)
        - 1.0 / (252.0 * x ** 6)
    )


def _max_norm_pdist(X: np.ndarray) -> np.ndarray:
    """Pairwise max-norm (L_inf) distance matrix, O(N^2 * d)."""
    diff = np.abs(X[:, None, :] - X[None, :, :])
    return diff.max(axis=2)


def kraskov_mi(X: np.ndarray, Y: np.ndarray, k: int = 3) -> float:
    """KSG-1 estimator (Kraskov, Stoegbauer, Grassberger 2004)."""
    n = X.shape[0]
    if n <= k + 1:
        return 0.0
    dx = _max_norm_pdist(X)
    dy = _max_norm_pdist(Y)
    # Joint distance is the max of the marginal distances under L_inf.
    djoint = np.maximum(dx, dy)
    np.fill_diagonal(djoint, np.inf)
    np.fill_diagonal(dx, np.inf)
    np.fill_diagonal(dy, np.inf)
    # Distance to the k-th nearest neighbour in joint space.
    eps = np.partition(djoint, kth=k - 1, axis=1)[:, k - 1]
    # KSG-1: count points strictly within eps in each marginal (excluding self).
    nx = (dx < eps[:, None]).sum(axis=1)
    ny = (dy < eps[:, None]).sum(axis=1)
    # Guard against zeros (would blow up digamma).
    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)
    psi_k = float(_digamma(np.array([float(k)]))[0])
    psi_n = float(_digamma(np.array([float(n)]))[0])
    mi = (
        psi_k
        - float(_digamma(nx + 1).mean())
        - float(_digamma(ny + 1).mean())
        + psi_n
    )
    return max(mi, 0.0)


# --- top-level verification -----------------------------------------------

@dataclass(frozen=True)
class FidelityVerdict:
    machine_path: Path
    baseline_path: Path
    reference_path: Path
    n_aligned: int
    n_components: int
    k: int
    mi_machine: float
    mi_baseline: float
    delta_hat: float
    bound_holds: bool


def verify_pair(
    machine_srt: Path,
    baseline_srt: Path,
    reference_srt: Path,
    *,
    align_by: str,
    n_components: int,
    k: int,
    n_min: int,
    n_max: int,
) -> FidelityVerdict | None:
    machine = parse_srt(machine_srt)
    baseline = parse_srt(baseline_srt)
    reference = parse_srt(reference_srt)
    if not (machine and baseline and reference):
        return None
    m_aligned, b_aligned, r_aligned = align_three(machine, baseline, reference, align_by)
    n = len(r_aligned)
    if n < (k + 2):
        return None
    # Build a single shared TF-IDF vocab over the union of all cues so the
    # embeddings live in the same space. Then SVD-reduce for tractable KNN.
    all_text = (
        [c.text for c in m_aligned]
        + [c.text for c in b_aligned]
        + [c.text for c in r_aligned]
    )
    X = build_tfidf(all_text, n_min=n_min, n_max=n_max)
    Z = reduce_svd(X, n_components=n_components)
    em_m = Z[0:n]
    em_b = Z[n : 2 * n]
    em_r = Z[2 * n : 3 * n]
    mi_m = kraskov_mi(em_m, em_r, k=k)
    mi_b = kraskov_mi(em_b, em_r, k=k)
    delta = mi_m - mi_b
    return FidelityVerdict(
        machine_path=machine_srt,
        baseline_path=baseline_srt,
        reference_path=reference_srt,
        n_aligned=n,
        n_components=em_m.shape[1],
        k=k,
        mi_machine=mi_m,
        mi_baseline=mi_b,
        delta_hat=delta,
        bound_holds=delta > 0.0,
    )


def print_verdict(v: FidelityVerdict) -> None:
    label = "PASS" if v.bound_holds else "FAIL"
    print(f"\npair: {v.machine_path.name}")
    print(f"  baseline:  {v.baseline_path}")
    print(f"  reference: {v.reference_path}")
    print(
        f"  n_aligned={v.n_aligned}  d={v.n_components}  k={v.k}  "
        f"I(M;R)={v.mi_machine:.4f}  I(B;R)={v.mi_baseline:.4f}"
    )
    print(f"  Delta_hat = {v.delta_hat:+.4f}   bound holds: {label}")


def aggregate(vs: Iterable[FidelityVerdict]) -> None:
    vs = list(vs)
    if not vs:
        return
    holds = sum(1 for v in vs if v.bound_holds)
    deltas = [v.delta_hat for v in vs]
    print("\n=== summary ===")
    print(f"pairs verified: {len(vs)}")
    print(f"bound holds:    {holds}/{len(vs)}")
    print(
        f"delta_hat: mean={np.mean(deltas):+.4f} "
        f"min={min(deltas):+.4f} max={max(deltas):+.4f}"
    )


def load_manifest(path: Path) -> list[tuple[Path, Path, Path]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    triples: list[tuple[Path, Path, Path]] = []
    for entry in data:
        triples.append((
            Path(entry["machine"]),
            Path(entry["baseline"]),
            Path(entry["reference"]),
        ))
    return triples


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--machine", type=Path, help="F.2-trained VoiDex SRT")
    parser.add_argument("--baseline", type=Path, help="per-cue baseline SRT")
    parser.add_argument("--reference", type=Path, help="human-reference SRT")
    parser.add_argument("--manifest", type=Path, help="JSON list of triples")
    parser.add_argument("--align-by", choices=("index", "timing"), default="timing")
    parser.add_argument("--k", type=int, default=3, help="KSG neighbour count (default 3)")
    parser.add_argument("--n-components", type=int, default=16,
                        help="SVD dim for cue embeddings (default 16)")
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=3)
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if any pair has Delta_hat <= 0")
    args = parser.parse_args(argv)

    if args.manifest is not None:
        triples = load_manifest(args.manifest)
    else:
        if not (args.machine and args.baseline and args.reference):
            parser.error("provide --machine + --baseline + --reference, or --manifest")
        triples = [(args.machine, args.baseline, args.reference)]

    verdicts: list[FidelityVerdict] = []
    for m, b, r in triples:
        for path in (m, b, r):
            if not path.is_file():
                print(f"error: not a file: {path}", file=sys.stderr)
                return 2
        v = verify_pair(
            m, b, r,
            align_by=args.align_by,
            n_components=args.n_components,
            k=args.k,
            n_min=args.ngram_min,
            n_max=args.ngram_max,
        )
        if v is None:
            print(f"warning: skipping unaligned/empty triple: {m}", file=sys.stderr)
            continue
        print_verdict(v)
        verdicts.append(v)

    aggregate(verdicts)
    if args.strict and any(not v.bound_holds for v in verdicts):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
