"""
Approach A - Semantic segmentation (embedding-similarity drops, TextTiling-style).

Embed every sentence; at each position compare the average embedding of the sentences just before
with those just after. A topic shift shows up as a dip in that similarity curve.
  1. smooth the curve, find its local minima
  2. score each minimum by its *depth* (how far the curve climbs on both sides - the TextTiling
     depth score, robust to the absolute similarity level of a given embedding model)
  3. keep the deepest minima whose depth clears mean + SENSITIVITY * std, at least min_chapter_s
     apart, and no more than duration / target_chapter_s chapters overall
The one knob is `sensitivity`; chapter length limits are in seconds, so it behaves the same on a
10-minute clip and a 3-hour podcast.

Usage:
    python -m podsum.segment.semantic data/runs/ep1/sentences.json data/runs/ep1/chapters_semantic.json
"""
from __future__ import annotations

import argparse
import functools
from dataclasses import dataclass

import numpy as np

from podsum import config
from podsum.schema import Chapter, Sentence, load_sentences, save_json
from podsum.segment.postprocess import build_chapters


@functools.lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed_sentences(sentences: list[Sentence], model_name: str | None = None) -> np.ndarray:
    model = _model(model_name or config.EMBED_MODEL)
    return model.encode([s.text for s in sentences], normalize_embeddings=True,
                        batch_size=64, show_progress_bar=False)


def similarity_curve(embeddings: np.ndarray, window: int = 3) -> np.ndarray:
    """Cosine similarity between the mean embedding of the `window` sentences before position i and the
    `window` after it. Low similarity = topic shift at i (a new chapter would start at sentence i)."""
    n = len(embeddings)
    sims = np.zeros(n)
    for i in range(1, n - 1):
        lo, hi = max(0, i - window), min(n, i + window)
        before = embeddings[lo:i].mean(axis=0)
        after = embeddings[i:hi].mean(axis=0)
        denom = (np.linalg.norm(before) * np.linalg.norm(after)) or 1e-8
        sims[i] = float(np.dot(before, after) / denom)
    sims[0] = sims[-1] = 1.0
    return sims


def smooth(sims: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1 or len(sims) < k:
        return sims.copy()
    pad = k // 2
    padded = np.pad(sims, pad, mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def depth_scores(sims: np.ndarray) -> dict[int, float]:
    """TextTiling depth score for every local minimum: (left peak - min) + (right peak - min)."""
    n = len(sims)
    out: dict[int, float] = {}
    for i in range(1, n - 1):
        if sims[i] < sims[i - 1] and sims[i] <= sims[i + 1]:
            left = i
            while left > 0 and sims[left - 1] >= sims[left]:
                left -= 1
            right = i
            while right < n - 1 and sims[right + 1] >= sims[right]:
                right += 1
            out[i] = float((sims[left] - sims[i]) + (sims[right] - sims[i]))
    return out


def select_boundaries(depths: dict[int, float], times: list[float], span_start: float, span_end: float,
                      min_chapter_s: float, target_chapter_s: float, sensitivity: float) -> list[int]:
    span = span_end - span_start
    max_boundaries = max(0, int(round(span / target_chapter_s)) - 1)
    if max_boundaries == 0 or not depths:
        return []
    vals = np.array(list(depths.values()))
    floor = float(vals.mean() + sensitivity * vals.std()) if len(vals) >= 4 else 0.0
    chosen: list[int] = []
    for depth, i in sorted(((d, i) for i, d in depths.items() if d >= floor and d > 0), reverse=True):
        t = times[i]
        if t - span_start < min_chapter_s or span_end - t < min_chapter_s:
            continue
        if any(abs(t - times[j]) < min_chapter_s for j in chosen):
            continue
        chosen.append(i)
        if len(chosen) >= max_boundaries:
            break
    return sorted(chosen)


@dataclass
class SemanticAnalysis:
    sims: np.ndarray            # smoothed similarity curve (for plotting)
    depths: dict[int, float]
    boundaries: list[int]       # sentence indices that start a new chapter (0 is implicit)
    window: int


def analyze(sentences: list[Sentence], embeddings: np.ndarray | None = None, window: int | None = None,
            min_chapter_s: float | None = None, target_chapter_s: float | None = None,
            sensitivity: float | None = None, model_name: str | None = None) -> SemanticAnalysis:
    min_chapter_s = config.MIN_CHAPTER_S if min_chapter_s is None else min_chapter_s
    target_chapter_s = config.TARGET_CHAPTER_S if target_chapter_s is None else target_chapter_s
    sensitivity = config.SEMANTIC_SENSITIVITY if sensitivity is None else sensitivity
    n = len(sentences)
    span_start, span_end = (sentences[0].start, sentences[-1].end) if n else (0.0, 0.0)
    if n < 12 or (span_end - span_start) < 2 * min_chapter_s:
        return SemanticAnalysis(np.ones(n), {}, [], window or 3)

    if embeddings is None:
        embeddings = embed_sentences(sentences, model_name)
    if window is None:                       # ~45 s of speech on each side of the candidate boundary
        avg = (span_end - span_start) / n
        window = int(min(10, max(3, round(45.0 / max(avg, 0.5)))))
    sims = similarity_curve(embeddings, window=window)
    sims[0], sims[-1] = sims[1], sims[-2]    # remove the edge sentinels so they can't fake a deep first/last dip
    sims = smooth(sims, 3)
    depths = depth_scores(sims)
    bounds = select_boundaries(depths, [s.start for s in sentences], span_start, span_end,
                               min_chapter_s, target_chapter_s, sensitivity)
    return SemanticAnalysis(sims, depths, bounds, window)


def segment_semantic(sentences: list[Sentence], embeddings: np.ndarray | None = None, **kw) -> list[int]:
    """Sentence indices that start a new chapter (0 is implicit)."""
    return analyze(sentences, embeddings, **kw).boundaries


def boundaries_to_chapters(sentences: list[Sentence], boundary_idxs: list[int], min_chapter_s: float = 0.0) -> list[Chapter]:
    return build_chapters(sentences, [(0, "")] + [(b, "") for b in boundary_idxs], min_chapter_s)


# Legacy fixed-threshold variant, kept because its behaviour is easy to explain and unit-test.
def find_local_minima(sims: np.ndarray, depth_threshold: float = 0.08, min_gap: int = 5) -> list[int]:
    n = len(sims)
    candidates = [i for i in range(2, n - 2) if sims[i] < sims[i - 1] and sims[i] <= sims[i + 1]]
    boundaries = []
    for i in candidates:
        left_peak = max(sims[max(0, i - 10):i + 1])
        right_peak = max(sims[i:min(n, i + 10)])
        if min(left_peak, right_peak) - sims[i] >= depth_threshold:
            boundaries.append(i)
    boundaries.sort()
    pruned: list[int] = []
    for b in boundaries:
        if pruned and b - pruned[-1] < min_gap:
            if sims[b] < sims[pruned[-1]]:
                pruned[-1] = b
        else:
            pruned.append(b)
    return pruned


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic (embedding-similarity) chapter boundary detection.")
    ap.add_argument("sentences_path")
    ap.add_argument("out_path")
    ap.add_argument("--min-chapter-s", type=float, default=config.MIN_CHAPTER_S)
    ap.add_argument("--target-chapter-s", type=float, default=config.TARGET_CHAPTER_S)
    ap.add_argument("--sensitivity", type=float, default=config.SEMANTIC_SENSITIVITY,
                    help="lower = more boundaries (depth cut-off = mean + sensitivity * std)")
    args = ap.parse_args()
    sentences = load_sentences(args.sentences_path)
    a = analyze(sentences, min_chapter_s=args.min_chapter_s, target_chapter_s=args.target_chapter_s,
                sensitivity=args.sensitivity)
    chapters = boundaries_to_chapters(sentences, a.boundaries, args.min_chapter_s)
    save_json(chapters, args.out_path)
    print(f"OK  {len(chapters)} chapters -> {args.out_path}")


if __name__ == "__main__":
    main()
