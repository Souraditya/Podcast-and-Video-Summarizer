"""
Evaluation - chapter boundary accuracy and summary quality.

Ground truth = the creator's own YouTube chapters (fetched at ingest; never seen by the generator).
  boundary_prf   precision / recall / F1 of predicted chapter starts within a tolerance window.
                 exclude_first=True drops the 0:00 start of both lists - it is always a trivial match and
                 would otherwise inflate every score.
  pk, window_diff  the standard text-segmentation error metrics (lower is better), computed on time.
                 Pk: probability that two points k seconds apart are wrongly judged to be in the same /
                 different chapter. WindowDiff: same idea, but compares the number of boundaries in the window.
  summary_rouge  ROUGE-1/2/L of each generated summary against the matching ground-truth chapter *title*
                 (coarse: titles are short, so absolute scores are low).

Usage:
    python -m podsum.evaluate.segmentation predicted_or_result.json ground_truth_chapters.json
"""
from __future__ import annotations

import argparse

import numpy as np

from podsum.schema import Chapter, load_chapters


def _starts(chs: list[Chapter], exclude_first: bool) -> list[float]:
    s = sorted(c.start for c in chs)
    return s[1:] if exclude_first and s and s[0] <= 1.0 else s


def boundary_prf(predicted: list[Chapter], ground_truth: list[Chapter], tolerance_s: float = 15.0,
                 exclude_first: bool = False) -> dict:
    """A predicted start is a hit if it is within tolerance_s of a not-yet-matched ground-truth start."""
    gt_starts, pred_starts = _starts(ground_truth, exclude_first), _starts(predicted, exclude_first)
    matched_gt: set[int] = set()
    offsets: list[float] = []
    tp = 0
    for p in pred_starts:
        candidates = [i for i in range(len(gt_starts)) if i not in matched_gt and abs(gt_starts[i] - p) <= tolerance_s]
        if candidates:
            best = min(candidates, key=lambda i: abs(gt_starts[i] - p))
            matched_gt.add(best)
            offsets.append(abs(gt_starts[best] - p))
            tp += 1
    fp, fn = len(pred_starts) - tp, len(gt_starts) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "mean_offset_s": float(np.mean(offsets)) if offsets else None, "tolerance_s": tolerance_s}


def _segment_ids(chs: list[Chapter], points: np.ndarray) -> np.ndarray:
    starts = np.array(sorted(c.start for c in chs))
    return np.searchsorted(starts, points, side="right") - 1


def _window_setup(predicted, ground_truth, duration_s, k_s):
    step = max(1.0, duration_s / 4000.0)
    pts = np.arange(0.0, duration_s, step)
    if k_s is None:
        k_s = max(step, duration_s / max(len(ground_truth), 1) / 2.0)     # half the mean reference chapter length
    k = max(1, int(round(k_s / step)))
    if len(pts) <= k:
        return None
    return _segment_ids(ground_truth, pts), _segment_ids(predicted, pts), k


def pk(predicted: list[Chapter], ground_truth: list[Chapter], duration_s: float, k_s: float | None = None) -> float | None:
    setup = _window_setup(predicted, ground_truth, duration_s, k_s)
    if setup is None:
        return None
    ref, hyp, k = setup
    n = len(ref) - k
    return float(np.mean((ref[:n] == ref[k:]) != (hyp[:n] == hyp[k:])))


def window_diff(predicted: list[Chapter], ground_truth: list[Chapter], duration_s: float, k_s: float | None = None) -> float | None:
    setup = _window_setup(predicted, ground_truth, duration_s, k_s)
    if setup is None:
        return None
    ref, hyp, k = setup
    n = len(ref) - k
    return float(np.mean((ref[k:] - ref[:n]) != (hyp[k:] - hyp[:n])))


def summary_rouge(predicted: list[Chapter], ground_truth: list[Chapter], tolerance_s: float = 15.0) -> dict:
    """ROUGE between each predicted chapter's summary and the nearest (within tolerance) ground-truth title."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = 0
    for p in predicted:
        if not p.summary:
            continue
        candidates = [g for g in ground_truth if abs(g.start - p.start) <= tolerance_s]
        if not candidates:
            continue
        gt = min(candidates, key=lambda g: abs(g.start - p.start))
        scores = scorer.score(gt.title, p.summary)
        for k in totals:
            totals[k] += scores[k].fmeasure
        n += 1
    if n == 0:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "n_matched": 0}
    return {**{k: v / n for k, v in totals.items()}, "n_matched": n}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate predicted chapters against ground truth.")
    ap.add_argument("predicted_path")
    ap.add_argument("ground_truth_path")
    ap.add_argument("--tolerance", type=float, default=15.0, help="seconds")
    ap.add_argument("--exclude-first", action="store_true", help="ignore the trivial 0:00 boundary")
    args = ap.parse_args()
    predicted, ground_truth = load_chapters(args.predicted_path), load_chapters(args.ground_truth_path)
    prf = boundary_prf(predicted, ground_truth, args.tolerance, args.exclude_first)
    print(f"Boundary  P={prf['precision']:.3f}  R={prf['recall']:.3f}  F1={prf['f1']:.3f}  "
          f"(tp={prf['tp']} fp={prf['fp']} fn={prf['fn']}, tolerance={args.tolerance}s)")
    if any(p.summary for p in predicted):
        r = summary_rouge(predicted, ground_truth, args.tolerance)
        print(f"Summary   ROUGE-1={r['rouge1']:.3f}  ROUGE-2={r['rouge2']:.3f}  ROUGE-L={r['rougeL']:.3f}  (matched {r['n_matched']} chapters)")


if __name__ == "__main__":
    main()
