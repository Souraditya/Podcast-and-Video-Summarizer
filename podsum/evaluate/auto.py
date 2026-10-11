"""
Automatic evaluation of one pipeline run, using whatever ground truth the source provides:
  * creator chapters (YouTube)      -> boundary P/R/F1 at several tolerances, Pk, WindowDiff, ROUGE
  * manual captions (YouTube)       -> approximate WER of the raw transcript
  * a hand-corrected reference text -> WER on a chosen time span (--reference-text)
If the video has none of these, evaluation is skipped and says why - it is never faked.
"""
from __future__ import annotations

from podsum.evaluate.segmentation import boundary_prf, pk, summary_rouge, window_diff
from podsum.evaluate.wer import evaluate_transcript
from podsum.schema import Chapter, Transcript

TOLERANCES = (15.0, 30.0, 60.0)


def creator_chapters_to_chapters(raw: list[dict]) -> list[Chapter]:
    return [Chapter(start=float(c["start"]), end=float(c.get("end") or c["start"]), title=c.get("title", "")) for c in raw]


def evaluate_chaptering(results: dict[str, list[Chapter]], ground_truth: list[Chapter], duration_s: float) -> dict:
    out = {"n_ground_truth_chapters": len(ground_truth), "methods": {}}
    for method, chapters in results.items():
        row = {"n_predicted_chapters": len(chapters), "boundary": {}}
        for tol in TOLERANCES:
            prf = boundary_prf(chapters, ground_truth, tol, exclude_first=True)
            row["boundary"][str(int(tol))] = {k: prf[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "mean_offset_s")}
        row["pk"] = pk(chapters, ground_truth, duration_s)
        row["window_diff"] = window_diff(chapters, ground_truth, duration_s)
        if any(c.summary for c in chapters):
            r = summary_rouge(chapters, ground_truth, 30.0)
            row["rouge_vs_creator_titles"] = {k: r[k] for k in ("rouge1", "rouge2", "rougeL", "n_matched")}
        out["methods"][method] = row
    return out


def evaluate_wer(transcript: Transcript, reference_text: str, start: float = 0.0, end: float | None = None) -> dict:
    return evaluate_transcript(transcript, reference_text, start=start, end=end)


def recommend_method(chaptering_eval: dict | None) -> dict | None:
    """Picks the better-scoring chaptering method for THIS video, from the boundary-F1 evaluation. This is a
    single-video signal, not a general verdict on either method - run on several videos before treating one
    method as reliably better than the other. Returns None when there is nothing to compare (fewer than 2
    methods were evaluated)."""
    methods = (chaptering_eval or {}).get("methods", {})
    if len(methods) < 2:
        return None

    def score(row: dict) -> float:
        # Average F1 across tolerances rather than a single tolerance: robust to a method being a few seconds
        # early/late on an otherwise-correct boundary, without ignoring tolerance entirely.
        return sum(row["boundary"][t]["f1"] for t in ("15", "30", "60")) / 3

    scores = {m: round(score(r), 4) for m, r in methods.items()}
    best = max(scores, key=scores.get)
    others = ", ".join(f"{m}={v:.3f}" for m, v in sorted(scores.items()) if m != best)
    tie = len(set(scores.values())) == 1
    return {"method": best, "scores": scores,
            "reason": "tied on boundary F1 (defaulting to the first method)" if tie else
                      f"higher mean boundary F1 ({scores[best]:.3f} vs {others}) against the creator's chapters"}


def format_recommendation(rec: dict | None) -> str | None:
    if not rec:
        return None
    return f"Recommended method for this video: {rec['method']}  ({rec['reason']})"


def format_evaluation(ev: dict) -> str:
    lines: list[str] = []
    ch = ev.get("chaptering")
    if ch:
        lines.append(f"Chapter boundaries vs the creator's {ch['n_ground_truth_chapters']} chapters (0:00 start excluded; higher F1 is better)")
        head = f"  {'method':<10}{'chapters':>9}" + "".join(f"{'F1@' + t + 's':>9}" for t in ("15", "30", "60")) + f"{'P@30':>8}{'R@30':>8}{'Pk':>8}{'WinDiff':>9}"
        lines.append(head)
        for m, r in ch["methods"].items():
            b = r["boundary"]
            f = lambda v: "   n/a  " if v is None else f"{v:8.3f}"
            lines.append(f"  {m:<10}{r['n_predicted_chapters']:>9}" + "".join(f"{b[t]['f1']:>9.3f}" for t in ("15", "30", "60"))
                         + f"{b['30']['precision']:>8.3f}{b['30']['recall']:>8.3f}" + f(r["pk"]) + " " + f(r["window_diff"]))
    else:
        lines.append(f"Chapter evaluation skipped: {ev.get('chaptering_skipped', 'no ground truth')}")
    w = ev.get("wer")
    if w:
        lines.append(f"Transcription WER {w['wer']:.3f}  (sub={w['substitutions']} del={w['deletions']} ins={w['insertions']})  "
                     f"[{w['reference']}]")
    else:
        lines.append(f"WER skipped: {ev.get('wer_skipped', 'no reference transcript')}")
    rec_line = format_recommendation(ev.get("recommended_method"))
    if rec_line:
        lines.append(rec_line)
    return "\n".join(lines)