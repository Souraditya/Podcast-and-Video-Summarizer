"""
Approach A vs Approach B — the ablation table for the report/viva.

Scores any number of predicted chapter files (a chapters list, or a
pipeline result.json) against one ground-truth chapters file, at several
boundary tolerances, and prints a side-by-side table.

Usage:
    python -m podsum.evaluate.compare data/eval/<video_id>_chapters.json \
        semantic=data/runs/ep1_semantic/result.json llm=data/runs/ep1_llm/result.json
"""
from __future__ import annotations

import argparse

from podsum.evaluate.segmentation import boundary_prf, summary_rouge
from podsum.schema import load_chapters

TOLERANCES = (15.0, 30.0, 60.0)


def compare(ground_truth_path: str, named_predictions: dict[str, str]) -> list[dict]:
    gt = load_chapters(ground_truth_path)
    rows = []
    for name, path in named_predictions.items():
        pred = load_chapters(path)
        row = {"method": name, "n_pred": len(pred), "n_gt": len(gt)}
        for tol in TOLERANCES:
            prf = boundary_prf(pred, gt, tolerance_s=tol)
            row[f"P@{int(tol)}"], row[f"R@{int(tol)}"], row[f"F1@{int(tol)}"] = prf["precision"], prf["recall"], prf["f1"]
        if any(c.summary for c in pred):
            rouge = summary_rouge(pred, gt, tolerance_s=30.0)
            row["ROUGE-1"], row["ROUGE-L"] = rouge["rouge1"], rouge["rougeL"]
        rows.append(row)
    return rows


def format_table(rows: list[dict]) -> str:
    cols = list(rows[0].keys())
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    widths = {c: max(len(c), *(len(fmt(r.get(c, "-"))) for r in rows)) for c in cols}
    lines = ["  ".join(c.ljust(widths[c]) for c in cols)]
    lines += ["  ".join(fmt(r.get(c, "-")).ljust(widths[c]) for c in cols) for r in rows]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare chaptering approaches against ground truth.")
    ap.add_argument("ground_truth_path")
    ap.add_argument("predictions", nargs="+", help="name=path pairs, e.g. semantic=data/runs/a/result.json")
    args = ap.parse_args()

    named = {}
    for item in args.predictions:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected name=path, got {item!r}")
        named[name] = path
    print(format_table(compare(args.ground_truth_path, named)))


if __name__ == "__main__":
    main()
