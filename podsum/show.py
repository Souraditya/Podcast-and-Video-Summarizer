"""
Inspect a finished run.

    python -m podsum.show summary data/runs/<id>/llm/result.json
    python -m podsum.show curve   data/runs/<id> [--show]     # similarity curve + detected (and creator) chapter lines
"""
from __future__ import annotations

import argparse
from pathlib import Path

from podsum.report import format_summary
from podsum.schema import load_json

__all__ = ["format_summary", "render_curve"]


def render_curve(times: list[float], sims: list[float], boundary_times: list[float], out_path: str,
                 gt_times: list[float] | None = None, show: bool = False) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot([t / 60 for t in times], sims, color="#0b1f3a", lw=1.2, label="similarity (before vs after window)")
    for k, t in enumerate(boundary_times):
        ax.axvline(t / 60, color="#f5a623", ls="--", lw=1.4, label="detected chapter start" if k == 0 else None)
    for k, t in enumerate(gt_times or []):
        ax.axvline(t / 60, color="#2a9d8f", ls=":", lw=1.6, label="creator chapter" if k == 0 else None)
    ax.set_xlabel("minutes"); ax.set_ylabel("cosine similarity"); ax.set_title("Topic shifts appear as dips in similarity")
    ax.legend(loc="lower left", fontsize=8); fig.tight_layout(); fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a finished run.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary"); s.add_argument("path")
    c = sub.add_parser("curve"); c.add_argument("run_dir"); c.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.cmd == "summary":
        print(format_summary(load_json(args.path)))
        return
    run_dir = Path(args.run_dir)
    a = load_json(run_dir / "semantic" / "analysis.json")
    meta = load_json(run_dir / "meta.json")
    gt = [ch["start"] for ch in meta.get("creator_chapters", [])] or None
    out = run_dir / "semantic" / "similarity_curve.png"
    render_curve(a["times"], a["sims"], [a["times"][b] for b in a["boundaries"]], str(out), gt, args.show)
    print(f"OK  {len(a['boundaries'])} boundaries plotted -> {out}")


if __name__ == "__main__":
    main()
