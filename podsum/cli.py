"""Command line for the full pipeline (also reachable as `python -m podsum`)."""
from __future__ import annotations

import argparse
import sys

from podsum import config
from podsum.data.ingest import IngestError
from podsum.llm import LLMConfigError
from podsum.pipeline import METHODS, PipelineError, run
from podsum.report import format_summary

EPILOG = """examples:
  python -m podsum "https://www.youtube.com/watch?v=XXXXXXXXXXX"
  python -m podsum lecture.mp4 --method llm --whisper-model medium.en
  python -m podsum doctor                      # check your setup first
"""


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    for stream in (sys.stdout, sys.stderr):      # video titles contain emoji/CJK; never crash on a cp1252 console or pipe
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if argv and argv[0] == "doctor":
        from podsum.doctor import main as doctor_main
        raise SystemExit(doctor_main(argv[1:]))

    ap = argparse.ArgumentParser(prog="python -m podsum", epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description="Turn any YouTube link or audio/video file into a transcript, auto-detected chapters "
                                             "with titles, chapter summaries and a TL;DR - whether or not the source has chapters.")
    ap.add_argument("source", help="YouTube (or other yt-dlp) URL, or a path to a local audio/video file")
    ap.add_argument("--out", default=None, help="run folder (default: data/runs/<video id or file name>)")
    ap.add_argument("--method", choices=list(METHODS) + ["both"], default="both",
                    help="chaptering approach: semantic (embeddings), llm (prompted), or both to compare (default)")
    ap.add_argument("--whisper-model", default=None, help=f"default {config.WHISPER_MODEL}")
    ap.add_argument("--device", default=None, help=f"cpu | cuda | auto (default {config.WHISPER_DEVICE})")
    ap.add_argument("--language", default=None, help="force a language code, e.g. hi (needs a non-.en model)")
    ap.add_argument("--cookies", dest="cookies_file", default=None, help="Netscape cookies.txt for youtube.com (bot check)")
    ap.add_argument("--cookies-from-browser", default=None, help="chrome | edge | firefox ...")
    ap.add_argument("--min-chapter-s", type=float, default=None, help=f"shortest allowed chapter (default {config.MIN_CHAPTER_S:.0f})")
    ap.add_argument("--target-chapter-s", type=float, default=None, help=f"typical chapter length (default {config.TARGET_CHAPTER_S:.0f})")
    ap.add_argument("--reference-text", default=None, help="hand-corrected transcript (text file) to compute WER against")
    ap.add_argument("--ref-start", type=float, default=0.0, help="seconds where the reference text starts")
    ap.add_argument("--ref-end", type=float, default=None, help="seconds where the reference text ends")
    ap.add_argument("--no-eval", action="store_true", help="skip evaluation even if ground truth is available")
    ap.add_argument("--force", action="store_true", help="ignore all cached stages and start over")
    ap.add_argument("--debug", action="store_true", help="show full tracebacks")
    a = ap.parse_args(argv)

    try:
        result = run(a.source, a.out, method=a.method, whisper_model=a.whisper_model, device=a.device, language=a.language,
                     cookies_file=a.cookies_file, cookies_from_browser=a.cookies_from_browser, force=a.force,
                     evaluate=not a.no_eval, reference_text=a.reference_text, ref_start=a.ref_start, ref_end=a.ref_end,
                     min_chapter_s=a.min_chapter_s, target_chapter_s=a.target_chapter_s)
    except (IngestError, LLMConfigError, PipelineError) as e:
        if a.debug:
            raise
        raise SystemExit(f"\nerror: {e}")
    except KeyboardInterrupt:
        raise SystemExit("\ninterrupted - cached stages are kept, re-run the same command to resume")
    except Exception as e:
        if a.debug:
            raise
        raise SystemExit(f"\nerror: {type(e).__name__}: {str(e)[:400]}\n(completed stages are cached - fix the cause and re-run the same command to resume; add --debug for the traceback)")

    results = result["results"] if result["method"] == "both" else {result["method"]: result}
    for m, r in results.items():
        print("\n" + "=" * 100 + f"\n{m.upper()} CHAPTERING\n" + "=" * 100)
        print(format_summary(r))
    run_dir = result["run_dir"]
    print(f"Outputs in {run_dir}\\  (per method: result.json, report.md, youtube_chapters.txt;  also transcript.json, "
          f"sentences.json, evaluation.json)")
    if result["warnings"]:
        print(f"{len(result['warnings'])} warning(s) - see above / report.md")


if __name__ == "__main__":
    main()