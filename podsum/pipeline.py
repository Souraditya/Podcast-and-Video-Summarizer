"""
End-to-end pipeline:  Ingest -> Transcribe -> Clean -> Segment -> Summarize -> Report.

Works on any YouTube (or yt-dlp-supported) link or local audio/video file, with or without existing
chapters: chapters and summaries are always generated from the transcript alone. The creator's chapters
and captions, when the source has them, are used ONLY to score the output afterwards.

Every stage is cached per run folder and keyed by a signature of its inputs and settings, so re-running the
same link only redoes what changed (e.g. switching method or LLM model never re-transcribes).

    python -m podsum "https://www.youtube.com/watch?v=XXXXXXXXXXX"            # both chaptering methods
    python -m podsum lecture.mp4 --method llm
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import time
from pathlib import Path
from typing import Callable

from podsum import config, llm
from podsum.clean import (DEFAULT_COMPRESSION_RATIO_THRESHOLD, DEFAULT_LOGPROB_THRESHOLD,
                          DEFAULT_NO_SPEECH_THRESHOLD, clean_pipeline)
from podsum.data.ingest import IngestError, clean_source, find_ffmpeg, ingest, run_dir_for
from podsum.evaluate.auto import (creator_chapters_to_chapters, evaluate_chaptering, evaluate_wer,
                                  format_evaluation, recommend_method)
from podsum.report import chapters_txt, markdown_report
from podsum.schema import (Chapter, fmt_ts, load_chapters, load_json, load_sentences, load_transcript,
                           save_json)
from podsum.segment.llm_chaptering import segment_llm
from podsum.segment.semantic import analyze, boundaries_to_chapters
from podsum.summarize import generate_tldr, summarize_chapters
from podsum.transcribe import transcribe

PIPELINE_VERSION = "1.0"     # bump when a stage's logic changes so old caches are not reused
PROMPT_VERSION = "1"
STAGES = ["Ingest", "Transcribe", "Clean", "Segment", "Summarize", "Report"]
METHODS = ("semantic", "llm")


class PipelineError(RuntimeError):
    """A run cannot produce a meaningful result (e.g. no speech found)."""


def _sig(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


class _Manifest:
    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.data = {"sigs": {}, "timings": {}}

    def fresh(self, key: str, sig: str, *files: Path, force: bool = False) -> bool:
        return (not force) and self.data["sigs"].get(key) == sig and all(Path(f).exists() for f in files)

    def mark(self, key: str, sig: str) -> None:
        self.data["sigs"][key] = sig
        self.save()

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")


def run(source: str, out_dir: str | Path | None = None, method: str = "both",
        whisper_model: str | None = None, device: str | None = None, language: str | None = None,
        cookies_file: str | None = None, cookies_from_browser: str | None = None, force: bool = False,
        evaluate: bool = True, reference_text: str | None = None, ref_start: float = 0.0,
        ref_end: float | None = None, min_chapter_s: float | None = None, target_chapter_s: float | None = None,
        on_progress: Callable[[int, str], None] | None = None) -> dict:
    """Returns the result dict of a single method, or {"method": "both", "results": {...}} for method="both"."""
    if method not in METHODS + ("both",):
        raise ValueError(f"unknown method {method!r} (use 'semantic', 'llm' or 'both')")
    methods = list(METHODS) if method == "both" else [method]
    source = clean_source(source)
    out_dir = Path(out_dir) if out_dir else run_dir_for(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    whisper_model = whisper_model or config.WHISPER_MODEL
    language = language or config.WHISPER_LANGUAGE
    min_s = config.MIN_CHAPTER_S if min_chapter_s is None else min_chapter_s
    target_s = config.TARGET_CHAPTER_S if target_chapter_s is None else target_chapter_s

    # Each warning is tagged with the method it came from (None = applies to every method - ingest/transcribe/
    # clean warnings). Without this, a per-method report would show warnings from the OTHER method too, which
    # reads as duplicated/contradictory noise (e.g. two "chapter 2 summarization failed" lines with different
    # timestamps - one per method's own chapter 2).
    warnings: list[tuple[str | None, str]] = []

    def warn(msg: str, method: str | None = None) -> None:
        print(f"      {msg}")
        warnings.append((method, msg))

    def warnings_for(method: str | None) -> list[str]:
        return [msg for meth, msg in warnings if meth in (None, method)]

    @contextlib.contextmanager
    def stage(i: int, label: str):
        print(f"[{i}/{len(STAGES)}] {label}")
        if on_progress:
            on_progress(i, label)
        t0 = time.time()
        yield
        manifest.data["timings"][label] = round(time.time() - t0, 1)
        manifest.save()
        print(f"      done in {time.time() - t0:.1f}s")

    # A brand-new run checks the LLM key/quota/network with ONE real call up front (fail in one second, not after
    # minutes of transcription). Every other call site only does the cheap, no-network check_ready(): a tight daily
    # free-tier quota (e.g. Gemini's ~20 requests/model/day) cannot afford a throwaway validation ping before every
    # stage, and the real chaptering/summarization calls already degrade gracefully (see summarize.py,
    # segment/llm_chaptering.py) if the key/quota turns out to be bad anyway.
    first_run = force or not (out_dir / "manifest.json").exists()
    llm_validated = False

    def ensure_llm(real_check: bool = False) -> None:
        nonlocal llm_validated
        llm.check_ready()
        if real_check and not llm_validated:
            llm.validate()
            llm_validated = True

    if first_run:
        ensure_llm()
    find_ffmpeg()
    manifest = _Manifest(out_dir / "manifest.json")

    # ---------------------------------------------------------------- 1. ingest
    with stage(1, "Ingest"):
        meta = ingest(source, out_dir, cookies_file=cookies_file, cookies_from_browser=cookies_from_browser, force=force)
        print(f"      {meta.title!r}  {fmt_ts(meta.duration)}"
              + (f"  ({len(meta.creator_chapters)} creator chapters kept for evaluation only)" if meta.creator_chapters else ""))

    # ---------------------------------------------------------------- 2. transcribe
    with stage(2, "Transcribe"):
        wav = Path(meta.audio_path)
        t_path = out_dir / "transcript.json"
        if meta.language and not meta.language.lower().startswith("en") and whisper_model.endswith(".en"):
            warn(f"video language is {meta.language!r} but the Whisper model {whisper_model!r} is English-only; "
                 f"use --whisper-model small (or medium) --language {meta.language[:2]}")
        sig_t = _sig(PIPELINE_VERSION, wav.stat().st_size, int(wav.stat().st_mtime), whisper_model, language)
        if manifest.fresh("transcribe", sig_t, t_path, force=force):
            print("      reusing cached transcript")
            transcript = load_transcript(t_path)
        else:
            ensure_llm(real_check=True)        # about to spend minutes: make sure the later LLM stages can run
            transcript = transcribe(str(wav), model_size=whisper_model, device=device, language=language)
            save_json(transcript, t_path)
            manifest.mark("transcribe", sig_t)
        print(f"      {len(transcript.segments)} segments, language={transcript.language}")

    # ---------------------------------------------------------------- 3. clean
    with stage(3, "Clean"):
        s_path = out_dir / "sentences.json"
        sig_c = _sig(sig_t, DEFAULT_NO_SPEECH_THRESHOLD, DEFAULT_LOGPROB_THRESHOLD, DEFAULT_COMPRESSION_RATIO_THRESHOLD)
        if manifest.fresh("clean", sig_c, s_path, force=force):
            sentences = load_sentences(s_path)
        else:
            sentences, dropped = clean_pipeline(transcript)
            lost = sum(s.end - s.start for s in dropped)
            if transcript.duration and lost / transcript.duration > 0.15:
                warn(f"{lost / transcript.duration:.0%} of the audio was dropped as non-speech/junk; check the audio quality")
            save_json(sentences, s_path)
            manifest.mark("clean", sig_c)
        if len(sentences) < 3:
            raise PipelineError("almost no speech was found in this media (fewer than 3 sentences after cleaning)")
        print(f"      {len(sentences)} sentences")

    # ---------------------------------------------------------------- 4. segment
    # One method's hard failure (e.g. a dead API key, or an exhausted daily quota hit mid-call) must never take the
    # other method down with it, and must never leave a method with chapters but no report. Each method's work is
    # therefore isolated in its own try/except; a method that fails here is dropped from `methods` (nothing to
    # summarize or report without chapters), with a clear warning recorded.
    chapters: dict[str, list[Chapter]] = {}
    seg_sigs: dict[str, str] = {}
    failed_methods: dict[str, str] = {}
    with stage(4, "Segment"):
        for m in list(methods):
            m_dir = out_dir / m
            m_dir.mkdir(exist_ok=True)
            c_path = m_dir / "chapters.json"
            seg_sigs[m] = _sig(sig_c, m, min_s, target_s,
                               config.EMBED_MODEL if m == "semantic" else (config.LLM_PROVIDER, config.LLM_MODEL, PROMPT_VERSION),
                               config.SEMANTIC_SENSITIVITY if m == "semantic" else None)
            try:
                if manifest.fresh(f"segment:{m}", seg_sigs[m], c_path, force=force):
                    chapters[m] = load_chapters(c_path)
                else:
                    if m == "semantic":
                        a = analyze(sentences, min_chapter_s=min_s, target_chapter_s=target_s)
                        chapters[m] = boundaries_to_chapters(sentences, a.boundaries, min_s)
                        save_json({"times": [s.start for s in sentences], "sims": [float(x) for x in a.sims],
                                   "boundaries": a.boundaries, "window": a.window}, m_dir / "analysis.json")
                    else:
                        ensure_llm()
                        chapters[m] = segment_llm(sentences, video_title=meta.title, min_chapter_s=min_s,
                                                  target_chapter_s=target_s, warn=lambda msg: warn(msg, method=m))
                    save_json(chapters[m], c_path)
                    manifest.mark(f"segment:{m}", seg_sigs[m])
                print(f"      {m}: {len(chapters[m])} chapters")
            except Exception as e:
                reason = llm.short_error(e)
                warn(f"warning: {m} chaptering failed entirely ({reason}); this method is skipped for this run", method=m)
                failed_methods[m] = reason
                methods.remove(m)

    # ---------------------------------------------------------------- 5. summarize
    # Per-chapter and TL;DR failures already degrade gracefully inside summarize_chapters()/generate_tldr() (an
    # extractive fallback, or joined titles). This try/except is the outer safety net for anything that still
    # escapes (e.g. check_ready() finding the key gone) - it falls back to the chapters AS SEGMENTED, with no LLM
    # summary, so a result/report always exists for a method that got as far as having chapters.
    summarized: dict[str, dict] = {}
    with stage(5, "Summarize"):
        for m in methods:
            u_path = out_dir / m / "summarized.json"
            sig_u = _sig(seg_sigs[m], config.LLM_PROVIDER, config.LLM_MODEL, PROMPT_VERSION)
            try:
                if manifest.fresh(f"summarize:{m}", sig_u, u_path, force=force):
                    summarized[m] = load_json(u_path)
                else:
                    ensure_llm()
                    m_warn = lambda msg: warn(msg, method=m)
                    chs = summarize_chapters(chapters[m], sentences, overwrite_titles=(m == "semantic"),
                                             video_title=meta.title, warn=m_warn)
                    summarized[m] = {"tldr": generate_tldr(chs, meta.title, warn=m_warn), "chapters": [c.__dict__ for c in chs]}
                    save_json(summarized[m], u_path)
                    manifest.mark(f"summarize:{m}", sig_u)
                print(f"      {m}: {len(summarized[m]['chapters'])} chapters summarized")
            except Exception as e:
                reason = llm.short_error(e)
                warn(f"warning: {m} summarization failed entirely ({reason}); using chapters without summaries", method=m)
                summarized[m] = {"tldr": "", "chapters": [{**c.__dict__, "summary": c.summary or ""} for c in chapters[m]]}

    # ---------------------------------------------------------------- 6. report + evaluation
    duration = meta.duration or transcript.duration or sentences[-1].end
    results: dict[str, dict] = {}
    with stage(6, "Report"):
        for m in methods:
            results[m] = {"source": meta.title, "url": meta.source, "video_id": meta.video_id, "duration": duration,
                          "method": m, "tldr": summarized[m]["tldr"], "chapters": summarized[m]["chapters"],
                          "warnings": warnings_for(m)}
            save_json(results[m], out_dir / m / "result.json")
            (out_dir / m / "report.md").write_text(markdown_report(results[m]), encoding="utf-8")
            (out_dir / m / "youtube_chapters.txt").write_text(chapters_txt(results[m]["chapters"]), encoding="utf-8")
        if "semantic" in methods:
            _try_plot_curve(out_dir, meta)

        ev = _evaluate(out_dir, meta, transcript, results, duration, evaluate, reference_text, ref_start, ref_end)
        if ev is not None:
            save_json(ev, out_dir / "evaluation.json")
        _write_best(out_dir, results, ev)

    if failed_methods:
        print(f"      note: {', '.join(failed_methods)} could not be completed this run (see warnings above); "
              f"re-run the same command later to retry just the missing method(s)")

    out = {"run_dir": str(out_dir), "evaluation": ev, "warnings": [msg for _, msg in warnings],
           "failed_methods": failed_methods}
    if not results:
        raise PipelineError(f"every chaptering method failed this run: {failed_methods}")
    if len(methods) == 1:
        return {**results[methods[0]], **out}
    return {"source": meta.title, "method": "both", "results": results, **out}


def _write_best(out_dir: Path, results: dict[str, dict], ev: dict | None) -> None:
    """Copies the evaluation's recommended method's output to out_dir/best/, so there is one obvious
    'final' answer to show even when both methods ran. Silently does nothing without 2+ results or a recommendation."""
    rec = (ev or {}).get("recommended_method")
    if not rec or rec["method"] not in results or len(results) < 2:
        return
    best_dir = out_dir / "best"
    best_dir.mkdir(exist_ok=True)
    m = rec["method"]
    save_json({**results[m], "chosen_by_evaluation": rec}, best_dir / "result.json")
    (best_dir / "report.md").write_text(markdown_report(results[m]) + f"\n\n*Chosen over the other method(s): {rec['reason']}*\n",
                                        encoding="utf-8")
    (best_dir / "youtube_chapters.txt").write_text(chapters_txt(results[m]["chapters"]), encoding="utf-8")


def _evaluate(out_dir, meta, transcript, results, duration, enabled, reference_text, ref_start, ref_end):
    if not enabled:
        return None
    ev: dict = {}
    gt = creator_chapters_to_chapters(meta.creator_chapters)
    if len(gt) >= 2:
        ev["chaptering"] = evaluate_chaptering({m: [Chapter(**c) for c in r["chapters"]] for m, r in results.items()}, gt, duration)
        rec = recommend_method(ev["chaptering"])
        if rec:
            ev["recommended_method"] = rec
    else:
        ev["chaptering_skipped"] = ("this source has no creator chapters to compare against (the pipeline still "
                                    "generated chapters; evaluate on a video that has them)")
    ref_text, label, start, end = None, "", 0.0, None
    if reference_text:
        ref_text, start, end = Path(reference_text).read_text(encoding="utf-8"), ref_start, ref_end
        label = f"vs your reference text, {fmt_ts(start)}-{fmt_ts(end) if end else 'end'}"
    elif meta.captions_path and Path(meta.captions_path).exists():
        ref_text = Path(meta.captions_path).read_text(encoding="utf-8")
        label = "vs the video's manual captions; approximate, captions are edited rather than verbatim"
    if ref_text:
        ev["wer"] = {**evaluate_wer(transcript, ref_text, start, end), "reference": label}
    else:
        ev["wer_skipped"] = "no manual captions on this video; pass --reference-text FILE [--ref-start S --ref-end S]"
    print(format_evaluation(ev))
    return ev


def _try_plot_curve(out_dir: Path, meta) -> None:
    try:
        from podsum.show import render_curve
        a = load_json(out_dir / "semantic" / "analysis.json")
        gt = [c["start"] for c in meta.creator_chapters] or None
        render_curve(a["times"], a["sims"], [a["times"][b] for b in a["boundaries"]],
                     str(out_dir / "semantic" / "similarity_curve.png"), gt)
    except Exception as e:      # matplotlib is optional; never fail a run over a plot
        print(f"      (similarity-curve plot skipped: {type(e).__name__}: {e})")


def main() -> None:
    from podsum.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()