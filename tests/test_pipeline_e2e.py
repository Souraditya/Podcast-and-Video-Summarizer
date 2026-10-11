import json

import pytest

from podsum.pipeline import PipelineError, run
from podsum.schema import Segment, Transcript
from tests.conftest import synthetic_transcript

CREATOR = [{"start": 0.0, "end": 600.0, "title": "Intro"}, {"start": 600.0, "end": 1200.0, "title": "Middle"},
           {"start": 1200.0, "end": 1800.0, "title": "End"}]
SRC = "https://youtu.be/AAAAAAAAAAA"


def test_both_methods_end_to_end_with_ground_truth(stub_pipeline, tmp_path):
    stub_pipeline.make_ingest(creator_chapters=CREATOR, captions=synthetic_transcript().text)
    out = tmp_path / "run"
    res = run(SRC, out, method="both")

    assert res["method"] == "both" and set(res["results"]) == {"semantic", "llm"}
    sem, llm = res["results"]["semantic"], res["results"]["llm"]
    assert len(sem["chapters"]) == len(llm["chapters"]) == 3
    assert [c["title"] for c in llm["chapters"]] == ["Topic 1", "Topic 2", "Topic 3"]      # LLM titles kept
    assert {c["title"] for c in sem["chapters"]} == {"Generated Title"}                    # semantic titled by summarizer
    assert all(c["summary"] for c in sem["chapters"] + llm["chapters"]) and sem["tldr"] and llm["tldr"]

    for name in ("meta.json", "transcript.json", "sentences.json", "evaluation.json", "manifest.json",
                 "semantic/result.json", "semantic/report.md", "semantic/youtube_chapters.txt", "semantic/analysis.json",
                 "llm/result.json", "llm/report.md", "llm/youtube_chapters.txt"):
        assert (out / name).exists(), name
    assert (out / "llm" / "youtube_chapters.txt").read_text().splitlines() == ["00:00 Topic 1", "10:00 Topic 2", "20:00 Topic 3"]
    assert "https://youtu.be/AAAAAAAAAAA?t=600" in (out / "llm" / "report.md").read_text()

    assert (out / "semantic" / "similarity_curve.png").stat().st_size > 1000
    ev = res["evaluation"]
    assert ev["chaptering"]["methods"]["llm"]["boundary"]["15"]["f1"] == 1.0
    assert ev["chaptering"]["methods"]["semantic"]["boundary"]["60"]["f1"] == 1.0
    assert ev["chaptering"]["methods"]["llm"]["pk"] == 0.0
    assert ev["wer"]["wer"] == 0.0


def test_generation_never_sees_creator_chapters(stub_pipeline, tmp_path):
    """Ground truth is for scoring only: nothing in any prompt may contain the creator's chapter titles."""
    stub_pipeline.make_ingest(creator_chapters=[{"start": 0.0, "end": 600.0, "title": "SECRET-CREATOR-TITLE"},
                                                {"start": 600.0, "end": 1800.0, "title": "SECRET-TWO"}])
    seen = []
    inner = stub_pipeline.llm.__call__

    def spy(prompt, system="", **kw):
        seen.append(prompt + system)
        return inner(prompt, system=system, **kw)

    import podsum.segment.llm_chaptering as lc, podsum.summarize as sm
    lc.complete = sm.complete = spy
    run(SRC, tmp_path / "run", method="both")
    assert seen and not any("SECRET" in s for s in seen)


def test_rerun_is_cached_and_settings_changes_invalidate_only_what_they_touch(stub_pipeline, tmp_path):
    out = tmp_path / "run"
    first = run(SRC, out, method="both", evaluate=False)
    calls_after_first = stub_pipeline.llm.calls
    assert stub_pipeline.counts["transcribe"] == 1 and calls_after_first > 0

    again = run(SRC, out, method="both", evaluate=False)                    # identical settings
    assert stub_pipeline.counts["transcribe"] == 1 and stub_pipeline.llm.calls == calls_after_first
    assert again["results"]["llm"]["chapters"] == first["results"]["llm"]["chapters"]

    run(SRC, out, method="llm", evaluate=False, min_chapter_s=120)          # new chapter settings: re-segment, don't re-transcribe
    assert stub_pipeline.counts["transcribe"] == 1 and stub_pipeline.llm.calls > calls_after_first

    run(SRC, out, method="semantic", evaluate=False, force=True)            # force: start over
    assert stub_pipeline.counts["transcribe"] == 2


def test_no_ground_truth_skips_evaluation_honestly(stub_pipeline, tmp_path):
    res = run(SRC, tmp_path / "run", method="semantic")
    assert "creator chapters" in res["evaluation"]["chaptering_skipped"] and "wer_skipped" in res["evaluation"]
    assert res["evaluation"].get("chaptering") is None


def test_single_method_result_has_the_shape_the_web_app_uses(stub_pipeline, tmp_path):
    steps = []
    res = run(SRC, tmp_path / "run", method="semantic", on_progress=lambda i, label: steps.append((i, label)), evaluate=False)
    assert res["method"] == "semantic" and res["tldr"] and len(res["chapters"]) == 3
    assert [i for i, _ in steps] == [1, 2, 3, 4, 5, 6]


def test_wer_against_a_reference_text_span(stub_pipeline, tmp_path):
    tr = synthetic_transcript()
    ref = tmp_path / "ref.txt"
    ref.write_text(" ".join(s.text for s in tr.segments[:2]), encoding="utf-8")        # the first 12 seconds only
    res = run(SRC, tmp_path / "run", method="semantic", reference_text=str(ref), ref_start=0.0, ref_end=12.0)
    assert res["evaluation"]["wer"]["wer"] == 0.0


def test_silent_media_fails_with_a_clear_error(stub_pipeline, tmp_path, monkeypatch):
    monkeypatch.setattr("podsum.pipeline.transcribe", lambda *a, **k: Transcript("en", 60.0, []))
    with pytest.raises(PipelineError, match="no speech|almost no speech"):
        run(SRC, tmp_path / "run", method="semantic")


def test_missing_api_key_fails_before_any_expensive_work(stub_pipeline, tmp_path, monkeypatch):
    from podsum.llm import LLMConfigError
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        run(SRC, tmp_path / "run", method="both")
    assert stub_pipeline.counts["transcribe"] == 0


def test_unknown_method_rejected(stub_pipeline, tmp_path):
    with pytest.raises(ValueError):
        run(SRC, tmp_path / "run", method="magic")


def test_cli_prints_both_methods_and_output_location(stub_pipeline, tmp_path, capsys):
    from podsum.cli import main
    stub_pipeline.make_ingest(creator_chapters=CREATOR)
    main([SRC, "--out", str(tmp_path / "run"), "--method", "both"])
    out = capsys.readouterr().out
    assert "SEMANTIC CHAPTERING" in out and "LLM CHAPTERING" in out and "TL;DR" in out
    assert "Outputs in" in out and "Chapter boundaries vs the creator's 3 chapters" in out


def test_cli_turns_expected_failures_into_one_clean_error_line(stub_pipeline, tmp_path, monkeypatch):
    from podsum.cli import main
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(SystemExit) as e:
        main([SRC, "--out", str(tmp_path / "run")])
    assert "ANTHROPIC_API_KEY" in str(e.value)


def test_invalid_key_is_caught_before_any_expensive_work(stub_pipeline, tmp_path, monkeypatch):
    from podsum.llm import LLMConfigError

    def reject(*a, **k):
        raise LLMConfigError("the API rejected your key (401)")
    monkeypatch.setattr("podsum.llm.validate", reject)
    with pytest.raises(LLMConfigError, match="401"):
        run(SRC, tmp_path / "run", method="both")
    assert stub_pipeline.counts["transcribe"] == 0


def test_cli_reports_unexpected_runtime_errors_cleanly(stub_pipeline, tmp_path, monkeypatch, capsys):
    """An error OUTSIDE the per-method safety net (here: ingest, stage 1, before any chaptering work) must still
    abort cleanly - only per-method LLM failures (stage 4/5) are caught and degrade gracefully."""
    from podsum.cli import main
    monkeypatch.setattr("podsum.pipeline.ingest", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    with pytest.raises(SystemExit) as e:
        main([SRC, "--out", str(tmp_path / "run"), "--method", "llm"])
    assert "RuntimeError: disk full" in str(e.value) and "cached" in str(e.value)


def test_one_methods_llm_failure_does_not_take_down_the_other(stub_pipeline, tmp_path, monkeypatch, capsys):
    """Core resilience behaviour, at the pipeline's outer safety net: whatever makes segment_llm() itself raise
    (ValueError/RuntimeError/a real API exception - everything inside already degrades per-chunk; see
    test_llm_chaptering_survives_garbage_replies for that) must not prevent the OTHER method's chapters,
    summaries and report from being produced."""
    from podsum.cli import main
    monkeypatch.setattr("podsum.pipeline.segment_llm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    main([SRC, "--out", str(tmp_path / "run"), "--method", "both"])             # must NOT raise SystemExit
    out = capsys.readouterr().out
    assert "SEMANTIC CHAPTERING" in out and "LLM CHAPTERING" not in out
    assert "llm chaptering failed entirely" in out and "boom" in out
    assert "note: llm could not be completed" in out
    assert (tmp_path / "run" / "semantic" / "report.md").exists()
    assert not (tmp_path / "run" / "llm" / "result.json").exists()


def test_fully_cached_rerun_needs_no_api_key_and_no_network(stub_pipeline, tmp_path, monkeypatch):
    """Demo safety: once a link has been processed, re-running it must work with the LLM unreachable."""
    from podsum.llm import LLMConfigError
    from podsum.pipeline import PipelineError
    out = tmp_path / "run"
    first = run(SRC, out, method="both", evaluate=False)
    calls = stub_pipeline.llm.calls

    def unreachable(*a, **k):
        raise LLMConfigError("no network")
    monkeypatch.setattr("podsum.llm.validate", unreachable)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    again = run(SRC, out, method="both", evaluate=False)
    assert stub_pipeline.llm.calls == calls
    assert again["results"]["llm"]["chapters"] == first["results"]["llm"]["chapters"]

    # new work that needs the LLM, with no key available at all, still fails - but as a clean PipelineError
    # (every requested method failed), not a crash, and the failure reason names the method:
    with pytest.raises(PipelineError, match="llm"):
        run(SRC, out, method="llm", evaluate=False, min_chapter_s=120)


def test_evaluation_recommends_a_method_and_writes_a_best_copy(stub_pipeline, tmp_path, monkeypatch):
    """The 'best/' folder is a convenience copy of whichever method scored higher on THIS video - not a
    standing verdict on either method in general."""
    stub_pipeline.make_ingest(creator_chapters=CREATOR)
    # Make the LLM method's boundaries land further from the creator's than semantic's, so recommend_method
    # has an unambiguous pick to make, and we can assert it end to end (not just in evaluate/auto.py's own tests).
    import podsum.segment.llm_chaptering as lc
    real_consolidate = lc.consolidate
    monkeypatch.setattr(lc, "consolidate", lambda *a, **k: [{"t": 0.0, "title": "All of it"}])  # 1 chapter: worse boundary F1

    out = tmp_path / "run"
    res = run(SRC, out, method="both")
    rec = res["evaluation"]["recommended_method"]
    assert rec["method"] == "semantic" and rec["scores"]["semantic"] > rec["scores"]["llm"]
    assert (out / "best" / "result.json").exists()
    best = json.loads((out / "best" / "result.json").read_text())
    assert best["method"] == "semantic" and best["chosen_by_evaluation"]["method"] == "semantic"
    assert "Chosen over the other method" in (out / "best" / "report.md").read_text()


def test_each_methods_report_only_shows_its_own_warnings(stub_pipeline, tmp_path, monkeypatch):
    """Regression: a report for method X must never show warnings that came from method Y's own chapters/
    stages (e.g. 'chapter 2 summarization failed' for the OTHER method's differently-timed chapter 2)."""
    real_summarize = __import__("podsum.summarize", fromlist=["summarize_chapter"]).summarize_chapter
    calls = {"n": 0}

    def flaky(chunk, video_title=""):
        calls["n"] += 1
        if calls["n"] == 2:                      # fail exactly once, on whichever method summarizes second
            raise RuntimeError("quota")
        return real_summarize(chunk, video_title)

    import podsum.summarize as sm
    monkeypatch.setattr(sm, "summarize_chapter", flaky)
    res = run(SRC, tmp_path / "run", method="both", evaluate=False)

    sem_warnings = " ".join(res["results"]["semantic"]["warnings"])
    llm_warnings = " ".join(res["results"]["llm"]["warnings"])
    # exactly one method actually hit the failure; its report must mention it, the other method's must not
    assert ("summarization failed" in sem_warnings) != ("summarization failed" in llm_warnings)
    # method-specific stage-4 warnings (LLM-chaptering-only) must never leak into the semantic report either
    assert "chaptering excerpt" not in sem_warnings