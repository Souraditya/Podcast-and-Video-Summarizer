from podsum.evaluate.wer import evaluate_transcript, transcript_text_between
from podsum.schema import Segment, Transcript, Word
from podsum.show import format_summary, render_curve


def _transcript():
    words = [Word(text=t, start=float(i), end=i + 0.9) for i, t in enumerate("the quick brown fox jumps over the lazy dog".split())]
    return Transcript(language="en", duration=9.0, segments=[Segment(id=0, start=0, end=9, text=" ".join(w.text for w in words), words=words)])


def test_wer_is_scored_on_the_reference_span_only():
    tr = _transcript()
    assert transcript_text_between(tr, 0, 3.95) == "the quick brown fox"
    assert evaluate_transcript(tr, "the quick brown fox", start=0, end=3.95)["wer"] == 0.0
    assert evaluate_transcript(tr, "the quick brown fox")["wer"] > 0.5   # full transcript vs short reference: inflated


def test_format_summary_lists_every_chapter():
    out = format_summary({"source": "s", "method": "llm", "tldr": "all about x",
                          "chapters": [{"start": 0, "end": 65, "title": "Intro", "summary": "hello"},
                                       {"start": 65, "end": 130, "title": "Main", "summary": "body"}]})
    assert "TL;DR" in out and "Intro" in out and "01:05" in out and out.count("[") == 2


def test_render_curve_writes_png(tmp_path):
    out = tmp_path / "c.png"
    render_curve([0, 30, 60, 90], [1, 0.4, 0.9, 1], [30], str(out), gt_times=[35])
    assert out.stat().st_size > 1000
