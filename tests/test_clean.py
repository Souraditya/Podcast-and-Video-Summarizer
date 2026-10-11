from podsum.clean import filter_segments, normalize_text, resegment_sentences
from podsum.schema import Segment, Transcript, Word


def make_transcript(segments):
    return Transcript(language="en", duration=100.0, segments=segments)


def test_filter_drops_confident_non_speech_but_keeps_speech_onset_after_silence():
    segs = [
        Segment(id=0, start=0, end=2, text="hello there", avg_logprob=-1.3, no_speech_prob=0.9, compression_ratio=1.0),
        Segment(id=1, start=2, end=4, text="real speech", avg_logprob=-0.2, no_speech_prob=0.1, compression_ratio=1.0),
        # Whisper quirk: no_speech_prob is high at a speech onset that follows silence, yet the text is confident
        Segment(id=2, start=4, end=6, text="onset after a pause", avg_logprob=-0.2, no_speech_prob=0.9, compression_ratio=1.0),
    ]
    kept, dropped = filter_segments(make_transcript(segs))
    assert [s.text for s in kept.segments] == ["real speech", "onset after a pause"]
    assert len(dropped) == 1


def test_filter_keeps_low_confidence_speech_but_drops_loops_and_silence():
    segs = [
        Segment(id=0, start=0, end=2, text="accented but real", avg_logprob=-1.4, no_speech_prob=0.05, compression_ratio=1.0),
        Segment(id=1, start=2, end=4, text="loop loop loop loop", avg_logprob=-0.1, no_speech_prob=0.0, compression_ratio=3.0),
        Segment(id=2, start=4, end=6, text="music", avg_logprob=-1.5, no_speech_prob=0.8, compression_ratio=1.0),
        Segment(id=3, start=6, end=8, text="fine", avg_logprob=-0.1, no_speech_prob=0.0, compression_ratio=1.0),
    ]
    kept, dropped = filter_segments(make_transcript(segs))
    assert [s.text for s in kept.segments] == ["accented but real", "fine"]
    assert len(dropped) == 2


def test_filter_drops_third_identical_repeat():
    segs = [Segment(id=i, start=i, end=i + 1, text="Thank you.") for i in range(4)]
    kept, dropped = filter_segments(make_transcript(segs))
    assert len(kept.segments) == 2 and len(dropped) == 2


def test_normalize_strips_fillers_and_fixes_case():
    assert normalize_text("um, so this is like, a test") == "So this is a test"
    assert normalize_text("you know, I really like this") == "I really like this"
    assert normalize_text("  multiple   spaces  ") == "Multiple spaces"


def test_resegment_splits_on_sentence_punctuation():
    words = [
        Word(text="Hello", start=0.0, end=0.3),
        Word(text="world.", start=0.3, end=0.6),
        Word(text="This", start=0.7, end=0.9),
        Word(text="is", start=0.9, end=1.0),
        Word(text="fine.", start=1.0, end=1.3),
    ]
    seg = Segment(id=0, start=0.0, end=1.3, text="Hello world. This is fine.",
                  avg_logprob=-0.1, no_speech_prob=0.0, compression_ratio=1.0, words=words)
    sentences = resegment_sentences(make_transcript([seg]))
    assert len(sentences) == 2
    assert sentences[0].text == "Hello world."
    assert sentences[0].start == 0.0 and sentences[0].end == 0.6
    assert sentences[1].text == "This is fine."
    assert sentences[1].start == 0.7 and sentences[1].end == 1.3


def test_resegment_handles_trailing_fragment_without_terminal_punctuation():
    words = [Word(text="Trailing", start=0.0, end=0.2), Word(text="thought", start=0.2, end=0.5)]
    seg = Segment(id=0, start=0.0, end=0.5, text="Trailing thought",
                  avg_logprob=-0.1, no_speech_prob=0.0, compression_ratio=1.0, words=words)
    sentences = resegment_sentences(make_transcript([seg]))
    assert len(sentences) == 1
    assert sentences[0].text == "Trailing thought"


def test_resegment_does_not_split_on_abbreviations():
    toks = "Dr. Smith visited the U.S. yesterday. It went well.".split()
    words = [Word(text=t, start=i * 0.3, end=i * 0.3 + 0.25) for i, t in enumerate(toks)]
    seg = Segment(id=0, start=0, end=3, text=" ".join(toks), words=words)
    sentences = resegment_sentences(make_transcript([seg]))
    assert [s.text for s in sentences] == ["Dr. Smith visited the U.S. yesterday.", "It went well."]


def test_resegment_splits_on_long_pause_and_caps_runaway_sentences():
    a = [Word(text="first", start=0.0, end=0.4), Word(text="part", start=0.5, end=0.9)]
    b = [Word(text="second", start=5.0, end=5.4), Word(text="part", start=5.5, end=5.9)]
    seg = Segment(id=0, start=0, end=6, text="first part second part", words=a + b)
    assert [s.text for s in resegment_sentences(make_transcript([seg]))] == ["First part", "Second part"]

    run = [Word(text=f"w{i}", start=i * 0.2, end=i * 0.2 + 0.15) for i in range(200)]   # no punctuation at all
    seg = Segment(id=0, start=0, end=40, text="x", words=run)
    assert len(resegment_sentences(make_transcript([seg]))) == 3   # 90 + 90 + 20


def test_resegment_uses_segment_text_when_words_are_missing():
    seg = Segment(id=0, start=10.0, end=14.0, text="No word data here. Still two sentences.")
    sentences = resegment_sentences(make_transcript([seg]))
    assert [s.text for s in sentences] == ["No word data here.", "Still two sentences."]
    assert sentences[0].start == 10.0 and sentences[-1].end == 14.0
