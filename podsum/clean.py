"""
Stage 3 - Clean.

Whisper output is a list of ASR utterances, not sentences, with a predictable set of defects.
Three separable jobs, each unit-tested on its own:

  1. filter_segments    drop hallucinated / non-speech segments using Whisper's own confidence signals
  2. normalize_text     strip filler disfluencies, fix whitespace and casing
  3. resegment_sentences  regroup word timestamps into real sentences (Whisper segment boundaries are
                          not sentence boundaries, and segmentation needs sentence-level units)

The raw transcript stays untouched on disk; cleaning works on a copy.

Usage:
    python -m podsum.clean data/runs/ep1/transcript.json data/runs/ep1/sentences.json
"""
from __future__ import annotations

import argparse
import re
from dataclasses import replace

from podsum.schema import Segment, Sentence, Transcript, Word, load_transcript, save_json

# Whisper's own rules of thumb (openai-whisper's decoding fallbacks):
DEFAULT_NO_SPEECH_THRESHOLD = 0.6        # ...AND a low logprob => the segment is silence/music, not speech
DEFAULT_LOGPROB_THRESHOLD = -1.0
DEFAULT_COMPRESSION_RATIO_THRESHOLD = 2.4  # highly compressible text => repetition loop
MAX_REPEATS = 2                          # the same line 3+ times in a row is a hallucination loop

MAX_SENTENCE_WORDS = 90                  # hard cap when the ASR emits no punctuation for a long stretch
SOFT_SENTENCE_WORDS = 60                 # past this, cut at the next comma
PAUSE_SPLIT_S = 2.0                      # a silence this long between words ends the sentence

_FILLER_INTERJECTIONS = re.compile(r"\b(um+|uh+|erm+|hm+)\b,?", re.IGNORECASE)
# "like" / "you know" / "I mean" are only fillers when set off by commas ("so, like, this happened");
# stripping them unconditionally would eat "I like this" / "you know the answer".
_FILLER_DISCOURSE_MARKERS = re.compile(r"\b(like|you know|i mean)\b,", re.IGNORECASE)
_TERMINAL = re.compile(r"[.!?][\"')\]]*$")
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "inc", "ltd", "approx", "fig", "etc"}
_INITIALISM = re.compile(r"^(?:[A-Za-z]\.){2,}$")   # U.S.  e.g.  i.e.


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def filter_segments(
    transcript: Transcript,
    no_speech_threshold: float = DEFAULT_NO_SPEECH_THRESHOLD,
    logprob_threshold: float = DEFAULT_LOGPROB_THRESHOLD,
    compression_ratio_threshold: float = DEFAULT_COMPRESSION_RATIO_THRESHOLD,
) -> tuple[Transcript, list[Segment]]:
    """Return (kept, dropped). A segment is junk if it is empty, is a repetition loop, is confidently
    non-speech (high no_speech_prob AND low logprob, Whisper's own silence rule), or repeats the previous
    line more than MAX_REPEATS times. A low logprob alone is NOT enough: accented or noisy real speech has
    low confidence too, and faster-whisper already retried those at higher temperature."""
    kept, dropped = [], []
    run_key, run_len = None, 0
    for seg in transcript.segments:
        key = _norm_key(seg.text)
        run_len = run_len + 1 if key and key == run_key else 1
        run_key = key
        is_junk = (
            not seg.text.strip()
            or seg.compression_ratio >= compression_ratio_threshold
            or (seg.no_speech_prob >= no_speech_threshold and seg.avg_logprob <= logprob_threshold)
            or run_len > MAX_REPEATS
        )
        (dropped if is_junk else kept).append(seg)
    return replace(transcript, segments=kept), dropped


def normalize_text(text: str, strip_fillers: bool = True) -> str:
    if strip_fillers:
        text = _FILLER_INTERJECTIONS.sub("", text)
        text = _FILLER_DISCOURSE_MARKERS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?])", r"\1", text)  # no space before punctuation
    if text:
        text = text[0].upper() + text[1:]
    return text


def _ends_sentence(token: str) -> bool:
    if not _TERMINAL.search(token):
        return False
    if token.endswith(".") and not token.endswith(".."):
        core = token.rstrip(".").lower()
        if core in _ABBREVIATIONS or _INITIALISM.match(token):
            return False
    return True


def _segment_words(seg: Segment) -> list[Word]:
    """Words with timestamps; segments without word data get their text spread evenly over the segment."""
    if seg.words:
        return seg.words
    toks = seg.text.split()
    if not toks:
        return []
    step = (seg.end - seg.start) / len(toks)
    return [Word(text=t, start=seg.start + i * step, end=seg.start + (i + 1) * step) for i, t in enumerate(toks)]


def resegment_sentences(transcript: Transcript, strip_fillers: bool = True) -> list[Sentence]:
    """Flatten every kept segment's words, then cut on sentence-ending punctuation (ignoring abbreviations),
    on long pauses, and - as a safety net for punctuation-free ASR - on very long runs."""
    words = [w for seg in transcript.segments for w in _segment_words(seg)]
    sentences: list[Sentence] = []
    buf: list[Word] = []

    def flush() -> None:
        nonlocal buf
        if buf:
            clean = normalize_text(" ".join(w.text for w in buf), strip_fillers)
            if clean:
                sentences.append(Sentence(idx=len(sentences), text=clean, start=buf[0].start, end=buf[-1].end))
        buf = []

    for w in words:
        if buf and w.start - buf[-1].end >= PAUSE_SPLIT_S:
            flush()
        buf.append(w)
        if _ends_sentence(w.text) or len(buf) >= MAX_SENTENCE_WORDS \
                or (len(buf) >= SOFT_SENTENCE_WORDS and w.text.endswith(",")):
            flush()
    flush()
    return sentences


def clean_pipeline(transcript: Transcript, strip_fillers: bool = True) -> tuple[list[Sentence], list[Segment]]:
    filtered, dropped = filter_segments(transcript)
    return resegment_sentences(filtered, strip_fillers=strip_fillers), dropped


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean a raw transcript into sentence-level units.")
    ap.add_argument("transcript_path")
    ap.add_argument("out_path")
    ap.add_argument("--keep-fillers", action="store_true")
    args = ap.parse_args()
    transcript = load_transcript(args.transcript_path)
    sentences, dropped = clean_pipeline(transcript, strip_fillers=not args.keep_fillers)
    save_json(sentences, args.out_path)
    lost = sum(s.end - s.start for s in dropped)
    print(f"OK  {len(sentences)} sentences ({len(dropped)} segments / {lost:.0f}s dropped as junk) -> {args.out_path}")


if __name__ == "__main__":
    main()
