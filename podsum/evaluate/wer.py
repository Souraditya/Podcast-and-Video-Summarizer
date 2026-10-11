"""
Evaluation Strategy #3 — Transcription accuracy (Word Error Rate).

Compares the raw transcript (podsum.transcribe output, before cleaning —
WER is a property of the ASR step) against a reference transcript, either
a known-good transcript or manual spot-check text.

Usage:
    python -m podsum.evaluate.wer data/raw/ep1/transcript.json data/eval/ep1_reference.txt
"""
from __future__ import annotations

import argparse

from podsum.schema import Transcript, load_transcript


def word_error_rate(hypothesis: str, reference: str) -> dict:
    import jiwer

    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])
    out = jiwer.process_words(reference, hypothesis,
                               reference_transform=transform, hypothesis_transform=transform)
    return {
        "wer": out.wer,
        "substitutions": out.substitutions,
        "deletions": out.deletions,
        "insertions": out.insertions,
        "hits": out.hits,
    }


def transcript_text_between(transcript: Transcript, start: float = 0.0, end: float | None = None) -> str:
    """Raw ASR text restricted to [start, end] seconds. A hand-corrected reference
    usually covers only a few minutes; scoring it against the whole transcript
    would count every other word as an insertion and inflate WER."""
    end = float("inf") if end is None else end
    words = [w.text for seg in transcript.segments for w in seg.words if w.start >= start and w.end <= end]
    if words:
        return " ".join(words)
    return " ".join(seg.text.strip() for seg in transcript.segments if seg.start >= start and seg.end <= end)


def evaluate_transcript(transcript: Transcript, reference_text: str,
                        start: float = 0.0, end: float | None = None) -> dict:
    return word_error_rate(transcript_text_between(transcript, start, end), reference_text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute WER against a reference transcript.")
    ap.add_argument("transcript_path", help="raw transcript.json from podsum.transcribe")
    ap.add_argument("reference_path", help="plain-text reference transcript")
    ap.add_argument("--start", type=float, default=0.0, help="start of the reference span, seconds")
    ap.add_argument("--end", type=float, default=None, help="end of the reference span, seconds (omit = to the end)")
    args = ap.parse_args()

    transcript = load_transcript(args.transcript_path)
    reference = open(args.reference_path, encoding="utf-8").read()
    result = evaluate_transcript(transcript, reference, start=args.start, end=args.end)
    print(f"WER: {result['wer']:.3f}  "
          f"(sub={result['substitutions']} del={result['deletions']} "
          f"ins={result['insertions']} hit={result['hits']})")


if __name__ == "__main__":
    main()
