"""
Stage 2 - Transcribe (faster-whisper).

Keeps word-level timestamps (chapter boundaries and player seek-points snap to real
times) and the per-segment confidence signals the cleaning stage uses to drop junk.
The raw output is never modified afterwards, so WER is always measured on true ASR text.

Usage:
    python -m podsum.transcribe data/runs/ep1/source.wav data/runs/ep1/transcript.json [--model small.en] [--device cpu]
"""
from __future__ import annotations

import argparse
import sys
import time

from podsum import config
from podsum.schema import Segment, Transcript, Word, fmt_ts, save_json


def _run(audio_path: str, model_size: str, device: str, compute_type: str | None,
         language: str | None, progress: bool) -> Transcript:
    from faster_whisper import WhisperModel

    ctype = compute_type or ("float16" if device == "cuda" else "int8")
    model = WhisperModel(model_size, device=device, compute_type=ctype)
    lang = "en" if model_size.endswith(".en") else language
    segments, info = model.transcribe(
        audio_path,
        language=lang,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,                       # drop long silences before they become junk segments
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,      # stops one bad window from poisoning (or looping through) the rest
    )

    total = float(info.duration or 0.0)
    out: list[Segment] = []
    started, last_pct = time.time(), -1
    tty = sys.stdout.isatty()
    for i, seg in enumerate(segments):
        words = [Word(text=w.word.strip(), start=w.start, end=w.end, prob=w.probability) for w in (seg.words or [])]
        out.append(Segment(id=i, start=seg.start, end=seg.end, text=seg.text.strip(),
                           avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob,
                           compression_ratio=seg.compression_ratio, words=words))
        if progress and total:
            pct = min(100, int(100 * seg.end / total))
            if pct != last_pct and (pct % 10 == 0 or tty):
                last_pct = pct
                print(f"      transcribing: {pct:3d}%  ({fmt_ts(seg.end)} / {fmt_ts(total)}, {time.time() - started:.0f}s elapsed)",
                      end="\r" if tty and pct < 100 else "\n", flush=True)
    return Transcript(language=info.language, duration=total, segments=out)


def transcribe(audio_path: str, model_size: str | None = None, device: str | None = None,
               compute_type: str | None = None, language: str | None = None, progress: bool = True) -> Transcript:
    model_size = model_size or config.WHISPER_MODEL
    device = device or config.WHISPER_DEVICE
    compute_type = compute_type or config.WHISPER_COMPUTE_TYPE
    language = language or config.WHISPER_LANGUAGE
    try:
        return _run(audio_path, model_size, device, compute_type, language, progress)
    except Exception as e:
        if device == "cpu":
            raise
        print(f"      warning: {device} transcription failed ({type(e).__name__}: {e}); retrying on CPU")
        return _run(audio_path, model_size, "cpu", None, language, progress)


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcribe a WAV file with faster-whisper.")
    ap.add_argument("audio_path")
    ap.add_argument("out_path")
    ap.add_argument("--model", default=config.WHISPER_MODEL, help="tiny.en/base.en/small.en/medium.en/distil-large-v3 (default: %(default)s)")
    ap.add_argument("--device", default=config.WHISPER_DEVICE, help="cpu | cuda | auto")
    ap.add_argument("--language", default=config.WHISPER_LANGUAGE, help="e.g. hi, es (needs a non-.en model)")
    args = ap.parse_args()
    t = transcribe(args.audio_path, model_size=args.model, device=args.device, language=args.language)
    save_json(t, args.out_path)
    print(f"OK  {len(t.segments)} segments, {t.duration:.1f}s, language={t.language} -> {args.out_path}")


if __name__ == "__main__":
    main()
