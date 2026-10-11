"""
Approach B - LLM-prompted chaptering.

Long transcripts do not fit one prompt, so this runs in two passes:
  1. PROPOSE  each overlapping excerpt of the timestamped transcript goes to the LLM, which returns
              chapter starts (a timestamp copied from the transcript + a short title) for that excerpt
  2. CONSOLIDATE  the (short) merged list of proposals goes back to the LLM once, to drop duplicates from
              the overlaps, merge over-fine chapters, and even out the granularity across the whole recording
Both passes are validated in code: timestamps snap to real sentence starts, chapters below the minimum
length are merged, and if the LLM misbehaves the code falls back to a deterministic merge.

Usage:
    python -m podsum.segment.llm_chaptering data/runs/ep1/sentences.json data/runs/ep1/chapters_llm.json
"""
from __future__ import annotations

import argparse

from podsum import config
from podsum.jsonutil import extract_json
from podsum.llm import complete, short_error
from podsum.schema import Chapter, Sentence, fmt_ts, load_sentences, parse_ts, save_json
from podsum.segment.postprocess import build_chapters, nearest_sentence

CHARS_PER_CHUNK = 9_000     # ~2.2k tokens of transcript per call
OVERLAP_SENTENCES = 6       # so a topic change right at a chunk seam is seen by both neighbours

PROPOSE_SYSTEM = (
    "You segment a spoken-word transcript (podcast, lecture, interview or talk) into chapters, the way a careful "
    "editor writes YouTube chapter markers.\n"
    "Rules:\n"
    "- Start a new chapter only where the discussion clearly moves to a different topic or segment (a new subject, "
    "a new question in an interview, intro -> main content, a sponsor read, the wrap-up). Do not split every sub-point.\n"
    "- A chapter is normally at least {min_min:.0f} minutes long; a typical one is about {target_min:.0f} minutes.\n"
    "- Every start MUST be a timestamp copied exactly from the beginning of a transcript line.\n"
    "- Titles: 3-7 words, specific to what is said, no timestamps, no quotation marks.\n"
    'Respond with ONLY a JSON array, no prose, no markdown fences: [{{"start": "MM:SS", "title": "..."}}, ...]'
)
CONSOLIDATE_SYSTEM = (
    "You are given candidate chapter starts for one recording, produced independently from overlapping excerpts, "
    "so some are duplicates or too fine-grained. Produce the final chapter list.\n"
    "Rules: keep only real topic changes; merge candidates that describe the same or adjacent sub-topics; every chapter "
    "must be at least {min_min:.0f} minutes long; aim for roughly {n_target} chapters (fewer is fine if the content "
    "is one continuous topic); keep the first candidate as the first chapter; every start MUST be copied exactly from "
    "the candidate list; titles 3-7 words.\n"
    'Respond with ONLY a JSON array, no prose, no markdown fences: [{{"start": "MM:SS", "title": "..."}}, ...]'
)


def _chunk_sentences(sentences: list[Sentence], chars_per_chunk: int, overlap: int = OVERLAP_SENTENCES) -> list[list[Sentence]]:
    chunks: list[list[Sentence]] = []
    cur: list[Sentence] = []
    cur_len = new = 0
    for s in sentences:
        cur.append(s)
        new += 1
        cur_len += len(s.text)
        if cur_len >= chars_per_chunk:
            chunks.append(cur)
            cur = cur[-overlap:]
            cur_len = sum(len(x.text) for x in cur)
            new = 0
    if new > 0:
        chunks.append(cur)
    return chunks


def _format_transcript(sentences: list[Sentence]) -> str:
    return "\n".join(f"[{fmt_ts(s.start)}] {s.text}" for s in sentences)


def _parse_proposals(raw: str) -> list[dict]:
    data = extract_json(raw)
    if isinstance(data, dict):                       # some models wrap the array: {"chapters": [...]}
        data = next((v for v in data.values() if isinstance(v, list)), [])
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array, got {type(data).__name__}")
    out = []
    for item in data:
        if isinstance(item, dict) and "start" in item:
            try:
                out.append({"t": parse_ts(str(item["start"])), "title": str(item.get("title", "")).strip()})
            except ValueError:
                continue
    return out


def _ask(system: str, prompt: str, max_tokens: int) -> list[dict]:
    """One LLM call, parsed; a second attempt if the reply was not usable JSON."""
    last: Exception | None = None
    for _ in range(2):
        try:
            return _parse_proposals(complete(prompt, system=system, max_tokens=max_tokens))
        except ValueError as e:
            last = e
    raise ValueError(str(last))


def propose_chapters_for_chunk(chunk: list[Sentence], part: int, parts: int, total_s: float, video_title: str,
                               min_s: float, target_s: float) -> list[dict]:
    header = (f"Video title: {video_title or '(unknown)'}\n"
              f"This excerpt is part {part} of {parts}, covering {fmt_ts(chunk[0].start)} to {fmt_ts(chunk[-1].end)} "
              f"of a {fmt_ts(total_s)} recording. Mark chapter starts inside this excerpt only. "
              + ("" if part == 1 else "If the topic at the very start simply continues from the previous part, do not add a chapter there.")
              + "\n\nTranscript:\n")
    system = PROPOSE_SYSTEM.format(min_min=min_s / 60, target_min=target_s / 60)
    return _ask(system, header + _format_transcript(chunk), 1500)


def consolidate(proposals: list[dict], total_s: float, video_title: str, min_s: float, target_s: float) -> list[dict]:
    n_target = max(1, round(total_s / target_s))
    lines = "\n".join(f"{fmt_ts(p['t'])} | {p['title']}" for p in proposals)
    prompt = f"Video title: {video_title or '(unknown)'}\nRecording length: {fmt_ts(total_s)}\n\nCandidates:\n{lines}"
    system = CONSOLIDATE_SYSTEM.format(min_min=min_s / 60, n_target=n_target)
    return _ask(system, prompt, 2500)


def segment_llm(sentences: list[Sentence], video_title: str = "", min_chapter_s: float | None = None,
                target_chapter_s: float | None = None, chars_per_chunk: int = CHARS_PER_CHUNK,
                warn=print) -> list[Chapter]:
    """Chapters (with titles) covering the whole transcript. Summaries are filled in later by summarize.py."""
    if not sentences:
        return []
    min_s = config.MIN_CHAPTER_S if min_chapter_s is None else min_chapter_s
    target_s = config.TARGET_CHAPTER_S if target_chapter_s is None else target_chapter_s
    span_start, span_end = sentences[0].start, sentences[-1].end
    total = span_end - span_start
    starts = [s.start for s in sentences]
    chunks = _chunk_sentences(sentences, chars_per_chunk)

    proposals: list[dict] = []
    for k, chunk in enumerate(chunks, 1):
        try:
            proposals += propose_chapters_for_chunk(chunk, k, len(chunks), span_end, video_title, min_s, target_s)
        except Exception as e:      # ValueError (bad JSON) AND real API errors (rate limit, quota, network) alike
            warn(f"warning: chaptering excerpt {k}/{len(chunks)} returned unusable output and was skipped ({short_error(e)})")
    proposals = [p for p in proposals if span_start - 5 <= p["t"] <= span_end + 5]
    proposals.sort(key=lambda p: p["t"])

    # snap to sentence starts, drop exact duplicates (from chunk overlap)
    snapped: dict[int, str] = {}
    for p in proposals:
        snapped.setdefault(nearest_sentence(starts, p["t"]), p["title"])
    merged = [{"t": starts[i], "title": t} for i, t in sorted(snapped.items())]

    final = merged
    if len(chunks) > 1 and len(merged) > 1:
        try:
            final = consolidate(merged, span_end, video_title, min_s, target_s) or merged
        except Exception as e:
            warn(f"warning: chapter consolidation failed ({short_error(e)}); using the merged proposals instead")

    idx_titles = [(nearest_sentence(starts, p["t"]), p["title"]) for p in final]
    if not idx_titles:
        warn("warning: the LLM proposed no chapters; falling back to a single chapter")
    return build_chapters(sentences, idx_titles, min_s)


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-prompted chapter boundary + title proposal.")
    ap.add_argument("sentences_path")
    ap.add_argument("out_path")
    ap.add_argument("--title", default="", help="video title, given to the LLM as context")
    args = ap.parse_args()
    chapters = segment_llm(load_sentences(args.sentences_path), video_title=args.title)
    save_json(chapters, args.out_path)
    print(f"OK  {len(chapters)} chapters -> {args.out_path}")


if __name__ == "__main__":
    main()