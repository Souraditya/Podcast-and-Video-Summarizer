"""
Stage 5 - Summarize (hierarchical).

A chapter's sentences are split into chunks; each chunk is summarized independently; the chunk summaries
are combined into one chapter title + summary (recursing if they are still too long). The same pattern
runs one level up, folding every chapter summary into a single TL;DR. Chapters are summarized in parallel.
If the LLM fails for a chapter, that chapter degrades to an extractive summary (its first sentences) and a
warning is recorded - one bad call never loses the whole run.

Usage:
    python -m podsum.summarize data/runs/ep1/sentences.json data/runs/ep1/chapters_llm.json data/runs/ep1/summarized.json
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

from podsum import config
from podsum.jsonutil import extract_json
from podsum.llm import complete, short_error
from podsum.schema import Chapter, Sentence, fmt_ts, load_chapters, load_sentences, save_json

CHUNK_SUMMARY_SYSTEM = (
    "Summarize this excerpt from a spoken-word transcript in 2-3 sentences. State the concrete points, claims, "
    "examples and names it contains, not vague generalities. Use only what the excerpt says. No preamble."
)
CHAPTER_COMBINE_SYSTEM = (
    "You are given a transcript excerpt or partial summaries of ONE chapter of a podcast/video. Write a single "
    "cohesive chapter summary using only the given content. Respond with ONLY JSON, no prose, no markdown "
    'fences: {"title": "3-7 word specific title", "summary": "3-5 sentence summary"}.'
)
TLDR_SYSTEM = (
    "You are given the chapter summaries of a full podcast/video episode. Write one overall TL;DR of 3-5 sentences "
    "covering the episode's arc and key takeaways. No preamble, no chapter-by-chapter listing. Use only the given content."
)

CHARS_PER_SUMMARY_CHUNK = 8_000
MAX_COMBINE_CHARS = 6_000       # if combined chunk summaries exceed this, summarize them again (another level)
MAX_TLDR_CHARS = 20_000


def _chunk_text(sentences: list[Sentence], chars_per_chunk: int) -> list[str]:
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        cur.append(s.text)
        cur_len += len(s.text)
        if cur_len >= chars_per_chunk:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def summarize_chunk(text: str) -> str:
    return complete(text, system=CHUNK_SUMMARY_SYSTEM, max_tokens=300).strip()


def _reduce(texts: list[str], max_chars: int) -> str:
    """Join texts; while the result is too long, summarize it piecewise and join again."""
    combined = " ".join(texts)
    for _ in range(5):                       # each pass shrinks the text; the bound only guards a misbehaving model
        if len(combined) <= max_chars:
            break
        pieces = [combined[i:i + CHARS_PER_SUMMARY_CHUNK] for i in range(0, len(combined), CHARS_PER_SUMMARY_CHUNK)]
        combined = " ".join(summarize_chunk(p) for p in pieces)
    return combined[:max_chars * 2]


def summarize_chapter(sentences: list[Sentence], video_title: str = "") -> tuple[str, str]:
    """(title, summary) for one chapter's sentences."""
    chunks = _chunk_text(sentences, CHARS_PER_SUMMARY_CHUNK)
    combined = chunks[0] if len(chunks) == 1 else _reduce([summarize_chunk(c) for c in chunks], MAX_COMBINE_CHARS)
    prompt = (f"Video title: {video_title}\n\n" if video_title else "") + combined
    for _ in range(2):
        try:
            data = extract_json(complete(prompt, system=CHAPTER_COMBINE_SYSTEM, max_tokens=500))
            if isinstance(data, dict) and str(data.get("summary", "")).strip():
                return str(data.get("title", "")).strip() or "Untitled", str(data["summary"]).strip()
        except ValueError:
            continue
    raise ValueError("the LLM did not return a usable title/summary")


def _extractive_fallback(sentences: list[Sentence], max_chars: int = 400) -> str:
    text = " ".join(s.text for s in sentences[:4])
    return text if len(text) <= max_chars else text[:max_chars].rsplit(" ", 1)[0] + "..."


def summarize_chapters(chapters: list[Chapter], sentences: list[Sentence], overwrite_titles: bool = False,
                       video_title: str = "", workers: int | None = None, warn=print) -> list[Chapter]:
    """Fills chapter.summary, and chapter.title when it is empty or overwrite_titles is set (the semantic
    approach produces boundaries only, so it needs titles from here)."""
    workers = workers or config.LLM_WORKERS

    def one(item: tuple[int, Chapter]) -> Chapter:
        i, ch = item
        chunk = sentences[ch.first_sentence: ch.last_sentence + 1]
        try:
            title, summary = summarize_chapter(chunk, video_title)
        except Exception as e:                       # network/quota/parse: degrade, don't crash the run
            warn(f"warning: chapter {i + 1} ({fmt_ts(ch.start)}): summarization failed "
                 f"({short_error(e)}); using an extractive fallback")
            title, summary = f"Chapter {i + 1}", _extractive_fallback(chunk)
            overwrite = not ch.title
        else:
            overwrite = overwrite_titles or not ch.title
        return Chapter(start=ch.start, end=ch.end, title=title if overwrite else ch.title, summary=summary,
                       first_sentence=ch.first_sentence, last_sentence=ch.last_sentence)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(one, enumerate(chapters)))


def generate_tldr(chapters: list[Chapter], video_title: str = "", warn=print) -> str:
    parts = [f"{c.title}: {c.summary}" for c in chapters if c.summary]
    if not parts:
        return ""
    try:
        combined = _reduce(["\n\n".join(parts)], MAX_TLDR_CHARS)
        prompt = (f"Video title: {video_title}\n\n" if video_title else "") + combined
        return complete(prompt, system=TLDR_SYSTEM, max_tokens=500).strip()
    except Exception as e:
        warn(f"warning: TL;DR generation failed ({short_error(e)}); joining chapter titles instead")
        return "Chapters: " + "; ".join(c.title for c in chapters)


def main() -> None:
    ap = argparse.ArgumentParser(description="Hierarchically summarize chapters + generate the overall TL;DR.")
    ap.add_argument("sentences_path")
    ap.add_argument("chapters_path")
    ap.add_argument("out_path")
    ap.add_argument("--overwrite-titles", action="store_true")
    ap.add_argument("--title", default="", help="video title, given to the LLM as context")
    args = ap.parse_args()
    sentences = load_sentences(args.sentences_path)
    chapters = summarize_chapters(load_chapters(args.chapters_path), sentences,
                                  overwrite_titles=args.overwrite_titles, video_title=args.title)
    save_json({"tldr": generate_tldr(chapters, args.title), "chapters": [c.__dict__ for c in chapters]}, args.out_path)
    print(f"OK  {len(chapters)} chapters summarized -> {args.out_path}")


if __name__ == "__main__":
    main()