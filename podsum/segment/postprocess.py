"""
Shared chapter construction for BOTH chaptering approaches, so the semantic and LLM methods produce
structurally identical output and are scored identically: contiguous chapters that cover every
sentence, start at the first sentence, snap to real sentence boundaries, and respect a minimum length.
"""
from __future__ import annotations

import bisect

from podsum.schema import Chapter, Sentence


def nearest_sentence(starts: list[float], t: float) -> int:
    """Index of the sentence whose start time is closest to t (`starts` sorted ascending)."""
    i = bisect.bisect_left(starts, t)
    if i == 0:
        return 0
    if i == len(starts):
        return len(starts) - 1
    return i if starts[i] - t < t - starts[i - 1] else i - 1


def build_chapters(sentences: list[Sentence], starts: list[tuple[int, str]], min_chapter_s: float = 0.0) -> list[Chapter]:
    """starts = [(first_sentence_index, title), ...] in any order. Returns contiguous chapters covering all
    sentences. The first chapter always begins at sentence 0; chapters shorter than min_chapter_s are merged
    into a neighbour (the merged chapter keeps the longer one's title)."""
    if not sentences:
        return []
    by_idx: dict[int, str] = {}
    for idx, title in sorted(starts, key=lambda s: s[0]):
        if 0 <= idx < len(sentences):
            by_idx.setdefault(idx, (title or "").strip())
    if 0 not in by_idx:
        first = min(by_idx) if by_idx else None
        by_idx[0] = by_idx[first] if first is not None and by_idx[first] else ""
    idxs = sorted(by_idx)
    # each entry: [first_idx, last_idx, title]
    chs = [[idxs[k], (idxs[k + 1] - 1) if k + 1 < len(idxs) else len(sentences) - 1, by_idx[idxs[k]]]
           for k in range(len(idxs))]

    def dur(c) -> float:
        return sentences[c[1]].end - sentences[c[0]].start

    while len(chs) > 1:
        k = min(range(len(chs)), key=lambda j: dur(chs[j]))
        if dur(chs[k]) >= min_chapter_s:
            break
        nb = k - 1 if k > 0 else k + 1
        a, b = sorted((k, nb))
        keep_title = chs[a][2] if dur(chs[a]) >= dur(chs[b]) else chs[b][2]
        chs[a] = [chs[a][0], chs[b][1], keep_title]
        del chs[b]

    return [Chapter(start=sentences[f].start, end=sentences[l].end, title=t, first_sentence=f, last_sentence=l)
            for f, l, t in chs]
