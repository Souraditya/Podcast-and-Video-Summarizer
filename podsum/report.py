"""Human-readable outputs: terminal summary, Markdown report, and a paste-ready YouTube chapter list."""
from __future__ import annotations

import textwrap

from podsum.schema import fmt_ts


def youtube_link(video_id: str | None, t: float) -> str | None:
    return f"https://youtu.be/{video_id}?t={int(t)}" if video_id else None


def format_summary(data: dict, width: int = 96) -> str:
    lines = [f"SOURCE : {data.get('source', '?')}", f"METHOD : {data.get('method', '?')}", ""]
    lines += ["TL;DR", *textwrap.wrap(data.get("tldr", "") or "(none)", width, initial_indent="  ", subsequent_indent="  "), ""]
    for i, c in enumerate(data["chapters"], 1):
        lines.append(f"{i:>2}. [{fmt_ts(c['start'])} - {fmt_ts(c['end'])}]  {c.get('title') or '(untitled)'}")
        if c.get("summary"):
            lines += textwrap.wrap(c["summary"], width, initial_indent="      ", subsequent_indent="      ")
        lines.append("")
    return "\n".join(lines)


def chapters_txt(chapters: list[dict]) -> str:
    """One 'timestamp title' line per chapter - the format YouTube reads from a video description."""
    return "\n".join(f"{fmt_ts(c['start'])} {c.get('title') or 'Chapter'}" for c in chapters) + "\n"


def markdown_report(result: dict) -> str:
    vid = result.get("video_id")
    out = [f"# {result['source']}", ""]
    meta = [f"method: **{result['method']}**", f"length: {fmt_ts(result.get('duration', 0))}",
            f"chapters: {len(result['chapters'])}"]
    if result.get("url"):
        meta.insert(0, f"source: {result['url']}")
    out += [" | ".join(meta), "", "## TL;DR", "", result.get("tldr") or "(none)", "", "## Chapters", ""]
    for c in result["chapters"]:
        link = youtube_link(vid, c["start"])
        stamp = f"[{fmt_ts(c['start'])}]({link})" if link else fmt_ts(c["start"])
        out += [f"### {stamp} {c.get('title') or 'Chapter'}", "", c.get("summary", ""), ""]
    if result.get("warnings"):
        out += ["## Warnings", ""] + [f"- {w}" for w in result["warnings"]] + [""]
    return "\n".join(out)
