"""Shared data structures + small JSON/time helpers used across every stage."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class Segment:
    """One Whisper segment (roughly one utterance) with its confidence signals."""
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0       # ~-0.3 confident, < -1.0 dubious
    no_speech_prob: float = 0.0    # near 1.0 -> probably silence/music, not speech
    compression_ratio: float = 1.0 # > 2.4 often signals repetition loops
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    """Raw ASR output. Cleaning never mutates this, so WER is always measured on the original."""
    language: str
    duration: float
    segments: list[Segment]

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)


@dataclass
class Sentence:
    """Unit of analysis for segmentation: a cleaned, time-stamped sentence."""
    idx: int
    text: str
    start: float
    end: float


@dataclass
class Chapter:
    start: float
    end: float
    title: str = ""
    summary: str = ""
    first_sentence: int = 0
    last_sentence: int = 0  # inclusive


# ---------------------------------------------------------------- JSON helpers
def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    elif isinstance(obj, list):
        obj = [asdict(o) if hasattr(o, "__dataclass_fields__") else o for o in obj]
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_transcript(path: str | Path) -> Transcript:
    d = load_json(path)
    segs = [Segment(**{**s, "words": [Word(**w) for w in s.get("words", [])]}) for s in d["segments"]]
    return Transcript(language=d["language"], duration=d["duration"], segments=segs)


def load_sentences(path: str | Path) -> list[Sentence]:
    return [Sentence(**s) for s in load_json(path)]


def load_chapters(path: str | Path) -> list[Chapter]:
    """Accepts a bare list of chapters, or a pipeline result.json / summarized.json
    style dict that wraps them under a "chapters" key."""
    data = load_json(path)
    if isinstance(data, dict):
        data = data.get("chapters", [])
    return [Chapter(**c) for c in data]


# ---------------------------------------------------------------- time helpers
def fmt_ts(seconds: float, force_hours: bool = False) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if (h or force_hours) else f"{m:02d}:{s:02d}"


def parse_ts(text: str) -> float:
    """'MM:SS', 'H:MM:SS' or plain seconds -> float seconds."""
    text = text.strip().strip("[]() ")
    parts = text.split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"bad timestamp: {text!r}")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total
