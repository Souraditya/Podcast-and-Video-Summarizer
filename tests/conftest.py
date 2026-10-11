import json
import os
import re
import shutil
from dataclasses import asdict
import wave

import numpy as np
import pytest

from podsum.data.ingest import IngestResult
from podsum.schema import Segment, Transcript, Word, fmt_ts

SENT_S = 6.0          # seconds per synthetic sentence
TOPIC_S = 600.0       # seconds per synthetic topic
N_TOPICS = 3


def write_wav(path, seconds=3.0, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


def synthetic_transcript(n_topics=N_TOPICS, topic_s=TOPIC_S, sent_s=SENT_S) -> Transcript:
    """n_topics x topic_s seconds of speech; one 6-word sentence per segment, distinct vocabulary per topic."""
    segs, i, t = [], 0, 0.0
    while t < n_topics * topic_s - 1e-6:
        topic = int(t // topic_s)
        toks = [f"topic{topic}word{k}" for k in range(5)] + [f"end{i}."]
        step = sent_s / len(toks)
        words = [Word(text=tok, start=t + j * step, end=t + (j + 1) * step - 0.05) for j, tok in enumerate(toks)]
        segs.append(Segment(id=i, start=t, end=t + sent_s, text=" ".join(toks), avg_logprob=-0.2,
                            no_speech_prob=0.01, compression_ratio=1.2, words=words))
        i += 1; t += sent_s
    return Transcript(language="en", duration=n_topics * topic_s, segments=segs)


def fake_embed(sentences, model_name=None):
    """One tight cluster per topic (topic = start // TOPIC_S), so similarity dips exactly at topic changes."""
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(N_TOPICS + 1, 16))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    out = []
    for k, s in enumerate(sentences):
        v = centers[int(s.start // TOPIC_S)] + np.random.default_rng(k).normal(scale=0.03, size=16)
        out.append(v / np.linalg.norm(v))
    return np.array(out)


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    """Tests must behave the same on every machine: ignore PODSUM_* / API-key variables from the developer's shell."""
    from podsum import config
    for name in [k for k in os.environ if k.startswith("PODSUM_")] + ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
        monkeypatch.delenv(name, raising=False)
    defaults = dict(LLM_PROVIDER="anthropic", LLM_MODEL="claude-sonnet-5", LLM_BASE_URL=None, LLM_API_KEY=None,
                    LLM_MIN_INTERVAL_S=0.0, LLM_WORKERS=4, COOKIES_FILE=None, MAX_DURATION_S=240 * 60.0,
                    WHISPER_MODEL="small.en", WHISPER_DEVICE="cpu", WHISPER_COMPUTE_TYPE=None, WHISPER_LANGUAGE=None,
                    MIN_CHAPTER_S=90.0, TARGET_CHAPTER_S=300.0, SEMANTIC_SENSITIVITY=0.25,
                    EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2", DATA_DIR="data")
    for name, value in defaults.items():
        monkeypatch.setattr(config, name, value)


class FakeLLM:
    """Routes on the system prompt, like the real prompts do; counts calls."""
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, system="", max_tokens=0, **kw):
        self.calls += 1
        if "segment a spoken-word transcript" in system:                     # approach B, pass 1
            stamps = set(re.findall(r"^\[(\d+:\d{2})\]", prompt, re.M))
            starts = [(fmt_ts(k * TOPIC_S), f"Topic {k + 1}") for k in range(N_TOPICS) if fmt_ts(k * TOPIC_S) in stamps]
            return "```json\n[" + ",".join(f'{{"start": "{s}", "title": "{t}"}}' for s, t in starts) + "]\n```"
        if "candidate chapter starts" in system:                             # approach B, pass 2
            cands = re.findall(r"^(\d+:\d{2}) \| (.*)$", prompt, re.M)
            return "[" + ",".join(f'{{"start": "{s}", "title": "{t}"}}' for s, t in cands) + "]"
        if "ONE chapter" in system:                                          # chapter title + summary
            return '{"title": "Generated Title", "summary": "A concise generated summary of the chapter."}'
        if "TL;DR" in system:
            return "An overall generated TL;DR of the episode."
        return "A short chunk summary."


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr("podsum.segment.llm_chaptering.complete", llm)
    monkeypatch.setattr("podsum.summarize.complete", llm)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("podsum.llm.validate", lambda *a, **k: None)     # no real network in tests
    return llm


@pytest.fixture
def stub_pipeline(monkeypatch, tmp_path, fake_llm):
    """Everything heavy (ffmpeg, Whisper, embeddings, LLM) replaced by deterministic fakes."""
    counts = {"transcribe": 0}

    def fake_transcribe(audio_path, **kw):
        counts["transcribe"] += 1
        return synthetic_transcript()

    monkeypatch.setattr("podsum.pipeline.transcribe", fake_transcribe)
    monkeypatch.setattr("podsum.segment.semantic.embed_sentences", fake_embed)
    monkeypatch.setattr("podsum.pipeline.find_ffmpeg", lambda: "ffmpeg")

    wav = tmp_path / "in.wav"
    write_wav(wav)

    def make_ingest(creator_chapters=None, captions=None):
        def fake_ingest(source, out_dir, **kw):
            out_dir.mkdir(parents=True, exist_ok=True)
            if not (out_dir / "source.wav").exists():        # like the real ingest, never rewrite existing audio
                shutil.copy(wav, out_dir / "source.wav")
            cap = None
            if captions:
                (out_dir / "captions.txt").write_text(captions, encoding="utf-8")
                cap = str(out_dir / "captions.txt")
            r = IngestResult(audio_path=str(out_dir / "source.wav"), title="Synthetic Talk", duration=N_TOPICS * TOPIC_S,
                             source=source, video_id="AAAAAAAAAAA", creator_chapters=creator_chapters or [], captions_path=cap)
            (out_dir / "meta.json").write_text(json.dumps(asdict(r)), encoding="utf-8")
            return r
        monkeypatch.setattr("podsum.pipeline.ingest", fake_ingest)

    make_ingest()
    return type("Stub", (), {"counts": counts, "llm": fake_llm, "make_ingest": staticmethod(make_ingest)})