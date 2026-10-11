import math
import shutil
import struct
import wave

import pytest

from podsum import config
from podsum.data import ingest as ing
from podsum.data.captions import parse_vtt
from podsum.data.ingest import IngestError, IngestResult, extract_youtube_id, friendly_ytdlp_error, run_dir_for

HAVE_FFMPEG = bool(shutil.which("ffmpeg"))


def test_extract_ids():
    assert extract_youtube_id("https://www.youtube.com/watch?v=8jPQjjsBbIc") == "8jPQjjsBbIc"
    assert extract_youtube_id("https://www.youtube.com/watch?v=8jPQjjsBbIc&list=PLx&t=42") == "8jPQjjsBbIc"
    assert extract_youtube_id("https://youtu.be/8jPQjjsBbIc?si=abc") == "8jPQjjsBbIc"
    assert extract_youtube_id("https://www.youtube.com/shorts/8jPQjjsBbIc") == "8jPQjjsBbIc"
    assert extract_youtube_id("https://example.com/watch?v=8jPQjjsBbIc") is None
    assert extract_youtube_id(r"C:\videos\clip.mp4") is None


def test_run_dir_is_stable_per_input(tmp_path):
    a = run_dir_for("https://youtu.be/8jPQjjsBbIc?si=x", str(tmp_path))
    b = run_dir_for("  https://www.youtube.com/watch?v=8jPQjjsBbIc&t=3  ", str(tmp_path))
    assert a == b == tmp_path / "runs" / "8jPQjjsBbIc"
    f = tmp_path / "My Talk (final).mp4"; f.write_bytes(b"x")
    assert run_dir_for(str(f), str(tmp_path)) == run_dir_for(str(f), str(tmp_path))
    assert run_dir_for("https://example.com/a.mp3", str(tmp_path)).name.startswith("url-")


def test_friendly_errors():
    assert "cookies" in friendly_ytdlp_error("ERROR: [youtube] x: Sign in to confirm you're not a bot")
    assert "private" in friendly_ytdlp_error("ERROR: Private video. Sign in").lower()
    assert "429" in friendly_ytdlp_error("HTTP Error 429: Too Many Requests")
    assert "Deno" in friendly_ytdlp_error("Requested format is not available")


def test_parse_vtt_strips_timing_tags_and_rolling_duplicates():
    vtt = ("WEBVTT\nKind: captions\nLanguage: en\n\n00:00:00.000 --> 00:00:02.000\n>> Hello <c>world</c> &amp; friends\n\n"
           "00:00:02.000 --> 00:00:04.000\nHello world & friends\n[Music]\nSecond line\n")
    assert parse_vtt(vtt) == "Hello world & friends Second line"


def _tone(path, rate=44100, seconds=2.0, channels=2):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels); w.setsampwidth(2); w.setframerate(rate)
        frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) * channels
                          for i in range(int(rate * seconds)))
        w.writeframes(frames)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_local_file_is_converted_to_16k_mono_and_reused(tmp_path, monkeypatch):
    src = tmp_path / "tone.wav"; _tone(src)
    out = tmp_path / "run"
    r = ing.ingest(str(src), out)
    with wave.open(str(out / "source.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
    assert abs(r.duration - 2.0) < 0.1 and r.title == "tone" and r.creator_chapters == []
    monkeypatch.setattr(ing, "to_wav", lambda *a, **k: pytest.fail("must reuse existing audio"))
    assert ing.ingest(str(src), out).title == "tone"


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_undecodable_file_raises_actionable_error(tmp_path):
    bad = tmp_path / "bad.mp4"; bad.write_bytes(b"this is not media")
    with pytest.raises(IngestError, match="could not decode"):
        ing.ingest(str(bad), tmp_path / "run")


def test_missing_file_and_overlong_media(tmp_path, monkeypatch):
    with pytest.raises(IngestError, match="not found"):
        ing.ingest(str(tmp_path / "nope.mp4"), tmp_path / "run")
    monkeypatch.setattr(ing, "_ingest_local", lambda p, o: IngestResult("x", "t", 99999.0, p))
    monkeypatch.setattr(config, "MAX_DURATION_S", 100.0)
    (tmp_path / "a.mp3").write_bytes(b"x")
    with pytest.raises(IngestError, match="limit"):
        ing.ingest(str(tmp_path / "a.mp3"), tmp_path / "run2")


def test_current_run_is_reused_without_network(tmp_path, monkeypatch):
    out = tmp_path / "run"; out.mkdir()
    (out / "source.wav").write_bytes(b"x")
    (out / "meta.json").write_text('{"audio_path": "old", "title": "T", "duration": 60.0, "source": "u", "schema": 2, '
                                   '"creator_chapters": [{"start": 0, "end": 30, "title": "a"}]}')
    monkeypatch.setattr(ing, "_ytdlp_run", lambda *a, **k: pytest.fail("no network expected"))
    r = ing.ingest("https://youtu.be/8jPQjjsBbIc", out)
    assert r.title == "T" and r.audio_path == str(out / "source.wav") and len(r.creator_chapters) == 1


def test_legacy_cache_audio_is_migrated_and_metadata_refreshed(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    legacy = tmp_path / "cache" / "8jPQjjsBbIc"; legacy.mkdir(parents=True)
    write = (legacy / "source.wav"); write.write_bytes(b"audio")
    monkeypatch.setattr(ing, "_refresh_metadata", lambda url, d, c, b: IngestResult(
        str(d / "source.wav"), "Refreshed", 60.0, url, creator_chapters=[{"start": 0.0, "end": 30.0, "title": "x"}]))
    out = tmp_path / "runs" / "8jPQjjsBbIc"
    r = ing.ingest("https://youtu.be/8jPQjjsBbIc", out)
    assert (out / "source.wav").read_bytes() == b"audio" and r.title == "Refreshed" and r.schema == ing.META_SCHEMA


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
def test_duration_comes_from_the_audio_when_the_site_reports_none(tmp_path, monkeypatch):
    """Regression: non-YouTube sites often give no duration; the run must not be rejected as 'empty'."""
    out = tmp_path / "run"; out.mkdir()
    _tone(out / "download.wav", seconds=3.0)
    monkeypatch.setattr(ing, "_ytdlp_run", lambda *a, **k: {"id": "x", "title": "T",
                                                             "requested_downloads": [{"filepath": str(out / "download.wav")}]})
    r = ing.ingest("https://example.com/a.mp3", out)
    assert abs(r.duration - 3.0) < 0.1 and not (out / "download.wav").exists()
