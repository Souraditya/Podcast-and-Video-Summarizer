import json
import sys
import types

import numpy as np
import pytest

from podsum.evaluate.segmentation import boundary_prf, pk, window_diff
from podsum.jsonutil import extract_json
from podsum.schema import Chapter, Sentence
from podsum.segment import llm_chaptering as lc
from podsum.segment.postprocess import build_chapters, nearest_sentence
from podsum.segment.semantic import analyze, boundaries_to_chapters, depth_scores, select_boundaries
from podsum.summarize import generate_tldr, summarize_chapters
from tests.conftest import N_TOPICS, SENT_S, TOPIC_S, fake_embed, synthetic_transcript


def sentences(n=300, step=SENT_S):
    return [Sentence(idx=i, text=f"sentence {i}.", start=i * step, end=i * step + step - 0.5) for i in range(n)]


# ---------------------------------------------------------------- json
def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert extract_json('Sure! Here you go:\n{"title": "x", "summary": "y"}\nHope that helps.') == {"title": "x", "summary": "y"}
    with pytest.raises(ValueError):
        extract_json("no json at all")


# ---------------------------------------------------------------- postprocess
def test_build_chapters_covers_everything_and_starts_at_zero():
    s = sentences(100)
    chs = build_chapters(s, [(40, "B"), (70, "C")], 0)
    assert [(c.first_sentence, c.last_sentence) for c in chs] == [(0, 39), (40, 69), (70, 99)]
    assert chs[0].title == "B"          # no explicit first title: inherits the earliest one rather than staying blank
    assert chs[0].start == 0.0 and chs[-1].end == s[-1].end


def test_build_chapters_merges_short_chapters_keeping_the_longer_title():
    s = sentences(100)                   # 6 s per sentence
    chs = build_chapters(s, [(0, "Intro"), (45, "Blip"), (50, "Main")], min_chapter_s=90)
    assert [(c.first_sentence, c.last_sentence, c.title) for c in chs] == [(0, 49, "Intro"), (50, 99, "Main")]
    assert build_chapters(s, [], 0)[0].last_sentence == 99
    assert build_chapters([], [(0, "x")], 0) == []


def test_nearest_sentence():
    starts = [0.0, 10.0, 20.0]
    assert [nearest_sentence(starts, t) for t in (-5, 4, 6, 19, 99)] == [0, 0, 1, 2, 2]


# ---------------------------------------------------------------- approach A
def test_semantic_finds_the_topic_changes_and_respects_limits():
    s = sentences(300)
    a = analyze(s, embeddings=fake_embed(s))
    assert len(a.boundaries) == N_TOPICS - 1
    assert all(abs(b - t) <= 3 for b, t in zip(a.boundaries, (100, 200)))
    times = [s[b].start for b in a.boundaries]
    assert all(t >= 90 for t in times) and np.diff(times).min() >= 90


def test_semantic_returns_nothing_for_short_clips_and_flat_content():
    s = sentences(8)
    assert analyze(s, embeddings=np.ones((8, 4))).boundaries == []
    s = sentences(300)
    flat = np.tile(np.array([1.0, 0, 0, 0]), (300, 1))
    assert analyze(s, embeddings=flat).boundaries == []


def test_select_boundaries_caps_count_by_duration():
    depths = {i: 1.0 + i * 1e-3 for i in range(10, 290, 10)}
    times = [i * 6.0 for i in range(300)]
    picked = select_boundaries(depths, times, 0.0, 1800.0, min_chapter_s=90, target_chapter_s=600, sensitivity=-9)
    assert len(picked) == 2                     # 1800 s / 600 s target = 3 chapters = 2 boundaries
    assert depth_scores(np.array([1.0, 0.5, 1.0, 0.9, 1.0]))[1] == pytest.approx(1.0)


def test_boundaries_to_chapters():
    chs = boundaries_to_chapters(sentences(300), [100, 200])
    assert [(c.first_sentence, c.last_sentence) for c in chs] == [(0, 99), (100, 199), (200, 299)]


# ---------------------------------------------------------------- approach B
def _sents_from_transcript():
    from podsum.clean import clean_pipeline
    return clean_pipeline(synthetic_transcript())[0]


def test_llm_chaptering_two_pass_finds_topics(fake_llm):
    s = _sents_from_transcript()
    chs = lc.segment_llm(s, video_title="T", chars_per_chunk=2000)          # forces several chunks + consolidation
    assert fake_llm.calls > 2
    assert [c.title for c in chs] == ["Topic 1", "Topic 2", "Topic 3"]
    assert [round(c.start) for c in chs] == [0, 600, 1200]


def test_llm_chaptering_survives_garbage_replies(monkeypatch):
    monkeypatch.setattr(lc, "complete", lambda *a, **k: "I cannot do that")
    warnings = []
    chs = lc.segment_llm(_sents_from_transcript(), warn=warnings.append)
    assert len(chs) == 1 and chs[0].first_sentence == 0 and warnings          # single chapter, not a crash


def test_llm_chunking_has_overlap_and_no_duplicate_tail():
    s = sentences(50)
    chunks = lc._chunk_sentences(s, chars_per_chunk=100, overlap=3)
    assert chunks[0][-3:] == chunks[1][:3]
    assert chunks[-1][-1] is s[-1] and len({id(x) for c in chunks for x in c}) == 50


# ---------------------------------------------------------------- summarize
def test_summarize_parallel_titles_and_tldr(fake_llm):
    s = sentences(60)
    chs = [Chapter(0, 100, "", "", 0, 29), Chapter(100, 200, "Keep me", "", 30, 59)]
    out = summarize_chapters(chs, s, overwrite_titles=False, workers=2)
    assert out[0].title == "Generated Title" and out[1].title == "Keep me" and out[1].summary
    assert generate_tldr(out).startswith("An overall")
    assert summarize_chapters(chs, s, overwrite_titles=True)[1].title == "Generated Title"


def test_summarize_degrades_to_extractive_fallback_on_llm_failure(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("quota exceeded")
    monkeypatch.setattr("podsum.summarize.complete", boom)
    warnings = []
    out = summarize_chapters([Chapter(0, 100, "", "", 0, 9)], sentences(10), warn=warnings.append)
    assert out[0].title == "Chapter 1" and out[0].summary.startswith("sentence 0.") and "quota" in warnings[0]
    assert generate_tldr(out, warn=warnings.append).startswith("Chapters:")


def test_long_chapter_is_summarized_hierarchically(monkeypatch, fake_llm):
    monkeypatch.setattr("podsum.summarize.CHARS_PER_SUMMARY_CHUNK", 200)
    long = [Sentence(idx=i, text="word " * 20, start=i, end=i + 1) for i in range(40)]
    summarize_chapters([Chapter(0, 40, "", "", 0, 39)], long)
    assert fake_llm.calls > 5                    # many chunk summaries + the combine call


# ---------------------------------------------------------------- metrics
def ch(t):
    return Chapter(start=float(t), end=float(t) + 60)


def test_boundary_prf_can_exclude_the_free_first_match():
    gt, pred = [ch(0), ch(600), ch(1200)], [ch(0), ch(1500)]
    assert boundary_prf(pred, gt, 15)["f1"] == pytest.approx(0.4)           # inflated by the 0:00 match
    strict = boundary_prf(pred, gt, 15, exclude_first=True)
    assert strict["tp"] == 0 and strict["f1"] == 0.0
    perfect = boundary_prf([ch(0), ch(602), ch(1195)], gt, 15, exclude_first=True)
    assert perfect["f1"] == 1.0 and perfect["mean_offset_s"] == pytest.approx(3.5)


def test_pk_and_windowdiff_zero_for_identical_and_high_for_wrong():
    gt = [ch(0), ch(600), ch(1200)]
    assert pk(gt, gt, 1800) == 0.0 and window_diff(gt, gt, 1800) == 0.0
    wrong = [ch(0), ch(300), ch(900)]
    assert pk(wrong, gt, 1800) > 0.2 and window_diff(wrong, gt, 1800) > 0.2
    assert pk([ch(0)], gt, 1800) > 0.2


# ---------------------------------------------------------------- transcribe wrapper (fake faster-whisper)
def _install_fake_whisper(monkeypatch, fail_on_cuda=False):
    class W:  # word
        def __init__(self, word, s, e): self.word, self.start, self.end, self.probability = word, s, e, 0.9

    class Seg:
        def __init__(self, i): self.start, self.end, self.text = i * 2.0, i * 2.0 + 2.0, f" Segment {i}. "; \
            self.avg_logprob, self.no_speech_prob, self.compression_ratio = -0.2, 0.01, 1.1; \
            self.words = [W(" Segment", i * 2.0, i * 2.0 + 1), W(f"{i}.", i * 2.0 + 1, i * 2.0 + 2)]

    class Model:
        def __init__(self, size, device, compute_type):
            self.device, self.ctype = device, compute_type
            Model.last = self
            if fail_on_cuda and device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found")

        def transcribe(self, path, **kw):
            Model.kwargs = kw
            return iter([Seg(0), Seg(1), Seg(2)]), types.SimpleNamespace(duration=6.0, language="en")

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=Model))
    return Model


def test_transcribe_maps_fields_and_uses_safe_decoding_options(monkeypatch):
    from podsum.transcribe import transcribe
    Model = _install_fake_whisper(monkeypatch)
    t = transcribe("x.wav", model_size="small.en", device="cpu", progress=False)
    assert [s.text for s in t.segments] == ["Segment 0.", "Segment 1.", "Segment 2."]
    assert t.segments[0].words[0].text == "Segment" and t.duration == 6.0
    assert Model.last.ctype == "int8"
    assert Model.kwargs["condition_on_previous_text"] is False and Model.kwargs["word_timestamps"] and Model.kwargs["language"] == "en"


def test_transcribe_falls_back_to_cpu_when_cuda_is_broken(monkeypatch):
    from podsum.transcribe import transcribe
    Model = _install_fake_whisper(monkeypatch, fail_on_cuda=True)
    t = transcribe("x.wav", model_size="small.en", device="cuda", progress=False)
    assert len(t.segments) == 3 and Model.last.device == "cpu"


# ---------------------------------------------------------------- LLM client contract
def test_anthropic_call_never_sends_sampling_params_the_sdk_rejects(monkeypatch):
    from podsum import llm
    captured = {}

    class Msgs:
        def create(self, *, max_tokens, messages, model, system=None):        # mirrors the real SDK: no temperature
            captured.update(model=model, system=system, max_tokens=max_tokens)
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="thinking", text="x"),
                                                  types.SimpleNamespace(type="text", text="hello")])

    monkeypatch.setattr(llm, "_anthropic_client", lambda: types.SimpleNamespace(messages=Msgs()))
    assert llm.complete("hi", system="sys", provider="anthropic", model="m", temperature=0.0) == "hello"
    assert captured == {"model": "m", "system": "sys", "max_tokens": 2000}


def test_check_ready_reports_missing_key(monkeypatch):
    from podsum import llm
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(llm.LLMConfigError, match="ANTHROPIC_API_KEY"):
        llm.check_ready("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    llm.check_ready("anthropic")


def test_validate_turns_a_401_into_an_actionable_message(monkeypatch):
    from podsum import llm

    class Auth(Exception):
        status_code = 401

    def boom(*a, **k):
        raise Auth("invalid x-api-key")
    monkeypatch.setattr(llm, "complete", boom)
    with pytest.raises(llm.LLMConfigError, match="rejected your key.*PODSUM_LLM_API_KEY"):
        llm.validate("anthropic", "m")
    monkeypatch.setattr(llm, "complete", lambda *a, **k: "ok")
    llm.validate("anthropic", "m")


def test_connection_errors_show_the_underlying_cause(monkeypatch):
    from podsum import llm

    class APIConnectionError(Exception):
        pass

    def boom(*a, **k):
        try:
            raise OSError("getaddrinfo failed")
        except OSError as inner:
            raise APIConnectionError("Connection error.") from inner
    monkeypatch.setattr(llm, "complete", boom)
    with pytest.raises(llm.LLMConfigError, match="getaddrinfo failed"):
        llm.validate("anthropic", "m")


# ---------------------------------------------------------------- free-tier / OpenAI-compatible providers
def _fake_openai(responses):
    """responses: list of (content, finish_reason) or Exception; records the kwargs of every call."""
    calls = []

    class Completions:
        def create(self, **kw):
            calls.append(kw)
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            content, finish = r
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content=content), finish_reason=finish)])

    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions())), calls


def test_generic_key_and_local_endpoint_pass_check_ready(monkeypatch):
    from podsum import config, llm
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config, "LLM_API_KEY", None)
    with pytest.raises(llm.LLMConfigError, match="PODSUM_LLM_API_KEY"):
        llm.check_ready("openai")
    monkeypatch.setattr(config, "LLM_API_KEY", "gemini-key")
    llm.check_ready("openai")
    monkeypatch.setattr(config, "LLM_API_KEY", None)
    monkeypatch.setattr(config, "LLM_BASE_URL", "http://localhost:11434/v1")
    llm.check_ready("openai")


def test_openai_client_uses_base_url_and_generic_key(monkeypatch):
    from podsum import config, llm
    monkeypatch.setattr(config, "LLM_API_KEY", "gem-key")
    monkeypatch.setattr(config, "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    llm._openai_client.cache_clear()
    try:
        c = llm._openai_client()
        assert "generativelanguage.googleapis.com" in str(c.base_url) and c.api_key == "gem-key"
    finally:
        llm._openai_client.cache_clear()


def test_thinking_model_that_spends_its_whole_budget_is_retried_with_more_tokens(monkeypatch):
    from podsum import llm
    client, calls = _fake_openai([("", "length"), ("the answer", "stop")])
    monkeypatch.setattr(llm, "_openai_client", lambda: client)
    assert llm.complete("q", provider="openai", model="m", max_tokens=500) == "the answer"
    assert [c["max_tokens"] for c in calls] == [500, 1500]
    client, calls = _fake_openai([("", "length")])
    monkeypatch.setattr(llm, "_openai_client", lambda: client)
    assert llm.complete("q", provider="openai", model="m", max_tokens=64, grow_on_truncation=False) == "" and len(calls) == 1


def test_429_is_waited_out_using_retry_after_and_quota_exhaustion_is_not_retried(monkeypatch):
    from podsum import llm

    class RateLimited(Exception):
        status_code = 429
        response = types.SimpleNamespace(headers={"retry-after": "2"})

    slept = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    client, calls = _fake_openai([RateLimited("slow down"), ("ok", "stop")])
    monkeypatch.setattr(llm, "_openai_client", lambda: client)
    assert llm.complete("q", provider="openai", model="m") == "ok" and slept == [2.0] and len(calls) == 2

    class NoQuota(Exception):
        status_code = 429
    client, calls = _fake_openai([NoQuota("insufficient_quota: billing")])
    monkeypatch.setattr(llm, "_openai_client", lambda: client)
    with pytest.raises(NoQuota):
        llm.complete("q", provider="openai", model="m")
    assert len(calls) == 1


def test_pacing_keeps_a_minimum_gap_between_calls(monkeypatch):
    from podsum import config, llm
    clock, slept = [100.0], []
    monkeypatch.setattr(config, "LLM_MIN_INTERVAL_S", 7.0)
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(llm.time, "sleep", lambda s: (slept.append(s), clock.__setitem__(0, clock[0] + s)))
    llm._last_call[0] = 0.0
    llm._pace(); clock[0] += 2.0; llm._pace()
    assert slept == [pytest.approx(5.0)]


# ---------------------------------------------------------------- daily-quota detection & short_error
def test_daily_quota_is_not_retried_like_a_per_minute_rate_limit(monkeypatch):
    from podsum import llm

    class DailyQuota(Exception):
        status_code = 429
        response = types.SimpleNamespace(headers={"retry-after": "47"})

    slept = []
    monkeypatch.setattr(llm.time, "sleep", slept.append)
    client, calls = _fake_openai([DailyQuota("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue: 20")])
    monkeypatch.setattr(llm, "_openai_client", lambda: client)
    with pytest.raises(DailyQuota):
        llm.complete("q", provider="openai", model="m")
    assert slept == [] and len(calls) == 1          # no wasted wait-and-retry on a cap that won't reset in seconds


def test_short_error_never_dumps_the_raw_api_response():
    from podsum import llm

    class DailyQuota(Exception):
        status_code = 429
    raw = DailyQuota("RESOURCE_EXHAUSTED: {'error': {'code': 429, " + "x" * 400 + "}}")
    assert llm.short_error(raw) == "rate-limited (429)"

    class Daily2(Exception):
        status_code = 429
    raw2 = Daily2("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier " + "x" * 400)
    out = llm.short_error(raw2)
    assert out == "daily free-tier quota used up for this model" and len(out) < 100

    assert llm.short_error(llm.LLMConfigError("no API key found. ...")) == "no API key found. ..."
    assert llm.short_error(RuntimeError("boom")).startswith("RuntimeError: boom")


# ---------------------------------------------------------------- choosing between the two methods
def _ch_eval(f1_semantic, f1_llm):
    row = lambda f1: {"n_predicted_chapters": 3, "boundary": {t: {"f1": f1} for t in ("15", "30", "60")},
                      "pk": 0.1, "window_diff": 0.1}
    return {"n_ground_truth_chapters": 3, "methods": {"semantic": row(f1_semantic), "llm": row(f1_llm)}}


def test_recommend_method_picks_the_higher_scoring_one():
    from podsum.evaluate.auto import format_recommendation, recommend_method
    rec = recommend_method(_ch_eval(f1_semantic=0.40, f1_llm=0.80))
    assert rec["method"] == "llm" and rec["scores"] == {"semantic": 0.4, "llm": 0.8}
    assert "higher mean boundary F1" in rec["reason"]
    assert "Recommended method for this video: llm" in format_recommendation(rec)


def test_recommend_method_handles_ties_and_single_method():
    from podsum.evaluate.auto import recommend_method
    tie = recommend_method(_ch_eval(0.5, 0.5))
    assert tie["method"] == "semantic" and "tied" in tie["reason"]
    assert recommend_method({"methods": {"semantic": {}}}) is None   # only one method evaluated: nothing to pick
    assert recommend_method(None) is None