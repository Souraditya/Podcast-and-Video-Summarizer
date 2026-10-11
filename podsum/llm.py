"""
Provider-agnostic LLM client used by chaptering approach B and by the summarizer.
Supports Anthropic directly or any OpenAI-compatible endpoint (OpenAI, Google Gemini, Groq, Ollama...).

Keys: PODSUM_LLM_API_KEY (any provider) or the provider's own variable (ANTHROPIC_API_KEY / OPENAI_API_KEY).
Never hardcode a key.

Free-tier friendly: PODSUM_LLM_MIN_INTERVAL_S paces calls; 429 responses are waited out and retried; and if a
"thinking" model spends its whole token budget before answering, the request is retried with a bigger budget.
"""
from __future__ import annotations

import functools
import os
import threading
import time

from podsum import config


class LLMConfigError(RuntimeError):
    pass


def _api_key(provider: str) -> str | None:
    return config.LLM_API_KEY or os.getenv("ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY")


def _is_local(url: str | None) -> bool:
    return bool(url) and any(h in url for h in ("localhost", "127.0.0.1"))


def check_ready(provider: str | None = None) -> None:
    """Fail fast (before minutes of transcription) if the LLM can't be called."""
    provider = provider or config.LLM_PROVIDER
    if provider not in ("anthropic", "openai"):
        raise LLMConfigError(f"unknown PODSUM_LLM_PROVIDER {provider!r} (use 'anthropic' or 'openai')")
    if _api_key(provider) or (provider == "openai" and _is_local(config.LLM_BASE_URL)):
        return
    var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    raise LLMConfigError(f"no API key found. In PowerShell:  $env:PODSUM_LLM_API_KEY = Read-Host \"Paste your API key\"  "
                         f"(or set {var})")


def _is_daily_quota_exhausted(e: Exception) -> bool:
    """Free tiers (e.g. Gemini) give a separate small quota PER MODEL PER DAY. That is not fixed by waiting a
    few seconds, so it must never be retried like an ordinary per-minute rate limit."""
    msg = str(e)
    return any(s in msg for s in ("PerDay", "per_day", "per-day")) or "requests per day" in msg.lower()


def _explain(e: Exception, provider: str, model: str) -> str:
    code = getattr(e, "status_code", None)
    if code == 401:
        return ("the API rejected your key (401). Check that it is the full key for THIS provider, with no quotes, spaces or '...', "
                "and set in THIS terminal window (PowerShell: $env:PODSUM_LLM_API_KEY = Read-Host \"Paste your API key\").")
    if code == 403:
        return f"the key is valid but not allowed to use model {model!r} (403). Try another model with PODSUM_LLM_MODEL."
    if code == 404:
        return f"model {model!r} was not found (404). Set PODSUM_LLM_MODEL to a model your account can use."
    if code == 429 and _is_daily_quota_exhausted(e):
        return (f"today's free-tier quota for model {model!r} is used up (resets after ~24h, counted separately per model). "
                f"Either wait, or switch to a different free model: $env:PODSUM_LLM_MODEL = \"<another free model name>\".")
    if code == 429:
        return ("rate limit reached (429). On a free tier set PODSUM_LLM_MIN_INTERVAL_S=7 and PODSUM_LLM_WORKERS=1, "
                "then wait a minute and re-run.")
    if any(w in type(e).__name__ for w in ("Connection", "Timeout")):
        cause = e.__cause__ or e.__context__
        detail = f" Underlying cause: {type(cause).__name__}: {str(cause)[:200]}." if cause else ""
        target = f" ({config.LLM_BASE_URL})" if config.LLM_BASE_URL else ""
        return (f"could not reach the LLM API{target} ({type(e).__name__}).{detail} Check your internet connection, VPN or proxy, "
                f"and that PODSUM_LLM_PROVIDER / PODSUM_LLM_BASE_URL are what you intend (Get-ChildItem Env:PODSUM*).")
    msg = str(e)
    if "credit balance is too low" in msg.lower():
        return "the account has no API credit. Add credit in the provider's billing page, or use a free provider (see README)."
    return f"{type(e).__name__}: {msg[:300]}"


def short_error(e: Exception) -> str:
    """One clean line for warnings/reports - never a raw multi-hundred-character API error blob."""
    if isinstance(e, LLMConfigError):
        return str(e)
    code = getattr(e, "status_code", None)
    if code == 429:
        return ("daily free-tier quota used up for this model" if _is_daily_quota_exhausted(e) else "rate-limited (429)")
    if code:
        return f"HTTP {code} from the LLM API"
    first_line = str(e).splitlines()[0] if str(e) else ""
    return f"{type(e).__name__}: {first_line[:160]}"


def validate(provider: str | None = None, model: str | None = None) -> None:
    """One tiny real request, so a wrong key, model name or provider fails in the first second instead of minutes into a run.
    Call this only right before expensive work (e.g. transcription) - it costs one real LLM call, which matters on a
    tight free-tier daily quota. check_ready() (cheap, no network) is enough for every other call site."""
    provider, model = provider or config.LLM_PROVIDER, model or config.LLM_MODEL
    try:
        complete("Reply with the single word: ok", max_tokens=64, provider=provider, model=model, grow_on_truncation=False)
    except Exception as e:
        raise LLMConfigError(_explain(e, provider, model)) from e


# ---------------------------------------------------------------- pacing and rate-limit handling
_pace_lock = threading.Lock()
_last_call = [0.0]


def _pace() -> None:
    gap = config.LLM_MIN_INTERVAL_S
    if gap <= 0:
        return
    with _pace_lock:
        wait = _last_call[0] + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()


def _retry_after(e: Exception) -> float | None:
    try:
        return float(e.response.headers.get("retry-after"))
    except Exception:
        return None


def _with_rate_limit_retry(fn, attempts: int = 5):
    for i in range(attempts):
        _pace()
        try:
            return fn()
        except Exception as e:
            if getattr(e, "status_code", None) != 429 or i == attempts - 1:
                raise
            if _is_daily_quota_exhausted(e) or "insufficient_quota" in str(e).lower():
                raise                          # a short wait never fixes a daily cap or an empty balance; fail now
            wait = min(60.0, _retry_after(e) or 15.0 * (i + 1))
            print(f"      rate-limited by the LLM provider; waiting {wait:.0f}s (retry {i + 1}/{attempts - 1})", flush=True)
            time.sleep(wait)


def complete(prompt: str, system: str = "", max_tokens: int = 2000, provider: str | None = None,
             model: str | None = None, temperature: float = 0.0, grow_on_truncation: bool = True) -> str:
    """`temperature` and `grow_on_truncation` apply to OpenAI-compatible providers only."""
    provider = provider or config.LLM_PROVIDER
    model = model or config.LLM_MODEL
    if provider == "anthropic":
        return _with_rate_limit_retry(lambda: _complete_anthropic(prompt, system, max_tokens, model))
    if provider == "openai":
        return _with_rate_limit_retry(lambda: _complete_openai(prompt, system, max_tokens, model, temperature, grow_on_truncation))
    raise LLMConfigError(f"unknown LLM provider: {provider!r}")


# ---------------------------------------------------------------- Anthropic
@functools.lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic

    kw = {"max_retries": 5, "timeout": 180.0}      # SDK retries 5xx with backoff
    if config.LLM_API_KEY:
        kw["api_key"] = config.LLM_API_KEY
    return anthropic.Anthropic(**kw)


def _complete_anthropic(prompt: str, system: str, max_tokens: int, model: str) -> str:
    # No sampling parameters are sent: current Claude models/SDKs manage them, and passing `temperature`
    # is rejected by the SDK. Determinism comes from the strict JSON prompts and from caching results per run.
    kwargs = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}])
    if system:
        kwargs["system"] = system
    resp = _anthropic_client().messages.create(**kwargs)
    return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


# ---------------------------------------------------------------- OpenAI-compatible (OpenAI, Gemini, Groq, Ollama...)
@functools.lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    kw = {"max_retries": 5, "timeout": 180.0}
    key = _api_key("openai")
    if config.LLM_BASE_URL:
        kw["base_url"] = config.LLM_BASE_URL
        kw["api_key"] = key or "not-needed"
    elif key:
        kw["api_key"] = key
    return OpenAI(**kw)


def _complete_openai(prompt: str, system: str, max_tokens: int, model: str, temperature: float, grow: bool) -> str:
    import openai

    client = _openai_client()
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    kwargs = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    text = ""
    for _ in range(6):
        try:
            resp = client.chat.completions.create(**kwargs)
        except openai.BadRequestError as e:          # some models reject max_tokens / temperature: adapt and retry
            msg = str(e).lower()
            if "max_completion_tokens" in msg and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                continue
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature")
                continue
            raise
        choice = resp.choices[0]
        text = choice.message.content or ""
        key = "max_tokens" if "max_tokens" in kwargs else "max_completion_tokens"
        if choice.finish_reason == "length" and grow and kwargs[key] < 16000:
            kwargs[key] = min(16000, kwargs[key] * 3)   # "thinking" models can spend the whole budget before answering
            continue
        return text
    return text