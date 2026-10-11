"""
Central defaults. Every value can be overridden with an environment variable
(no code edits) and most also with a CLI flag.
"""
import os

# huggingface_hub cannot create symlinks on Windows without Developer Mode; it then
# copies files instead, but prints a long warning. Silence it.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _get(name: str, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return cast(raw)


# ---- transcription (faster-whisper)
WHISPER_MODEL = _get("PODSUM_WHISPER_MODEL", "small.en")   # tiny.en/base.en/small.en/medium.en/distil-large-v3; drop ".en" for other languages
WHISPER_DEVICE = _get("PODSUM_WHISPER_DEVICE", "cpu")       # cpu | cuda | auto  (cuda falls back to cpu automatically if its libraries are missing)
WHISPER_COMPUTE_TYPE = _get("PODSUM_WHISPER_COMPUTE", None) # default: int8 on cpu, float16 on cuda
WHISPER_LANGUAGE = _get("PODSUM_LANGUAGE", None)            # None = auto-detect (".en" models are always English)

# ---- segmentation
EMBED_MODEL = _get("PODSUM_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MIN_CHAPTER_S = _get("PODSUM_MIN_CHAPTER_S", 90.0, float)        # no chapter shorter than this
TARGET_CHAPTER_S = _get("PODSUM_TARGET_CHAPTER_S", 300.0, float) # typical chapter length; sets the maximum chapter count
SEMANTIC_SENSITIVITY = _get("PODSUM_SEMANTIC_SENSITIVITY", 0.25, float)  # depth cut-off = mean + this * std; lower = more boundaries

# ---- LLM (chapter titles/boundaries for approach B, and all summaries)
LLM_PROVIDER = _get("PODSUM_LLM_PROVIDER", "anthropic")     # "anthropic" | "openai" (any OpenAI-compatible endpoint)
LLM_MODEL = _get("PODSUM_LLM_MODEL", "claude-sonnet-5")     # e.g. claude-haiku-4-5-20251001 for cheaper runs
LLM_BASE_URL = _get("PODSUM_LLM_BASE_URL", None)            # only for provider="openai" (Groq, Ollama, ...)
LLM_WORKERS = _get("PODSUM_LLM_WORKERS", 4, int)            # parallel summary calls (use 1 on free tiers)
LLM_API_KEY = _get("PODSUM_LLM_API_KEY", None)              # any provider's key; overrides ANTHROPIC_API_KEY / OPENAI_API_KEY
LLM_MIN_INTERVAL_S = _get("PODSUM_LLM_MIN_INTERVAL_S", 0.0, float)  # minimum gap between LLM calls; ~7 keeps free tiers under ~9 requests/min

# ---- data
DATA_DIR = _get("PODSUM_DATA_DIR", "data")
COOKIES_FILE = _get("PODSUM_COOKIES_FILE", None)            # Netscape cookies.txt for YouTube's bot check
MAX_DURATION_S = _get("PODSUM_MAX_MINUTES", 240, float) * 60  # refuse anything longer (guards against 10-hour streams)