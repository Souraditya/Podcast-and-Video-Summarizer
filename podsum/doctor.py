"""Pre-flight check: `python -m podsum doctor` tells you exactly what is missing before you spend 10 minutes on a run."""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

from podsum import config


def _check_import(mod: str, pip_name: str | None = None, required: bool = True) -> tuple[bool, str]:
    try:
        m = importlib.import_module(mod)
        return True, f"{mod} {getattr(m, '__version__', '')}".strip()
    except Exception as e:
        tag = "MISSING" if required else "optional, missing"
        return not required, f"{mod}: {tag} -> python -m pip install {pip_name or mod}  ({type(e).__name__})"


def main(argv=None) -> int:
    ok_all = True

    def line(ok: bool, text: str, warn: bool = False) -> None:
        nonlocal ok_all
        ok_all = ok_all and (ok or warn)
        print(f"  [{'ok' if ok else ('..' if warn else '!!')}] {text}")

    print("podsum doctor")
    line(sys.version_info >= (3, 9), f"python {sys.version.split()[0]} (need 3.9+)")
    for mod, pip, req in [("yt_dlp", "yt-dlp[default]", True), ("faster_whisper", "faster-whisper", True),
                          ("sentence_transformers", "sentence-transformers", True), ("numpy", None, True),
                          ("jiwer", None, True), ("rouge_score", "rouge-score", True),
                          ("matplotlib", None, False), ("imageio_ffmpeg", "imageio-ffmpeg", False)]:
        ok, msg = _check_import(mod, pip, req)
        line(ok, msg, warn=not req)

    provider = config.LLM_PROVIDER
    ok, msg = _check_import("anthropic" if provider == "anthropic" else "openai")
    line(ok, msg)
    from podsum import llm
    try:
        llm.check_ready()
        line(True, f"LLM key found; provider={provider}, model={config.LLM_MODEL}"
                   + (f", endpoint={config.LLM_BASE_URL}" if config.LLM_BASE_URL else ""))
    except llm.LLMConfigError as e:
        line(False, str(e))
    if provider == "openai" and config.LLM_BASE_URL and not config.LLM_BASE_URL.startswith(("http://localhost", "http://127.")):
        line(False, "free tiers are rate-limited: set PODSUM_LLM_WORKERS=1 and PODSUM_LLM_MIN_INTERVAL_S=7 if you see 429 errors", warn=True)

    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ff = None
    line(bool(ff), f"ffmpeg: {ff}" if ff else "ffmpeg NOT found -> winget install Gyan.FFmpeg (then reopen the terminal) or pip install imageio-ffmpeg")
    line(bool(shutil.which("deno")), "deno found (yt-dlp needs a JS runtime for YouTube)" if shutil.which("deno")
         else "deno not found -> winget install DenoLand.Deno (needed for most YouTube links; local files work without it)", warn=True)
    if config.COOKIES_FILE:
        line(Path(config.COOKIES_FILE).is_file(), f"cookies file: {config.COOKIES_FILE}")
    data = Path(config.DATA_DIR)
    try:
        data.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(data).free / 1e9
        line(free > 2.0, f"data dir '{data}' writable, {free:.1f} GB free")
    except OSError as e:
        line(False, f"cannot write to data dir '{data}': {e}")
    print(f"\n  whisper model: {config.WHISPER_MODEL} on {config.WHISPER_DEVICE}; embeddings: {config.EMBED_MODEL}")
    print("  (models download on first use, so the very first run needs internet and a few hundred MB)")
    print("\nAll required checks passed." if ok_all else "\nFix the [!!] items above, then run doctor again.")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())