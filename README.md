# Podcast / Video Summarizer with Auto-Chaptering

Give it any YouTube link (or an audio/video file). It transcribes the speech, **detects chapters from the
content itself** - whether or not the video has chapters - titles them, summarizes every chapter, and writes an overall TL;DR.
When the video happens to have creator chapters or manual captions, they are used **only to score the output**, never to
produce it.

```
link / file -> [1 Ingest] -> [2 Transcribe] -> [3 Clean] -> [4 Segment] -> [5 Summarize] -> [6 Report + Evaluate]
               yt-dlp+ffmpeg  faster-whisper    filter,       A: embeddings   hierarchical      result.json, report.md,
               16 kHz WAV     word timestamps   sentences     B: LLM prompt   LLM summaries     youtube_chapters.txt, metrics
```

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python -m podsum doctor                                    # checks everything before you start
python -m podsum "https://www.youtube.com/watch?v=XXXXXXXXXXX"
python -m podsum lecture.mp4 --method llm                  # local files work the same way
```

The first run downloads the Whisper and embedding models (a few hundred MB). YouTube links usually also need
[Deno](https://deno.com) (`winget install DenoLand.Deno`) and sometimes cookies (see below).

Default is `--method both`: it runs the two chaptering approaches on the *same* transcript so you can compare them.
Use `--method semantic` (no LLM calls for chaptering) or `--method llm` to run one.

## What you get (`data/runs/<video id or file name>/`)

| File | Contents |
|---|---|
| `transcript.json` | raw Whisper output: segments, word timestamps, confidence signals (never modified) |
| `sentences.json` | cleaned sentence-level transcript |
| `<method>/result.json` | TL;DR + chapters (start, end, title, summary) |
| `<method>/report.md` | readable report; chapter timestamps are clickable YouTube links |
| `<method>/youtube_chapters.txt` | `00:00 Title` lines you can paste into a video description |
| `semantic/similarity_curve.png` | the similarity curve with detected (and creator) chapter lines |
| `evaluation.json` | metrics, when ground truth exists |
| `manifest.json` | stage signatures and timings |

Re-running the same link reuses every stage whose inputs and settings are unchanged (changing method, chapter
length or LLM model never re-transcribes). `--force` starts over. Ctrl+C keeps completed stages.

## How chapters are detected without existing chapters

**A - semantic.** Every sentence is embedded (all-MiniLM-L6-v2). At each position the average embedding of the sentences
just before is compared with the average of those just after; a topic change is a dip in that similarity curve. Dips are
scored by TextTiling depth, and the deepest ones are kept subject to a minimum chapter length (90 s) and a maximum chapter
count (duration / 300 s). One knob: `PODSUM_SEMANTIC_SENSITIVITY`.

**B - LLM.** The timestamped transcript is sent in overlapping excerpts; the LLM proposes chapter starts (timestamps it must
copy from the transcript) and titles; a second short call consolidates duplicates and unifies granularity. Code validates every
timestamp, snaps it to a real sentence, enforces the minimum length, and falls back gracefully if the LLM misbehaves.

Both produce the same chapter structure; chapters cover the whole transcript. Summaries are hierarchical (chunks -> chapter -> TL;DR),
run in parallel, and one failed LLM call degrades a single chapter to an extractive summary instead of losing the run.

## Evaluation (automatic when the source has ground truth)

* **Chapter boundaries** vs the creator's chapters: precision / recall / F1 at 15, 30 and 60 s tolerance, with the trivial 0:00 start
  excluded from both sides; plus Pk and WindowDiff (standard segmentation error rates, lower is better).
* **Summaries**: ROUGE against the matching creator chapter titles (coarse - titles are short).
* **Which method to use**: when both methods ran on a video with creator chapters, the run also prints a one-line
  recommendation ("Recommended method for this video: ...") based on mean boundary F1, saved as `recommended_method`
  in `evaluation.json`. Its matching output is also copied to `<run>/best/`. This is a per-video signal, not a general
  verdict - check it across a few videos before treating one method as reliably better than the other.
* **Transcription**: WER against the video's manual captions (approximate: captions are edited, not verbatim), or against your own
  hand-corrected text: `--reference-text ref.txt --ref-start 0 --ref-end 180`.

If a video has no creator chapters/captions, evaluation says so and is skipped - pick a video that has chapters when you want numbers.

## Settings

Everything has a CLI flag or an environment variable (`python -m podsum --help`; defaults in `podsum/config.py`).

| Variable | Default | |
|---|---|---|
| `PODSUM_WHISPER_MODEL` | `small.en` | `medium.en` / `distil-large-v3` are more accurate but slower; drop `.en` for other languages |
| `PODSUM_WHISPER_DEVICE` | `cpu` | `cuda` falls back to CPU automatically if its libraries are missing |
| `PODSUM_LLM_MODEL` | `claude-sonnet-5` | cheaper: `claude-haiku-4-5-20251001` |
| `PODSUM_LLM_PROVIDER` / `PODSUM_LLM_BASE_URL` | `anthropic` | `openai` for any OpenAI-compatible endpoint (see "Free LLM option") |
| `PODSUM_LLM_API_KEY` | none | key for any provider (overrides `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) |
| `PODSUM_LLM_WORKERS` / `PODSUM_LLM_MIN_INTERVAL_S` | `4` / `0` | use `1` / `7` on rate-limited free tiers |
| `PODSUM_MIN_CHAPTER_S` / `PODSUM_TARGET_CHAPTER_S` | `90` / `300` | chapter length limits |
| `PODSUM_MAX_MINUTES` | `240` | refuses longer media |
| `PODSUM_COOKIES_FILE` | none | default cookies file for YouTube |

## Free LLM option (no paid credit)

The Anthropic API needs paid credit. The pipeline also works with any OpenAI-compatible endpoint; Google's Gemini API has a free
tier (Flash models, rate-limited, no card; free-tier prompts may be used by Google to improve its products). Get a key in Google
AI Studio, then in PowerShell (a fresh window, so old variables are gone):

```powershell
$env:PODSUM_LLM_PROVIDER = "openai"
$env:PODSUM_LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
$env:PODSUM_LLM_MODEL = "gemini-3.8-flash"        # pick a Flash/Flash-Lite model that AI Studio lists as free for your key
$env:PODSUM_LLM_API_KEY = Read-Host "Paste your Gemini API key"
$env:PODSUM_LLM_WORKERS = "1"; $env:PODSUM_LLM_MIN_INTERVAL_S = "7"
python -m podsum doctor
```

Per-minute rate limits (429) are waited out and retried automatically. A **daily** quota being used up (Gemini's free
tier gives a separate small quota PER MODEL PER DAY) is different: it is not fixed by waiting a few seconds, so it is
never retried - the run fails that stage immediately with a clear message, and (since the cap is *per model*) switching
to a different free model name has its own separate quota:

```powershell
$env:PODSUM_LLM_MODEL = "<another free model name from AI Studio>"
```

A local Ollama server also works (`PODSUM_LLM_BASE_URL=http://localhost:11434/v1`), but small local models write
weaker titles and summaries.

### If the quota runs out mid-run

A failure in one chaptering method (segmentation or summarization) is isolated to that method - the other method's
chapters, summaries and report are still produced, with a clear warning recorded instead of losing the whole run.
Finished stages stay cached, so re-running the exact same command later only retries what is still missing.

## Cookies (YouTube "confirm you're not a bot")

Export cookies for youtube.com from a logged-in browser (a "cookies.txt" extension, ideally from a private window), then pass
`--cookies youtube_cookies.txt` or set `PODSUM_COOKIES_FILE`. If your export is a JSON file, filter it down to YouTube only:
`python tools\cookies_json_to_txt.py cookies.json youtube_cookies.txt`. Cookie files are secrets - they are in `.gitignore`; never share them.

## Known limits

Thresholds (junk filtering, chapter sensitivity) are sensible defaults, not tuned on a large set - validate on several videos before
claiming generality. Summaries are LLM-written and not fact-checked against the transcript. The default Whisper/embedding
models are English; other languages need a multilingual Whisper model and a multilingual embedding model.

## Tests

`python -m pytest -q` - runs offline with deterministic fakes for Whisper, embeddings and the LLM. Real downloads, Whisper and the
LLM are exercised only when you run the pipeline.

## Layout

`podsum/` - `data/ingest.py` `transcribe.py` `clean.py` `segment/{semantic,llm_chaptering,postprocess}.py` `summarize.py`
`evaluate/{segmentation,wer,auto,compare}.py` `pipeline.py` `report.py` `cli.py` `doctor.py`; each stage module also runs standalone
(`python -m podsum.clean in.json out.json`). `app.py` + `web/` is the optional web UI (deferred).