# TTB Label Verification

Proof-of-concept web app for checking alcohol label images against submitted application data. The app extracts visible label text with a vision model, compares each extracted field with deterministic rules, and returns a clear `APPROVED` or `NEEDS REVIEW` result.

## Live Demo

Deployed URL: https://ttb-label-verification-production-6496.up.railway.app


## What This Does

- Accepts a label image plus seven application fields.
- Extracts label text into structured JSON.
- Compares every field and shows per-field `PASS` or `FAIL`.
- Shows expected-vs-found values on failures.
- Surfaces the extracted government warning text and heading-weight check for review.
- Supports up to 300 uploaded labels, sent to the backend in bounded groups with combined results.
- Returns readable errors for bad input, timeouts, provider limits, and missing configuration.

## Demo Checklist

- Single-label check: upload one image, fill the seven fields, and click `Check Label`.
- Batch check: open `Batch Labels`, add label images one at a time or all together, fill each row, and click `Check All Labels`.
- Exact-warning behavior: capitalization, wording, punctuation, and a bold `GOVERNMENT WARNING:` heading are required; line-break layout is normalized because label OCR often wraps warning text.
- Error behavior: submit without an image or with a wrong file type to see a plain-English 4xx response.

## Tech Stack

- Python 3.12
- FastAPI
- Pydantic
- Plain HTML, CSS, and JavaScript
- OpenAI vision model through the OpenAI Responses API
- Railway deployment
- Pytest test suite

## Architecture at a Glance

`main.py` assembles FastAPI and static assets. Route handlers validate uploads and delegate concurrent batch work; `comparison.py` contains deterministic matching. Vision interfaces remain in `vision.py`, while provider errors and HEIC/JPEG preprocessing live in focused modules. The frontend fetches runtime limits from `/health`, chunks large upload sets into bounded API requests, and warms `/health/deep` while the user fills the form; no database or persistent file storage is used.

## Environment Variables

Real secrets must live in local environment variables or Railway service variables. Do not commit `.env`.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `APP_ENV` | No | `local` | Environment label returned by `/health`. |
| `LOG_LEVEL` | No | `INFO` | Backend logging level. |
| `OPENAI_API_KEY` | Yes | none | Authenticates extraction and deep health checks. |
| `OPENAI_MODEL` | No | `gpt-5.4-nano` | Pins the deployed vision model. |
| `OPENAI_TIMEOUT_SECONDS` | No | `4.5` | Bounds provider latency for the five-second target. |
| `IMAGE_MAX_LONG_SIDE` | No | `768` | Maximum image dimension sent to the model. |
| `IMAGE_JPEG_QUALITY` | No | `70` | JPEG quality after preprocessing. |
| `BATCH_CONCURRENCY` | No | `3` | Maximum concurrent batch model calls. |
| `BATCH_MAX_LABELS` | No | `10` | Per-request batch cap returned by `/health` and enforced by the API. |
| `BATCH_UPLOAD_MAX_LABELS` | No | `300` | Maximum labels the browser accepts and divides into bounded requests. |

`OPENAI_API_KEY` is required for real vision extraction. `.env.example` lists variable names only.

Configured deployed model: `gpt-5.4-nano` (replaced `gpt-4o-mini` on 2026-07-13).

Model verification: on 2026-07-13, [OpenAI's model documentation](https://developers.openai.com/api/docs/models/gpt-5.4-nano) listed `gpt-5.4-nano` as supporting image input, the Responses API, and structured outputs. The production deploy also passed `/health/deep` against OpenAI's Models API with that exact model name.

Model-name smoke check:

```bash
curl https://YOUR-DEPLOYED-URL/health/deep
```

That endpoint calls OpenAI's Models API for the configured `OPENAI_MODEL`. It should return HTTP `200` before a deploy is considered healthy.

## Run Locally

Install dependencies with Python 3.12:

```bash
uv sync --python python3.12
```

Run the app:

```bash
export OPENAI_MODEL=gpt-5.4-nano
export OPENAI_TIMEOUT_SECONDS=4.5
export IMAGE_MAX_LONG_SIDE=768
export IMAGE_JPEG_QUALITY=70
export BATCH_CONCURRENCY=3
export BATCH_MAX_LABELS=10
export BATCH_UPLOAD_MAX_LABELS=300
uv run uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Set `OPENAI_API_KEY` in your shell or local `.env` before running. Do not commit that value.

Open:

```text
http://127.0.0.1:8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/deep
```

## Run Tests

```bash
uv run pytest
node --check frontend/app.js
```

The current test suite covers the comparison engine, vision service parsing and error mapping, single-label endpoint, batch endpoint, frontend static behavior, and health route.

## Live Smoke Check

Run one real image through the deployed frontend API, comparison engine, and configured model. The command exits non-zero for HTTP errors or a verdict other than `APPROVED`:

```bash
uv run python scripts/live_smoke.py --url https://ttb-label-verification-production-6496.up.railway.app
```

## API Examples

Create the local sample image if needed:

```bash
uv run python scripts/run_vision_sample.py --create-sample-only
```

Single-label request:

```bash
curl -sS -X POST http://127.0.0.1:8000/verify \
  -F "image=@samples/sample_label.jpg;type=image/jpeg" \
  -F 'application_data={"brand_name":"SUNSET RIDGE","class_type":"CABERNET SAUVIGNON","producer":"North Valley Estate Winery LLC","country_of_origin":"USA","abv":"45%","net_contents":"750 mL","government_warning":"GOVERNMENT WARNING: EXACTLY AS PRINTED"}'
```

Batch requests use `images`, `image_ids`, and `application_data`. Each `image_ids` value maps an uploaded image to one application-data row.

## Approach

The backend is stateless. It receives uploaded image bytes and application data, validates the request, asks the vision service for a structured `ExtractedLabel`, compares the extracted label against the submitted application data, and returns a `VerificationResult`.

The vision service:

- downscales and re-encodes images before the model call;
- uses OpenAI structured JSON output instead of ad hoc string parsing;
- returns partial data when fields are unreadable;
- maps timeouts, rate limits, malformed output, and missing configuration into readable API errors.

The comparison engine is pure Python over typed Pydantic models, so it is unit-testable without model calls.

## Comparison Rules

- Brand name, product type, and the complete producer/bottler name-and-address statement: fuzzy token-sort matching with threshold `90`.
- Country of origin: normalized country synonyms, including `USA`, `United States`, and common producing-country variants.
- Alcohol content: numeric ABV normalization, so `45%` can match `45% Alc./Vol. (90 Proof)` or `90 Proof`.
- Net contents: unit normalization, so `750 mL` can match `750ml`, and `355 mL` can match `12 FL OZ`.
- Government warning: strict wording, capitalization, and punctuation, plus a visually bold `GOVERNMENT WARNING:` heading; whitespace layout is normalized to avoid false failures from OCR line wrapping. Unreadable heading weight requires review.

Any field failure makes the overall verdict `NEEDS_REVIEW`.

## Batch Processing

The browser accepts up to `BATCH_UPLOAD_MAX_LABELS` labels and sends them sequentially in groups of `BATCH_MAX_LABELS`. Each backend group processes labels concurrently with bounded provider concurrency. One bad label does not fail its group; the browser combines all item results into one approved, needs-review, error, and total summary. Large batches are throughput workflows and are not subject to the five-second single-label target.

## Security

- API keys are read from environment variables only.
- `.env` and `.env.*` files are ignored by Git.
- `.env.example` contains placeholders only.
- The production key is configured in Railway, not in the repository.
- Error responses do not expose stack traces.

Secret-handling audit: tracked source, examples, tests, and documentation contain no production API key; `.gitignore` excludes `.env` and `.env.*` except the placeholder-only `.env.example`.

## Performance Notes

Single-label verification has a hard target of under `5` seconds on the deployed URL. The backend provider timeout defaults to `4.5` seconds. The frontend shows a delayed status message after `5` seconds and keeps the request open for up to `8` seconds to accommodate hosting cold-start and network overhead.

Railway now uses `/health/deep` for deployment readiness, and the browser calls that provider check in the background on page load. This removes avoidable first-use setup from the submit path, but a free-tier host or upstream provider can still experience infrastructure delays; performance must be remeasured after each deployment.

Measure deployed single-label latency after every deploy:

```bash
uv run python scripts/measure_single_label_latency.py \
  --url https://YOUR-DEPLOYED-URL \
  --samples 20
```

Current OpenAI deployment measurement:

| Date | URL | Model | Samples | Successful | Timeouts | p50 | p95 | Script |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-07-13 | `https://ttb-label-verification-production-6496.up.railway.app` | `gpt-5.4-nano` | 20 | 20 | 0 | 1,652 ms | 2,233 ms | `scripts/measure_single_label_latency.py` |
| 2026-07-13 | `https://ttb-label-verification-production-6496.up.railway.app` | `gpt-5.4-nano` | 20 | 19 | 1 | 1,883 ms | 3,105 ms | `scripts/measure_single_label_latency.py` |
| 2026-07-12 | `https://ttb-label-verification-production-6496.up.railway.app` | `gpt-4o-mini` | 20 | 20 | 0 | 2,839 ms | 3,195 ms | `scripts/measure_single_label_latency.py` |

The latest Nano run completed all 20 requests and met the under-5-second target. Across both Nano runs, 39 of 40 requests succeeded; one request reached the 4.5-second provider timeout. The older `gpt-4o-mini` row is retained as a historical comparison rather than an active configuration.

Batch mode can take longer because several labels are processed together. Concurrency is bounded with `BATCH_CONCURRENCY`.

## Assumptions

- One image represents one label record.
- The app is stateless and does not store uploaded images or results.
- Application data is supplied as seven required fields.
- Reviewers will use a configured `OPENAI_API_KEY` for real extraction.
- Railway provides production environment variables.

## Limitations

- Vision/OCR can misread small or blurry text and can confuse a prominent fanciful name with a smaller class/type designation.
- A tested 503 x 373 composite image containing front label, back label, and bottle views produced tiny compliance text. The model read `5.3%` instead of the visible `13.5%` and classified `STORMCHASER` instead of `Chardonnay`. The application correctly surfaced these as expected-versus-found failures rather than silently approving them.
- Higher-resolution, tightly cropped label images improve extraction. The service currently uses low image detail and JPEG preprocessing to meet the five-second latency requirement, so perfect OCR cannot be guaranteed.
- OpenAI quota or rate limits can temporarily block live testing.
- Deployment readiness and browser warm-up reduce cold-start risk, but free-tier infrastructure cannot guarantee every request will complete under five seconds.
- The app does not persist history or support user accounts.
- The app does not currently support multiple images for one label because that slowed model calls during testing.
- This is a proof of concept, not a full TTB production workflow.

## Tradeoffs

- Images are downscaled for speed and cost, which can reduce accuracy on very small text.
- Browser-side chunking supports peak upload sets without one oversized request, but processing hundreds of labels takes longer and can encounter provider rate limits.
- Runtime configuration is exposed through `/health` only for non-secret UI limits.
- The app is stateless, so it cannot provide history or audit persistence.

## Approach and Tools

AI-assisted work produced initial scaffolding, extraction prompts, repetitive tests, and documentation drafts. Hand-written and manually reviewed work defines the response contracts, deterministic comparison rules, exact-warning behavior, request validation, error mapping, security boundaries, deployment configuration, and acceptance measurements. OpenAI performs label-text extraction only; application verdicts come from the deterministic Python comparison engine.
