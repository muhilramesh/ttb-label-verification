# TTB Label Verification

Proof-of-concept web app for checking alcohol label images against submitted application data. The app extracts visible label text with a vision model, compares each extracted field with deterministic rules, and returns a clear `APPROVED` or `NEEDS REVIEW` result.

## Live Demo

Deployed URL: https://ttb-label-verification-production-6496.up.railway.app


## What This Does

- Accepts a label image plus seven application fields.
- Extracts label text into structured JSON.
- Compares every field and shows per-field `PASS` or `FAIL`.
- Shows expected-vs-found values on failures.
- Surfaces the extracted government warning text for review.
- Supports batch checking with per-label results and summary counts.
- Returns readable errors for bad input, timeouts, provider limits, and missing configuration.

## Demo Checklist

- Single-label check: upload one image, fill the seven fields, and click `Check Label`.
- Batch check: open `Batch Labels`, add label images one at a time or all together, fill each row, and click `Check All Labels`.
- Exact-warning behavior: capitalization, wording, and punctuation are strict; line-break layout is normalized because label OCR often wraps warning text.
- Error behavior: submit without an image or with a wrong file type to see a plain-English 4xx response.

## Tech Stack

- Python 3.12
- FastAPI
- Pydantic
- Plain HTML, CSS, and JavaScript
- OpenAI vision model through the OpenAI Responses API
- Railway deployment
- Pytest test suite

## Environment Variables

Real secrets must live in local environment variables or Railway service variables. Do not commit `.env`.

| Variable | Default | Required | Notes |
| --- | --- | --- | --- |
| `APP_ENV` | `local` | No | App environment label returned by `/health`. |
| `LOG_LEVEL` | `INFO` | No | Backend logging level. |
| `OPENAI_API_KEY` | none | Yes | Required for real vision extraction and `/health/deep`. |
| `OPENAI_MODEL` | `gpt-4o-mini` | No | Exact deployed model name. |
| `OPENAI_TIMEOUT_SECONDS` | `4.5` | No | Backend provider timeout. Keep production at or below `4.5` for the <5s target. |
| `IMAGE_MAX_LONG_SIDE` | `768` | No | Max image dimension before the model call. |
| `IMAGE_JPEG_QUALITY` | `70` | No | JPEG quality after preprocessing. |
| `BATCH_CONCURRENCY` | `3` | No | Max concurrent model calls for batch verification. |

`OPENAI_API_KEY` is required for real vision extraction. `.env.example` lists variable names only.

Configured deployed model: `gpt-4o-mini`.

Model documentation verification: on 2026-07-12, OpenAI's Responses API reference listed `gpt-4o-mini` as an accepted model, and OpenAI's image/vision guide listed `GPT-4o-mini` as supporting image detail modes. The production deploy also passed `/health/deep` against OpenAI's Models API on that date.

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
export OPENAI_MODEL=gpt-4o-mini
export OPENAI_TIMEOUT_SECONDS=4.5
export IMAGE_MAX_LONG_SIDE=768
export IMAGE_JPEG_QUALITY=70
export BATCH_CONCURRENCY=3
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

- Brand name, product type, and producer name: fuzzy token-sort matching with threshold `90`.
- Country of origin: normalized country synonyms, including `USA`, `United States`, and common producing-country variants.
- Alcohol content: numeric ABV normalization, so `45%` can match `45% Alc./Vol. (90 Proof)` or `90 Proof`.
- Net contents: unit normalization, so `750 mL` can match `750ml`, and `355 mL` can match `12 FL OZ`.
- Government warning: strict wording, capitalization, and punctuation; whitespace layout is normalized to avoid false failures from OCR line wrapping.

Any field failure makes the overall verdict `NEEDS_REVIEW`.

## Batch Processing

Batch mode processes labels concurrently with a bounded concurrency limit. One bad label does not fail the whole batch; each item returns either a result or an item-level error. The summary reports approved, needs-review, error, and total counts.

## Security

- API keys are read from environment variables only.
- `.env` and `.env.*` files are ignored by Git.
- `.env.example` contains placeholders only.
- The production key is configured in Railway, not in the repository.
- Error responses do not expose stack traces.

## Performance Notes

Single-label verification has a hard target of under `5` seconds on the deployed URL. The backend provider timeout defaults to `4.5` seconds, and the frontend aborts a single-label request after `5` seconds.

Measure deployed single-label latency after every deploy:

```bash
uv run python scripts/measure_single_label_latency.py \
  --url https://YOUR-DEPLOYED-URL \
  --samples 20
```

Current OpenAI deployment measurement:

| Date | URL | Model | Samples | Successful | Timeouts | p50 | p95 | Script |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-07-12 | `https://ttb-label-verification-production-6496.up.railway.app` | `gpt-4o-mini` | 20 | 20 | 0 | 2,839 ms | 3,195 ms | `scripts/measure_single_label_latency.py` |

The measured p50 and p95 meet the under-5-second target, and all 20 requests completed successfully.

Batch mode can take longer because several labels are processed together. Concurrency is bounded with `BATCH_CONCURRENCY`.

## Assumptions

- One image represents one label record.
- The app is stateless and does not store uploaded images or results.
- Application data is supplied as seven required fields.
- Reviewers will use a configured `OPENAI_API_KEY` for real extraction.
- Railway provides production environment variables.

## Limitations

- Vision/OCR can misread small or blurry warning text.
- OpenAI quota or rate limits can temporarily block live testing.
- The app does not persist history or support user accounts.
- The app does not currently support multiple images for one label because that slowed model calls during testing.
- This is a proof of concept, not a full TTB production workflow.
