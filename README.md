# Vera Bot — magicpin AI Challenge

## Approach

**Architecture**: Trigger-routed LLM composer using **Gemini 2.0 Flash** with rule-based fast paths.

### How it works

```
Judge → POST /v1/context  →  In-memory store (category, merchant, customer, trigger)
Judge → POST /v1/tick     →  Trigger loop → compose() → Gemini 2.0 Flash → validated action
Judge → POST /v1/reply    →  Rule fast-paths → LLM reply if needed
```

### Composer (composer.py)

1. **Trigger routing** — 25 trigger `kind` values each get a specific composition hint describing exactly what information to use and what CTA shape to produce.
2. **4-context prompt** — Category (voice, peer_stats, digest), Merchant (identity, performance, offers, signals, conversation_history), Trigger (kind, payload, urgency), Customer (if present).
3. **Digest item injection** — When trigger references `top_item_id`, the matching digest item is pulled from category context and added to the prompt directly.
4. **Post-LLM validation** — Body length enforced (≤320 chars), URLs stripped, suppression_key from trigger, `send_as` and `cta` validated.
5. **Fallback** — Rule-based composer fires if LLM is unavailable.

### Reply Handler (reply_handler.py)

Rule-based fast paths (O(1), no LLM needed):
- **Auto-reply detection** — Pattern-matches 10+ canned phrases in EN + HI. First auto → `send` (one prompt for owner). Second → `wait 24h`. Third → `end`.
- **Opt-out / hostile** — Detected via phrase list → `end` immediately.
- **Intent transition** — "let's do it / karo / go ahead" → LLM in `execute_intent` mode (no re-qualification allowed).
- **Out-of-scope** — GST/legal/insurance → politely decline + redirect.
- **LLM reply** — Everything else → Gemini with 6-turn history context.

### Key design decisions

1. **Gemini 2.0 Flash** — 2-5s latency, well within 10s budget. Temperature=0 for determinism.
2. **Suppression** — `suppression_key` is stored after each tick action; won't re-send same trigger to same merchant.
3. **Idempotency** — `POST /v1/context` with same `(context_id, version)` is a no-op. Higher version replaces atomically.
4. **Stateful in-memory** — Contexts + conversations persist across calls (no restarts during test).
5. **Teardown** — `POST /v1/teardown` wipes all state cleanly.

## What additional context would have helped

- **Conversation history format** from production Vera — knowing how past turns were encoded would let me use richer history.
- **Merchant reply distribution** — knowing the % of auto-replies vs real replies helps tune fast-path thresholds.
- **Category-specific offer pricing ranges** — to generate more grounded offers for the fallback composer.

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY to .env
python main.py
```

Or with uvicorn:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8080
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/healthz` | Liveness probe |
| GET | `/v1/metadata` | Bot identity |
| POST | `/v1/context` | Receive context push (idempotent) |
| POST | `/v1/tick` | Periodic wake-up, returns actions |
| POST | `/v1/reply` | Handle merchant/customer reply |
| POST | `/v1/teardown` | Wipe state (optional) |

## Files

```
├── main.py          # FastAPI server, all 5 endpoints
├── composer.py      # LLM message composer (Gemini 2.0 Flash)
├── reply_handler.py # Multi-turn reply handler
├── test_bot.py      # Local integration test
├── requirements.txt
├── Dockerfile       # Railway deployment
├── railway.json     # Railway config
└── .env.example
```
