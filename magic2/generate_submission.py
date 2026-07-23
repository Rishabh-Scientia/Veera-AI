#!/usr/bin/env python3
"""
generate_submission.py
======================
Reads the expanded dataset + test_pairs.json and calls the running bot
to compose messages for all 30 canonical test pairs.
Writes submission.jsonl (one JSON line per test pair).

Usage:
    python generate_submission.py --bot http://localhost:8080 --out submission.jsonl
"""

import io
import sys
# Force UTF-8 stdout so Indian characters (₹, etc.) don't crash on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urlrequest

EXPANDED = Path(__file__).parent / "expanded"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def post(url: str, body: dict) -> dict:
    req = urlrequest.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def get(url: str) -> dict:
    with urlrequest.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def push_context(bot_url: str, scope: str, cid: str, payload: dict, version: int = 2) -> bool:
    try:
        resp = post(f"{bot_url}/v1/context", {
            "scope": scope, "context_id": cid,
            "version": version, "payload": payload, "delivered_at": _now()
        })
        return resp.get("accepted", False)
    except Exception as e:
        print(f"    push {scope}/{cid} failed: {e}")
        return False


def compose_for_pair(bot_url: str, pair: dict,
                     cats: dict, merchants: dict, customers: dict, triggers_by_id: dict,
                     version_base: int) -> dict | None:
    """
    Push only the contexts needed for this pair, then call /v1/tick.
    """
    test_id = pair["test_id"]
    tid = pair["trigger_id"]
    mid = pair["merchant_id"]
    cust_id = pair.get("customer_id")

    trg = triggers_by_id.get(tid)
    merchant = merchants.get(mid)
    if not trg or not merchant:
        print(f"  [{test_id}] SKIP: missing trigger or merchant data")
        return None

    cat_slug = merchant.get("category_slug", "")
    cat = cats.get(cat_slug)
    if not cat:
        print(f"  [{test_id}] SKIP: missing category {cat_slug}")
        return None

    v = version_base  # unique version per pair to avoid idempotency skip

    # Push category
    push_context(bot_url, "category", cat_slug, cat, v)

    # Push merchant
    push_context(bot_url, "merchant", mid, merchant, v)

    # Push customer if needed
    if cust_id:
        cust = customers.get(cust_id)
        if cust:
            push_context(bot_url, "customer", cust_id, cust, v)

    # Push trigger with unique context_id per pair to avoid suppression conflicts
    # Extend expiry to 2027 so no trigger is considered expired during submission
    pair_tid = f"{tid}_pair_{test_id}"
    trg_payload = dict(trg)
    trg_payload["suppression_key"] = f"{trg.get('suppression_key', tid)}:pair_{test_id}"
    trg_payload["expires_at"] = "2027-12-31T00:00:00Z"  # ensure not expired
    push_context(bot_url, "trigger", pair_tid, trg_payload, v)

    # Tick
    resp = post(f"{bot_url}/v1/tick", {
        "now": _now(),
        "available_triggers": [pair_tid]
    })
    actions = resp.get("actions", [])
    return actions[0] if actions else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bot", default="http://localhost:8080", help="Bot base URL")
    parser.add_argument("--out", default="submission.jsonl", help="Output file")
    args = parser.parse_args()

    bot_url = args.bot.rstrip("/")
    out_path = Path(args.out)

    # Health check
    try:
        h = get(f"{bot_url}/v1/healthz")
        print(f"Bot health: {h.get('status')} | uptime: {h.get('uptime_seconds')}s", flush=True)
    except Exception as e:
        print(f"ERROR: Bot not reachable at {bot_url}: {e}")
        sys.exit(1)

    # Load expanded dataset into memory
    print("Loading expanded dataset...", flush=True)
    cats, merchants, customers, triggers_by_id = {}, {}, {}, {}

    for f in (EXPANDED / "categories").glob("*.json"):
        d = load_json(f); cats[d["slug"]] = d

    for f in (EXPANDED / "merchants").glob("*.json"):
        d = load_json(f); merchants[d["merchant_id"]] = d

    for f in (EXPANDED / "customers").glob("*.json"):
        d = load_json(f); customers[d["customer_id"]] = d

    for f in (EXPANDED / "triggers").glob("*.json"):
        d = load_json(f); triggers_by_id[d["id"]] = d

    print(f"  {len(cats)} cats | {len(merchants)} merchants | "
          f"{len(customers)} customers | {len(triggers_by_id)} triggers", flush=True)

    # Load test pairs
    pairs_file = EXPANDED / "test_pairs.json"
    if not pairs_file.exists():
        print(f"ERROR: test_pairs.json not found in {EXPANDED}")
        sys.exit(1)

    pairs = load_json(pairs_file)["pairs"]
    print(f"\nComposing messages for {len(pairs)} test pairs...\n", flush=True)

    results = []

    for i, pair in enumerate(pairs):
        test_id = pair["test_id"]
        print(f"[{i+1:02d}/{len(pairs)}] {test_id} | merchant={pair['merchant_id'][:25]} | "
              f"trigger={pair['trigger_id'][:30]}", flush=True)

        try:
            action = compose_for_pair(
                bot_url, pair,
                cats, merchants, customers, triggers_by_id,
                version_base=10 + i,  # unique version per pair
            )
            if action:
                body = action.get("body", "")
                print(f"         OK ({len(body)} chars): {body[:70]}...", flush=True)
                results.append({
                    "test_id": test_id,
                    "trigger_id": pair["trigger_id"],
                    "merchant_id": pair["merchant_id"],
                    "customer_id": pair.get("customer_id"),
                    "body": body,
                    "cta": action.get("cta", "open_ended"),
                    "send_as": action.get("send_as", "vera"),
                    "suppression_key": action.get("suppression_key", ""),
                    "rationale": action.get("rationale", ""),
                })
            else:
                print(f"         WARN: no action returned", flush=True)
                results.append({
                    "test_id": test_id, "trigger_id": pair["trigger_id"],
                    "merchant_id": pair["merchant_id"], "customer_id": pair.get("customer_id"),
                    "body": "", "cta": "none", "send_as": "vera",
                    "suppression_key": "", "rationale": "Bot returned no action",
                })
        except Exception as e:
            print(f"         ERROR: {e}", flush=True)
            results.append({
                "test_id": test_id, "trigger_id": pair["trigger_id"],
                "merchant_id": pair["merchant_id"], "body": f"[Error: {e}]",
                "cta": "none", "send_as": "vera", "suppression_key": "", "rationale": ""
            })

    # Write JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    non_empty = [r for r in results if r.get("body") and not r["body"].startswith("[Error")]
    avg_len = sum(len(r["body"]) for r in non_empty) / max(len(non_empty), 1)
    too_long = [r for r in non_empty if len(r["body"]) > 320]
    with_urls = [r for r in non_empty if "http" in r.get("body", "")]

    print(f"\nWrote {len(results)} lines to {out_path}", flush=True)
    print(f"  OK:            {len(non_empty)}/{len(results)}", flush=True)
    print(f"  Avg body len:  {avg_len:.0f} chars", flush=True)
    print(f"  Body >320:     {len(too_long)} (should be 0)", flush=True)
    print(f"  With URLs:     {len(with_urls)} (should be 0)", flush=True)


if __name__ == "__main__":
    main()
