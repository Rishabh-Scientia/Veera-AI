"""Integration test for Vera bot — validates all mandatory endpoints."""
import json
import sys
import argparse
import urllib.request
import urllib.error

parser = argparse.ArgumentParser()
parser.add_argument("--bot", default="http://localhost:8080")
args = parser.parse_args()
BASE = args.bot.rstrip("/")

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def post(path, data, expect_status=200):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(data, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            body = json.loads(r.read())
            return r.status, body
    except urllib.error.HTTPError as e:
        body = {}
        try:
            body = json.loads(e.read())
        except Exception:
            pass
        return e.code, body

def get(path):
    with urllib.request.urlopen(BASE + path) as r:
        return r.status, json.loads(r.read())

def check(name, condition, detail=""):
    mark = PASS if condition else FAIL
    msg = f"{mark} {name}"
    if detail:
        msg += f" | {detail}"
    print(msg)
    results.append((name, condition))
    return condition

# ─────────────────────────────────────────────
print("\n=== 1. GET /v1/healthz ===")
s, h = get("/v1/healthz")
print(json.dumps(h, indent=2))
check("healthz status=ok", h.get("status") == "ok")
check("healthz has contexts_loaded", "contexts_loaded" in h)

# ─────────────────────────────────────────────
print("\n=== 2. GET /v1/metadata ===")
s, m = get("/v1/metadata")
print(json.dumps(m, indent=2))
check("metadata has team_name", bool(m.get("team_name")))
check("metadata has model", bool(m.get("model")))
check("metadata has approach", bool(m.get("approach")))

# ─────────────────────────────────────────────
print("\n=== 3. POST /v1/context — push category ===")
cat = {
    "scope": "category", "context_id": "dentists", "version": 1,
    "delivered_at": "2026-04-29T10:00:00Z",
    "payload": {
        "slug": "dentists",
        "voice": {"tone": "peer_clinical", "vocab_taboo": ["guaranteed", "cure"]},
        "offer_catalog": [{"id": "den_001", "title": "Dental Cleaning @ \u20b9299", "value": "299"}],
        "peer_stats": {"avg_rating": 4.4, "avg_ctr": 0.030, "avg_calls_30d": 12},
        "digest": [{"id": "d_2026W17_jida_fluoride", "kind": "research",
                    "title": "3-month fluoride recall cuts caries 38% better than 6-month",
                    "source": "JIDA Oct 2026, p.14", "trial_n": 2100,
                    "patient_segment": "high_risk_adults"}],
        "seasonal_beats": [{"month_range": "Nov-Feb", "note": "exam-stress bruxism spike"}],
        "trend_signals": [{"query": "clear aligners delhi", "delta_yoy": 0.62}]
    }
}
s, r = post("/v1/context", cat)
print(json.dumps(r, indent=2))
check("push category accepted", r.get("accepted") is True, f"HTTP {s}")

# ─────────────────────────────────────────────
print("\n=== 4. POST /v1/context — push merchant ===")
merch = {
    "scope": "merchant", "context_id": "m_001_drmeera_dentist_delhi", "version": 1,
    "delivered_at": "2026-04-29T10:00:00Z",
    "payload": {
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "category_slug": "dentists",
        "identity": {"name": "Dr. Meera's Dental Clinic", "city": "Delhi",
                     "locality": "Lajpat Nagar", "verified": True,
                     "languages": ["en", "hi"], "owner_first_name": "Meera"},
        "subscription": {"status": "active", "plan": "Pro", "days_remaining": 82},
        "performance": {"window_days": 30, "views": 2410, "calls": 18, "ctr": 0.021,
                        "delta_7d": {"views_pct": 0.18, "calls_pct": -0.05}},
        "offers": [{"id": "o_meera_001", "title": "Dental Cleaning @ \u20b9299", "status": "active"}],
        "conversation_history": [],
        "customer_aggregate": {"total_unique_ytd": 540, "lapsed_180d_plus": 78,
                               "retention_6mo_pct": 0.38, "high_risk_adult_count": 124},
        "signals": ["stale_posts:22d", "ctr_below_peer_median", "high_risk_adult_cohort"]
    }
}
s, r = post("/v1/context", merch)
print(json.dumps(r, indent=2))
check("push merchant accepted", r.get("accepted") is True, f"HTTP {s}")

# ─────────────────────────────────────────────
print("\n=== 5. POST /v1/context — push trigger ===")
trg = {
    "scope": "trigger", "context_id": "trg_001_research_digest_dentists", "version": 1,
    "delivered_at": "2026-04-29T10:05:00Z",
    "payload": {
        "id": "trg_001_research_digest_dentists",
        "scope": "merchant", "kind": "research_digest", "source": "external",
        "merchant_id": "m_001_drmeera_dentist_delhi", "customer_id": None,
        "payload": {"category": "dentists", "top_item_id": "d_2026W17_jida_fluoride"},
        "urgency": 2, "suppression_key": "research:dentists:2026-W17",
        "expires_at": "2026-05-10T00:00:00Z"
    }
}
s, r = post("/v1/context", trg)
print(json.dumps(r, indent=2))
check("push trigger accepted", r.get("accepted") is True, f"HTTP {s}")

# ─────────────────────────────────────────────
print("\n=== 6. POST /v1/tick ===")
s, tick = post("/v1/tick", {"now": "2026-04-29T10:10:00Z",
                             "available_triggers": ["trg_001_research_digest_dentists"]})
print(json.dumps(tick, indent=2, ensure_ascii=False))
actions = tick.get("actions", [])
check("tick returns actions list", isinstance(actions, list))
check("tick has at least 1 action", len(actions) >= 1)

conv_id = None
if actions:
    a = actions[0]
    conv_id = a.get("conversation_id")
    body = a.get("body", "")
    check("action has conversation_id", bool(conv_id))
    check("action has merchant_id", bool(a.get("merchant_id")))
    check("action has send_as", a.get("send_as") in ("vera", "merchant_on_behalf"))
    check("action has trigger_id", bool(a.get("trigger_id")))
    check("action has cta", bool(a.get("cta")))
    check("action has suppression_key", bool(a.get("suppression_key")))
    check("action has rationale", bool(a.get("rationale")))
    check("body <= 320 chars", len(body) <= 320, f"{len(body)} chars")
    check("body has no URLs", "http" not in body.lower())
    print(f"\nBody preview: {body[:120]}...")

# ─────────────────────────────────────────────
print("\n=== 7. POST /v1/reply — merchant engaged ===")
if conv_id:
    s, rep = post("/v1/reply", {
        "conversation_id": conv_id,
        "merchant_id": "m_001_drmeera_dentist_delhi",
        "from_role": "merchant",
        "message": "Yes please send the abstract",
        "received_at": "2026-04-29T10:15:00Z",
        "turn_number": 2
    })
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    check("reply has action field", rep.get("action") in ("send", "wait", "end"))
    if rep.get("action") == "send":
        check("reply body <= 320", len(rep.get("body","")) <= 320)

# ─────────────────────────────────────────────
print("\n=== 8. POST /v1/reply — auto-reply detection ===")
s, arep = post("/v1/reply", {
    "conversation_id": "conv_auto_test_001",
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "from_role": "merchant",
    "message": "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly.",
    "received_at": "2026-04-29T10:16:00Z",
    "turn_number": 1
})
print(json.dumps(arep, indent=2, ensure_ascii=False))
check("auto-reply: not 'end' immediately", arep.get("action") in ("send", "wait"))

# ─────────────────────────────────────────────
print("\n=== 9. POST /v1/reply — opt-out ===")
s, erep = post("/v1/reply", {
    "conversation_id": "conv_end_test_001",
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "from_role": "merchant",
    "message": "Stop messaging me. Not interested.",
    "received_at": "2026-04-29T10:17:00Z",
    "turn_number": 1
})
print(json.dumps(erep, indent=2, ensure_ascii=False))
check("opt-out: action=end", erep.get("action") == "end")

# ─────────────────────────────────────────────
print("\n=== 10. POST /v1/reply — intent transition ===")
s, irep = post("/v1/reply", {
    "conversation_id": conv_id or "conv_intent_test",
    "merchant_id": "m_001_drmeera_dentist_delhi",
    "from_role": "merchant",
    "message": "Ok let's do it, go ahead",
    "received_at": "2026-04-29T10:18:00Z",
    "turn_number": 3
})
print(json.dumps(irep, indent=2, ensure_ascii=False))
check("intent transition: action=send", irep.get("action") == "send")
check("intent transition: binary cta", irep.get("cta") in ("binary_confirm_cancel", "binary_yes_no"))

# ─────────────────────────────────────────────
print("\n=== 11. Idempotency — same version re-push ===")
s, r = post("/v1/context", cat)  # version 1 again
print(json.dumps(r, indent=2))
check("idempotent re-push accepted", r.get("accepted") is True, f"HTTP {s}")

# ─────────────────────────────────────────────
print("\n=== 12. Stale version — returns 409 ===")
cat_old = dict(cat, version=0)
s, r = post("/v1/context", cat_old)
print(f"HTTP {s}: {json.dumps(r, indent=2)}")
check("stale version returns 409", s == 409)
check("stale version reason field", r.get("reason") == "stale_version")

# ─────────────────────────────────────────────
print("\n=== 13. Version bump — higher version accepted ===")
cat_v2 = dict(cat, version=2)
s, r = post("/v1/context", cat_v2)
print(json.dumps(r, indent=2))
check("version bump accepted", r.get("accepted") is True, f"HTTP {s}")

# ─────────────────────────────────────────────
print("\n=== 14. Healthz after contexts loaded ===")
s, h2 = get("/v1/healthz")
print(json.dumps(h2, indent=2))
check("contexts_loaded > 0", sum(h2.get("contexts_loaded", {}).values()) > 0)

# ─────────────────────────────────────────────
print("\n" + "="*50)
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"\nRESULT: {passed}/{total} checks passed")
if passed == total:
    print("ALL TESTS PASSED")
else:
    failed = [name for name, ok in results if not ok]
    print(f"FAILED: {failed}")
    sys.exit(1)
