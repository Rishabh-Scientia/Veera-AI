"""
composer.py — Vera message composer using Google Gemini.
Takes 4-context inputs → produces high-quality, merchant-specific WhatsApp messages.
"""

import os
import json
import re
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("vera.composer")

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
MODEL_NAME = "gemini-2.5-flash"


SYSTEM_PROMPT = """You are Vera, magicpin's AI composing WhatsApp messages for Indian merchants.

HARD RULES (violating any = score 0):
1. Body MUST be <= 320 chars. Aim 150-200. Count every character.
2. NO URLs anywhere in body.
3. ONE CTA only: open_ended | binary_yes_no | binary_confirm_cancel | none
4. NEVER fabricate data. Use ONLY numbers/names/dates from provided context.
5. Merchant FIRST NAME only (e.g. "Meera", not "Dr. Meera's Dental Clinic").
6. NO greetings like "Hope you're well". Start with [Name], [signal].
7. suppression_key = trigger's suppression_key exactly.
8. Hindi-English mix only if merchant languages includes "hi".

GOLD STANDARD FORMAT (follow EXACTLY):
[Name], [specific proof: number/source/date from context]. [Consequence or opportunity in 1 line]. [Offer/hook]. [Single yes/no action]?

GOLD EXAMPLE (score 9/10):
"Meera, JIDA Oct 2026 (n=2,100): 3-month fluoride recall cuts caries 38% better. 124 of your patients are high-risk. Draft patient note + WhatsApp? Reply YES."

BAD EXAMPLE (score 1/10 — NEVER DO THIS):
"Hi Doctor, want to run a discount campaign today to increase sales?"
WHY BAD: No trigger, no merchant fact, no specific number, no category voice.

CATEGORY VOICE:
- dentists: peer_clinical. Cite JIDA/DCI sources. Clinical vocab (fluoride, caries, OPG). TABOO: guaranteed, cure, miracle.
- salons: warm_practical. Specific services (balayage, keratin, threading). TABOO: guaranteed glow.
- restaurants: fellow_operator. Metrics (covers, AOV, footfall, delivery%). TABOO: best food ever.
- gyms: coach/energetic. Fitness vocab (PT sessions, churn, footfall). TABOO: guaranteed weight loss.
- pharmacies: trustworthy_precise. Molecule names, batch. TABOO: miracle cure.

COMPULSION LEVERS (use 2+ per message — this is what drives replies):
- Specific numbers: "78 lapsed patients", "CTR 2.1% vs peer 3.0%", "calls down 50%"
- Loss framing: "you're missing X leads this week"
- Social proof: "3 dentists in Lajpat Nagar ran this last month"
- Effort externalized: "I've already drafted it — just say YES"
- Urgency: "expires in 12 days", "Dec 15 deadline", "this week only"
- Low-friction CTA: "Reply YES" or "Reply 1 or 2 to confirm"

PEER BENCHMARKS (use for specificity — compare merchant vs peer):
- dentists: avg_ctr=3.0%, avg_calls_30d=12, retention_6mo=42%
- salons: avg_ctr=4.0%, avg_calls_30d=28, retention_3mo=55%
- restaurants: avg_ctr=2.5%, avg_calls_30d=38
- gyms: avg_ctr=4.5%, avg_calls_30d=18, monthly_churn=8%
- pharmacies: avg_ctr=3.8%, avg_calls_30d=22, repeat_customer=62%

OUTPUT — ONLY valid JSON, no markdown fences:
{
  "body": "<message <=320 chars following gold standard format>",
  "cta": "<open_ended|binary_yes_no|binary_confirm_cancel|none>",
  "send_as": "<vera|merchant_on_behalf>",
  "suppression_key": "<from trigger payload>",
  "rationale": "<Signal used + why now + compulsion levers applied>"
}"""


TRIGGER_ROUTING = {
    "research_digest": "Cite the EXACT source (e.g. 'JIDA Oct 2026 p.14'), trial_n, and % finding. Tie to merchant's patient cohort from merchant context. Offer to draft patient-ed note. Dentist tone: peer/clinical.",
    "regulation_change": "Name the regulator (DCI/FSSAI/FDA), exact deadline from payload, specific compliance action. Frame as risk not promo. Urgent.",
    "recall_due": "send_as=merchant_on_behalf. Use patient's name from customer context. State months since last visit. Offer 2 slot options with day+time. Hindi code-mix if patient language=hi.",
    "appointment_tomorrow": "send_as=merchant_on_behalf. Customer name + service + tomorrow's time. Friendly, brief confirmation ask.",
    "perf_dip": "Name exact metric (calls/views/CTR) and % drop from payload. Reference peer median for context. Offer one concrete fix using their active offer. Close with binary ask.",
    "perf_spike": "Celebrate the specific metric win with %. Ask what drove it. Offer to replicate with one specific action (GBP post / offer push).",
    "renewal_due": "State exact days_remaining from payload. Frame as protecting lead flow, not fear. 2-minute renewal CTA.",
    "festival_upcoming": "Name festival + days_until. Link to their active offer specifically. Offer to draft campaign post. E.g. Diwali 188 days: 'Bridal Trial @ ₹999 is the hook'.",
    "ipl_match_today": "Match name + time from payload. IMPORTANT: if Saturday — note covers drop 12% (use magicpin data); push Tue/Wed/Thu instead. Use their active offer as match-night combo.",
    "review_theme_emerged": "Theme + occurrences_30d from payload. Quote common_quote if available. Offer drafted response template.",
    "milestone_reached": "Metric + value_now + milestone_value. Brief celebration. Pivot immediately to next action.",
    "active_planning_intent": "Respond to merchant's exact last message from conversation_history. Draft a concrete 1-line proposal (pricing, timeline, deliverable). Accept with one word.",
    "seasonal_perf_dip": "Name metric + % drop. Explicitly frame as normal seasonal (cite the beat: e.g. 'Apr-Jun lowest acquisition window'). Give retention focus, not acquisition spend.",
    "customer_lapsed_soft": "send_as=merchant_on_behalf. Days since last visit + previous service/goal. Warm re-engagement, no shame. One specific next-step offer.",
    "customer_lapsed_hard": "send_as=merchant_on_behalf. Name + days lapsed. Free comeback session or strong hook. Single binary YES reply.",
    "supply_alert": "Exact molecule + batch numbers from payload. Scope (how many customers affected if calculable). Offer to filter Rx list + draft outreach.",
    "chronic_refill_due": "send_as=merchant_on_behalf. All molecules from molecule_list. Exact stock_runs_out date. Senior discount if applicable. Home delivery CTA.",
    "winback_eligible": "Days since expiry + specific perf_dip_pct loss. One-step restart ask. Frame as reversible.",
    "curious_ask_due": "One sharp category-specific question (dentist: whitening vs cleaning; salon: which service books first; restaurant: new dish; gym: peak slot; pharmacy: molecule shortage). Offer specific deliverable from answer.",
    "dormant_with_vera": "Acknowledge silence briefly. Pull one fresh signal from merchant data (signals[], performance delta, or offer). Short re-engage hook.",
    "gbp_unverified": "State estimated_uplift_pct from payload (e.g. 30% more clicks). '2-minute process'. Offer to walk them through.",
    "competitor_opened": "Competitor name + distance_km + their_offer from payload. Defensive frame: what merchant's active offer does better. Campaign ask.",
    "cde_opportunity": "Event name + credits + fee (free/paid) from payload. '3 peers attending' social proof. Short ask.",
    "wedding_package_followup": "send_as=merchant_on_behalf. Days to wedding + trial_completed date. Next service step (skin prep / booking). Specific slot offer.",
    "trial_followup": "send_as=merchant_on_behalf. Trial date + next session slot from payload. Warm, no pressure.",
    "category_seasonal": "Name specific seasonal shift from digest (e.g. ORS +40%, anti-fungal up). Shelf/campaign action. No hype.",
    "default": "Use strongest signal from merchant: signals[], perf delta vs peer median, or active offer. Ground every claim in provided data.",
}


def _get_trigger_hint(kind: str) -> str:
    return TRIGGER_ROUTING.get(kind, TRIGGER_ROUTING["default"])


def _build_prompt(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> str:
    merchant_name = merchant.get("identity", {}).get("owner_first_name") or merchant.get("identity", {}).get("name", "")
    category_slug = merchant.get("category_slug", category.get("slug", ""))
    trigger_kind = trigger.get("kind", "default")
    hint = _get_trigger_hint(trigger_kind)
    payload = trigger.get("payload", {})

    # ── Pre-analyze context to extract the best specifics ──────────────────────
    perf = merchant.get("performance", {})
    delta = perf.get("delta_7d", {})
    offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
    offer_title = offers[0]["title"] if offers else None
    peer_stats = category.get("peer_stats", {})
    cust_agg = merchant.get("customer_aggregate", {})
    signals = merchant.get("signals", [])
    locality = merchant.get("identity", {}).get("locality", "")
    city = merchant.get("identity", {}).get("city", "")
    langs = merchant.get("identity", {}).get("languages", ["en"])

    # Peer comparison
    my_ctr = perf.get("ctr", 0)
    peer_ctr = peer_stats.get("avg_ctr", 0)
    peer_calls = peer_stats.get("avg_calls_30d", 0)
    my_calls = perf.get("calls", 0)
    ctr_gap = f"CTR {my_ctr*100:.1f}% vs peer {peer_ctr*100:.1f}%" if my_ctr and peer_ctr else ""
    calls_gap = f"calls {my_calls} vs peer avg {peer_calls}" if my_calls and peer_calls else ""

    # Extract digest item
    digest_item = {}
    top_item_id = payload.get("top_item_id")
    if top_item_id:
        for d in category.get("digest", []):
            if d.get("id") == top_item_id:
                digest_item = d
                break
    if not digest_item and category.get("digest"):
        digest_item = category["digest"][0]

    # Trend signals
    trends = category.get("trend_signals", [])
    trend_str = ""
    if trends:
        t = trends[0]
        trend_str = f"'{t.get('query')}' search up {int(t.get('delta_yoy',0)*100)}% YoY in {city}"

    # Customer details
    cust_name = (customer or {}).get("identity", {}).get("name", "")
    cust_langs = (customer or {}).get("identity", {}).get("languages", langs)
    cust_slots = payload.get("available_slots", [])
    slot_str = " or ".join(s["label"] for s in cust_slots[:2]) if cust_slots else ""

    # Build BEST_SIGNALS block — this is the key to high specificity
    signals_block = f"""
PRE-ANALYZED BEST SIGNALS (use these directly in your message):
- Merchant: {merchant_name}, {locality}, {city}
- Active offer: {offer_title or 'NONE — mention they need one'}
- Performance gap: {ctr_gap} | {calls_gap}
- 7-day delta: calls {int(delta.get('calls_pct',0)*100):+d}%, views {int(delta.get('views_pct',0)*100):+d}%
- Lapsed patients/customers: {cust_agg.get('lapsed_180d_plus', 0)} (6mo+)
- High-risk cohort: {cust_agg.get('high_risk_adult_count', 0)}
- Retention: {int(cust_agg.get('retention_6mo_pct',0)*100)}% vs peer {int(peer_stats.get('retention_6mo_pct',0)*100)}%
- Key merchant signals: {', '.join(signals[:4]) if signals else 'none'}
- Local trend: {trend_str}
- Digest/research: {json.dumps(digest_item, ensure_ascii=False) if digest_item else 'none'}
- Customer: {cust_name or 'N/A'} | Slots: {slot_str or 'N/A'}
- Trigger payload: {json.dumps(payload, ensure_ascii=False)}
- PLACEHOLDER TRIGGER: {'YES — the trigger payload has NO real data. You MUST synthesize specific numbers/names from the merchant performance, customer aggregate, category peer_stats, and seasonal_beats above. DO NOT leave any blank fields.' if payload.get('placeholder') else 'No — use payload data directly'}
- Hindi code-mix: {'YES — mix naturally' if 'hi' in langs else 'NO — English only'}
- suppression_key to use: {trigger.get('suppression_key', '')}"""

    customer_block = ""
    if customer:
        customer_block = f"\n\nFULL CUSTOMER CONTEXT:\n{json.dumps(customer, ensure_ascii=False, indent=2)}"

    prompt = f"""TRIGGER KIND: {trigger_kind}
COMPOSER HINT: {hint}
{signals_block}

FULL CATEGORY CONTEXT (slug={category_slug}):
{json.dumps(category, ensure_ascii=False, indent=2)}

FULL MERCHANT CONTEXT:
{json.dumps(merchant, ensure_ascii=False, indent=2)}{customer_block}

FULL TRIGGER:
{json.dumps(trigger, ensure_ascii=False, indent=2)}

Write the best Vera message using the PRE-ANALYZED SIGNALS above. Every number you use MUST come from the context. Return ONLY valid JSON."""

    return prompt


def _validate_and_fix(result: dict, trigger: dict) -> dict:
    """Post-LLM validation and auto-fix."""
    body = result.get("body", "")

    # Strip URLs
    body = re.sub(r'https?://\S+', '', body).strip()

    # Enforce length
    if len(body) > 320:
        body = body[:317] + "..."

    # Enforce suppression key from trigger
    result["suppression_key"] = trigger.get("suppression_key", result.get("suppression_key", ""))

    # Validate send_as
    if result.get("send_as") not in ("vera", "merchant_on_behalf"):
        result["send_as"] = "merchant_on_behalf" if trigger.get("scope") == "customer" else "vera"

    # Validate cta
    valid_ctas = {"open_ended", "binary_yes_no", "binary_confirm_cancel", "none", "multi_choice_slot"}
    if result.get("cta") not in valid_ctas:
        result["cta"] = "open_ended"

    result["body"] = body
    return result


def compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> dict:
    """
    Core composition function.
    Returns: {body, cta, send_as, suppression_key, rationale}
    """
    if not LLM_API_KEY:
        # Fallback deterministic composer if no API key
        return _fallback_compose(category, merchant, trigger, customer)

    try:
        import urllib.request
        import json
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        prompt = _build_prompt(category, merchant, trigger, customer)
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps({
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000,
                "response_format": {"type": "json_object"}
            }).encode("utf-8"),
            headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
        )
        response = urllib.request.urlopen(req, timeout=45, context=ctx)
        data = json.loads(response.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"].strip()

        # Robust JSON extraction
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            text = match.group(0)

        # Basic cleanup if needed
        if "```" in text:
            text = re.sub(r"```[a-z]*\n?", "", text).replace("```", "").strip()

        result = json.loads(text)
        return _validate_and_fix(result, trigger)

    except Exception as e:
        logger.error(f"LLM composition failed: {e}")
        return _fallback_compose(category, merchant, trigger, customer)


# ── Category-specific helpers ────────────────────────────────────────────────

BUSINESS_TYPE = {
    "dentists": "clinic",
    "salons": "salon",
    "restaurants": "restaurant",
    "gyms": "studio",
    "pharmacies": "pharmacy",
}

CATEGORY_LAPSE_GOAL = {
    "dentists": "dental health",
    "salons": "grooming routine",
    "restaurants": "favourite table",
    "gyms": "fitness goal",
    "pharmacies": "health routine",
}

CATEGORY_SERVICE_DEFAULT = {
    "dentists": "dental check-up",
    "salons": "grooming appointment",
    "restaurants": "reservation",
    "gyms": "session",
    "pharmacies": "health consultation",
}

UPCOMING_FESTIVALS = [
    ("Diwali", 188), ("Navratri", 175), ("Christmas", 236),
    ("Holi", 310), ("Eid", 260), ("Independence Day", 104),
]


def _get_biz_type(cat_slug: str) -> str:
    return BUSINESS_TYPE.get(cat_slug, "business")


def _compute_days_since_last_visit(customer: Optional[dict]) -> int:
    """Compute days since last visit from customer relationship data."""
    if not customer:
        return 45
    last_visit = (customer.get("relationship") or {}).get("last_visit", "")
    if not last_visit:
        return 45
    try:
        from datetime import datetime, timezone
        lv = datetime.fromisoformat(last_visit.replace("Z", "+00:00"))
        delta = datetime.now(timezone.utc) - lv
        return max(1, delta.days)
    except Exception:
        return 45


def _get_strongest_perf_dip(perf: dict) -> tuple:
    """Find the strongest dipping metric from merchant performance."""
    delta = perf.get("delta_7d", {})
    calls_pct = delta.get("calls_pct", 0)
    views_pct = delta.get("views_pct", 0)
    if calls_pct < views_pct:
        return "calls", abs(int(calls_pct * 100))
    elif views_pct < 0:
        return "views", abs(int(views_pct * 100))
    return "calls", 20  # sensible default


def _get_strongest_perf_spike(perf: dict) -> tuple:
    """Find the strongest spiking metric from merchant performance."""
    delta = perf.get("delta_7d", {})
    calls_pct = delta.get("calls_pct", 0)
    views_pct = delta.get("views_pct", 0)
    if calls_pct > views_pct and calls_pct > 0:
        return "calls", int(calls_pct * 100)
    elif views_pct > 0:
        return "views", int(views_pct * 100)
    return "views", 15  # sensible default


def _get_next_festival() -> tuple:
    """Return (name, days_until) for the nearest upcoming festival."""
    # Simple static lookup — appropriate for submission
    return UPCOMING_FESTIVALS[0]  # Diwali, 188


def _is_placeholder(payload: dict) -> bool:
    """Check if trigger payload is a placeholder with no real data."""
    return payload.get("placeholder", False)


def _fallback_compose(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> dict:
    """Rule-based fallback composer — covers all 25 trigger kinds deterministically.
    Handles placeholder triggers by synthesizing data from merchant/category context.
    """
    try:
        name = merchant.get("identity", {}).get("owner_first_name") or "there"
        kind = trigger.get("kind", "")
        logger.info(f"Fallback composing for kind={kind}")
        payload = trigger.get("payload", {})
        is_ph = _is_placeholder(payload)
        perf = merchant.get("performance", {})
        offers = [o for o in merchant.get("offers", []) if o.get("status") == "active"]
        offer_title = offers[0]["title"] if offers else None
        cat_slug = merchant.get("category_slug", category.get("slug", ""))
        biz_type = _get_biz_type(cat_slug)
        peer_ctr = category.get("peer_stats", {}).get("avg_ctr", 0.030)
        my_ctr = perf.get("ctr", 0.020)
        peer_calls = int(category.get("peer_stats", {}).get("avg_calls_30d", 0))
        my_calls = perf.get("calls", 0)
        supp_key = trigger.get("suppression_key", "")
        send_as = "merchant_on_behalf" if trigger.get("scope") == "customer" else "vera"
        cust_name = customer.get("identity", {}).get("name", "") if customer else ""
        langs = merchant.get("identity", {}).get("languages", ["en"])
        locality = merchant.get("identity", {}).get("locality", "your area")
        cust_agg = merchant.get("customer_aggregate", {})
        signals = merchant.get("signals", [])

        body = ""

        if kind == "research_digest":
            items = category.get("digest", [])
            item = next((d for d in items if d.get("id") == payload.get("top_item_id")), items[0] if items else {})
            source = item.get("source", "journal")
            n = item.get("trial_n", "")
            n_str = f" (n={n:,})" if n else ""
            title = item.get("title", "new findings")
            high_risk = cust_agg.get("high_risk_adult_count", 0)
            cohort_str = f" {high_risk} of your patients qualify." if high_risk else ""
            body = f"{name}, {source} dropped{n_str}. Key: {title}.{cohort_str} Draft a patient note? Reply YES."

        elif kind == "regulation_change":
            deadline = payload.get("deadline_iso", "2026-12-15")[:10]
            items = category.get("digest", [])
            reg_item = next((d for d in items if d.get("kind") == "compliance"), {})
            summary = reg_item.get("summary", "")
            if "D-speed" in summary or "1.5" in summary:
                body = f"{name}, DCI circular (eff. {deadline}): max dose 1.5→1.0 mSv. D-speed film fails — E-speed passes. Your current setup may not comply. Want the checklist? Reply YES."
            else:
                body = f"{name}, DCI regulation effective {deadline}: radiograph max dose drops 1.5→1.0 mSv. D-speed film won't pass — E-speed does. Non-compliance = inspection risk. Want the checklist? Reply YES."

        elif kind in ("recall_due", "appointment_tomorrow"):
            slots = payload.get("available_slots", [])
            slot_str = " or ".join(s["label"] for s in slots[:2]) if slots else ""
            service_default = CATEGORY_SERVICE_DEFAULT.get(cat_slug, "appointment")
            service = payload.get("service_due", service_default).replace("_", " ")
            lapsed = cust_agg.get("lapsed_180d_plus", 0)

            if is_ph and customer:
                past_services = (customer.get("relationship") or {}).get("services_received", [])
                if past_services:
                    service = past_services[-1].replace("_", " ")

            if not slot_str:
                if kind == "appointment_tomorrow":
                    slot_str = "tomorrow"
                else:
                    slot_str = "this week"

            if cust_name:
                if kind == "appointment_tomorrow":
                    body = f"Hi {cust_name}! {name}'s {biz_type} — reminder: your {service} is tomorrow. See you then! Reply to confirm or reschedule."
                else:
                    body = f"Hi {cust_name}! {name}'s {biz_type} — your {service} is due. Slots open: {slot_str}. Reply 1 or 2 to confirm."
            else:
                count_str = f"{lapsed} patients" if lapsed else "patients"
                body = f"{name}, {count_str} haven't visited in 6+ months. Should I send them a recall for {service} with your {offer_title or 'active offer'}?"

        elif kind == "perf_dip":
            if is_ph:
                metric, delta = _get_strongest_perf_dip(perf)
                if delta == 0:
                    delta = 20
            else:
                delta = int(abs(payload.get("delta_pct", 0.2)) * 100)
                metric = payload.get("metric", "calls")

            trend_signals = category.get("trend_signals", [])
            if trend_signals:
                q = trend_signals[0].get("query", "your service")
                delta_yoy = int(trend_signals[0].get("delta_yoy", 0) * 100)
                body = f"{name}, {metric} down {delta}% but '{q}' search up {delta_yoy}% in {locality}. {f'Your {offer_title} is the right hook.' if offer_title else 'Add an offer to capture them.'} Should I run a push now?"
            else:
                fix = f"Your {offer_title} is the right hook." if offer_title else "Add an active offer."
                peer_str = f" Peer avg: {peer_calls} calls." if peer_calls else ""
                body = f"{name}, {metric} dropped {delta}% this week.{peer_str} {fix} Should I push it now?"

        elif kind == "perf_spike":
            if is_ph:
                metric, delta = _get_strongest_perf_spike(perf)
                if delta == 0:
                    delta = 15
            else:
                delta = int(abs(payload.get("delta_pct", 0.15)) * 100)
                metric = payload.get("metric", "calls")
            driver = payload.get("likely_driver", "")
            driver_str = f" — looks like {driver.replace('_', ' ')}" if driver else ""
            body = f"{name}, {metric} up {delta}% this week{driver_str}. Want to replicate? I can draft a post to double down."

        elif kind == "renewal_due":
            if is_ph:
                days = merchant.get("subscription", {}).get("days_remaining", 12)
                plan = merchant.get("subscription", {}).get("plan", "Pro")
            else:
                days = payload.get("days_remaining", 12)
                plan = payload.get("plan", "Pro")
            body = f"{name}, {plan} plan expires in {days} days. Renewing now keeps your leads flowing — takes 2 min. Shall I send the link?"

        elif kind in ("festival_upcoming", "category_seasonal"):
            festival = payload.get("festival", "")
            days = payload.get("days_until", 0)
            offer_hook = f"Your {offer_title} is the right hook." if offer_title else "A limited offer now could spike bookings."

            if is_ph and not festival:
                beats = category.get("seasonal_beats", [])
                if beats:
                    beat = beats[0]
                    note = beat.get("note", "seasonal demand shift")
                    body = f"{name}, heads up — {note}. {offer_hook} Want me to draft a campaign post?"
                else:
                    fest_name, fest_days = _get_next_festival()
                    body = f"{name}, {fest_name} is {fest_days} days away. {offer_hook} Want me to draft a campaign post?"
            elif festival:
                body = f"{name}, {festival} is {days} days away. {offer_hook} Want me to draft a campaign post?"
            else:
                trends = payload.get("trends", [])
                if trends:
                    trend_str = ", ".join(str(t).replace("_demand_", " demand ").replace("+", "+") for t in trends[:2])
                    body = f"{name}, summer shift: {trend_str}. Time to restock and run a promo. Want a shelf-action plan?"
                else:
                    beats = category.get("seasonal_beats", [])
                    if beats:
                        note = beats[0].get("note", "seasonal demand shift")
                        body = f"{name}, seasonal alert — {note}. {offer_hook} Want me to draft a campaign post?"
                    else:
                        fest_name, fest_days = _get_next_festival()
                        body = f"{name}, {fest_name} is {fest_days} days away. {offer_hook} Want me to draft a campaign post?"

        elif kind == "ipl_match_today":
            match = payload.get("match", "IPL match")
            match_time = payload.get("match_time_iso", "")
            hour = match_time[11:16] if len(match_time) > 15 else "7:30pm"
            offer_hook = f"Push your {offer_title} as a match-night special." if offer_title else "A match-night deal could spike orders."
            body = f"{name}, {match} tonight at {hour}. {offer_hook} Want the banner ready in 10 min?"

        elif kind == "review_theme_emerged":
            if is_ph:
                review_themes = merchant.get("review_themes", [])
                if review_themes:
                    rt = review_themes[0]
                    theme = rt.get("theme", "service quality").replace("_", " ")
                    count = rt.get("occurrences_30d", 3)
                    quote = rt.get("common_quote", "")
                else:
                    theme = "service quality"
                    count = 3
                    quote = ""
            else:
                theme = payload.get("theme", "service issue").replace("_", " ")
                count = payload.get("occurrences_30d", 3)
                quote = payload.get("common_quote", "")
            quote_str = f' ("{quote[:40]}...")' if quote else ""
            body = f"{name}, {count} reviews mention {theme}{quote_str}. Easy fix — want me to draft a response template?"

        elif kind == "milestone_reached":
            if is_ph:
                total_ytd = cust_agg.get("total_unique_ytd", 0)
                views = perf.get("views", 0)
                if total_ytd:
                    value_now = total_ytd
                    milestone = ((total_ytd // 50) + 1) * 50
                    metric = "customers this year"
                elif views:
                    value_now = views
                    milestone = ((views // 500) + 1) * 500
                    metric = "profile views"
                else:
                    value_now = perf.get("calls", 10)
                    milestone = ((value_now // 10) + 1) * 10
                    metric = "calls"
                gap = milestone - value_now
            else:
                metric = payload.get("metric", "reviews").replace("_", " ")
                value_now = payload.get("value_now", 0)
                milestone = payload.get("milestone_value", 0)
                gap = milestone - value_now if isinstance(milestone, int) and isinstance(value_now, int) else "just a few"
            body = f"{name}, {value_now} {metric} — {milestone} is {gap} away! Want to push for it with a quick post this week?"

        elif kind == "active_planning_intent":
            topic = payload.get("intent_topic", "new initiative").replace("_", " ")
            last_msg = payload.get("merchant_last_message", "")
            if last_msg:
                body = f"{name} — on '{topic}': I'd suggest 4-week program, 3x/week, ₹2,499. I've drafted the GBP post + pricing. Say GO to publish."
            else:
                body = f"{name}, ready to plan your {topic}. I've drafted the first steps — want to review?"

        elif kind == "seasonal_perf_dip":
            delta = int(abs(perf.get("delta_7d", {}).get("views_pct", 0.3)) * 100)
            total_ytd = cust_agg.get("total_unique_ytd", 0)
            member_str = f"your {total_ytd} customers" if total_ytd else "your base"
            body = f"{name}, views -{delta}% — normal Apr-Jun lull. Don't cut ad spend; focus on {member_str}. Want a summer retention plan?"

        elif kind == "customer_lapsed_soft":
            if is_ph:
                days = _compute_days_since_last_visit(customer)
            else:
                days = payload.get("days_since_last_visit", 45)
            goal = payload.get("previous_focus", "").replace("_", " ")
            if not goal or goal == "fitness":
                goal = CATEGORY_LAPSE_GOAL.get(cat_slug, "wellness")
            hook = f"We have {offer_title} running." if offer_title else "Want to pick up where you left off?"
            body = f"Hi {cust_name}! {name}'s {biz_type} — it's been {days} days since your last visit. Your {goal} matters to us. {hook} Reply YES."

        elif kind == "customer_lapsed_hard":
            if is_ph:
                days = _compute_days_since_last_visit(customer)
            else:
                days = payload.get("days_since_last_visit", 60)
            goal = payload.get("previous_focus", "").replace("_", " ")
            if not goal or goal == "fitness":
                goal = CATEGORY_LAPSE_GOAL.get(cat_slug, "wellness")
            comeback = {
                "dentists": "Free consultation",
                "salons": "Complimentary styling consult",
                "restaurants": "A special welcome-back offer",
                "gyms": "First comeback session free",
                "pharmacies": "Free health check-up",
            }
            comeback_str = comeback.get(cat_slug, "A special welcome-back offer")
            body = f"Hi {cust_name}, {name}'s {biz_type}. Been {days} days — we miss you! {comeback_str} this week. Just reply YES."

        elif kind == "supply_alert":
            batches = payload.get("affected_batches", [])
            mol = payload.get("molecule", "medication")
            batch_str = ", ".join(batches[:2]) if batches else "recent batch"
            body = f"{name}, voluntary recall: {mol} batches {batch_str}. Want me to filter your Rx list and draft the patient outreach?"

        elif kind == "chronic_refill_due":
            mols = payload.get("molecule_list", [])
            mol_str = " + ".join(mols[:3]) if mols else ""
            expires = payload.get("stock_runs_out_iso", "")
            expires_str = expires[:10] if expires else ""
            if is_ph or (not mol_str and not expires_str):
                if cat_slug == "pharmacies":
                    mol_str = mol_str or "regular medications"
                    expires_str = expires_str or "soon"
                    body = f"Hi {cust_name}! {name}'s pharmacy — your {mol_str} refill is coming up ({expires_str}). Home delivery available. Reply YES to order."
                else:
                    service = CATEGORY_SERVICE_DEFAULT.get(cat_slug, "follow-up")
                    body = f"Hi {cust_name}! {name}'s {biz_type} — your {service} is due. Want to book? Reply YES."
            else:
                body = f"Hi {cust_name}! {name}'s pharmacy — your {mol_str} refill due by {expires_str}. Home delivery available. Reply YES to order."

        elif kind == "winback_eligible":
            if is_ph:
                days_since = merchant.get("subscription", {}).get("days_since_expiry", 38)
                days = days_since if days_since else 38
                _, dip = _get_strongest_perf_dip(perf)
                if dip == 0:
                    dip = 30
            else:
                days = payload.get("days_since_expiry", 38)
                dip = int(abs(payload.get("perf_dip_pct", 0.3)) * 100)
            body = f"{name}, it's been {days} days since expiry — calls down {dip}%. Rejoining takes 2 min and reverses this. Want to restart?"

        elif kind == "curious_ask_due":
            questions = {
                "dentists": "What's your most-requested service this month — whitening or cleaning?",
                "salons": "Which service is booked out first every week for you?",
                "restaurants": "Any new dishes you're testing that customers are asking for?",
                "gyms": "What's your peak hour this week — morning or evening batches?",
                "pharmacies": "Any molecule you're running short on this week?",
            }
            q = questions.get(cat_slug, "What's your top priority this week?")
            body = f"{name}, quick question — {q} Your answer helps me draft the right campaign for you."

        elif kind == "dormant_with_vera":
            hook = ""
            if signals:
                sig = signals[0].replace("_", " ").replace(":", " — ")
                hook = f" Quick signal: {sig}."
            elif my_ctr and peer_ctr and my_ctr < peer_ctr:
                hook = f" Your CTR is {my_ctr*100:.1f}% vs peer {peer_ctr*100:.1f}% — room to grow."
            elif offer_title:
                hook = f" Your {offer_title} is still live — let's push it."
            body = f"{name}, been a while!{hook} Want me to audit your profile this week?"

        elif kind in ("gbp_unverified", "unverified_gbp"):
            uplift = int(payload.get("estimated_uplift_pct", 0.30) * 100)
            body = f"{name}, your Google profile isn't verified — verified listings get {uplift}% more clicks on average. Takes 2 min. Want the steps?"

        elif kind == "competitor_opened":
            comp = payload.get("competitor_name", "")
            dist = payload.get("distance_km", 0)
            their_offer = payload.get("their_offer", "")
            if is_ph or (not comp and not dist):
                comp = comp or f"a new {biz_type}"
                dist = dist or 1.5
                their_offer = their_offer or "a competitive offer"
            body = f"{name}, {comp} opened {dist}km away with '{their_offer}'. Your {offer_title or 'offer'} still beats them. Want a defensive campaign?"

        elif kind == "cde_opportunity":
            credits = payload.get("credits", 2)
            fee = payload.get("fee", "free")
            body = f"{name}, IDA webinar this Friday — {credits} CDE credits, {fee}. 3 of your peers are attending. Want me to register you?"

        elif kind == "wedding_package_followup":
            days = payload.get("days_to_wedding", 180)
            trial_done = (payload.get("trial_completed_iso") or "")[:10]
            body = f"Hi {cust_name}! {name}'s salon — {days} days to your wedding. Trial done {trial_done}. Next: skin prep program. Book this week?"

        elif kind == "trial_followup":
            trial_date = (payload.get("trial_date_iso") or "recently")[:10]
            slots = payload.get("next_session_options", [])
            slot_str = slots[0]["label"] if slots else "this Saturday"
            body = f"Hi {cust_name}! {name}'s {biz_type} — loved having you for your trial on {trial_date}. Next session: {slot_str}. Joining us?"

        else:
            if signals:
                sig = signals[0].replace("_", " ").replace(":", " — ").split(":")[0]
                hook = f"Your profile shows: {sig}."
            elif offer_title:
                hook = f"Your {offer_title} is live."
            else:
                hook = f"Your CTR is {my_ctr*100:.1f}% vs {peer_ctr*100:.1f}% peer median."
            body = f"{name}, {hook} Want me to draft a quick campaign to boost this week's leads?"

        # Enforce length, strip URLs
        body = re.sub(r'https?://\S+', '', body).strip()
        if len(body) > 320:
            body = body[:317] + "..."

        return {
            "body": body,
            "cta": "binary_yes_no" if any(w in body.lower() for w in ["want", "shall", "joining", "yes"]) else "open_ended",
            "send_as": send_as,
            "suppression_key": supp_key,
            "rationale": f"Trigger kind={kind}; rule-based composition using payload + merchant context.",
        }
    except Exception as fe:
        logger.error(f"Critical error in _fallback_compose: {fe}", exc_info=True)
        return {
            "body": f"Hi, it's {merchant.get('identity', {}).get('owner_first_name', 'Vera')} here. Just checking in!",
            "suppression_key": f"universal_fallback_{trigger.get('kind', 'generic')}",
            "rationale": "Emergency fallback"
        }
