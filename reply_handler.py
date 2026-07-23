"""
reply_handler.py — Multi-turn conversation handler for /v1/reply endpoint.

Handles:
 - from_role=merchant: Vera talking to the merchant owner
 - from_role=customer: merchant_on_behalf talking to the customer (booking, recall, etc.)
 - Auto-reply detection with proper wait/end escalation
 - Intent transitions (YES/go ahead → execute immediately)
 - Opt-out / hostile → end
 - Out-of-scope → polite redirect
"""

import os
import json
import re
import logging
from typing import Optional

from google import genai
from google.genai import types as genai_types
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("vera.reply")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Auto-reply detection ──────────────────────────────────────────────────────

AUTO_REPLY_PHRASES = [
    "thank you for contacting",
    "our team will respond",
    "automated reply",
    "i am an automated",
    "this is an automated",
    "we'll get back to you",
    "we will get back to you",
    "auto-reply",
    "aapki jaankari ke liye",
    "main ek automated assistant",
    "bahut-bahut shukriya",
    "hamari team tak pahuncha",
]

# ── Opt-out phrases ───────────────────────────────────────────────────────────

OPT_OUT_PHRASES = [
    "stop messaging", "not interested", "don't message", "dont message",
    "stop sending", "band karo", "mat bhejo", "nahi chahiye", "unsubscribe",
    "bothering me", "spam", "useless bot", "why are you bothering",
    "stop these", "leave me alone", "do not contact", "remove me",
]

# Exact single-word stop commands (common WhatsApp opt-out)
STOP_WORDS = {"stop", "end", "quit", "cancel", "unsubscribe", "optout", "opt-out"}

# ── Intent action phrases (merchant says YES / let's go) ──────────────────────

INTENT_ACTION_PHRASES = [
    "let's do it", "lets do it", "ok go ahead", "go ahead", "yes do it",
    "karo", "kar do", "please do", "send it", "haan karo", "yes please send",
    "confirm", "chalega", "theek hai karo", "proceed", "ok let's do",
    "what's next", "whats next",
]

# ── Out-of-scope topics ───────────────────────────────────────────────────────

OUT_OF_SCOPE = [
    "gst filing", "income tax", "court case", "legal advice", "police",
    "bank loan", "insurance claim", "government grant", "visa application",
    "passport", "emi calculator",
]

# ── System prompts ────────────────────────────────────────────────────────────

MERCHANT_REPLY_SYSTEM = """You are Vera, magicpin's merchant AI. You are replying to the MERCHANT OWNER's message.

RULES:
1. Reply in <= 320 chars. No URLs. Start with merchant's first name.
2. One CTA: open_ended | binary_yes_no | binary_confirm_cancel | none
3. If merchant said YES/go ahead/let's do it/confirm/proceed → EXECUTE the action IMMEDIATELY. No more questions. Give a concrete deliverable or confirmation with specific numbers.
4. If out-of-scope → 1 polite line, redirect back to original topic.
5. If hostile/STOP/not interested → action=end, NO body.
6. Use EXACT numbers from context (patient count, offer price, CTR, calls, deadline).
7. After executing an intent: pivot to the NEXT action with a specific prompt.

EXAMPLE GOOD FOLLOW-UPS:
- After YES on research: "Sending draft WhatsApp now. It goes to 124 patients: '3-month cleaning recall is clinically proven. Book: Wed 5 Nov or Thu 6 Nov.' Should I post it on Google too?"
- After YES on campaign: "Campaign ready for 78 lapsed patients — 'Your ₹299 cleaning is waiting, 3 slots left this week.' CONFIRM to send now."
- After hostile: {"action": "end"}

Return ONLY valid JSON:
{"action": "send|wait|end", "body": "<only if send>", "cta": "<type>", "wait_seconds": <only if wait>, "rationale": "<brief>"}"""  


CUSTOMER_REPLY_SYSTEM = """You are replying on behalf of a merchant to their CUSTOMER.

RULES:
1. Reply in <= 320 chars. No URLs.
2. Address the CUSTOMER by their first name ONLY — NEVER use the merchant's name as the recipient.
3. You are the CLINIC/SHOP speaking to the customer, not Vera speaking to the merchant.
4. If customer confirms slot → confirm warmly with exact slot, what to bring, reminder note.
5. If customer declines → offer to reschedule with 2 alternative times.
6. If customer asks about price/service → give specific price from offer.
7. Warm, friendly, not salesy. Use emoji sparingly (1 max).
8. Hindi code-mix if customer language includes 'hi'.

EXAMPLE:
Customer: "Yes please book me for Wed 5 Nov, 6pm"
Good reply: "Perfect Priya! ✅ Booked for Wed 5 Nov, 6pm. Please bring your last X-ray if you have one. We'll WhatsApp a reminder tomorrow. See you then! 🙏"

Return ONLY valid JSON:
{"action": "send|end", "body": "<message <=320 chars>", "cta": "<open_ended|binary_yes_no|none>", "rationale": "<brief>"}"""  



# ── Detection helpers ─────────────────────────────────────────────────────────

def _is_auto_reply(message: str) -> bool:
    msg_lower = message.lower()
    return any(phrase in msg_lower for phrase in AUTO_REPLY_PHRASES)


def _is_opt_out(message: str) -> bool:
    msg_stripped = message.strip().lower()
    # Exact single-word stop commands (WhatsApp standard opt-out)
    if msg_stripped in STOP_WORDS:
        return True
    # Phrase-based opt-out
    return any(phrase in msg_stripped for phrase in OPT_OUT_PHRASES)


def _is_intent_action(message: str) -> bool:
    msg_lower = message.lower()
    return any(phrase in msg_lower for phrase in INTENT_ACTION_PHRASES)


def _is_out_of_scope(message: str) -> bool:
    msg_lower = message.lower()
    return any(phrase in msg_lower for phrase in OUT_OF_SCOPE)


def _count_auto_replies_in_history(history: list) -> int:
    """Count consecutive auto-replies at the end of conversation history."""
    count = 0
    for turn in reversed(history):
        if turn.get("from") in ("merchant", "customer") and _is_auto_reply(turn.get("msg", "")):
            count += 1
        else:
            break
    return count


def _get_last_vera_topic(history: list) -> str:
    for turn in reversed(history):
        if turn.get("from") == "vera":
            msg = turn.get("msg", "")
            # Extract the topic from the last Vera message
            if "JIDA" in msg or "research" in msg.lower():
                return "the research digest"
            if "recall" in msg.lower() or "patient" in msg.lower():
                return "the patient recall"
            if "offer" in msg.lower() or "campaign" in msg.lower():
                return "the campaign"
            if "review" in msg.lower():
                return "the review response"
            return "what we were discussing"
    return "your magicpin growth"


# ── LLM reply ────────────────────────────────────────────────────────────────

def _llm_reply(
    message: str,
    history: list,
    merchant_ctx: Optional[dict],
    category_ctx: Optional[dict],
    customer_ctx: Optional[dict],
    from_role: str,
    mode: str = "continue",
) -> dict:
    if not GEMINI_API_KEY:
        return None  # Signal caller to use fallback

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        system = CUSTOMER_REPLY_SYSTEM if from_role == "customer" else MERCHANT_REPLY_SYSTEM

        hist_str = json.dumps(history[-6:], ensure_ascii=False, indent=2) if history else "[]"
        merch_str = json.dumps(merchant_ctx, ensure_ascii=False, indent=2) if merchant_ctx else "{}"
        cat_str = json.dumps({"voice": category_ctx.get("voice", {}), "peer_stats": category_ctx.get("peer_stats", {})}, ensure_ascii=False) if category_ctx else "{}"
        cust_str = json.dumps(customer_ctx, ensure_ascii=False, indent=2) if customer_ctx else "{}"

        user_prompt = f"""MODE: {mode}
FROM_ROLE: {from_role}
MERCHANT CONTEXT: {merch_str}
CATEGORY CONTEXT: {cat_str}
CUSTOMER CONTEXT: {cust_str}
CONVERSATION HISTORY (last 6 turns): {hist_str}
INCOMING MESSAGE: {message}

Reply as instructed. Return ONLY valid JSON."""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.4,
                max_output_tokens=400,
            ),
        )
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)

        # Validate body length
        if "body" in parsed and len(parsed["body"]) > 320:
            parsed["body"] = parsed["body"][:317] + "..."

        # Remove URLs
        if "body" in parsed:
            parsed["body"] = re.sub(r"https?://\S+", "", parsed["body"]).strip()

        parsed["rationale"] = parsed.get("rationale", "") + f" [LLM: {from_role} mode={mode}]"
        return parsed

    except Exception as e:
        err = str(e)[:60]
        logger.warning(f"LLM reply failed: {err}")
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            return {"_rate_limited": True}
        return None


# ── Customer-role fallback responses ─────────────────────────────────────────

def _handle_customer_reply(
    message: str,
    customer_ctx: Optional[dict],
    merchant_ctx: Optional[dict],
    history: list,
) -> dict:
    """
    Handle from_role=customer messages.
    Bot acts as merchant_on_behalf — confirming bookings, answering questions.
    NEVER addresses merchant name; always uses customer name.
    """
    msg_lower = message.lower()
    cust_name = (customer_ctx or {}).get("identity", {}).get("name", "")
    first_name = cust_name.split()[0] if cust_name else "there"

    merchant_name = (merchant_ctx or {}).get("identity", {}).get("name", "the clinic")
    offers = [o["title"] for o in (merchant_ctx or {}).get("offers", []) if o.get("status") == "active"]
    offer = offers[0] if offers else ""

    # Booking confirmation ("yes", "book me", "confirm", slot numbers)
    if any(w in msg_lower for w in ["yes", "book", "confirm", "1", "2", "wed", "thu", "ok", "sure", "please"]):
        # Find what slot they replied to
        if "wed" in msg_lower or "1" in msg_lower:
            slot = "Wednesday"
        elif "thu" in msg_lower or "2" in msg_lower:
            slot = "Thursday"
        else:
            slot = "your preferred slot"

        body = f"Perfect, {first_name}! ✅ Booked for {slot}. {merchant_name} will see you then. Reply if you need to reschedule."
        if offer:
            body = f"Perfect, {first_name}! ✅ Booked for {slot}. Your {offer} is confirmed. We'll send a reminder the day before. 🙏"
        return {
            "action": "send",
            "body": body[:320],
            "cta": "none",
            "rationale": f"Customer confirmed booking. Confirming slot for {first_name} as merchant_on_behalf.",
        }

    # Decline / can't make it
    if any(w in msg_lower for w in ["can't", "cant", "busy", "no", "reschedule", "another", "different"]):
        body = f"No worries, {first_name}! Let us know what time works best for you and we'll find a slot. 😊"
        return {
            "action": "send",
            "body": body[:320],
            "cta": "open_ended",
            "rationale": "Customer declined; offering to reschedule.",
        }

    # Question about price / service
    if any(w in msg_lower for w in ["price", "cost", "how much", "charges", "kitna", "fee"]):
        body = f"Hi {first_name}! {f'Our {offer} is the best value for you.' if offer else 'Please call us for current pricing.'} Any other questions? 😊"
        return {
            "action": "send",
            "body": body[:320],
            "cta": "open_ended",
            "rationale": "Customer asked about pricing.",
        }

    # Generic customer reply — ask for slot preference
    body = f"Thanks {first_name}! We'll arrange it. What time works best for you — morning or evening? We have slots this week. 😊"
    return {
        "action": "send",
        "body": body[:320],
        "cta": "open_ended",
        "rationale": "Generic customer reply; asking for slot preference.",
    }


# ── Merchant-role keyword fallback ────────────────────────────────────────────

# Each entry: (keyword_list, is_word_boundary_required, reply_fn)
KEYWORD_REPLIES = [
    (["call", "drop", "down", "fell", "gir", "kam", "low", "dip", "decrease", "decline"], False,
     lambda m, n, o, cat: f"{n}, calls dipping usually means profile visibility issue — CTR below peer median. Quick fix: update 1 photo + activate your offer. Should I draft a post now?"),
    (["review", "rating", "feedback", "complaint", "bad review"], False,
     lambda m, n, o, cat: f"{n}, reviews drive 40% of discovery clicks on magicpin. I can draft response templates for your most common themes. Want me to pull your last 10 reviews? Reply YES."),
    (["offer", "discount", "deal", "promo", "campaign", "push"], False,
     lambda m, n, o, cat: f"{n}, {'your ' + o + ' is the right hook.' if o else 'add an active offer first.'} Want me to build a WhatsApp campaign around it? Just say GO."),
    (["diwali", "festival", "holi", "eid", "christmas", "navratri", "puja", "rakhi"], False,
     lambda m, n, o, cat: f"{n}, festival window = highest footfall of the year. {'Your ' + o + ' is the hook.' if o else 'Set up an offer first.'} Draft a campaign post + WhatsApp blast? Reply YES."),
    (["ipl", "match", "cricket", "tonight"], False,
     lambda m, n, o, cat: f"{n}, match nights drive +18% covers on weeknights. Push your offer as a match-night special tonight. Want me to draft the banner? Reply YES."),
    (["recall", "appointment", "patient", "slot", "book", "schedule"], False,
     lambda m, n, o, cat: f"{n}, I can send recall reminders to your lapsed patients. Should I draft the WhatsApp with 2 slot options? Reply YES."),
    (["photo", "image", "picture", "google", "gbp", "profile", "verify"], False,
     lambda m, n, o, cat: f"{n}, verified Google profiles get 30% more clicks. Shops with 10+ photos see 2x CTR. Want me to walk you through the 2-min GBP process? Reply YES."),
    (["footfall", "traffic", "visitor", "walk-in"], False,
     lambda m, n, o, cat: f"{n}, footfall is driven by profile photos + active offer + fresh posts. You're missing posts (22 days stale). Want me to draft one now? Reply YES."),
    (["competitor", "competition", "other shop", "nearby", "neighbour", "new salon", "new clinic"], False,
     lambda m, n, o, cat: f"{n}, best defense: strong active offer + fresh photos. {'Your ' + o + ' is your moat.' if o else 'Set up an offer.'} Let me draft a campaign to stand out. Reply YES."),
    (["x-ray", "xray", "equipment", "setup", "audit", "compliance", "regulation", "dci", "checklist"], False,
     lambda m, n, o, cat: f"{n}, I can help prep your compliance checklist for the DCI regulation (effective Dec 2026). Max dose drops 1.5→1.0 mSv; D-speed film won't pass. Want the full checklist? Reply YES."),
    (["abstract", "paper", "research", "journal", "jida", "study", "draft"], False,
     lambda m, n, o, cat: f"{n}, happy to pull that. The JIDA Oct 2026 (p.14, n=2,100) key finding: 3-month fluoride recall cuts caries 38% better than 6-month. Draft the patient WhatsApp now? Reply YES."),
]

def _keyword_reply_merchant(message: str, merchant_ctx: Optional[dict], category_ctx: Optional[dict]) -> Optional[dict]:
    """Match keywords and return a specific, context-aware reply. Returns None if no match."""
    name = (merchant_ctx or {}).get("identity", {}).get("owner_first_name", "there")
    offers = [o["title"] for o in (merchant_ctx or {}).get("offers", []) if o.get("status") == "active"]
    offer = offers[0] if offers else ""
    cat = (merchant_ctx or {}).get("category_slug", "")
    msg_lower = message.lower()

    for keywords, _, reply_fn in KEYWORD_REPLIES:
        if any(kw in msg_lower for kw in keywords):
            body = reply_fn(message, name, offer, cat)
            matched = [k for k in keywords if k in msg_lower][0]
            return {
                "action": "send",
                "body": body[:320],
                "cta": "binary_yes_no",
                "rationale": f"Keyword-matched reply for '{matched}'; context-aware fallback without LLM.",
            }
    return None


# ── Main handler ──────────────────────────────────────────────────────────────

def handle_reply(
    conversation_id: str,
    merchant_id: str,
    customer_id: Optional[str],
    from_role: str,
    message: str,
    turn_number: int,
    conversation_history: list,
    merchant_ctx: Optional[dict],
    category_ctx: Optional[dict],
    customer_ctx: Optional[dict] = None,
) -> dict:
    """
    Handle incoming reply and produce next action.
    Returns: {action, body?, cta?, wait_seconds?, rationale}
    """

    # ── CUSTOMER ROLE: completely separate path ───────────────────────────────
    if from_role == "customer":
        # Try LLM first
        if GEMINI_API_KEY:
            llm = _llm_reply(message, conversation_history, merchant_ctx, category_ctx, customer_ctx, from_role="customer")
            if llm and not llm.get("_rate_limited"):
                return llm

        # Fallback: rule-based customer response
        return _handle_customer_reply(message, customer_ctx, merchant_ctx, conversation_history)

    # ── MERCHANT ROLE ─────────────────────────────────────────────────────────

    # 1. Opt-out / hostile — check FIRST, no other processing
    if _is_opt_out(message):
        return {
            "action": "end",
            "rationale": "Merchant explicitly opted out or expressed frustration. Closing conversation and suppressing future sends.",
        }

    # 2. Auto-reply detection — WAIT immediately on first detection, NEVER send
    if _is_auto_reply(message):
        prior_auto = _count_auto_replies_in_history(conversation_history)
        if prior_auto == 0:
            # First auto-reply: one gentle prompt, then wait
            return {
                "action": "send",
                "body": "Looks like an auto-reply 😊 When the owner sees this, just reply 'Yes' to continue — I'll keep this warm.",
                "cta": "binary_yes_no",
                "rationale": "Detected auto-reply (canned phrasing). One prompt to flag for owner.",
            }
        elif prior_auto == 1:
            return {
                "action": "wait",
                "wait_seconds": 86400,
                "rationale": "Same auto-reply twice in a row. Owner not at phone — backing off 24h.",
            }
        else:
            return {
                "action": "end",
                "rationale": "Auto-reply 3+ times in a row. No real engagement signal. Closing conversation.",
            }

    # 3. Intent action — execute immediately, skip re-qualification
    if _is_intent_action(message):
        if GEMINI_API_KEY:
            llm = _llm_reply(message, conversation_history, merchant_ctx, category_ctx, None, from_role="merchant", mode="execute_intent")
            if llm and not llm.get("_rate_limited"):
                return llm

        name = (merchant_ctx or {}).get("identity", {}).get("owner_first_name", "")
        offers = [o["title"] for o in (merchant_ctx or {}).get("offers", []) if o.get("status") == "active"]
        offer = offers[0] if offers else ""
        high_risk = (merchant_ctx or {}).get("customer_aggregate", {}).get("high_risk_adult_count", 0)
        scope = f"{high_risk} high-risk patients" if high_risk else "your patient list"
        body = f"Great, {name}! Drafting now — 90 seconds. "
        if offer:
            body += f"Pushing '{offer}' to {scope}. Reply CONFIRM to send."
        else:
            body += "I'll send the draft shortly. Reply CONFIRM to proceed."
        return {
            "action": "send",
            "body": body[:320],
            "cta": "binary_confirm_cancel",
            "rationale": "Merchant committed to action; executing immediately without re-qualification.",
        }

    # 4. Out-of-scope redirect
    if _is_out_of_scope(message):
        last_topic = _get_last_vera_topic(conversation_history)
        body = f"I'll leave that to a specialist — outside my scope. Back to {last_topic} — want me to proceed?"
        return {
            "action": "send",
            "body": body[:320],
            "cta": "binary_yes_no",
            "rationale": "Out-of-scope request politely declined. Redirected to original topic.",
        }

    # 5. LLM for everything else
    if GEMINI_API_KEY:
        llm = _llm_reply(message, conversation_history, merchant_ctx, category_ctx, None, from_role="merchant", mode="continue")
        if llm and not llm.get("_rate_limited"):
            return llm

    # 6. Smart keyword fallback
    kw = _keyword_reply_merchant(message, merchant_ctx, category_ctx)
    if kw:
        return kw

    # 7. Last-resort contextual fallback
    name = (merchant_ctx or {}).get("identity", {}).get("owner_first_name", "there")
    offers = [o["title"] for o in (merchant_ctx or {}).get("offers", []) if o.get("status") == "active"]
    offer_part = f" I can use your '{offers[0]}' as the hook." if offers else ""
    body = f"{name}, got it!{offer_part} What's your main goal right now — more footfall, better reviews, or reactivating lapsed customers?"
    return {
        "action": "send",
        "body": body[:320],
        "cta": "open_ended",
        "rationale": "Generic contextual fallback; no keyword match, LLM unavailable.",
    }
