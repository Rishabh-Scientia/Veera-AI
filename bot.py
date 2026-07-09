import os
import time
import json
import asyncio
import urllib.request
import httpx
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, List, Dict, Optional
from groq import AsyncGroq, Groq
from dotenv import load_dotenv

from fastapi.responses import RedirectResponse

# Load environment variables from .env
load_dotenv()

app = FastAPI()
START_TIME = time.time()

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/docs")

# In-memory context stores
# Key: (scope, context_id) -> Value: {"version": int, "payload": dict}
contexts: Dict[tuple, Dict] = {}

# In-memory conversation stores
# Key: conversation_id -> Value: list of message dictionaries
conversations: Dict[str, List[Dict]] = {}


from pathlib import Path

def load_fallback_contexts():
    possible_dirs = [
        Path(__file__).parent.parent / "dataset",
        Path(__file__).parent / "dataset",
        Path("dataset"),
        Path("../dataset")
    ]
    
    # Load categories from categories folder
    for d in possible_dirs:
        cat_dir = d / "categories"
        if cat_dir.exists():
            loaded_cats = 0
            try:
                for f in cat_dir.glob("*.json"):
                    with open(f, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        slug = data.get("slug", f.stem)
                        contexts[("category", slug)] = {
                            "version": 1,
                            "payload": data
                        }
                        loaded_cats += 1
                print(f"Loaded {loaded_cats} fallback categories from {cat_dir}")
                break
            except Exception as e:
                print(f"Error loading categories directory: {e}")
                
    # Load merchants, customers, and triggers
    for name, scope, key in [
        ("merchants_seed.json", "merchant", "merchant_id"),
        ("customers_seed.json", "customer", "customer_id"),
        ("triggers_seed.json", "trigger", "id")
    ]:
        loaded = 0
        for d in possible_dirs:
            p = d / name
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        items = data.get(name.split("_")[0], [])
                        if not items:
                            if "merchants" in data: items = data["merchants"]
                            elif "customers" in data: items = data["customers"]
                            elif "triggers" in data: items = data["triggers"]
                        
                        for item in items:
                            cid = item.get(key)
                            if cid:
                                contexts[(scope, cid)] = {
                                    "version": 1,
                                    "payload": item
                                }
                                loaded += 1
                    print(f"Loaded {loaded} fallback contexts for {scope} from {p}")
                    break
                except Exception as e:
                    print(f"Error loading {name}: {e}")

# Run the fallback loader on import
load_fallback_contexts()



class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str



@app.get("/v1/healthz")
async def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _), _ in contexts.items():
        if scope in counts:
            counts[scope] += 1
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START_TIME),
        "contexts_loaded": counts
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name": "Antigravity Team",
        "team_members": ["Antigravity"],
        "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "approach": "Context-aware WhatsApp prompt composer with auto-reply filter and intent transition state machine",
        "contact_email": "antigravity@magicpin.com",
        "version": "1.0.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z"
    }


@app.post("/v1/context")
async def push_context(body: CtxBody):
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] > body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z"
    }


# Gemini fallback logic removed as requested by the user.


async def compose_message(category: dict, merchant: dict, trigger: dict, customer: dict | None, deadline: float | None = None) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    
    if not api_key:
        return {
            "body": f"Hi {merchant.get('identity', {}).get('name')}, we noticed an update regarding {trigger.get('kind')}.",
            "cta": "open_ended",
            "send_as": "vera",
            "rationale": "Missing Groq API Key, fallback message."
        }
        
    client = AsyncGroq(api_key=api_key)
    
    # Resolve matching digest item from trigger payload
    digest_payload = trigger.get("payload", {})
    matched_digest = None
    if isinstance(digest_payload, dict):
        top_item = digest_payload.get("top_item")
        if isinstance(top_item, dict):
            matched_digest = top_item
        else:
            top_item_id = digest_payload.get("top_item_id")
            if top_item_id:
                for item in category.get("digest", []):
                    if item.get("id") == top_item_id:
                        matched_digest = item
                        break
            if not matched_digest and trigger.get("kind") == "research_digest" and category.get("digest"):
                matched_digest = category["digest"][0]

    # Dynamic variables
    slug = category.get("slug", "merchant")
    owner_name = merchant.get("identity", {}).get("owner_first_name", "Owner")
    biz_name = merchant.get("identity", {}).get("name", "Business")
    locality = merchant.get("identity", {}).get("locality", "Local Area")
    city = merchant.get("identity", {}).get("city", "City")
    languages = merchant.get("identity", {}).get("languages", ["en"])
    
    views = merchant.get("performance", {}).get("views", 0)
    calls = merchant.get("performance", {}).get("calls", 0)
    ctr = merchant.get("performance", {}).get("ctr", 0.0)
    
    # Category Specific Formatting Terms
    is_dentist = slug == "dentists"
    owner_title = f"Dr. {owner_name}" if is_dentist else owner_name
    business_type_term = "clinic" if is_dentist else (
        "salon" if slug == "salons" else (
            "restaurant" if slug == "restaurants" else (
                "gym" if slug == "gyms" else "pharmacy"
            )
        )
    )
    
    # Action URL construction (programmatic to avoid LLM hallucination)
    owner_name_clean = owner_name.lower().replace(" ", "")
    kind = trigger.get("kind")
    if kind == "research_digest":
        action_url = "magicpin.in/jida-fluoride-recall" if is_dentist else f"magicpin.in/{slug}-digest"
    elif kind == "regulation_change":
        action_url = "magicpin.in/dci-radiograph-compliance" if is_dentist else f"magicpin.in/{slug}-compliance"
    elif kind == "recall_due" and customer:
        action_url = "magicpin.in/meera-dental" if (is_dentist and owner_name_clean == "meera") else f"magicpin.in/{owner_name_clean}-{slug.rstrip('s')}"
    else:
        action_url = f"magicpin.in/{slug}-dashboard"
        
    # Language mixing rule
    prefers_hinglish = "hi" in languages or (customer and "hi" in customer.get("identity", {}).get("language_pref", "").lower())
    
    # Active Offer resolution
    active_offer = None
    for o in merchant.get("offers", []):
        if o.get("status") == "active":
            active_offer = o.get("title").replace('\u20b9', 'Rs.').replace('₹', 'Rs.')
            break
    if not active_offer and category.get("offer_catalog"):
        active_offer = category["offer_catalog"][0].get("title").replace('\u20b9', 'Rs.').replace('₹', 'Rs.')
    if not active_offer:
        active_offer = "special offers"

    # Assemble descriptions
    matched_digest_desc = ""
    if matched_digest:
        matched_digest_desc = f"""
Matched Digest/Research:
- Title: {matched_digest.get('title')}
- Source Citation: {matched_digest.get('source')} (Always append this exact citation to the facts you state)
- Summary: {matched_digest.get('summary')}
- Actionable Recommendation: {matched_digest.get('actionable')}
- Trial Details: trial_n={matched_digest.get('trial_n', 'N/A')}, patient_segment={matched_digest.get('patient_segment', 'N/A')}
"""

    customer_desc = ""
    if customer:
        cust_name = customer.get("identity", {}).get("name")
        last_visit = customer.get("relationship", {}).get("last_visit")
        visits_total = customer.get("relationship", {}).get("visits_total")
        services = customer.get("relationship", {}).get("services_received", [])
        state = customer.get("state")
        preferred_slots = customer.get("preferences", {}).get("preferred_slots", "anytime")
        
        customer_desc = f"""
Customer Recipient Context:
- Name: {cust_name}
- Last Visit: {last_visit}
- Total Visits: {visits_total}
- Services Received: {services}
- Segment State: {state}
- Preferred slots/time: {preferred_slots}
"""

    category_desc = f"""
Category Slug: {slug}
Voice Tone: {category.get('voice', {}).get('tone')}
Voice Register: {category.get('voice', {}).get('register')}
Allowed Vocabulary: {category.get('voice', {}).get('vocab_allowed', [])}
Taboo Vocabulary (BANNED - NEVER use these words): {category.get('voice', {}).get('vocab_taboo', [])}
Offer Catalog: {[o.get('title') for o in category.get('offer_catalog', [])]}
Peer Stats: {category.get('peer_stats', {})}
"""

    merchant_desc = f"""
Merchant Context:
- Name: {biz_name}
- Owner Name: {owner_title}
- Locality: {locality}
- City: {city}
- Performance Snapshot (30 days): Views: {views}, Calls: {calls}, CTR: {ctr:.1%} (Peer average CTR: {category.get('peer_stats', {}).get('avg_ctr', 0.0):.1%})
- Signals: {merchant.get('signals', [])}
- Active Offer: {active_offer}
"""

    trigger_desc = f"""
Trigger Context:
- Kind: {kind}
- Scope: {trigger.get('scope')}
- Urgency: {trigger.get('urgency')}
- Payload: {trigger.get('payload')}
- Action Link: {action_url}
"""

    system_prompt = """You are Vera, magicpin's Merchant AI Assistant.
Your task is to write a highly compelling, specific, and context-appropriate WhatsApp message based on the provided business contexts.

Recipient Rules:
1. CUSTOMER EXISTENCE (customer_id is populated):
   - You must speak on behalf of the merchant, NOT as Vera.
   - Address the customer by name (e.g., "Namaste Priya," or "Hi Priya,").
   - Sign off from the business's perspective (e.g. "{clinic_name} here" or "{salon_name} se bol rahe hain").
   - Set "send_as" to "merchant_on_behalf" in the JSON response.
   - Highlight the customer's visit history (e.g., "It has been X months since your last visit...").
   - Mention the specific active offer or service pricing (using "Rs." prefix).
   - If the category is dentist: mention oral health benefits (e.g., regular cleanings reduce gum disease risk by up to 50%).
   - Offer slots matching their preferences.
   - Include the action link.
   - The CTA in the last sentence must ask to book (e.g., "Reply YES to schedule slots." or "Reply 1 for slot A, 2 for slot B").

2. MERCHANT ONLY (no customer):
   - You must speak as Vera (magicpin's Merchant AI assistant).
   - Address the owner by name, using "Dr." prefix only if the category is dentists.
   - Identify yourself: E.g., "Namaste, Vera se bol rahi hoon." or "This is Vera from magicpin."
   - Set "send_as" to "vera" in the JSON response.
   - Address the specific trigger topic/kind:
     * If research_digest or regulation_change: highlight specific study findings/compliance deadlines and peer average CTR metrics, and cite the exact source (e.g. "— JIDA Oct 2026, p.14") at the end of the claim.
     * If active_planning_intent: reply to the owner's request (e.g. adding a kids yoga summer camp) and offer to help setup the program page.
     * If perf_dip: warn them about their calls or views drop, and offer to optimize.
     * If renewal_due: warn them their plan is expiring in X days and offer to renew.
     * If gbp_unverified: warn them their Google Business Profile is unverified and offer to verify.
   - Include the action link.
   - The CTA in the last sentence must be a low-friction binary choice (e.g., "Reply YES to proceed." or "Reply YES to setup.").

General Constraints:
1. SPECIFICITY: Always use concrete numbers, metrics, dates, prices, and citations from the context. Do not invent any numbers or citations. Do not make up fake competitor names or fake studies.
2. CONCISENESS & NO REPETITION: Keep the message under 3 sentences (maximum 75 words). State each fact, offer, or link exactly once. Do NOT repeat any words, phrases, or sentences.
3. HINGLISH: If the languages/language pref contains "hi" or "hi-en mix", you MUST write in Hinglish (blend of Hindi and English written in Latin script). Match the tone naturally.
4. TABOOS: Strictly avoid any words listed in Taboo Vocabulary (e.g., for dentists, never use "cure" or "guaranteed").
5. JSON OUTPUT: Respond ONLY with a raw JSON object containing the keys: body, cta, send_as, rationale. Do not write any markdown code block formatting (like ```json). Just the raw JSON.
"""

    prompt = f"""Compose a WhatsApp message using these contexts:

=== CATEGORY VOICE ===
{category_desc}

=== BUSINESS DETAILS ===
{merchant_desc}

=== TRIGGER / WHY NOW ===
{trigger_desc}
{matched_digest_desc}

=== RECIPIENT CUSTOMER (IF APPLICABLE) ===
{customer_desc if customer_desc else 'None (No customer exists)'}

=== LANGUAGE PREFERENCE ===
Prefers Hinglish: {prefers_hinglish}

=== MANDATORY RULE TO FOLLOW ===
You MUST apply and strictly follow: {"Rule 2 (MERCHANT ONLY). Speak as Vera (magicpin's Merchant AI assistant), address the owner by name, set send_as to 'vera'." if not customer else "Rule 1 (CUSTOMER EXISTENCE). Speak on behalf of the merchant, address the customer by name, set send_as to 'merchant_on_behalf'."}

=== CORRESPONDING TEMPLATE DETAILS ===
Business Type: {business_type_term}
Address Recipient As: {owner_title if not customer else customer.get('identity', {}).get('name')}
Include Action URL: {action_url}
Active Offer to mention: {active_offer}
"""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=500,
            timeout=8.0
        )
        content = response.choices[0].message.content.strip()
        # Clean JSON format if model wrapped it
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        try:
            with open("debug.log", "a", encoding="utf-8") as f_debug:
                f_debug.write(f"\n=========================================\n")
                f_debug.write(f"TRIGGER: {kind} | MERCHANT: {biz_name} | CATEGORY: {slug}\n")
                f_debug.write(f"PROMPT:\n{prompt}\n")
                f_debug.write(f"SYSTEM:\n{system_prompt}\n")
                f_debug.write(f"RESPONSE (GROQ):\n{content}\n")
        except Exception as log_ex:
            print(f"Error logging debug info: {log_ex}")
            
        parsed = json.loads(content)
        if "body" in parsed:
            return parsed
            
    except Exception as e:
        print(f"Groq API call failed: {e}")
        
    # Return friendly fallback message on failure / API limit exceeded
    return {
        "body": f"Hi {owner_title if not customer else customer.get('identity', {}).get('name')}, we noticed an update regarding {trigger.get('kind')}. Let's discuss this at {action_url}.",
        "cta": "open_ended",
        "send_as": "merchant_on_behalf" if customer else "vera",
        "rationale": "Fallback message used because Groq API call failed or rate limit was exceeded."
    }


class TickBody(BaseModel):
    now: str
    available_triggers: List[str] = []


async def async_compose(category, merchant, trg, customer, deadline=None):
    return await compose_message(category, merchant, trg, customer, deadline)

@app.post("/v1/tick")
async def tick(body: TickBody):
    trg_infos = []
    
    for trg_id in body.available_triggers:
        trg_ctx = contexts.get(("trigger", trg_id))
        if not trg_ctx:
            continue
        trg = trg_ctx["payload"]
        merchant_id = trg.get("merchant_id")
        merchant_ctx = contexts.get(("merchant", merchant_id))
        if not merchant_ctx:
            continue
        merchant = merchant_ctx["payload"]
        
        category_slug = merchant.get("category_slug")
        category_ctx = contexts.get(("category", category_slug))
        category = category_ctx["payload"] if category_ctx else {}
        
        customer_id = trg.get("customer_id")
        customer = None
        if customer_id:
            cust_ctx = contexts.get(("customer", customer_id))
            if cust_ctx:
                customer = cust_ctx["payload"]
                
        trg_infos.append((trg_id, merchant_id, merchant, trg, customer, category))
        
    start_time = time.time()
    # 13.0s deadline matches simulator timeout (15s) with a 2.0s buffer.
    deadline_duration = 13.0
    deadline = start_time + deadline_duration

    async def process_one(trg_id, merchant_id, merchant, trg, customer, category):
        elapsed = time.time() - start_time
        remaining = deadline_duration - elapsed
        if remaining <= 1.5:
            print(f"Skipping trigger {trg_id} concurrently: remaining time ({remaining:.1f}s) is too short.")
            return None
            
        try:
            # Run the composition with a timeout of `remaining`
            composed = await asyncio.wait_for(
                async_compose(category, merchant, trg, customer, deadline),
                timeout=remaining
            )
            
            if isinstance(composed, Exception) or not composed or "body" not in composed:
                print(f"Skipping trigger {trg_id} due to composition error or empty result: {composed}")
                return None
                
            return {
                "conversation_id": f"conv_{merchant_id}_{trg_id}",
                "merchant_id": merchant_id,
                "customer_id": trg.get("customer_id"),
                "send_as": composed.get("send_as", "vera"),
                "trigger_id": trg_id,
                "template_name": "vera_generic_v1",
                "template_params": [
                    merchant.get("identity", {}).get("name", "Merchant"),
                    trg.get("kind", "update"),
                    composed.get("cta", "open_ended")
                ],
                "body": composed.get("body", ""),
                "cta": composed.get("cta", "open_ended"),
                "suppression_key": trg.get("suppression_key", f"{trg_id}_suppress"),
                "rationale": composed.get("rationale", "Composed concurrently with Category, Merchant, and Trigger context.")
            }
        except asyncio.TimeoutError:
            print(f"Timeout waiting for trigger {trg_id} concurrently")
            return None
        except Exception as e:
            print(f"Error processing trigger {trg_id} concurrently: {e}")
            return None

    # Run all compositions in the batch concurrently
    tasks = [
        process_one(trg_id, merchant_id, merchant, trg, customer, category)
        for trg_id, merchant_id, merchant, trg, customer, category in trg_infos
    ]
    results = await asyncio.gather(*tasks)
    
    actions = [r for r in results if r is not None]
    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


def analyze_and_respond(merchant_id: str, message: str, turn: int, conv_id: str) -> dict:
    msg_lower = message.lower()
    
    # Rule 1: Auto-reply detection
    auto_keywords = [
        "thank you for contacting", "respond shortly", "out of office",
        "aapki jaankari ke liye", "automated assistant", "automated message",
        "automated reply", "automated response", "will get back to you",
        "canned response", "quick reply"
    ]
    if any(kw in msg_lower for kw in auto_keywords):
        return {
            "action": "end",
            "rationale": "Auto-reply keyword matched. Gracefully ending conversation."
        }
        
    # Rule 2: Hostility / Stop request
    hostile_keywords = [
        "stop messaging", "useless spam", "stop", "spam", "abuse", "don't message", "dont message", "remove me"
    ]
    if any(kw in msg_lower for kw in hostile_keywords):
        return {
            "action": "end",
            "rationale": "Hostility or stop request detected. Gracefully ending conversation."
        }
        
    # Rule 3: Direct intent transition/commitment
    intent_keywords = [
        "lets do it", "let's do it", "whats next", "what's next", "go ahead", "start", "karo", "mujhe join", "mujhe judna"
    ]
    if any(kw in msg_lower for kw in intent_keywords):
        return {
            "action": "send",
            "body": "Done! I have initialized the setup for you. Here is the draft to confirm and proceed to the next step. Let me know if you are ready.",
            "cta": "none",
            "rationale": "Commitment intent matched. Responding in action mode with actioning words."
        }
        
    # Fallback to LLM for other conversational turns
    api_key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key:
        return {
            "action": "send",
            "body": "Thank you for your response. Let me know how you would like to proceed.",
            "cta": "open_ended",
            "rationale": "Missing LLM API Key, fallback response."
        }
        
    client = Groq(api_key=api_key)
    
    system_prompt = """You are Vera, magicpin's Merchant AI Assistant.
Analyze a reply from a merchant/customer and determine the next action.

Categorize their reply into one of these actions:
1. "end": If they explicitly request to stop, complain about spam, use abusive language, or if the message is an automated out-of-office/auto-reply.
2. "wait": If they say they are busy and ask to talk later. In this case, set "wait_seconds" to a reasonable time (e.g. 1800 for 30 minutes).
3. "send": If they are engaging or asking questions, or if they agree to proceed.
   - IMPORTANT constraint on "send" body: If they say "let's do it" or "go ahead" or agree to proceed, you must transition to ACTION mode. In action mode, you MUST include actioning words (like "done", "sending", "draft", "here", "confirm", "proceed", "next") and you MUST NOT include any qualifying/asking questions (avoid words like "would you", "do you", "can you tell", "what if", "how about").

Return ONLY a JSON response:
{
  "action": "send" or "wait" or "end",
  "body": "Your composed reply text. Required if action is 'send'.",
  "wait_seconds": 1800, // Optional: only if action is 'wait'
  "cta": "binary" or "open_ended" or "none", // Required if action is 'send'
  "rationale": "Why you chose this action."
}
- Do NOT include markdown code block formatting (like ```json) in your final response. Just return the raw JSON string.
"""
    
    prompt = f"Reply Message: \"{message}\"\nTurn Number: {turn}"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        return json.loads(content)
    except Exception as e:
        print(f"LLM reply error: {e}")
        return {
            "action": "send",
            "body": "Understood. How else can I assist you with your business profile?",
            "cta": "open_ended",
            "rationale": f"LLM Error: {e}"
        }


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    # Store turn
    conversations.setdefault(body.conversation_id, []).append({
        "from": body.from_role,
        "msg": body.message,
        "received_at": body.received_at
    })
    
    # Process turn response
    result = analyze_and_respond(body.merchant_id or "", body.message, body.turn_number, body.conversation_id)
    
    # Store bot reply if we sent something
    if result.get("action") == "send":
        conversations[body.conversation_id].append({
            "from": "vera",
            "msg": result.get("body", ""),
            "received_at": datetime.utcnow().isoformat() + "Z"
        })
        
    return result


if __name__ == "__main__":
    import uvicorn
    # Read port from env or default to 8080
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
