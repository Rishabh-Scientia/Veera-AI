import os
import json
from urllib import request as urlrequest, error as urlerror

api_key = os.getenv("GROQ_API_KEY", "")
model = "llama-3.3-70b-versatile"
prompt = "hello"

messages = [{"role": "user", "content": prompt}]
req = urlrequest.Request(
    "https://api.groq.com/openai/v1/chat/completions",
    data=json.dumps({"model": model, "messages": messages,
                    "temperature": 0.2, "max_tokens": 1500}).encode("utf-8"),
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
)
try:
    resp = urlrequest.urlopen(req, timeout=45)
    data = json.loads(resp.read().decode("utf-8"))
    print("Success")
except urlerror.HTTPError as e:
    print(e.code)
    print(e.read().decode("utf-8"))
