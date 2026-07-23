import urllib.request
import json
import os

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
key = "AIzaSyCVeWb4GtGUUPyCKZRWeytymH1eZMg4sXQ"
models = [
    "gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash-001", "gemini-1.5-flash-002",
    "gemini-1.5-pro", "gemini-1.5-pro-latest",
    "gemini-2.0-flash", "gemini-2.0-flash-exp"
]


for model in ["gemini-2.0-flash-lite", "gemini-2.5-flash"]:
    print(f"Testing {model}...")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": "Say OK"}]}]
    }).encode("utf-8")
    
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"  [SUCCESS] {model}: {data['candidates'][0]['content']['parts'][0]['text']}")
    except Exception as e:
        print(f"  [FAIL] {model}: {e}")


