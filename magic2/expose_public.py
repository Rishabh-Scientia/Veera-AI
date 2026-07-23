"""
expose_public.py — Exposes localhost:8080 via ngrok tunnel.
Run this while vera_bot server is running.
"""
import sys
import time

try:
    from pyngrok import ngrok, conf
except ImportError:
    print("Install pyngrok: pip install pyngrok")
    sys.exit(1)

# If you have an ngrok authtoken, set it here:
# ngrok.set_auth_token("YOUR_TOKEN_HERE")

print("Opening ngrok tunnel to localhost:8080 ...")
tunnel = ngrok.connect(8080, "http")
public_url = tunnel.public_url

# Prefer https
if public_url.startswith("http://"):
    public_url = public_url.replace("http://", "https://", 1)

print("\n" + "="*60)
print(f"  PUBLIC URL: {public_url}")
print("="*60)
print("\nEndpoints live at:")
print(f"  GET  {public_url}/v1/healthz")
print(f"  GET  {public_url}/v1/metadata")
print(f"  POST {public_url}/v1/context")
print(f"  POST {public_url}/v1/tick")
print(f"  POST {public_url}/v1/reply")
print("\nSubmit this URL to Magicpin:")
print(f"  {public_url}")
print("\nPress Ctrl+C to stop tunnel.\n")

try:
    while True:
        time.sleep(5)
except KeyboardInterrupt:
    print("\nClosing tunnel...")
    ngrok.disconnect(tunnel.public_url)
    ngrok.kill()
