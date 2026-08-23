import os

print("1. Checking GROQ_API_KEY...")
key = os.environ.get("GROQ_API_KEY")
if not key:
    print("   ❌ GROQ_API_KEY is NOT set in this terminal session.")
    raise SystemExit(1)
else:
    print(f"   ✅ Key found, starts with: {key[:10]}...")

print("\n2. Checking groq package...")
from groq import Groq
print("   ✅ groq package installed")

print("\n3. Making a real API call...")
try:
    client = Groq(api_key=key)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=50,
        messages=[{"role": "user", "content": "Say 'API working' and nothing else."}],
    )
    print(f"   ✅ API call succeeded. Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"   ❌ API call failed: {e}")
    raise SystemExit(1)

print("\nAll checks passed!")
