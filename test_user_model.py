from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1", api_key="sk-no-key-required", timeout=120
)

model = "hf.co/Mikkkkoooo/qwen35-4b-private-analyst-full-corpus:Q4_K_M"

messages = [
    {
        "role": "system",
        "content": "You are a senior equity research analyst. Based ONLY on the provided context, deliver concise, evidence-based analytical commentary. Cite specific data points inline as [S1], [S2] when using evidence. Do NOT copy full sentences. Paraphrase and synthesize. If the context is insufficient, say so.",
    },
    {
        "role": "user",
        "content": "Context:\nAAA 2016 revenue grew 20% YoY to VND 45,600bn. Net profit margin was 12%. EPS reached VND 1,270. The company operates in packaging manufacturing with exports to EU, US, and Japan.\n\nTask: Deliver expert equity research commentary based on the context above.",
    },
]

print(f"Testing model: {model}")
resp = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=500,
    stream=True,
    temperature=0.3,
)
content = ""
for chunk in resp:
    cd = chunk.model_dump()
    delta = cd["choices"][0]["delta"]
    if delta.get("content"):
        content += delta["content"]
print("Response:")
print(content if content else "EMPTY")
