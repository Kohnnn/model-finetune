from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1", api_key="sk-no-key-required", timeout=120
)

# Use the user's fine-tuned model which works on Ollama
messages = [
    {
        "role": "system",
        "content": "You are a senior equity research analyst. Based ONLY on the provided context, deliver concise, evidence-based analytical commentary. Cite specific data points inline as [S1], [S2] when using evidence. Do NOT copy full sentences. Paraphrase and synthesize.",
    },
    {
        "role": "user",
        "content": "Context:\nAAA 2016 revenue grew 20% YoY. Net profit margin was 12%.\n\nTask: Deliver expert equity research commentary.",
    },
]
resp = client.chat.completions.create(
    model="hf.co/Mikkkkoooo/qwen35-4b-private-analyst-full-corpus:Q4_K_M",
    messages=messages,
    max_tokens=500,
    stream=True,
    temperature=0.3,
)
content = ""
reasoning = ""
for chunk in resp:
    cd = chunk.model_dump()
    delta = cd["choices"][0]["delta"]
    if delta.get("content"):
        content += delta["content"]
    if delta.get("reasoning"):
        reasoning += delta["reasoning"]
print("Content:", repr(content[:800]) if content else "EMPTY")
print("Reasoning:", repr(reasoning[:300]) if reasoning else "EMPTY")
