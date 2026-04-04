from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1", api_key="sk-no-key-required", timeout=120
)

# No system prompt - embed instructions in user message
user_msg = """You are a senior equity research analyst. Based ONLY on the provided context, deliver concise, evidence-based analytical commentary. Cite specific data points inline as [S1], [S2] when using evidence. Do NOT copy full sentences. Paraphrase and synthesize. If the context is insufficient, say so.

Context:
AAA 2016 revenue grew 20% YoY to 45,600 tons. Net profit margin was 12%. The company operates in packaging manufacturing.

Task: Deliver expert equity research commentary based on the context above."""

messages = [{"role": "user", "content": user_msg}]
resp = client.chat.completions.create(
    model="qwen3.5:9b",
    messages=messages,
    max_tokens=400,
    stream=True,
    temperature=0.3,
)
content = ""
for chunk in resp:
    cd = chunk.model_dump()
    delta = cd["choices"][0]["delta"]
    if delta.get("content"):
        content += delta["content"]
print("Content:", content)
