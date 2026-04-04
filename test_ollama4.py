from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1", api_key="sk-no-key-required", timeout=120
)

user_msg = """You are a senior equity research analyst. Based ONLY on the provided context, deliver concise, evidence-based analytical commentary. Cite specific data points inline as [S1] when using evidence. Do NOT copy full sentences. Paraphrase and synthesize.

Context:
AAA 2016 revenue grew 20% YoY to 45,600 tons. Net profit margin was 12%.

Task: Deliver expert equity research commentary."""

messages = [{"role": "user", "content": user_msg}]
resp = client.chat.completions.create(
    model="qwen3.5:9b",
    messages=messages,
    max_tokens=400,
    stream=True,
    temperature=0.3,
)
content = ""
reasoning = ""
tool_calls = ""
for chunk in resp:
    cd = chunk.model_dump()
    delta = cd["choices"][0]["delta"]
    if delta.get("content"):
        content += delta["content"]
    if delta.get("reasoning"):
        reasoning += delta["reasoning"]
    if delta.get("tool_calls"):
        tool_calls += str(delta["tool_calls"])
print("Content (len=%d):" % len(content), repr(content[-500:]) if content else "EMPTY")
print(
    "Reasoning (len=%d):" % len(reasoning),
    repr(reasoning[-300:]) if reasoning else "EMPTY",
)
print("Tool calls:", tool_calls[:200] if tool_calls else "EMPTY")
