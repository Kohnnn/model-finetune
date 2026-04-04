from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1", api_key="sk-no-key-required", timeout=120
)
messages = [
    {"role": "system", "content": "You are a helpful assistant. Keep responses short."},
    {
        "role": "user",
        "content": "What is the square root of 144? Just give the number.",
    },
]
resp = client.chat.completions.create(
    model="qwen3.5:9b",
    messages=messages,
    max_tokens=100,
    stream=True,
    extra_body={"options": {"num_predict": 100}},
)
content = ""
reasoning = ""
for chunk in resp:
    cd = chunk.model_dump()
    delta = cd["choices"][0]["delta"]
    c = delta.get("content", "")
    r = delta.get("reasoning", "")
    content += c
    reasoning += r
print("Content:", content)
print("Reasoning (last 200):", reasoning[-200:] if reasoning else "EMPTY")
