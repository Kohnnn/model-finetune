from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1", api_key="sk-no-key-required", timeout=120
)

TEST_PROMPT = "What is 2+2? Answer in exactly one word."

models_to_test = [
    "qwen3.5:9b",
    "hf.co/Mikkkkoooo/qwen35-4b-private-analyst-full-corpus:Q4_K_M",
    "glm-4.7-flash:latest",
]

for model in models_to_test:
    print(f"\n=== Testing {model} ===")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": TEST_PROMPT}],
            max_tokens=100,
            stream=True,
            timeout=60,
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
        print(f"  content: {repr(content[:100]) if content else 'EMPTY'}")
        print(f"  reasoning: {repr(reasoning[:100]) if reasoning else 'EMPTY'}")
    except Exception as e:
        print(f"  ERROR: {e}")
