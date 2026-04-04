from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1", api_key="sk-no-key-required", timeout=120
)

TEST_CASES = [
    (
        "qwen3.5:9b",
        [{"role": "user", "content": "What is 2+2? Answer in exactly one word."}],
    ),
    (
        "hf.co/Mikkkkoooo/qwen35-4b-private-analyst-full-corpus:Q4_K_M",
        [{"role": "user", "content": "What is 2+2? Answer in exactly one word."}],
    ),
]

for model, messages in TEST_CASES:
    print(f"\n=== {model} ===")
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=200,
                stream=True,
                temperature=0.3,
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
            print(
                f"  attempt {attempt + 1}: content={repr(content[:100]) if content else 'EMPTY'}, reasoning={repr(reasoning[:100]) if reasoning else 'EMPTY'}"
            )
            break
        except Exception as e:
            print(f"  attempt {attempt + 1}: ERROR: {e}")
