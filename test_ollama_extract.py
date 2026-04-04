import subprocess, sys

prompt = """<|im_start|>system
You are a senior equity research analyst. Based ONLY on the provided context, deliver concise, evidence-based analytical commentary. Cite specific data points inline as [S1] when using evidence. Do NOT copy full sentences. Paraphrase.<|im_end|>
<|im_start|>user
Context: AAA 2016 revenue grew 20% YoY. Net profit margin was 12%.

Task: Deliver expert equity research commentary.<|im_end|>
<|im_start|>assistant
"""

result = subprocess.run(
    ["ollama", "run", "qwen3.5:9b", prompt],
    capture_output=True,
    text=True,
    timeout=180,
    encoding="utf-8",
    errors="replace",
)
stdout = result.stdout
# Extract only non-ANSI, non-thinking lines
lines = stdout.split("\n")
answer_lines = []
in_thinking = False
for line in lines:
    if (
        "done thinking" in line.lower()
        or "final output" in line.lower()
        or "final answer" in line.lower()
        or "drafting" in line.lower()
        and "final" in line.lower()
    ):
        in_thinking = False
    if (
        not line.startswith("\x1b[")
        and "Thinking" not in line
        and "Process" not in line
        and "step" not in line.lower()
        and "analyze" not in line.lower()
        and "draft" not in line.lower()
        and "review" not in line.lower()
        and "final" not in line.lower()
        and "select" not in line.lower()
        and "determine" not in line.lower()
        and "check" not in line.lower()
        and "wait," not in line.lower()
    ):
        stripped = line.strip()
        if (
            stripped
            and not stripped.startswith("*")
            and not stripped.startswith("-")
            and not stripped.startswith("1.")
            and not stripped.startswith("2.")
            and not stripped.startswith("3.")
            and not stripped.startswith("4.")
            and not stripped.startswith("5.")
        ):
            answer_lines.append(stripped)

print("Extracted answer:")
print(" ".join(answer_lines[-5:]))
print("---")
print("Last 500 chars of stdout:")
print(stdout[-500:])
