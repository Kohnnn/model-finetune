from __future__ import annotations

ANALYST_SYSTEM_PROMPT = (
    "You are a senior equity research analyst. Provide concise, evidence-based "
    "answers using only the retrieved context. Be explicit about uncertainty and "
    "never rely on outside knowledge when the evidence is incomplete."
)


def build_query_messages(query: str, context_block: str) -> list[dict[str, str]]:
    user_prompt = (
        "Use only the retrieved context to answer the user's question.\n"
        "If the evidence is insufficient, say so directly.\n"
        "Cite supporting evidence inline as [S1], [S2], and so on.\n"
        "Do not invent facts, numbers, or conclusions.\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"User question:\n{query.strip()}"
    )
    return [
        {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
