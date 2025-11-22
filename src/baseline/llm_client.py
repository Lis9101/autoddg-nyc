"""
Gemini client for Baseline AutoDDG.
"""

import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in .env")

GEMINI_MODEL_NAME = "gemini-2.0-flash"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL_NAME)

# ----------------------------------------------------------
# GLOBAL SAFETY LIMIT: Prevent gigantic prompts causing 400
# "input token count exceeds max (1048575)"
# ----------------------------------------------------------

# Gemini accepts up to around 1 million tokens,
# but we stay WAY below that. ~120k chars ≈ 30–40k tokens.
MAX_PROMPT_CHARS = 120000

RAISE_ON_TRUNCATION = False  # flip to True while debugging

def _truncate_prompt(prompt: str) -> str:
    if len(prompt) <= MAX_PROMPT_CHARS:
        return prompt

    msg = (
        f"[WARN] Truncating prompt from {len(prompt)} chars "
        f"to {MAX_PROMPT_CHARS} chars to avoid token overflow."
    )
    print(msg)

    if RAISE_ON_TRUNCATION:
        raise RuntimeError(
            f"Prompt too long ({len(prompt)} chars); caller should shrink inputs."
        )

    return (
        prompt[:MAX_PROMPT_CHARS] +
        f"\n\n...[prompt truncated from {len(prompt)} to {MAX_PROMPT_CHARS} chars]..."
    )

# Main LLM call
def call_llm(prompt: str, temperature: float = 0.2) -> str:
    # 🔥 Always truncate before sending to Gemini
    prompt = _truncate_prompt(prompt)

    response = model.generate_content(
        prompt,
        generation_config={"temperature": temperature},
    )
    if not response.candidates:
        return ""

    parts = response.candidates[0].content.parts
    texts = [p.text for p in parts if hasattr(p, "text")]
    return "\n".join(texts).strip()
