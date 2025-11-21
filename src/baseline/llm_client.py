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


def call_llm(prompt: str, temperature: float = 0.2) -> str:
    response = model.generate_content(
        prompt,
        generation_config={"temperature": temperature},
    )
    if not response.candidates:
        return ""

    parts = response.candidates[0].content.parts
    texts = [p.text for p in parts if hasattr(p, "text")]
    return "\n".join(texts).strip()

