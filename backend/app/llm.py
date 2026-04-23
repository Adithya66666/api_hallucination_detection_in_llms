import os
from pathlib import Path

from openai import OpenAI

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
if ENV_PATH.exists():
    with ENV_PATH.open("r", encoding="utf-8") as dotenv_file:
        for line in dotenv_file:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "Missing OPENAI_API_KEY environment variable. "
        "Set it in backend/.env or in your system environment."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

def get_llm_response(query: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Answer factually and clearly."},
                {"role": "user", "content": query},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc