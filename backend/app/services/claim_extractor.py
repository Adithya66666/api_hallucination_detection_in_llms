import json
import re
import nltk

from app.llm import client

# Make sure punkt is downloaded once
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

REFERRING_START = re.compile(r'^(This|That|These|Those|It|They|Its|He|She|His|Her|Their|Such)\b', re.I)
MAX_CLAIMS = 8


def _parse_json_array(response: str):
    response = response.strip()
    if not response:
        return []

    try:
        parsed = json.loads(response)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if item]
    except json.JSONDecodeError:
        pass

    start = response.find("[")
    end = response.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = response[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item]
        except json.JSONDecodeError:
            pass

    return []


def _local_claim_candidates(text: str):
    sentences = nltk.sent_tokenize(text)
    claims = []

    for s in sentences:
        s = s.replace("\n", " ").strip()
        if len(s) < 20 or s.isnumeric():
            continue

        if REFERRING_START.match(s) and claims:
            claims[-1] = claims[-1].rstrip(' .') + '. ' + s
            continue

        claims.append(s)
        if len(claims) >= MAX_CLAIMS:
            break

    return claims


def extract_claims(text: str):
    text = text.strip()
    if not text:
        return []

    candidates = _local_claim_candidates(text)
    if not candidates:
        return []

    prompt = (
        "Extract the main factual claims from the following text. "
        "Return only a JSON array of clear, standalone claim sentences. "
        "Do not include explanations, numbering, or extra commentary. "
        "If a sentence refers to earlier text with pronouns like 'this', 'that', 'it', or vague phrases like 'at that time', "
        "rewrite the claim so it is fully explicit and includes the referenced subject from context. "
        "Use only information present in the text. "
        "Each claim must be a complete, self-contained factual statement.\n\n"
        "Text:\n"
        f"{text}\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a claim extraction assistant. "
                        "Extract concise factual claims from the text. "
                        "Return only the claims as a valid JSON array of standalone sentences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=400,
        )

        claims = _parse_json_array(response.choices[0].message.content)
        if claims:
            return claims[:MAX_CLAIMS]
    except Exception as exc:
        print("Claim extraction failed:", exc)

    return candidates

