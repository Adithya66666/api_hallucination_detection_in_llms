from sentence_transformers import SentenceTransformer, util
import re

from app.llm import client

model = SentenceTransformer('all-MiniLM-L6-v2')


def verify_claim(claim: str, sentences: list):
    """
    Compare a claim against individual evidence sentences.

    Sentence-level similarity is much more accurate than comparing against
    large text blocks, because the model was trained on sentence pairs.

    Returns:
        (label, top_evidence_sentences)
        label: "Supported" | "Not Enough Info" | "Refuted"
    """
    if not sentences:
        return "Not Enough Info", [], "No candidate evidence available for this claim."

    claim_emb = model.encode(claim, convert_to_tensor=True)
    candidate_sentences = [item for item in sentences if len(item.get("sentence", item).split()) >= 5]
    if not candidate_sentences:
        return "Not Enough Info", [], "No sufficiently long evidence sentences were available."

    sentence_texts = []
    for item in candidate_sentences:
        sentence = item.get("sentence") if isinstance(item, dict) else item
        sentence_texts.append(sentence)

    sentence_embeddings = model.encode(sentence_texts, convert_to_tensor=True)
    similarity_scores = util.cos_sim(claim_emb, sentence_embeddings)[0].tolist()

    scored = []
    for idx, item in enumerate(candidate_sentences):
        score = float(similarity_scores[idx]) if idx < len(similarity_scores) else 0.0
        scored.append((score, item))

    if not scored:
        return "Not Enough Info", [], "No evidence matched the claim semantically."

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, _ = scored[0]

    # Sentence-level thresholds (higher than chunk-level because matches are tighter)
    if best_score >= 0.45:
        label = "Supported"
    elif best_score >= 0.22:
        label = "Not Enough Info"
    else:
        label = "Refuted"

    top_evidence = []
    for score, item in scored[:3]:
        if isinstance(item, dict):
            evidence_entry = {
                "sentence": item["sentence"],
                "page": item.get("page"),
                "score": round(float(score), 3),
            }
        else:
            evidence_entry = {"sentence": item, "page": None, "score": round(float(score), 3)}
        top_evidence.append(evidence_entry)

    llm_label, llm_rationale = validate_claim_with_context(claim, top_evidence, label)
    return llm_label, top_evidence, llm_rationale


def _parse_validation_response(raw: str):
    raw = raw.strip()
    normalized = raw.replace("\n", " ").strip()
    match = re.search(r"label\s*:\s*(supported|refuted|not enough info)", raw, re.IGNORECASE)
    if match:
        value = match.group(1).lower()
        if value == "supported":
            return "Supported", normalized
        if value == "refuted":
            return "Refuted", normalized
        return "Not Enough Info", normalized

    label = None
    if "supported" in normalized.lower():
        label = "Supported"
    elif "refuted" in normalized.lower() or "false" in normalized.lower():
        label = "Refuted"
    elif "not enough" in normalized.lower() or "unknown" in normalized.lower() or "insufficient" in normalized.lower():
        label = "Not Enough Info"
    return label, normalized


def validate_claim_with_context(claim: str, evidence: list, fallback_label: str):
    prompt = (
        "You are a fact verification assistant.\n"
        "Evaluate the claim only using the evidence sentences provided below.\n"
        "Do not introduce new external facts.\n"
        "Return a short verification label and a brief rationale.\n\n"
        "Claim:\n"
        f"{claim}\n\n"
        "Evidence:\n"
        + "\n".join([f"- {item['sentence']}" for item in evidence])
        + "\n\n"
        "Answer format:\n"
        "Label: Supported | Refuted | Not Enough Info\n"
        "Rationale: ...\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a fact-checking assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=250,
        )
        raw = response.choices[0].message.content
        label, rationale = _parse_validation_response(raw)
        return label or fallback_label, rationale
    except Exception as exc:
        return fallback_label, f"LLM validation failed: {exc}"