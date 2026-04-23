import wikipedia
import wikipediaapi
import re
import nltk
import json
from urllib.parse import quote_plus
from urllib.request import urlopen

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.llm import client

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='fact-checker/1.0'
)


def _get_sentences(page) -> list:
    """Split a Wikipedia page into individual sentences (first ~4000 chars)."""
    raw = (page.text or "")[:4000]
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        sentences = nltk.sent_tokenize(raw)
    except Exception:
        sentences = [s.strip() for s in raw.split('.') if s.strip()]
    return [s.strip() for s in sentences if len(s.split()) >= 6]


def _collect_page_sentences(title: str, max_snippets: int = 40) -> list:
    try:
        page = wiki.page(title)
    except Exception:
        return []
    if not page.exists():
        return []

    sentences = _get_sentences(page)
    return [
        {
            "page": title,
            "sentence": sentence,
            "source": "wikipedia",
        }
        for sentence in sentences[:max_snippets]
    ]


def _flatten_related_topics(items: list) -> list:
    flat = []
    for item in items or []:
        if isinstance(item, dict) and "Topics" in item:
            flat.extend(_flatten_related_topics(item.get("Topics") or []))
        elif isinstance(item, dict):
            flat.append(item)
    return flat


def _duckduckgo_snippets(query: str, max_snippets: int = 20) -> list:
    if not query.strip():
        return []

    url = (
        "https://api.duckduckgo.com/?q="
        f"{quote_plus(query)}"
        "&format=json&no_html=1&skip_disambig=1"
    )

    try:
        with urlopen(url, timeout=6) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    except Exception:
        return []

    snippets = []
    abstract = (payload.get("AbstractText") or "").strip()
    if len(abstract.split()) >= 6:
        snippets.append(
            {
                "page": payload.get("Heading") or "DuckDuckGo",
                "sentence": abstract,
                "source": "duckduckgo",
            }
        )

    related = _flatten_related_topics(payload.get("RelatedTopics") or [])
    for item in related:
        text = (item.get("Text") or "").strip()
        if len(text.split()) < 6:
            continue
        first_sentence = re.split(r"(?<=[.!?])\s+", text)[0].strip()
        if len(first_sentence.split()) < 6:
            continue
        snippets.append(
            {
                "page": "DuckDuckGo Related",
                "sentence": first_sentence,
                "source": "duckduckgo",
            }
        )
        if len(snippets) >= max_snippets:
            break

    return snippets[:max_snippets]


def _rank_by_claim_similarity(claim: str, evidence: list, keep: int = 120) -> list:
    if not claim.strip() or not evidence:
        return evidence[:keep]

    seen = set()
    deduped = []
    for item in evidence:
        sentence = (item.get("sentence") or "").strip()
        key = sentence.lower()
        if not sentence or key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    if not deduped:
        return []

    sentences = [item["sentence"] for item in deduped]
    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform([claim] + sentences)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten().tolist()
    except Exception:
        sims = [0.0] * len(sentences)

    ranked = []
    for index, item in enumerate(deduped):
        score = float(sims[index]) if index < len(sims) else 0.0
        ranked.append((score, item))

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    output = []
    for score, item in ranked[:keep]:
        enriched = dict(item)
        enriched["score"] = round(score, 3)
        output.append(enriched)
    return output


def _parse_json_array(raw: str) -> list:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
        except json.JSONDecodeError:
            return []
    return []


def _llm_fallback_snippets(claim: str, max_snippets: int = 8) -> list:
    prompt = (
        "Provide brief factual evidence snippets relevant to this claim. "
        "Return only a JSON array of 4 to 8 standalone factual sentences. "
        "Do not include commentary, markdown, or labels.\n\n"
        f"Claim: {claim}\n"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You provide concise factual snippets."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=250,
        )
        raw = response.choices[0].message.content
        sentences = _parse_json_array(raw)
    except Exception:
        return []

    snippets = []
    for sentence in sentences:
        if len(sentence.split()) < 6:
            continue
        snippets.append(
            {
                "page": "LLM Fallback",
                "sentence": sentence,
                "source": "llm",
            }
        )
        if len(snippets) >= max_snippets:
            break
    return snippets


def _search_titles(entities: list, claim: str, max_pages: int = 10) -> list:
    titles = []
    for entity in entities:
        entity = entity.strip()
        if entity:
            titles.append(entity)
            try:
                for title in wikipedia.search(entity, results=3):
                    if title not in titles:
                        titles.append(title)
            except Exception:
                continue

    if len(titles) < max_pages:
        try:
            for title in wikipedia.search(claim, results=5):
                if title not in titles:
                    titles.append(title)
        except Exception:
            pass

    return titles[:max_pages]


def get_evidence(claim: str, entities: list) -> list:
    """
    Return a list of individual Wikipedia sentences relevant to the claim.
    Sentence-level granularity allows the verifier to do accurate comparisons.
    """
    evidence = []
    titles = _search_titles(entities, claim)
    for title in titles:
        evidence.extend(_collect_page_sentences(title))

    if len(evidence) < 30:
        evidence.extend(_duckduckgo_snippets(claim, max_snippets=24))

    ranked = _rank_by_claim_similarity(claim, evidence, keep=120)

    top_score = ranked[0].get("score", 0.0) if ranked else 0.0
    if len(ranked) < 8 or top_score < 0.06:
        ranked.extend(_llm_fallback_snippets(claim, max_snippets=8))
        ranked = _rank_by_claim_similarity(claim, ranked, keep=120)

    # Keep the most claim-relevant evidence to improve verifier precision.
    return ranked
