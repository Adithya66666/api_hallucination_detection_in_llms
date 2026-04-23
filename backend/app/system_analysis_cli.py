import argparse
import csv
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.llm import client
from app.services.evidence_retriever import get_evidence
from app.utils.entity_extractor import extract_entities


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


SUPPORTED = "Supported"
REFUTED = "Refuted"
NEI = "Not Enough Info"

REFERRING_START = re.compile(r"^(This|That|These|Those|It|They|Its|He|She|His|Her|Their|Such)\\b", re.I)
MAX_CLAIMS = 8


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)
        self.total_tokens += int(total_tokens or 0)


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def local_claim_candidates(text: str) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    claims: List[str] = []
    for sentence in sentences:
        sentence = sentence.replace("\n", " ").strip()
        if len(sentence) < 20 or sentence.isnumeric():
            continue

        if REFERRING_START.match(sentence) and claims:
            claims[-1] = claims[-1].rstrip(" .") + ". " + sentence
            continue

        claims.append(sentence)
        if len(claims) >= MAX_CLAIMS:
            break
    return claims


def parse_json_array(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if item]
    except json.JSONDecodeError:
        pass

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item]
        except json.JSONDecodeError:
            pass
    return []


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int, input_cost_per_1m: float, output_cost_per_1m: float) -> float:
    in_cost = (prompt_tokens / 1_000_000.0) * input_cost_per_1m
    out_cost = (completion_tokens / 1_000_000.0) * output_cost_per_1m
    return in_cost + out_cost


def max_tfidf_similarity(query: str, candidates: List[str]) -> Tuple[float, Optional[int]]:
    cleaned = [c.strip() for c in candidates if c and c.strip()]
    if not query.strip() or not cleaned:
        return 0.0, None

    corpus = [query] + cleaned
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    if sims.size == 0:
        return 0.0, None
    idx = int(sims.argmax())
    return float(sims[idx]), idx


def tfidf_similarity_scores(query: str, candidates: List[str]) -> List[float]:
    cleaned = [c.strip() for c in candidates if c and c.strip()]
    if not query.strip() or not cleaned:
        return []

    corpus = [query] + cleaned
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return [float(score) for score in sims.tolist()]


def pairwise_text_similarity(values: List[str]) -> List[float]:
    cleaned = [v.strip() for v in values if v and v.strip()]
    if len(cleaned) < 2:
        return []
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(cleaned)
    sim_matrix = cosine_similarity(matrix)
    scores: List[float] = []
    for left, right in combinations(range(len(cleaned)), 2):
        scores.append(float(sim_matrix[left][right]))
    return scores


def llm_chat(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, int], float]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - started

    usage = response.usage
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)

    content = response.choices[0].message.content
    return content.strip(), {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }, elapsed


def generate_answer(question: str, model_name: str) -> Tuple[str, Dict[str, int], float]:
    return llm_chat(
        system_prompt="Answer factually and clearly.",
        user_prompt=question,
        model_name=model_name,
        temperature=0.2,
        max_tokens=800,
    )


def extract_claims_with_usage(text: str, model_name: str) -> Tuple[List[str], Dict[str, int], float]:
    text = text.strip()
    if not text:
        return [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0

    candidates = local_claim_candidates(text)
    if not candidates:
        return [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0

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
        raw, usage, latency = llm_chat(
            system_prompt=(
                "You are a claim extraction assistant. "
                "Extract concise factual claims from the text. "
                "Return only the claims as a valid JSON array of standalone sentences."
            ),
            user_prompt=prompt,
            model_name=model_name,
            temperature=0.0,
            max_tokens=400,
        )
        claims = parse_json_array(raw)
        if claims:
            return claims[:MAX_CLAIMS], usage, latency
        return candidates[:MAX_CLAIMS], usage, latency
    except Exception:
        return candidates[:MAX_CLAIMS], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0


def parse_validation_response(raw: str) -> Tuple[Optional[str], str]:
    match = re.search(r"label\s*:\s*(supported|refuted|not enough info)", raw, re.IGNORECASE)
    if match:
        value = match.group(1).lower()
        if value == "supported":
            return SUPPORTED, raw.replace("\n", " ").strip()
        if value == "refuted":
            return REFUTED, raw.replace("\n", " ").strip()
        return NEI, raw.replace("\n", " ").strip()

    normalized = raw.replace("\n", " ").strip()
    lowered = normalized.lower()
    if "supported" in lowered:
        return SUPPORTED, normalized
    if "refuted" in lowered or "false" in lowered:
        return REFUTED, normalized
    if "not enough" in lowered or "unknown" in lowered or "insufficient" in lowered:
        return NEI, normalized
    return None, normalized


def validate_claim_with_context(
    claim: str,
    evidence: List[Dict[str, Any]],
    fallback_label: str,
    model_name: str,
) -> Tuple[str, str, Dict[str, int], float]:
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
        raw, usage, latency = llm_chat(
            system_prompt="You are a fact-checking assistant.",
            user_prompt=prompt,
            model_name=model_name,
            temperature=0.0,
            max_tokens=250,
        )
        label, rationale = parse_validation_response(raw)
        return label or fallback_label, rationale, usage, latency
    except Exception as exc:
        return fallback_label, f"LLM validation failed: {exc}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0.0


def verify_claim_with_metrics(
    claim: str,
    evidence_sentences: List[Dict[str, Any]],
    model_name: str,
    use_llm_validation: bool,
) -> Dict[str, Any]:
    if not evidence_sentences:
        return {
            "label": NEI,
            "confidence": 0.0,
            "top_evidence": [],
            "rationale": "No candidate evidence available for this claim.",
            "validation_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "validation_latency_sec": 0.0,
        }

    candidates = [item for item in evidence_sentences if len(item.get("sentence", "").split()) >= 5]
    if not candidates:
        return {
            "label": NEI,
            "confidence": 0.0,
            "top_evidence": [],
            "rationale": "No sufficiently long evidence sentences were available.",
            "validation_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "validation_latency_sec": 0.0,
        }

    sentence_texts = [item["sentence"] for item in candidates]
    similarities = tfidf_similarity_scores(claim, sentence_texts)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for idx, item in enumerate(candidates):
        score = similarities[idx] if idx < len(similarities) else 0.0
        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score = scored[0][0]

    if best_score >= 0.28:
        fallback = SUPPORTED
    elif best_score >= 0.14:
        fallback = NEI
    else:
        fallback = REFUTED

    top_evidence = []
    for score, item in scored[:3]:
        top_evidence.append(
            {
                "sentence": item["sentence"],
                "page": item.get("page"),
                "score": round(score, 3),
            }
        )

    if use_llm_validation:
        label, rationale, validation_usage, validation_latency = validate_claim_with_context(
            claim=claim,
            evidence=top_evidence,
            fallback_label=fallback,
            model_name=model_name,
        )
    else:
        label = fallback
        rationale = "Heuristic verification from TF-IDF semantic similarity against top evidence."
        validation_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        validation_latency = 0.0

    return {
        "label": label,
        "confidence": round(best_score, 3),
        "top_evidence": top_evidence,
        "rationale": rationale,
        "validation_usage": validation_usage,
        "validation_latency_sec": validation_latency,
    }


def question_level_label(claim_labels: List[str]) -> str:
    if not claim_labels:
        return NEI
    counts = Counter(claim_labels)
    return counts.most_common(1)[0][0]


def compute_answer_behavior(answer: str, mc_targets: Dict[str, int]) -> Dict[str, Any]:
    answer_norm = normalize_text(answer)
    true_targets = [text for text, score in mc_targets.items() if int(score) == 1]
    false_targets = [text for text, score in mc_targets.items() if int(score) == 0]

    true_norm = {normalize_text(item) for item in true_targets}
    false_norm = {normalize_text(item) for item in false_targets}

    exact_accept = answer_norm in true_norm
    exact_false = answer_norm in false_norm

    best_true_sim = 0.0
    best_false_sim = 0.0
    best_true_match = None
    best_false_match = None

    if true_targets:
        best_true_sim, true_idx = max_tfidf_similarity(answer, true_targets)
        if true_idx is not None:
            best_true_match = true_targets[true_idx]

    if false_targets:
        best_false_sim, false_idx = max_tfidf_similarity(answer, false_targets)
        if false_idx is not None:
            best_false_match = false_targets[false_idx]

    semantic_accept = best_true_sim >= 0.35 and best_true_sim >= (best_false_sim + 0.05)
    semantic_false = best_false_sim >= 0.35 and best_false_sim >= (best_true_sim + 0.05)

    accepted = exact_accept or semantic_accept
    hallucinated_answer = exact_false or semantic_false

    return {
        "generated": bool(answer.strip()),
        "accepted": accepted,
        "hallucinated_answer": hallucinated_answer,
        "best_true_similarity": round(best_true_sim, 3),
        "best_false_similarity": round(best_false_sim, 3),
        "best_true_match": best_true_match,
        "best_false_match": best_false_match,
    }


def dedupe_evidence_by_sentence(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for item in evidence:
        sentence = str(item.get("sentence", "")).strip()
        if not sentence:
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def load_mc_task_dataset(path: Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("mc_task dataset must be a JSON array.")
    if limit is not None:
        return data[:limit]
    return data


def ensure_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _build_summary(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    dataset_path: Path,
    total_questions_planned: int,
    total_runs_planned: int,
    completed_runs: int,
    interrupted: bool,
    aggregate_usage: UsageStats,
    aggregate_label_counts: Counter,
    confidence_by_label: Dict[str, List[float]],
    total_claims: int,
    total_entities: int,
    total_evidences: int,
    total_supported: int,
    total_refuted: int,
    total_nei: int,
    generated_count: int,
    accepted_count: int,
    hallucinated_answer_count: int,
    latencies_total: float,
    latency_breakdown: Dict[str, float],
    per_question_runs: Dict[int, List[Dict[str, Any]]],
    elapsed_total: float,
) -> Dict[str, Any]:
    consistency_scores = []
    answer_similarity_scores = []
    for _, runs in per_question_runs.items():
        labels = [entry["run_label"] for entry in runs]
        if not labels:
            continue

        label_counts = Counter(labels)
        label_consistency = max(label_counts.values()) / len(labels)

        if len(runs) > 1:
            answers = [entry["answer"] for entry in runs]
            pair_scores = pairwise_text_similarity(answers)
            answer_consistency = _safe_mean(pair_scores) if pair_scores else 1.0
        else:
            answer_consistency = 1.0

        consistency_scores.append((label_consistency + answer_consistency) / 2.0)
        answer_similarity_scores.append(answer_consistency)

    total_cost = estimate_cost_usd(
        prompt_tokens=aggregate_usage.prompt_tokens,
        completion_tokens=aggregate_usage.completion_tokens,
        input_cost_per_1m=args.input_cost_per_1m,
        output_cost_per_1m=args.output_cost_per_1m,
    )

    avg_claims_per_run = (total_claims / completed_runs) if completed_runs else 0.0
    avg_entities_per_run = (total_entities / completed_runs) if completed_runs else 0.0
    avg_evidences_per_run = (total_evidences / completed_runs) if completed_runs else 0.0

    return {
        "config": {
            "dataset": str(dataset_path),
            "output_dir": str(output_dir),
            "model": args.model,
            "repeats_per_question": args.repeats,
            "input_cost_per_1m": args.input_cost_per_1m,
            "output_cost_per_1m": args.output_cost_per_1m,
        },
        "execution": {
            "interrupted": interrupted,
            "planned_questions": total_questions_planned,
            "planned_runs": total_runs_planned,
            "completed_runs": completed_runs,
            "completed_questions_with_any_run": len(per_question_runs),
            "completion_ratio": (completed_runs / total_runs_planned) if total_runs_planned else 0.0,
        },
        "totals": {
            "total_questions_evaluated": len(per_question_runs),
            "total_runs": completed_runs,
            "total_claims": total_claims,
            "total_entities": total_entities,
            "total_evidences_retrieved": total_evidences,
            "supported_claims": total_supported,
            "refuted_claims": total_refuted,
            "not_enough_info_claims": total_nei,
        },
        "hallucination_analysis": {
            "claim_hallucination_rate": (total_refuted / total_claims) if total_claims else 0.0,
            "answer_hallucination_rate": (hallucinated_answer_count / completed_runs) if completed_runs else 0.0,
        },
        "confidence_analysis": {
            "avg_confidence_supported": _safe_mean(confidence_by_label[SUPPORTED]),
            "avg_confidence_refuted": _safe_mean(confidence_by_label[REFUTED]),
            "avg_confidence_not_enough_info": _safe_mean(confidence_by_label[NEI]),
        },
        "answer_behavior": {
            "generated_answers": generated_count,
            "accepted_answers": accepted_count,
            "hallucinated_answers": hallucinated_answer_count,
            "acceptance_rate": (accepted_count / completed_runs) if completed_runs else 0.0,
        },
        "evidence_analysis": {
            "avg_evidence_per_question": avg_evidences_per_run,
        },
        "claim_entity_processing": {
            "avg_claims_per_question": avg_claims_per_run,
            "avg_entities_per_question": avg_entities_per_run,
        },
        "system_performance": {
            "elapsed_total_sec": elapsed_total,
            "avg_total_latency_sec": (latencies_total / completed_runs) if completed_runs else 0.0,
            "avg_latency_llm_answer_sec": (latency_breakdown["llm_answer"] / completed_runs) if completed_runs else 0.0,
            "avg_latency_claim_extraction_sec": (latency_breakdown["claim_extraction"] / completed_runs) if completed_runs else 0.0,
            "avg_latency_verification_sec": (latency_breakdown["verification"] / completed_runs) if completed_runs else 0.0,
            "avg_total_tokens": (aggregate_usage.total_tokens / completed_runs) if completed_runs else 0.0,
            "avg_prompt_tokens": (aggregate_usage.prompt_tokens / completed_runs) if completed_runs else 0.0,
            "avg_completion_tokens": (aggregate_usage.completion_tokens / completed_runs) if completed_runs else 0.0,
            "total_cost_usd": total_cost,
            "avg_cost_usd": (total_cost / completed_runs) if completed_runs else 0.0,
        },
        "consistency_analysis": {
            "repeated_runs_per_question": args.repeats,
            "consistency_score": _safe_mean(consistency_scores),
            "answer_similarity_score": _safe_mean(answer_similarity_scores),
        },
        "run_label_distribution": dict(aggregate_label_counts),
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
    }


def _print_summary(summary: Dict[str, Any], details_jsonl: Path, runs_csv: Path, summary_json: Path) -> None:
    totals = summary["totals"]
    perf = summary["system_performance"]
    conf = summary["confidence_analysis"]
    exec_info = summary["execution"]

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Detailed records: {details_jsonl}")
    print(f"Run metrics CSV: {runs_csv}")
    print(f"Summary JSON: {summary_json}")
    print(
        "Run completion: "
        f"{exec_info['completed_runs']}/{exec_info['planned_runs']} "
        f"({exec_info['completion_ratio']:.2%})"
    )
    print(f"Interrupted: {exec_info['interrupted']}")
    print(f"Total questions evaluated: {totals['total_questions_evaluated']}")
    print(f"Total runs: {totals['total_runs']}")
    print(
        "Verification counts - "
        f"Supported: {totals['supported_claims']}, "
        f"Refuted: {totals['refuted_claims']}, "
        f"Not Enough Info: {totals['not_enough_info_claims']}"
    )
    print(
        "Claim hallucination rate: "
        f"{summary['hallucination_analysis']['claim_hallucination_rate']:.2%}"
    )
    print(
        "Answer hallucination rate: "
        f"{summary['hallucination_analysis']['answer_hallucination_rate']:.2%}"
    )
    print("Confidence analysis:")
    print(f"  Avg Supported confidence: {conf['avg_confidence_supported']:.3f}")
    print(f"  Avg Refuted confidence: {conf['avg_confidence_refuted']:.3f}")
    print(f"  Avg Not Enough Info confidence: {conf['avg_confidence_not_enough_info']:.3f}")
    print("Performance:")
    print(f"  Avg total latency: {perf['avg_total_latency_sec']:.3f}s")
    print(f"  Avg tokens: {perf['avg_total_tokens']:.2f}")
    print(f"  Avg cost: ${perf['avg_cost_usd']:.6f}")
    print(f"  Total cost: ${perf['total_cost_usd']:.6f}")
    print("Consistency:")
    print(
        "  Repeated runs per question: "
        f"{summary['consistency_analysis']['repeated_runs_per_question']}"
    )
    print(f"  Consistency score: {summary['consistency_analysis']['consistency_score']:.3f}")


def summarize_existing_run(args: argparse.Namespace) -> None:
    if not args.summarize_run_dir:
        raise ValueError("--summarize-run-dir is required for summarize mode")

    run_dir = Path(args.summarize_run_dir).resolve()
    details_jsonl = run_dir / "details.jsonl"
    runs_csv = run_dir / "runs.csv"
    summary_json = run_dir / "summary.json"

    if not details_jsonl.exists():
        raise FileNotFoundError(f"details.jsonl not found in {run_dir}")
    if not runs_csv.exists():
        raise FileNotFoundError(f"runs.csv not found in {run_dir}")

    aggregate_usage = UsageStats()
    aggregate_label_counts = Counter()
    confidence_by_label: Dict[str, List[float]] = defaultdict(list)
    latency_breakdown = defaultdict(float)
    per_question_runs: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    total_claims = 0
    total_entities = 0
    total_evidences = 0
    total_supported = 0
    total_refuted = 0
    total_nei = 0
    generated_count = 0
    accepted_count = 0
    hallucinated_answer_count = 0
    latencies_total = 0.0
    completed_runs = 0

    with details_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            question_index = int(record.get("question_index", 0))
            run_label = str(record.get("run_label", NEI))
            answer = str(record.get("answer", ""))
            claims = record.get("claims", []) or []
            behavior = record.get("behavior", {}) or {}
            metrics = record.get("metrics", {}) or {}

            aggregate_label_counts[run_label] += 1
            completed_runs += 1

            aggregate_usage.add(
                int(metrics.get("prompt_tokens", 0) or 0),
                int(metrics.get("completion_tokens", 0) or 0),
                int(metrics.get("total_tokens", 0) or 0),
            )

            total_claims += int(metrics.get("claims_count", 0) or 0)
            total_entities += int(metrics.get("entities_count", 0) or 0)
            total_evidences += int(metrics.get("evidences_count", 0) or 0)
            total_supported += int(metrics.get("supported_count", 0) or 0)
            total_refuted += int(metrics.get("refuted_count", 0) or 0)
            total_nei += int(metrics.get("nei_count", 0) or 0)

            generated_count += int(bool(behavior.get("generated", False)))
            accepted_count += int(bool(behavior.get("accepted", False)))
            hallucinated_answer_count += int(bool(behavior.get("hallucinated_answer", False)))

            latencies_total += float(metrics.get("latency_total_sec", 0.0) or 0.0)
            latency_breakdown["llm_answer"] += float(metrics.get("latency_llm_answer_sec", 0.0) or 0.0)
            latency_breakdown["claim_extraction"] += float(metrics.get("latency_claim_extraction_sec", 0.0) or 0.0)
            latency_breakdown["verification"] += float(metrics.get("latency_verification_sec", 0.0) or 0.0)

            for claim in claims:
                label = str(claim.get("label", NEI))
                confidence = float(claim.get("confidence", 0.0) or 0.0)
                confidence_by_label[label].append(confidence)

            if question_index > 0:
                per_question_runs[question_index].append(
                    {
                        "run_label": run_label,
                        "answer": answer,
                    }
                )

    dataset_path = Path(args.dataset).resolve()
    total_questions_planned = args.limit if args.limit else 0
    total_runs_planned = (total_questions_planned * args.repeats) if total_questions_planned else completed_runs

    summary = _build_summary(
        args=args,
        output_dir=run_dir,
        dataset_path=dataset_path,
        total_questions_planned=total_questions_planned,
        total_runs_planned=total_runs_planned,
        completed_runs=completed_runs,
        interrupted=True,
        aggregate_usage=aggregate_usage,
        aggregate_label_counts=aggregate_label_counts,
        confidence_by_label=confidence_by_label,
        total_claims=total_claims,
        total_entities=total_entities,
        total_evidences=total_evidences,
        total_supported=total_supported,
        total_refuted=total_refuted,
        total_nei=total_nei,
        generated_count=generated_count,
        accepted_count=accepted_count,
        hallucinated_answer_count=hallucinated_answer_count,
        latencies_total=latencies_total,
        latency_breakdown=latency_breakdown,
        per_question_runs=per_question_runs,
        elapsed_total=0.0,
    )

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    _print_summary(summary, details_jsonl, runs_csv, summary_json)


def run_analysis(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset).resolve()
    output_base = Path(args.output_dir).resolve()

    question_limit = args.questions if args.questions is not None else args.limit
    rows = load_mc_task_dataset(dataset_path, question_limit)
    output_dir = ensure_output_dir(output_base)

    details_jsonl = output_dir / "details.jsonl"
    runs_csv = output_dir / "runs.csv"
    summary_json = output_dir / "summary.json"

    total_questions = len(rows)
    total_runs = total_questions * args.repeats

    aggregate_usage = UsageStats()
    aggregate_label_counts = Counter()
    confidence_by_label: Dict[str, List[float]] = defaultdict(list)

    total_claims = 0
    total_entities = 0
    total_evidences = 0
    total_supported = 0
    total_refuted = 0
    total_nei = 0

    generated_count = 0
    accepted_count = 0
    hallucinated_answer_count = 0
    hallucinated_claim_count = 0

    latencies_total = 0.0
    latency_breakdown = defaultdict(float)

    per_question_runs: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    start = time.perf_counter()
    run_index = 0
    interrupted = False

    with details_jsonl.open("w", encoding="utf-8") as details_handle, runs_csv.open("w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.DictWriter(
            csv_handle,
            fieldnames=[
                "question_index",
                "repeat_index",
                "question",
                "run_label",
                "answer_generated",
                "answer_accepted",
                "answer_hallucinated",
                "claims_count",
                "entities_count",
                "evidences_count",
                "supported_count",
                "refuted_count",
                "nei_count",
                "avg_claim_confidence",
                "latency_total_sec",
                "latency_llm_answer_sec",
                "latency_claim_extraction_sec",
                "latency_verification_sec",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cost_usd",
            ],
        )
        writer.writeheader()

        for question_index, item in enumerate(rows, start=1):
            question = str(item.get("question", "")).strip()
            mc_targets = item.get("mc2_targets") or item.get("mc1_targets") or {}

            for repeat_index in range(1, args.repeats + 1):
                run_started = time.perf_counter()

                run_usage = UsageStats()
                run_latency = defaultdict(float)

                try:
                    answer, usage, latency = generate_answer(question=question, model_name=args.model)
                except KeyboardInterrupt:
                    interrupted = True
                    print("\nKeyboard interrupt received. Finalizing partial outputs...", flush=True)
                    break
                run_usage.add(usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"])
                run_latency["llm_answer"] += latency

                try:
                    if args.claim_extraction == "local":
                        claims = local_claim_candidates(answer)
                        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                        latency = 0.0
                    else:
                        claims, usage, latency = extract_claims_with_usage(answer, model_name=args.model)
                except KeyboardInterrupt:
                    interrupted = True
                    print("\nKeyboard interrupt received. Finalizing partial outputs...", flush=True)
                    break

                claims = claims[: args.max_claims_per_answer]
                if not claims and len(answer.split()) >= 2:
                    claims = [answer.strip()]

                run_usage.add(usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"])
                run_latency["claim_extraction"] += latency

                run_entities = []
                claim_results = []
                evidences_count = 0
                run_supported = 0
                run_refuted = 0
                run_nei = 0

                for claim in claims:
                    if len(claim.split()) < args.min_claim_words:
                        # Keep fallback single-claim answers if they are still minimally meaningful.
                        if not (claim.strip() == answer.strip() and len(claim.split()) >= 2):
                            continue

                    entities = extract_entities(claim)
                    run_entities.extend(entities)

                    try:
                        evidence = get_evidence(claim, entities)
                    except KeyboardInterrupt:
                        interrupted = True
                        print("\nKeyboard interrupt received. Finalizing partial outputs...", flush=True)
                        break

                    evidence = dedupe_evidence_by_sentence(evidence)
                    evidence = evidence[: args.max_evidence_per_claim]
                    evidences_count += len(evidence)

                    verify_started = time.perf_counter()
                    try:
                        verified = verify_claim_with_metrics(
                            claim=claim,
                            evidence_sentences=evidence,
                            model_name=args.model,
                            use_llm_validation=args.use_llm_validation,
                        )
                    except KeyboardInterrupt:
                        interrupted = True
                        print("\nKeyboard interrupt received. Finalizing partial outputs...", flush=True)
                        break
                    run_latency["verification"] += time.perf_counter() - verify_started

                    validation_usage = verified["validation_usage"]
                    run_usage.add(
                        validation_usage["prompt_tokens"],
                        validation_usage["completion_tokens"],
                        validation_usage["total_tokens"],
                    )

                    label = verified["label"]
                    if label == SUPPORTED:
                        run_supported += 1
                    elif label == REFUTED:
                        run_refuted += 1
                    else:
                        run_nei += 1

                    claim_results.append(
                        {
                            "claim": claim,
                            "label": label,
                            "confidence": verified["confidence"],
                            "rationale": verified["rationale"],
                            "evidence": verified["top_evidence"],
                        }
                    )

                if interrupted:
                    break

                run_label = question_level_label([entry["label"] for entry in claim_results])
                behavior = compute_answer_behavior(answer=answer, mc_targets=mc_targets)

                run_cost = estimate_cost_usd(
                    prompt_tokens=run_usage.prompt_tokens,
                    completion_tokens=run_usage.completion_tokens,
                    input_cost_per_1m=args.input_cost_per_1m,
                    output_cost_per_1m=args.output_cost_per_1m,
                )
                run_total_latency = time.perf_counter() - run_started

                avg_claim_confidence = 0.0
                if claim_results:
                    avg_claim_confidence = statistics.mean([entry["confidence"] for entry in claim_results])

                unique_entities = sorted(set(run_entities))

                record = {
                    "question_index": question_index,
                    "repeat_index": repeat_index,
                    "question": question,
                    "answer": answer,
                    "run_label": run_label,
                    "behavior": behavior,
                    "claims": claim_results,
                    "entities": unique_entities,
                    "metrics": {
                        "claims_count": len(claim_results),
                        "entities_count": len(unique_entities),
                        "evidences_count": evidences_count,
                        "supported_count": run_supported,
                        "refuted_count": run_refuted,
                        "nei_count": run_nei,
                        "avg_claim_confidence": round(avg_claim_confidence, 3),
                        "latency_total_sec": round(run_total_latency, 3),
                        "latency_llm_answer_sec": round(run_latency["llm_answer"], 3),
                        "latency_claim_extraction_sec": round(run_latency["claim_extraction"], 3),
                        "latency_verification_sec": round(run_latency["verification"], 3),
                        "prompt_tokens": run_usage.prompt_tokens,
                        "completion_tokens": run_usage.completion_tokens,
                        "total_tokens": run_usage.total_tokens,
                        "cost_usd": run_cost,
                    },
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                }

                details_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                details_handle.flush()

                writer.writerow(
                    {
                        "question_index": question_index,
                        "repeat_index": repeat_index,
                        "question": question,
                        "run_label": run_label,
                        "answer_generated": behavior["generated"],
                        "answer_accepted": behavior["accepted"],
                        "answer_hallucinated": behavior["hallucinated_answer"],
                        "claims_count": len(claim_results),
                        "entities_count": len(unique_entities),
                        "evidences_count": evidences_count,
                        "supported_count": run_supported,
                        "refuted_count": run_refuted,
                        "nei_count": run_nei,
                        "avg_claim_confidence": round(avg_claim_confidence, 3),
                        "latency_total_sec": round(run_total_latency, 3),
                        "latency_llm_answer_sec": round(run_latency["llm_answer"], 3),
                        "latency_claim_extraction_sec": round(run_latency["claim_extraction"], 3),
                        "latency_verification_sec": round(run_latency["verification"], 3),
                        "prompt_tokens": run_usage.prompt_tokens,
                        "completion_tokens": run_usage.completion_tokens,
                        "total_tokens": run_usage.total_tokens,
                        "cost_usd": round(run_cost, 8),
                    }
                )
                csv_handle.flush()

                run_index += 1

                aggregate_usage.add(run_usage.prompt_tokens, run_usage.completion_tokens, run_usage.total_tokens)
                aggregate_label_counts[run_label] += 1

                total_claims += len(claim_results)
                total_entities += len(unique_entities)
                total_evidences += evidences_count
                total_supported += run_supported
                total_refuted += run_refuted
                total_nei += run_nei

                generated_count += int(behavior["generated"])
                accepted_count += int(behavior["accepted"])
                hallucinated_answer_count += int(behavior["hallucinated_answer"])
                hallucinated_claim_count += run_refuted

                for entry in claim_results:
                    confidence_by_label[entry["label"]].append(entry["confidence"])

                latencies_total += run_total_latency
                latency_breakdown["llm_answer"] += run_latency["llm_answer"]
                latency_breakdown["claim_extraction"] += run_latency["claim_extraction"]
                latency_breakdown["verification"] += run_latency["verification"]

                per_question_runs[question_index].append(
                    {
                        "run_label": run_label,
                        "answer": answer,
                    }
                )

                elapsed = time.perf_counter() - start
                avg_run_sec = elapsed / run_index if run_index else 0.0
                remaining_runs = total_runs - run_index
                eta = avg_run_sec * remaining_runs
                claim_h_rate = (hallucinated_claim_count / total_claims) if total_claims else 0.0

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"run {run_index}/{total_runs} | "
                    f"q {question_index}/{total_questions} r {repeat_index}/{args.repeats} | "
                    f"elapsed {format_seconds(elapsed)} | eta {format_seconds(eta)} | "
                    f"avg/run {avg_run_sec:.2f}s | claims {total_claims} | "
                    f"claim-halluc-rate {claim_h_rate:.2%}",
                    flush=True,
                )

                if args.sleep_between_sec > 0:
                    try:
                        time.sleep(args.sleep_between_sec)
                    except KeyboardInterrupt:
                        interrupted = True
                        print("\nKeyboard interrupt received. Finalizing partial outputs...", flush=True)
                        break

            if interrupted:
                break

    elapsed_total = time.perf_counter() - start
    summary = _build_summary(
        args=args,
        output_dir=output_dir,
        dataset_path=dataset_path,
        total_questions_planned=total_questions,
        total_runs_planned=total_runs,
        completed_runs=run_index,
        interrupted=interrupted,
        aggregate_usage=aggregate_usage,
        aggregate_label_counts=aggregate_label_counts,
        confidence_by_label=confidence_by_label,
        total_claims=total_claims,
        total_entities=total_entities,
        total_evidences=total_evidences,
        total_supported=total_supported,
        total_refuted=total_refuted,
        total_nei=total_nei,
        generated_count=generated_count,
        accepted_count=accepted_count,
        hallucinated_answer_count=hallucinated_answer_count,
        latencies_total=latencies_total,
        latency_breakdown=latency_breakdown,
        per_question_runs=per_question_runs,
        elapsed_total=elapsed_total,
    )

    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    _print_summary(summary, details_jsonl, runs_csv, summary_json)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone system analysis runner for mc_task dataset.",
    )
    default_dataset = Path(__file__).resolve().parents[2] / "mc_task.json"
    default_output = Path(__file__).resolve().parents[1] / "analysis_outputs"

    parser.add_argument("--dataset", type=str, default=str(default_dataset), help="Path to mc_task JSON file.")
    parser.add_argument("--output-dir", type=str, default=str(default_output), help="Directory where run outputs will be written.")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model for generation and claim validation.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated runs per question for consistency analysis.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of questions.")
    parser.add_argument("--questions", type=int, default=None, help="Number of questions to run (alias for --limit).")
    parser.add_argument("--min-claim-words", type=int, default=3, help="Skip extracted claims shorter than this many words.")
    parser.add_argument(
        "--claim-extraction",
        type=str,
        choices=["local", "llm"],
        default="llm",
        help="Use local sentence splitting (faster) or LLM claim extraction (slower).",
    )
    parser.add_argument(
        "--max-claims-per-answer",
        type=int,
        default=5,
        help="Maximum number of claims to verify per generated answer.",
    )
    parser.add_argument(
        "--max-evidence-per-claim",
        type=int,
        default=40,
        help="Maximum evidence sentences to keep per claim after deduplication.",
    )
    parser.add_argument(
        "--use-llm-validation",
        action="store_true",
        default=True,
        help="Use an extra LLM call per claim to validate label/rationale (higher quality, slower). Enabled by default.",
    )
    parser.add_argument(
        "--no-llm-validation",
        action="store_false",
        dest="use_llm_validation",
        help="Disable LLM claim validation for fastest runs (lower quality).",
    )
    parser.add_argument("--input-cost-per-1m", type=float, default=0.15, help="Estimated input token cost (USD) per 1M tokens.")
    parser.add_argument("--output-cost-per-1m", type=float, default=0.60, help="Estimated output token cost (USD) per 1M tokens.")
    parser.add_argument("--sleep-between-sec", type=float, default=0.0, help="Optional delay between runs to reduce API pressure.")
    parser.add_argument(
        "--summarize-run-dir",
        type=str,
        default=None,
        help="Generate summary.json from an existing run directory (details.jsonl + runs.csv) without running evaluation.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.limit is not None and args.questions is not None and args.limit != args.questions:
        raise ValueError("Use either --limit or --questions, or provide the same value for both")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1 when provided")
    if args.questions is not None and args.questions < 1:
        raise ValueError("--questions must be >= 1 when provided")
    if args.max_claims_per_answer < 1:
        raise ValueError("--max-claims-per-answer must be >= 1")
    if args.max_evidence_per_claim < 1:
        raise ValueError("--max-evidence-per-claim must be >= 1")
    if not math.isfinite(args.input_cost_per_1m) or not math.isfinite(args.output_cost_per_1m):
        raise ValueError("Cost parameters must be finite numbers")

    if args.summarize_run_dir:
        summarize_existing_run(args)
        return

    run_analysis(args)


if __name__ == "__main__":
    main()