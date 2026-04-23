# source venv/Scripts/activate
# uvicorn app.main:app --reload
# python run_system_analysis.py --dataset ../mc_task.json --questions 100 --repeats 1


# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# from app.llm import get_llm_response
# from app.services.claim_extractor import extract_claims
# from app.services.evidence_retriever import get_evidence
# from app.services.verifier import verify_claim
# from app.utils.entity_extractor import extract_entities

# app = FastAPI()

# origins = [
#     "http://localhost:5173",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class QueryRequest(BaseModel):
#     query: str

# class VerifyRequest(BaseModel):
#     text: str

# @app.post("/chat")
# def chat(request: QueryRequest):
#     response = get_llm_response(request.query)

#     return {
#         "response": response
#     }

# @app.post("/verify")
# def verify(request: VerifyRequest):
#     text = request.text

#     claims = extract_claims(text)

#     results = []

#     for claim in claims:
#         if len(claim.split()) < 5:
#             continue

#         entities = extract_entities(claim)
#         evidences = get_evidence(claim, entities)
#         label, best_evidence = verify_claim(claim, evidences)

#         results.append({
#             "claim": claim,
#             "label": label,
#             "evidence": best_evidence
#         })

#     return {
#         "verification": results
#     }



from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.llm import get_llm_response
from app.services.claim_extractor import extract_claims
from app.services.evidence_retriever import get_evidence
from app.services.verifier import verify_claim
from app.utils.entity_extractor import extract_entities

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class VerifyRequest(BaseModel):
    text: str

@app.post("/chat")
def chat(request: QueryRequest):
    response = get_llm_response(request.query)
    return {"response": response}

@app.post("/verify")
def verify(request: VerifyRequest):
    text = request.text
    claims = extract_claims(text)
    results = []

    for claim in claims:
        if len(claim.split()) < 5:
            continue
        entities = extract_entities(claim)
        evidences = get_evidence(claim, entities)
        label, best_evidence, rationale = verify_claim(claim, evidences)
        results.append({
            "claim": claim,
            "label": label,
            "evidence": best_evidence,
            "rationale": rationale,
        })

    truth_score = 0
    if results:
        supported_count = sum(1 for item in results if item["label"] == "Supported")
        truth_score = round(100 * supported_count / len(results))

    return {"verification": results, "truth_score": truth_score}

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fact-checker backend is running."}

# ----------------- NEW /ask ENDPOINT -----------------
@app.post("/ask")
def ask(request: QueryRequest):
    # Step 1: Get LLM response
    llm_response = get_llm_response(request.query)

    # Step 2: Extract and verify claims in the LLM response
    claims = extract_claims(llm_response)
    verification_results = []

    for claim in claims:
        if len(claim.split()) < 5:
            continue
        entities = extract_entities(claim)
        evidences = get_evidence(claim, entities)
        label, best_evidence, rationale = verify_claim(claim, evidences)
        verification_results.append({
            "claim": claim,
            "label": label,
            "evidence": best_evidence,
            "rationale": rationale,
        })

    truth_score = 0
    if verification_results:
        supported_count = sum(1 for item in verification_results if item["label"] == "Supported")
        truth_score = round(100 * supported_count / len(verification_results))

    return {
        "response": llm_response,
        "verification": verification_results,
        "truth_score": truth_score,
    }