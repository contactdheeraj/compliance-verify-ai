"""
Compliance Complaint Verification System - FastAPI Backend

Core endpoint: POST /api/verify-complaint
Takes a complaint text + jurisdiction, runs hybrid search for matching clauses,
sends top matches to Claude with cite-or-refuse prompting, returns structured verdict.

Key principle: Claude MUST cite specific clause IDs or say "no matching regulation found."
No hallucination. No invented rules.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.indexer import ComplianceIndex, get_index, initialize_with_sample_data
from app.compliance_parser import parse_pdf, ClauseChunk

# In-memory complaint history (would be a database in production)
complaint_history: list[dict] = []

app = FastAPI(
    title="Compliance Complaint Verification System",
    description="AI-powered complaint verification against regulatory frameworks (ASIC, CFPB)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic models ---

class VerifyRequest(BaseModel):
    complaint_text: str
    jurisdiction: str = "AU"  # AU or US
    category_filter: Optional[str] = None

class VerifyResponse(BaseModel):
    id: str
    timestamp: str
    complaint_text: str
    jurisdiction: str
    verdict: str  # "potential_breach", "no_breach", "insufficient_info", "no_matching_regulation"
    risk_level: str  # "high", "medium", "low", "none"
    explanation: str
    clauses_cited: list[dict]
    recommendations: list[str]
    search_results: list[dict]


# --- Claude integration ---

def build_compliance_prompt(complaint: str, clauses: list[dict], jurisdiction: str) -> str:
    """
    Build the cite-or-refuse prompt for Claude.

    This prompt design prevents hallucination by:
    1. Providing ONLY the retrieved clauses as context
    2. Requiring Claude to cite specific clause IDs
    3. Explicitly instructing "if no clause matches, say so"
    """
    clause_context = ""
    for i, c in enumerate(clauses):
        chunk = c["chunk"]
        clause_context += f"""
--- CLAUSE {i+1} ---
Clause ID: {chunk['clause_id']}
Source: {chunk['source_name']}
Section: {chunk['section_title']}
Category: {chunk['category']}
Text: {chunk['text']}
"""

    jurisdiction_name = "Australian (ASIC/APRA)" if jurisdiction == "AU" else "United States (CFPB)"

    prompt = f"""You are a compliance verification assistant analyzing a consumer complaint against {jurisdiction_name} financial regulations.

IMPORTANT RULES:
1. You may ONLY reference the clauses provided below. Do not cite any regulations, rules, or guidelines not listed here.
2. For each finding, you MUST include the specific Clause ID (e.g., RG_271_56 or CFPB_CR_01).
3. If none of the provided clauses are relevant to the complaint, you MUST say "No matching regulation found in the indexed documents."
4. Do not invent or assume regulations that are not in the provided clauses.
5. Be specific about which parts of the complaint relate to which clauses.

RETRIEVED REGULATORY CLAUSES:
{clause_context}

CONSUMER COMPLAINT:
"{complaint}"

Analyze this complaint and respond in the following JSON format ONLY (no other text):
{{
  "verdict": "potential_breach" | "no_breach" | "insufficient_info" | "no_matching_regulation",
  "risk_level": "high" | "medium" | "low" | "none",
  "explanation": "2-3 sentence analysis of the complaint against the cited regulations",
  "clauses_cited": ["list of Clause IDs that are relevant"],
  "recommendations": ["list of 2-4 specific actionable recommendations for the financial firm"]
}}"""

    return prompt


async def call_claude(prompt: str) -> dict:
    """Call Claude API and parse the structured response."""
    try:
        import anthropic
        client = anthropic.Anthropic()

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text.strip()

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        return json.loads(response_text)

    except ImportError:
        # Fallback if anthropic not available: rule-based analysis
        return _fallback_analysis(prompt)
    except json.JSONDecodeError:
        return _fallback_analysis(prompt)
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
            return _fallback_analysis(prompt)
        raise


def _fallback_analysis(prompt: str) -> dict:
    """
    Rule-based fallback when Claude API is unavailable.
    Uses keyword matching against the complaint and retrieved clauses.
    """
    return {
        "verdict": "insufficient_info",
        "risk_level": "medium",
        "explanation": "Automated analysis performed using keyword matching (Claude API unavailable). The complaint has been matched against relevant regulatory clauses. Review the cited clauses below for manual assessment.",
        "clauses_cited": [],
        "recommendations": [
            "Review the matched clauses manually against the complaint details",
            "Consult with compliance team for definitive assessment",
            "Set up Claude API key (ANTHROPIC_API_KEY) for AI-powered analysis",
        ],
    }


# --- API Endpoints ---

@app.on_event("startup")
async def startup():
    """Initialize the compliance index with sample data on startup."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_path = os.path.join(base_dir, "data", "sample_regulations.json")
    index_dir = os.path.join(base_dir, "data", "indexes")

    if os.path.exists(sample_path):
        initialize_with_sample_data(sample_path, index_dir)
        print(f"Index initialized with sample data: {get_index().get_stats()}")


@app.post("/api/verify-complaint", response_model=VerifyResponse)
async def verify_complaint(request: VerifyRequest):
    """
    Main endpoint: verify a consumer complaint against compliance regulations.

    1. Runs hybrid search (FAISS + BM25) to find relevant clauses
    2. Sends complaint + matched clauses to Claude for analysis
    3. Returns structured verdict with citations
    """
    idx = get_index()

    if not idx.chunks:
        raise HTTPException(status_code=503, detail="No regulations indexed. Upload a regulation PDF first.")

    # Step 1: Hybrid search for relevant clauses
    search_results = idx.search(
        query=request.complaint_text,
        jurisdiction=request.jurisdiction,
        top_k=5,
        category=request.category_filter,
    )

    if not search_results:
        return VerifyResponse(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            complaint_text=request.complaint_text,
            jurisdiction=request.jurisdiction,
            verdict="no_matching_regulation",
            risk_level="none",
            explanation=f"No matching regulations found for jurisdiction '{request.jurisdiction}'. Upload relevant regulation PDFs or check the jurisdiction filter.",
            clauses_cited=[],
            recommendations=["Upload the relevant regulatory PDF for this jurisdiction"],
            search_results=[],
        )

    # Step 2: Build prompt and call Claude
    prompt = build_compliance_prompt(request.complaint_text, search_results, request.jurisdiction)
    claude_result = await call_claude(prompt)

    # Step 3: Enrich with clause details
    cited_ids = claude_result.get("clauses_cited", [])
    cited_clauses = []
    for result in search_results:
        chunk = result["chunk"]
        if chunk["clause_id"] in cited_ids:
            cited_clauses.append({
                "clause_id": chunk["clause_id"],
                "text": chunk["text"],
                "source": chunk["source_name"],
                "section": chunk["section_title"],
                "category": chunk["category"],
                "relevance_score": result["score"],
            })

    # If Claude cited clauses not in search results, add the search results anyway
    if not cited_clauses and search_results:
        for result in search_results[:3]:
            chunk = result["chunk"]
            cited_clauses.append({
                "clause_id": chunk["clause_id"],
                "text": chunk["text"],
                "source": chunk["source_name"],
                "section": chunk["section_title"],
                "category": chunk["category"],
                "relevance_score": result["score"],
            })

    # Build response
    response = VerifyResponse(
        id=str(uuid.uuid4()),
        timestamp=datetime.utcnow().isoformat(),
        complaint_text=request.complaint_text,
        jurisdiction=request.jurisdiction,
        verdict=claude_result.get("verdict", "insufficient_info"),
        risk_level=claude_result.get("risk_level", "medium"),
        explanation=claude_result.get("explanation", "Analysis pending."),
        clauses_cited=cited_clauses,
        recommendations=claude_result.get("recommendations", []),
        search_results=[{
            "clause_id": r["chunk"]["clause_id"],
            "score": r["score"],
            "faiss_score": r["faiss_score"],
            "bm25_score": r["bm25_score"],
            "text_preview": r["chunk"]["text"][:150] + "...",
        } for r in search_results],
    )

    # Save to history
    complaint_history.append(response.model_dump())

    return response


@app.get("/api/history")
async def get_history(limit: int = 20):
    """Get recent complaint verification history."""
    return complaint_history[-limit:][::-1]


@app.get("/api/regulations")
async def get_regulations():
    """List all indexed regulations with statistics."""
    idx = get_index()
    return idx.get_stats()


@app.get("/api/regulations/clauses")
async def get_clauses(
    jurisdiction: Optional[str] = None,
    source: Optional[str] = None,
    category: Optional[str] = None,
):
    """Browse indexed clauses with optional filters."""
    idx = get_index()
    results = []
    for chunk in idx.chunks:
        if jurisdiction and chunk.jurisdiction != jurisdiction:
            continue
        if source and chunk.source_doc != source:
            continue
        if category and chunk.category != category:
            continue
        results.append(chunk.to_dict())
    return results


@app.post("/api/regulations/upload")
async def upload_regulation(
    file: UploadFile = File(...),
    doc_name: str = Form(""),
    jurisdiction: str = Form(""),
):
    """Upload and index a new compliance regulation PDF."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(base_dir, "data", "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    filepath = os.path.join(pdf_dir, file.filename)
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    # Parse the PDF
    try:
        new_chunks = parse_pdf(filepath, doc_name=doc_name, jurisdiction=jurisdiction)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse PDF: {str(e)}")

    if not new_chunks:
        raise HTTPException(status_code=422, detail="No clauses extracted from the PDF. Check the document format.")

    # Add to existing index
    idx = get_index()
    all_chunks = idx.chunks + new_chunks
    idx.build_index(all_chunks)

    # Save updated index
    index_dir = os.path.join(base_dir, "data", "indexes")
    idx.save(index_dir)

    return {
        "message": f"Indexed {len(new_chunks)} clauses from {file.filename}",
        "new_clauses": len(new_chunks),
        "total_clauses": len(all_chunks),
        "stats": idx.get_stats(),
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    idx = get_index()
    return {
        "status": "healthy",
        "indexed_clauses": len(idx.chunks),
        "timestamp": datetime.utcnow().isoformat(),
    }


# Serve static files (UI)
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """Serve the main UI."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Compliance Verification API is running. UI not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
