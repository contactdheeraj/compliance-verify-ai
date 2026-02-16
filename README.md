# ComplianceVerify AI

## Complaint Verification System for Financial Regulatory Compliance

**DISCLAIMER: This is a demonstration system for educational and evaluation purposes only. It does not constitute legal advice. Compliance determinations must be verified by qualified compliance and legal professionals.**

---

### What It Does

ComplianceVerify AI takes a consumer complaint, matches it against indexed financial regulations using hybrid AI search, and returns a structured compliance verdict with specific clause citations.

The system addresses a common failure in compliance AI systems: LLM hallucination. Instead of letting the AI guess which regulations apply, ComplianceVerify uses a **cite-or-refuse** approach. The AI either cites specific clause IDs from the indexed regulations, or explicitly states that no matching regulation was found.

### The Problem It Solves

Many banks and financial institutions use AI systems for compliance monitoring. These systems frequently suffer from:

1. **Page-based PDF chunking** that splits regulatory clauses mid-sentence, destroying context
2. **Semantic-only search** that misses exact clause references and regulatory keywords
3. **LLM hallucination** where the AI invents plausible-sounding but incorrect regulatory references

ComplianceVerify fixes all three:

- **Clause-boundary chunking**: The PDF parser detects clause markers (e.g., "RG 271.27", "CPS 220.14") and keeps each clause as an atomic unit
- **Hybrid search (FAISS + BM25)**: Combines semantic similarity with keyword matching for higher recall and precision
- **Cite-or-refuse prompting**: Claude receives only the retrieved clauses and must cite specific IDs or say "no match"

### Architecture

```
Consumer Complaint
        |
        v
  [Hybrid Search Engine]
  FAISS (TF-IDF vectors) + BM25 (keyword)
  Weighted ensemble scoring (50/50)
        |
        v
  Top 5 matching clauses
  with metadata (clause ID, section, category, jurisdiction)
        |
        v
  [Claude API - Cite or Refuse]
  Structured JSON verdict:
  - verdict (potential_breach / no_breach / insufficient_info)
  - risk_level (high / medium / low)
  - explanation with cited clause IDs
  - actionable recommendations
        |
        v
  Web UI with verdict display,
  clause citations, search diagnostics
```

### Regulations Included (Demo)

| Regulation | Jurisdiction | Clauses | Focus |
|------------|-------------|---------|-------|
| ASIC RG 271 | Australia | 11 | Internal Dispute Resolution requirements |
| ASIC REP 802 | Australia | 4 | Enforcement findings on complaint handling failures |
| CFPB Guidelines | United States | 6 | Consumer complaint handling processes |

The system accepts any compliance PDF through the upload endpoint. The clause-aware parser auto-detects regulation structure.

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set Claude API key (optional - falls back to rule-based analysis)
export ANTHROPIC_API_KEY=your_key_here

# Start the server
chmod +x run.sh
./run.sh

# Open http://localhost:8000
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/verify-complaint` | POST | Submit complaint for verification |
| `/api/regulations` | GET | List indexed regulations and statistics |
| `/api/regulations/clauses` | GET | Browse individual clauses with filters |
| `/api/regulations/upload` | POST | Upload and index a new regulation PDF |
| `/api/history` | GET | Recent verification history |
| `/api/health` | GET | System health check |

### Tech Stack

- **Backend**: Python, FastAPI
- **Search**: FAISS (TF-IDF vectors), BM25 (rank-bm25), Ensemble scoring
- **AI**: Claude API (Anthropic) with cite-or-refuse prompting
- **PDF Processing**: pdfplumber with clause-boundary detection
- **Frontend**: Single-page HTML/CSS/JS (no framework dependency)

### Project Structure

```
compliance-verifier/
  app/
    __init__.py
    compliance_parser.py   # Clause-aware PDF parser
    indexer.py              # FAISS + BM25 hybrid search
    main.py                 # FastAPI application
  data/
    pdfs/                   # Uploaded regulation PDFs
    indexes/                # Serialized FAISS + BM25 indexes
    sample_regulations.json # Demo regulation clauses
  static/
    index.html              # Web UI
  requirements.txt
  run.sh
  README.md
```

### Adding Real Regulation PDFs

Upload through the API:

```bash
curl -X POST http://localhost:8000/api/regulations/upload \
  -F "file=@path/to/regulation.pdf" \
  -F "doc_name=ASIC_RG_271" \
  -F "jurisdiction=AU"
```

Or use the clause-aware parser directly:

```python
from app.compliance_parser import parse_pdf

chunks = parse_pdf("path/to/regulation.pdf", doc_name="ASIC_RG_271", jurisdiction="AU")
print(f"Extracted {len(chunks)} clauses")
for c in chunks:
    print(f"  [{c.clause_id}] {c.category}: {c.text[:80]}...")
```

---

Built by Dee | Solutions Architect | 2026
