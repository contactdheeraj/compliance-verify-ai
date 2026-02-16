"""
Compliance PDF Parser - Clause-Aware Chunking

Fixes the core problem in Aparna's ANZ system: page-based chunking that splits
clauses mid-sentence. This parser detects clause boundaries using regex patterns
and preserves each clause as an atomic unit with full metadata.

Supported regulation formats:
- ASIC RG-style (e.g., "RG 271.27", "RG 271.36")
- APRA CPS/SPS-style (e.g., "CPS 220.14", "SPS 220.5")
- CFPB section-style (e.g., "Section 5", numbered paragraphs)
- Generic numbered clauses (e.g., "1.", "1.1", "(a)", "(i)")
"""

import re
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


@dataclass
class ClauseChunk:
    clause_id: str
    text: str
    section_title: str
    source_doc: str
    source_name: str
    page_number: int
    category: str
    jurisdiction: str
    effective_date: str = ""

    def to_dict(self):
        return asdict(self)


# Regex patterns for detecting clause boundaries in different regulation styles
CLAUSE_PATTERNS = [
    # ASIC RG style: "RG 271.27" or "RG 271.27(a)"
    (r'(RG\s+\d+\.\d+(?:\([a-z]\))?)', "ASIC_RG"),
    # APRA CPS/SPS style: "CPS 220.14" or "SPS 220.5"
    (r'((?:CPS|SPS|APS|GPS|LPS|HPS)\s+\d+\.\d+)', "APRA"),
    # CFPB section references
    (r'(Section\s+\d+(?:\.\d+)?)', "CFPB"),
    # Generic numbered clauses: "1.", "1.1", "1.1.1"
    (r'(?:^|\n)(\d+(?:\.\d+)*\.)\s', "GENERIC"),
    # Lettered sub-clauses: "(a)", "(b)", "(i)", "(ii)"
    (r'(\([a-z]\)|\([ivxlc]+\))', "SUB_CLAUSE"),
]

# Category keywords for auto-classification
CATEGORY_KEYWORDS = {
    "timeframe": ["days", "calendar days", "business days", "timeframe", "within", "deadline", "response time", "maximum"],
    "definition": ["means", "defined as", "definition", "expression of dissatisfaction", "complaint is"],
    "response_content": ["must include", "IDR response", "written response", "content requirements", "reasons for"],
    "identification": ["identify", "recogni", "capture", "classify", "broad interpretation"],
    "recordkeeping": ["record", "recording", "log", "document", "maintain records", "systems for recording"],
    "reporting": ["report to", "reporting", "senior management", "board", "regular report"],
    "enforcement_finding": ["review found", "non-compliant", "failed to meet", "did not comply", "proportion of"],
    "channel_handling": ["social media", "online", "channel", "phone", "email", "in person"],
    "scope": ["scope", "applies to", "covers", "expansive approach"],
    "exception": ["exception", "does not apply", "exempt", "excluded"],
    "transparency": ["published", "public", "database", "disclosure"],
    "monitoring": ["monitor", "analyze", "assess", "review", "supervision"],
    "intake": ["receive", "intake", "submit", "filing", "submission"],
    "referral": ["refer", "referral", "transfer", "forward", "another agency"],
}


def classify_category(text: str) -> str:
    """Auto-classify a clause into a category based on keyword matching."""
    text_lower = text.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score
    if scores:
        return max(scores, key=scores.get)
    return "general"


def detect_jurisdiction(text: str, filename: str = "") -> str:
    """Detect jurisdiction from content or filename."""
    combined = (text + " " + filename).upper()
    if any(kw in combined for kw in ["ASIC", "APRA", "AFCA", "RG 271", "CPS ", "SPS ", "AUSTRALIAN"]):
        return "AU"
    if any(kw in combined for kw in ["CFPB", "FEDERAL RESERVE", "DODD-FRANK", "SEC ", "OCC "]):
        return "US"
    if any(kw in combined for kw in ["FCA ", "PRA ", "FSA "]):
        return "UK"
    return "UNKNOWN"


def extract_section_title(text: str, prev_text: str = "") -> str:
    """Extract a section title from nearby text or generate one from content."""
    # Look for common heading patterns
    heading_patterns = [
        r'^([A-Z][A-Za-z\s]+(?:of|for|and|the|in|to|by)\s[A-Za-z\s]+)$',
        r'^([A-Z][A-Za-z\s]{5,60})$',
    ]
    lines = text.split('\n')
    for line in lines[:3]:
        line = line.strip()
        for pattern in heading_patterns:
            match = re.match(pattern, line)
            if match:
                return match.group(1).strip()

    # Fallback: first 8 words
    words = text.split()[:8]
    return " ".join(words) + "..."


def parse_pdf(filepath: str, doc_name: str = "", jurisdiction: str = "") -> list[ClauseChunk]:
    """
    Parse a compliance PDF into clause-level chunks with metadata.

    This is the core fix for Aparna's ANZ problem. Instead of splitting by page,
    we detect clause boundaries and keep each clause as one unit.
    """
    if not HAS_PDFPLUMBER:
        raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")

    chunks = []
    full_text = ""
    page_texts = []

    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            page_texts.append((i + 1, page_text))
            full_text += page_text + "\n"

    if not jurisdiction:
        jurisdiction = detect_jurisdiction(full_text, filepath)

    if not doc_name:
        doc_name = os.path.basename(filepath).replace(".pdf", "").replace(" ", "_")

    # Try structured clause extraction first
    clause_chunks = _extract_clauses_structured(full_text, page_texts, doc_name, jurisdiction)

    if clause_chunks:
        return clause_chunks

    # Fallback: paragraph-based chunking with overlap
    return _extract_clauses_paragraph(full_text, page_texts, doc_name, jurisdiction)


def _extract_clauses_structured(
    full_text: str,
    page_texts: list,
    doc_name: str,
    jurisdiction: str
) -> list[ClauseChunk]:
    """Extract clauses using regex pattern matching on clause IDs."""
    chunks = []

    # Find all clause markers and their positions
    markers = []
    for pattern, style in CLAUSE_PATTERNS:
        if style == "SUB_CLAUSE":
            continue  # Sub-clauses get merged into parent
        for match in re.finditer(pattern, full_text):
            markers.append({
                "id": match.group(1).strip().rstrip('.'),
                "start": match.start(),
                "style": style,
            })

    if len(markers) < 3:
        return []  # Not enough structure, use fallback

    # Sort by position
    markers.sort(key=lambda m: m["start"])

    # Extract text between consecutive markers
    for i, marker in enumerate(markers):
        start = marker["start"]
        end = markers[i + 1]["start"] if i + 1 < len(markers) else len(full_text)
        text = full_text[start:end].strip()

        if len(text) < 20:
            continue  # Skip tiny fragments

        # Determine page number
        page_num = _find_page_number(start, page_texts, full_text)

        # Clean up clause ID
        clause_id = marker["id"].replace(" ", "_").replace(".", "_")

        chunk = ClauseChunk(
            clause_id=clause_id,
            text=text,
            section_title=extract_section_title(text),
            source_doc=doc_name,
            source_name=doc_name.replace("_", " "),
            page_number=page_num,
            category=classify_category(text),
            jurisdiction=jurisdiction,
        )
        chunks.append(chunk)

    return chunks


def _extract_clauses_paragraph(
    full_text: str,
    page_texts: list,
    doc_name: str,
    jurisdiction: str
) -> list[ClauseChunk]:
    """Fallback: split by paragraphs with smart boundary detection."""
    chunks = []
    paragraphs = re.split(r'\n\s*\n', full_text)

    clause_counter = 1
    for para in paragraphs:
        para = para.strip()
        if len(para) < 50:
            continue  # Skip headers, footers, short fragments

        # Find page number for this paragraph
        page_num = 1
        para_pos = full_text.find(para)
        if para_pos >= 0:
            page_num = _find_page_number(para_pos, page_texts, full_text)

        clause_id = f"{doc_name}_P{clause_counter:03d}"

        chunk = ClauseChunk(
            clause_id=clause_id,
            text=para,
            section_title=extract_section_title(para),
            source_doc=doc_name,
            source_name=doc_name.replace("_", " "),
            page_number=page_num,
            category=classify_category(para),
            jurisdiction=jurisdiction,
        )
        chunks.append(chunk)
        clause_counter += 1

    return chunks


def _find_page_number(position: int, page_texts: list, full_text: str) -> int:
    """Find which page a character position falls on."""
    current_pos = 0
    for page_num, page_text in page_texts:
        current_pos += len(page_text) + 1  # +1 for newline separator
        if position < current_pos:
            return page_num
    return page_texts[-1][0] if page_texts else 1


def load_sample_regulations(filepath: str) -> list[ClauseChunk]:
    """Load pre-built sample regulation data from JSON."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [ClauseChunk(**item) for item in data]


def chunks_to_json(chunks: list[ClauseChunk]) -> str:
    """Serialize chunks to JSON for storage."""
    return json.dumps([c.to_dict() for c in chunks], indent=2)


if __name__ == "__main__":
    # Test with sample data
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_regulations.json")
    if os.path.exists(sample_path):
        chunks = load_sample_regulations(sample_path)
        print(f"Loaded {len(chunks)} sample clauses")
        for c in chunks[:3]:
            print(f"  [{c.clause_id}] {c.category} ({c.jurisdiction}) - {c.text[:60]}...")
    else:
        print("Sample data not found. Run from project root.")
