#!/bin/bash
# ComplianceVerify AI - Startup Script
# Run from the project root: ./run.sh

echo "============================================"
echo "  ComplianceVerify AI - Starting Server"
echo "============================================"
echo ""

# Check dependencies
python3 -c "import pdfplumber, faiss, fastapi, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install pdfplumber faiss-cpu rank-bm25 fastapi uvicorn anthropic scikit-learn python-multipart --break-system-packages -q
fi

echo "Starting FastAPI server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

cd "$(dirname "$0")"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
