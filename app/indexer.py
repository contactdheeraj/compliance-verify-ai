"""
Hybrid Search Indexer - FAISS + BM25

This is the second fix for Aparna's ANZ problem. Her system used only
semantic search, which missed keyword-specific matches like clause numbers
and exact regulatory terms. This hybrid approach combines:

1. FAISS with TF-IDF vectors: captures semantic similarity without needing
   a 500MB transformer model. TF-IDF on legal text performs well because
   regulatory language has distinctive vocabulary.

2. BM25: keyword search that catches exact terms, clause references, and
   specific phrases that semantic search might miss.

Ensemble scoring: weighted combination (configurable, default 50/50).
"""

import json
import os
import pickle
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from typing import Optional

from app.compliance_parser import ClauseChunk, load_sample_regulations


class ComplianceIndex:
    """Hybrid FAISS + BM25 index for compliance clause retrieval."""

    def __init__(self, semantic_weight: float = 0.5):
        self.chunks: list[ClauseChunk] = []
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight

    def build_index(self, chunks: list[ClauseChunk]):
        """Build both FAISS and BM25 indexes from clause chunks."""
        self.chunks = chunks
        texts = [c.text for c in chunks]

        if not texts:
            return

        # --- FAISS index using TF-IDF vectors ---
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # unigrams + bigrams for better phrase matching
            sublinear_tf=True,
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.tfidf_matrix = tfidf_matrix

        # Convert sparse TF-IDF to dense numpy and normalize for cosine similarity
        dense_vectors = tfidf_matrix.toarray().astype('float32')
        norms = np.linalg.norm(dense_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        dense_vectors = dense_vectors / norms

        # Build FAISS inner product index (cosine similarity on normalized vectors)
        dimension = dense_vectors.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(dense_vectors)

        # --- BM25 index ---
        tokenized = [text.lower().split() for text in texts]
        self.bm25_index = BM25Okapi(tokenized)

    def search(
        self,
        query: str,
        jurisdiction: Optional[str] = None,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> list[dict]:
        """
        Hybrid search: combines FAISS semantic scores with BM25 keyword scores.

        Returns ranked list of {chunk, score, faiss_score, bm25_score}.
        """
        if not self.chunks:
            return []

        n_chunks = len(self.chunks)

        # --- FAISS semantic search ---
        query_tfidf = self.tfidf_vectorizer.transform([query])
        query_vector = query_tfidf.toarray().astype('float32')
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        faiss_scores_raw, faiss_indices = self.faiss_index.search(query_vector, min(n_chunks, 50))
        faiss_scores_raw = faiss_scores_raw[0]
        faiss_indices = faiss_indices[0]

        # Normalize FAISS scores to 0-1
        faiss_score_map = {}
        max_faiss = max(faiss_scores_raw) if len(faiss_scores_raw) > 0 else 1
        min_faiss = min(faiss_scores_raw) if len(faiss_scores_raw) > 0 else 0
        range_faiss = max_faiss - min_faiss if max_faiss != min_faiss else 1

        for idx, score in zip(faiss_indices, faiss_scores_raw):
            if idx >= 0:
                faiss_score_map[int(idx)] = (score - min_faiss) / range_faiss

        # --- BM25 keyword search ---
        query_tokens = query.lower().split()
        bm25_scores_raw = self.bm25_index.get_scores(query_tokens)

        # Normalize BM25 scores to 0-1
        max_bm25 = max(bm25_scores_raw) if len(bm25_scores_raw) > 0 else 1
        min_bm25 = min(bm25_scores_raw) if len(bm25_scores_raw) > 0 else 0
        range_bm25 = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1

        bm25_score_map = {}
        for i, score in enumerate(bm25_scores_raw):
            bm25_score_map[i] = (score - min_bm25) / range_bm25

        # --- Combine scores ---
        combined = []
        for i in range(n_chunks):
            chunk = self.chunks[i]

            # Apply jurisdiction filter
            if jurisdiction and chunk.jurisdiction != jurisdiction:
                continue

            # Apply category filter
            if category and chunk.category != category:
                continue

            f_score = faiss_score_map.get(i, 0.0)
            b_score = bm25_score_map.get(i, 0.0)
            ensemble = (self.semantic_weight * f_score) + (self.keyword_weight * b_score)

            combined.append({
                "chunk": chunk.to_dict(),
                "score": round(float(ensemble), 4),
                "faiss_score": round(float(f_score), 4),
                "bm25_score": round(float(b_score), 4),
            })

        # Sort by combined score descending
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]

    def save(self, directory: str):
        """Save indexes to disk for fast reload."""
        os.makedirs(directory, exist_ok=True)

        # Save chunks
        with open(os.path.join(directory, "chunks.json"), "w") as f:
            json.dump([c.to_dict() for c in self.chunks], f)

        # Save FAISS index
        if self.faiss_index:
            faiss.write_index(self.faiss_index, os.path.join(directory, "faiss.index"))

        # Save TF-IDF vectorizer and BM25
        with open(os.path.join(directory, "tfidf.pkl"), "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)

        with open(os.path.join(directory, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25_index, f)

        # Save config
        config = {
            "semantic_weight": self.semantic_weight,
            "keyword_weight": self.keyword_weight,
            "n_chunks": len(self.chunks),
        }
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f)

    def load(self, directory: str):
        """Load indexes from disk."""
        # Load chunks
        with open(os.path.join(directory, "chunks.json"), "r") as f:
            data = json.load(f)
        self.chunks = [ClauseChunk(**item) for item in data]

        # Load FAISS
        index_path = os.path.join(directory, "faiss.index")
        if os.path.exists(index_path):
            self.faiss_index = faiss.read_index(index_path)

        # Load TF-IDF
        with open(os.path.join(directory, "tfidf.pkl"), "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)

        # Load BM25
        with open(os.path.join(directory, "bm25.pkl"), "rb") as f:
            self.bm25_index = pickle.load(f)

        # Load config
        with open(os.path.join(directory, "config.json"), "r") as f:
            config = json.load(f)
        self.semantic_weight = config["semantic_weight"]
        self.keyword_weight = config["keyword_weight"]

    def get_stats(self) -> dict:
        """Return index statistics."""
        jurisdictions = {}
        categories = {}
        sources = {}
        for c in self.chunks:
            jurisdictions[c.jurisdiction] = jurisdictions.get(c.jurisdiction, 0) + 1
            categories[c.category] = categories.get(c.category, 0) + 1
            sources[c.source_doc] = sources.get(c.source_doc, 0) + 1

        return {
            "total_clauses": len(self.chunks),
            "jurisdictions": jurisdictions,
            "categories": categories,
            "sources": sources,
        }


# Singleton index instance
_global_index: Optional[ComplianceIndex] = None


def get_index() -> ComplianceIndex:
    """Get or create the global compliance index."""
    global _global_index
    if _global_index is None:
        _global_index = ComplianceIndex()
    return _global_index


def initialize_with_sample_data(sample_path: str, index_dir: str = None):
    """Load sample regulations and build the index."""
    idx = get_index()
    chunks = load_sample_regulations(sample_path)
    idx.build_index(chunks)
    if index_dir:
        idx.save(index_dir)
    return idx


if __name__ == "__main__":
    # Test the indexer
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_regulations.json")
    idx = initialize_with_sample_data(sample_path)

    print(f"Index stats: {json.dumps(idx.get_stats(), indent=2)}")
    print()

    # Test queries
    test_queries = [
        ("Bank didn't respond to my complaint for 45 days", "AU"),
        ("They didn't tell me why my complaint was rejected", "AU"),
        ("Company took 3 months to respond to my complaint", "US"),
        ("I posted on their Facebook page and they ignored it", "AU"),
    ]

    for query, jurisdiction in test_queries:
        print(f"Query: {query} [{jurisdiction}]")
        results = idx.search(query, jurisdiction=jurisdiction, top_k=3)
        for r in results:
            print(f"  [{r['chunk']['clause_id']}] score={r['score']} "
                  f"(faiss={r['faiss_score']}, bm25={r['bm25_score']}) "
                  f"- {r['chunk']['text'][:80]}...")
        print()
