"""Experience Retrieval Module (Section 3.3, Step 1).

Computes E^{(k)} = Top-k { s(f(q), f(r)) : r in E }
where s(u, v) = <u, v> / (||u|| ||v||) is cosine similarity.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


class ExperienceRetriever:
    """Retrieve top-k expert experiences for a given query.

    Supports both in-memory and file-backed experience bases.
    Uses cosine similarity between query and rule embeddings.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        experience_path: Optional[str] = None,
    ):
        self.model = SentenceTransformer(embedding_model)
        self.experiences: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        if experience_path:
            self.load_experiences(experience_path)

    def load_experiences(self, path: str) -> None:
        """Load experience base from JSONL file."""
        self.experiences = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.experiences.append(json.loads(line))
        self._build_index()

    def add_experiences(self, experiences: List[Dict[str, Any]]) -> None:
        """Add experiences and rebuild the index."""
        self.experiences.extend(experiences)
        self._build_index()

    def _build_index(self) -> None:
        """Build embedding index over all stored experiences."""
        if not self.experiences:
            self.embeddings = None
            return
        texts = [e.get("rule", "") or " ".join(e.get("sentences", []))
                 for e in self.experiences]
        self.embeddings = self.model.encode(texts, show_progress_bar=False)

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """Retrieve top-k experiences for a query.

        Returns:
            A tuple of (retrieved_experiences, similarity_scores).
        """
        if self.embeddings is None or len(self.experiences) == 0:
            return [], []

        query_emb = self.model.encode([query], show_progress_bar=False)
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        normed = self.embeddings / norms
        query_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)
        scores = (normed @ query_norm.T).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [self.experiences[i] for i in top_indices]
        sim_scores = [float(scores[i]) for i in top_indices]
        return results, sim_scores

    def compute_confidence(self, scores: List[float]) -> float:
        """Compute average retrieval confidence from top-k scores."""
        if not scores:
            return 0.0
        return sum(scores) / len(scores)
