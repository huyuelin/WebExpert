"""Preference-Optimized Planning with Coverage-Aware SFT (Section 3.4).

Implements:
  - L_plan: Facet-aligned token-weighted planning loss (Eq. 3)
  - L_ret: Contrastive retrieval objective (Eq. 4)
  - Coverage-aware SFT
  - Pairwise preference learning for DPO
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class PreferenceOptimizer:
    """Prepare and manage preference-optimized training data.

    Implements the training objectives described in Section 3.4:
    - L_plan: Token-weighted planning loss with facet alignment
    - L_ret: Contrastive retrieval margin loss
    - Coverage-aware SFT
    - Pairwise preference pairs for DPO training

    Note: Actual model training uses Pai-Megatron-Patch with
    full-parameter fine-tuning of QwQ-32B.  This module handles
    data preparation, preference pair curation, and hard-negative
    mining.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        score_margin: float = 0.05,
        top_k_negatives: int = 64,
    ):
        self.temperature = temperature
        self.score_margin = score_margin
        self.top_k_negatives = top_k_negatives

    def compute_facet_weights(
        self,
        tokens: List[str],
        facet_indicators: Dict[str, List[str]],
    ) -> List[float]:
        """Compute per-token facet alignment weights w(y_t; phi(E^{(k)})).

        phi(E^{(k)}) maps retrieved experiences to facet indicators
        (time, region, policy, industry) and associated keywords.
        w(.) up-weights tokens that activate these indicators and
        down-weights off-facet tokens.
        """
        all_facet_terms = set()
        for terms in facet_indicators.values():
            all_facet_terms.update(t.lower() for t in terms)

        weights = []
        for token in tokens:
            token_lower = token.lower().strip()
            if any(term in token_lower for term in all_facet_terms):
                weights.append(2.0)
            else:
                weights.append(0.5)
        return weights

    def mine_hard_negatives(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        candidate_ids: List[str],
        positive_id: str,
    ) -> List[str]:
        """Mine hard negatives for the contrastive retrieval loss.

        Hard negatives are candidates from top-k FAISS candidates
        (excluding positives) with score margin within 0.05,
        refreshed every epoch.
        """
        q_emb = np.array(query_embedding)
        c_embs = np.array(candidate_embeddings)
        q_norm = q_emb / max(np.linalg.norm(q_emb), 1e-10)
        c_norms = c_embs / np.maximum(
            np.linalg.norm(c_embs, axis=1, keepdims=True), 1e-10
        )
        scores = c_norms @ q_norm

        pos_idx = None
        for i, cid in enumerate(candidate_ids):
            if cid == positive_id:
                pos_idx = i
                break
        if pos_idx is None:
            return []

        pos_score = scores[pos_idx]
        hard_neg_ids = []
        for i, (score, cid) in enumerate(zip(scores, candidate_ids)):
            if cid == positive_id:
                continue
            if pos_score - score <= self.score_margin:
                hard_neg_ids.append(cid)
        return hard_neg_ids

    def compute_coverage(
        self,
        generated_plan: List[Dict[str, str]],
        facet_vocabulary: Dict[str, List[str]],
    ) -> float:
        """Compute facet coverage of a generated query plan."""
        if not generated_plan or not facet_vocabulary:
            return 0.0
        covered_dims = set()
        plan_text = " ".join(
            q.get("query", "") + " " + q.get("intent", "")
            for q in generated_plan
        ).lower()
        for dim, terms in facet_vocabulary.items():
            for term in terms:
                if term.lower() in plan_text:
                    covered_dims.add(dim)
                    break
        return len(covered_dims) / max(len(facet_vocabulary), 1)

    def create_preference_pairs(
        self,
        question: str,
        plans: List[List[Dict[str, str]]],
        facet_vocabulary: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        """Create pairwise preference pairs from multiple candidate plans.

        Positives emphasize facet-aligned plans; negatives suppress
        off-facet or redundant plans.  Ranking uses a composite of
        coverage and facet alignment scores.
        """
        scored = []
        for plan in plans:
            coverage = self.compute_coverage(plan, facet_vocabulary)
            plan_text = " ".join(
                q.get("query", "") + " " + q.get("intent", "")
                for q in plan
            ).lower()
            all_terms = set()
            for terms in facet_vocabulary.values():
                all_terms.update(t.lower() for t in terms)
            alignment = sum(1 for t in all_terms if t in plan_text) / max(
                len(all_terms), 1
            )
            score = 0.6 * coverage + 0.4 * alignment
            scored.append((plan, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        pairs = []
        for i in range(len(scored)):
            for j in range(i + 1, len(scored)):
                if scored[i][1] > scored[j][1]:
                    pairs.append({
                        "question": question,
                        "chosen": scored[i][0],
                        "rejected": scored[j][0],
                        "chosen_score": scored[i][1],
                        "rejected_score": scored[j][1],
                    })
        return pairs

    def save_preference_data(self, pairs: List[Dict[str, Any]], path: str) -> None:
        """Save preference pairs to JSONL for DPO training."""
        with open(path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    def load_preference_data(self, path: str) -> List[Dict[str, Any]]:
        """Load preference pairs from JSONL."""
        pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        return pairs
