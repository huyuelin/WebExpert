"""Experience Gate with Confidence-Based Fallback (Section 3.3, Step 2).

The experience gate biases decoding toward active facets when
retrieval confidence is high (avg cosine sim >= theta).
When confidence < theta (default 0.3), it falls back to generic
query generation to avoid over-constraint.

P(z | q, E^{(k)}) = Prod_{j=1}^{M} P(z_j | z_{<j}, q, E^{(k)})
"""

from typing import Any, Dict, List, Optional


class ExperienceGate:
    """Lightweight experience gate for inference-time facet biasing.

    The gate computes retrieval confidence as the average cosine
    similarity of top-k experiences.  When confidence is above
    threshold theta, the gate biases decoding toward active facets;
    when below, it falls back to generic query generation.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        top_k: int = 5,
    ):
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k

    def compute_confidence(self, scores: List[float]) -> float:
        """Compute retrieval confidence from top-k similarity scores.

        Confidence = mean(top-k cosine similarities).
        Calibrated on validation set; default threshold theta = 0.3.
        """
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def gate(
        self,
        query: str,
        retrieved_experiences: List[Dict[str, Any]],
        scores: List[float],
        active_facets: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """Apply the experience gate to decide whether to bias decoding.

        Returns a dict with:
        - 'mode': 'facet_biased' or 'generic_fallback'
        - 'confidence': retrieval confidence score
        - 'active_facets': facets to bias toward (if mode is facet_biased)
        - 'experiences': the retrieved experiences
        """
        confidence = self.compute_confidence(scores)
        mode = "facet_biased" if confidence >= self.confidence_threshold else "generic_fallback"

        if mode == "facet_biased" and active_facets:
            # Collect facet-keywords from retrieved experiences
            facet_keywords = self._extract_facet_keywords(
                retrieved_experiences, active_facets
            )
        else:
            facet_keywords = {}

        return {
            "mode": mode,
            "confidence": confidence,
            "active_facets": active_facets if mode == "facet_biased" else {},
            "facet_keywords": facet_keywords,
            "experiences": retrieved_experiences[:self.top_k],
        }

    def _extract_facet_keywords(
        self,
        experiences: List[Dict[str, Any]],
        active_facets: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """Extract facet-relevant keywords from retrieved experiences.

        Maps phi(E^{(k)}) to facet indicators (time, region, policy,
        industry) and associated keywords for biasing query generation.
        """
        keywords: Dict[str, List[str]] = {
            "time": [], "region": [], "policy": [], "industry": []
        }
        for exp in experiences:
            facets = exp.get("facets", {})
            for dim in keywords:
                val = facets.get(dim, "")
                if val and val not in ("ongoing", "universal", ""):
                    keywords[dim].append(val)
        # Merge with active facets from the query
        for dim, vals in active_facets.items():
            for v in vals:
                if v not in keywords.get(dim, []):
                    keywords.setdefault(dim, []).append(v)
        return keywords

    def format_experience_context(self, gate_result: Dict[str, Any]) -> str:
        """Format gate result into a prompt-ready context string."""
        if gate_result["mode"] == "generic_fallback":
            return ""
        parts = []
        for dim, kws in gate_result["facet_keywords"].items():
            if kws:
                parts.append(f"{dim}: {', '.join(kws[:3])}")
        exp_texts = []
        for exp in gate_result["experiences"][:3]:
            rule = exp.get("rule", "")
            if rule:
                exp_texts.append(rule)
        context = ""
        if parts:
            context += "Active facets: " + "; ".join(parts) + "\n"
        if exp_texts:
            context += "Expert rules:\n" + "\n".join(f"- {r}" for r in exp_texts)
        return context
