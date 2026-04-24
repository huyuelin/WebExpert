"""Schema-Light Facet Induction (Contribution iii of the paper).

Bootstraps time/region/policy/industry facets from weak supervision
and corpus statistics instead of static hand-written lexicons.
This reduces manual schema dependence and enables automatic
discovery of domain-specific facet vocabularies.
"""

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


class FacetInducer:
    """Induce facet vocabularies from corpus statistics and weak supervision.

    Facet dimensions:
    - time: temporal scope (e.g., '2024', 'ongoing', '1990-2020')
    - region: geographic scope (e.g., 'US', 'EU', 'China', 'universal')
    - policy: regulatory/policy references (e.g., 'GDPR', 'FDA approval')
    - industry: L2 industry classification (e.g., 'biotech', 'fintech')

    Unlike static lexicon-based approaches, FacetInducer discovers
    facet values from the corpus itself using frequency analysis,
    pattern matching, and optional weak supervision signals.
    """

    # Weak supervision patterns for facet extraction
    TIME_PATTERNS = [
        re.compile(r"\b(19|20)\d{2}\b"),           # Years
        re.compile(r"\b(19|20)\d{2}[-–](19|20)\d{2}\b"),  # Year ranges
        re.compile(r"\b(Q[1-4]\s*(19|20)\d{2})\b"),  # Quarters
        re.compile(r"\b(january|february|march|april|may|june|july|"
                   r"august|september|october|november|december)\b", re.I),
        re.compile(r"\b(recent|current|ongoing|latest|historical)\b", re.I),
    ]

    REGION_PATTERNS = [
        re.compile(r"\b(US|USA|United States|America)\b", re.I),
        re.compile(r"\b(EU|Europe|European Union)\b", re.I),
        re.compile(r"\b(China|Chinese)\b", re.I),
        re.compile(r"\b(UK|United Kingdom|Britain)\b", re.I),
        re.compile(r"\b(global|worldwide|international|universal)\b", re.I),
    ]

    POLICY_KEYWORDS = {
        "act", "law", "regulation", "directive", "mandate", "policy",
        "compliance", "standard", "approval", "certification", "framework",
        "guideline", "ordinance", "legislation",
    }

    def __init__(
        self,
        corpus: Optional[List[str]] = None,
        domain_hints: Optional[Dict[str, List[str]]] = None,
        min_frequency: int = 2,
        max_facets_per_dim: int = 20,
    ):
        self.min_frequency = min_frequency
        self.max_facets_per_dim = max_facets_per_dim
        self.domain_hints = domain_hints or {}
        self.facet_vocabulary: Dict[str, List[str]] = {
            "time": [],
            "region": [],
            "policy": [],
            "industry": [],
        }
        if corpus:
            self.induce_from_corpus(corpus)

    def induce_from_corpus(self, texts: List[str]) -> Dict[str, List[str]]:
        """Induce facet vocabularies from a text corpus.

        Uses frequency analysis over pattern-matched candidates
        and weak supervision from domain hints.  Only candidates
        appearing at least min_frequency times are retained.
        """
        time_candidates: Counter = Counter()
        region_candidates: Counter = Counter()
        policy_candidates: Counter = Counter()
        industry_candidates: Counter = Counter()

        for text in texts:
            for pat in self.TIME_PATTERNS:
                for m in pat.finditer(text):
                    time_candidates[m.group(0)] += 1
            for pat in self.REGION_PATTERNS:
                for m in pat.finditer(text):
                    region_candidates[m.group(0)] += 1
            # Policy: look for capitalized phrases near policy keywords
            words = text.split()
            for i, w in enumerate(words):
                if w.lower().rstrip(".,;:") in self.POLICY_KEYWORDS:
                    # Grab preceding 1-3 word phrase as policy name
                    start = max(0, i - 3)
                    phrase = " ".join(words[start:i + 1]).rstrip(".,;:")
                    if phrase:
                        policy_candidates[phrase] += 1

        # Add domain hints (weak supervision)
        for dim, hints in self.domain_hints.items():
            counter = {"time": time_candidates, "region": region_candidates,
                       "policy": policy_candidates, "industry": industry_candidates}
            if dim in counter:
                for hint in hints:
                    counter[dim][hint] += self.min_frequency

        # Filter by frequency and build vocabulary
        self.facet_vocabulary["time"] = [
            t for t, c in time_candidates.most_common(self.max_facets_per_dim)
            if c >= self.min_frequency
        ]
        self.facet_vocabulary["region"] = [
            t for t, c in region_candidates.most_common(self.max_facets_per_dim)
            if c >= self.min_frequency
        ]
        self.facet_vocabulary["policy"] = [
            t for t, c in policy_candidates.most_common(self.max_facets_per_dim)
            if c >= self.min_frequency
        ]
        self.facet_vocabulary["industry"] = list(self.domain_hints.get("industry", []))

        return self.facet_vocabulary

    def tag_text(self, text: str) -> Dict[str, List[str]]:
        """Tag a text with active facets from the induced vocabulary."""
        active: Dict[str, List[str]] = {"time": [], "region": [], "policy": [], "industry": []}
        text_lower = text.lower()
        for dim, vocab in self.facet_vocabulary.items():
            for term in vocab:
                if term.lower() in text_lower:
                    active[dim].append(term)
        return active

    def facetize_rule(self, rule_text: str, sentences: List[str] = None) -> Dict[str, str]:
        """Facetize a distilled rule into (time, region, policy, industry).

        Returns a structured facet dict with the most specific
        match for each dimension.
        """
        full_text = rule_text
        if sentences:
            full_text = rule_text + " " + " ".join(sentences)
        active = self.tag_text(full_text)

        facets = {
            "time": active["time"][0] if active["time"] else "ongoing",
            "region": active["region"][0] if active["region"] else "universal",
            "policy": ", ".join(active["policy"][:3]) if active["policy"] else "",
            "industry": active["industry"][0] if active["industry"] else "",
        }
        return facets
