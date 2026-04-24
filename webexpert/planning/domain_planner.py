"""Domain-Grounded Query Generation (Section 3.3, Step 2).

Produces a multi-query plan z = (z_1, ..., z_M) conditioned on
q and E^{(k)}, with an experience gate that biases decoding toward
active facets.

P(z | q, E^{(k)}) = Prod_{j=1}^{M} P(z_j | z_{<j}, q, E^{(k)})
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional


class DomainPlanner:
    """Generate domain-grounded multi-query plans.

    Given a question q and retrieved experiences E^{(k)}, produces
    a structured search plan with domain, query, and intent fields.
    The experience gate biases query generation toward active facets
    when retrieval confidence is high, and falls back to generic
    generation when confidence is low.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        max_queries: int = 3,
    ):
        self.llm_fn = llm_fn
        self.max_queries = max_queries

    def build_planning_prompt(
        self,
        question: str,
        experience_context: str = "",
        facet_bias: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """Build the domain analysis prompt.

        Incorporates expert experiences and active facet keywords
        to ground the query plan in domain-specific terminology.
        """
        facet_str = ""
        if facet_bias:
            parts = []
            for dim, kws in facet_bias.items():
                if kws:
                    parts.append(f"{dim}: {', '.join(kws[:3])}")
            if parts:
                facet_str = "Active facets to cover: " + "; ".join(parts)

        return f"""You are a domain-aware web search planner. Given a question and expert experiences, generate {self.max_queries} targeted search queries.

{experience_context}
{facet_str}

### Question ###
{question}

### Output Format ###
For each search query, provide:
- Domain: the L1/L2 category
- Query: specific search keywords using domain terminology
- Intent: what information this query aims to find

**Query 1:** Domain: ...
Search keywords: ...
Intent: ...

**Query 2:** Domain: ...
Search keywords: ...
Intent: ...
"""

    def parse_plan(self, raw_response: str) -> List[Dict[str, str]]:
        """Parse the LLM response into structured query plans.

        Supports both English and Chinese output formats.
        """
        targets = []

        # English format
        pattern_en = re.compile(
            r"\*\*Query\s*\d+\s*[：:]\s*\*\*\s*Domain\s*[：:]\s*([^\n]+)\s*\n"
            r"\s*Search\s+keywords\s*[：:]\s*([^\n]+)\s*\n"
            r"\s*Intent\s*[：:]\s*([^\n]+)",
            re.IGNORECASE,
        )
        for m in pattern_en.finditer(raw_response):
            targets.append({
                "domain": m.group(1).strip(),
                "query": m.group(2).strip(),
                "intent": m.group(3).strip(),
            })

        # Chinese format fallback
        if not targets:
            pattern_zh = re.compile(
                r"\*\*第[^*]*?搜索查询及意图[：:]\s*\*\*\s*领域[：:]\s*([^\n]+)\s*\n"
                r"\s*(?:\*\*)?搜索关键[词字](?:\*\*)?[：:]\s*([^\n]+)\s*\n"
                r"\s*(?:\*\*)?意图(?:\*\*)?[：:]\s*([^\n]+)",
            )
            for m in pattern_zh.finditer(raw_response):
                targets.append({
                    "domain": m.group(1).strip(),
                    "query": m.group(2).strip(),
                    "intent": m.group(3).strip(),
                })

        return targets[:self.max_queries]

    def plan(
        self,
        question: str,
        experience_context: str = "",
        facet_bias: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, str]]:
        """Generate a domain-grounded multi-query plan.

        Args:
            question: The input question q.
            experience_context: Formatted experience context from the gate.
            facet_bias: Active facet keywords for biasing generation.

        Returns:
            A list of dicts with domain, query, and intent.
        """
        prompt = self.build_planning_prompt(question, experience_context, facet_bias)
        if self.llm_fn is None:
            raise RuntimeError("No LLM function provided. Set llm_fn before calling plan().")
        response = self.llm_fn(prompt=prompt, stop=[], echo_stream=False)
        return self.parse_plan(response)
