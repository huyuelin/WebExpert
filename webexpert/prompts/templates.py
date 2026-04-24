"""Prompt templates for WebExpert inference and training."""


def get_expert_planning_prompt(
    question: str,
    experience_context: str = "",
    facet_keywords: dict | None = None,
    max_queries: int = 3,
) -> str:
    """Build the domain analysis prompt for query planning."""
    facet_str = ""
    if facet_keywords:
        parts = []
        for dim, kws in facet_keywords.items():
            if kws:
                parts.append(f"{dim}: {', '.join(kws[:3])}")
        if parts:
            facet_str = "Active facets to cover: " + "; ".join(parts)

    return f"""You are a domain-aware web search planner. Given a question and expert experiences, generate {max_queries} targeted search queries.

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
"""


def get_deep_exploration_prompt(search_query: str, search_intent: str, search_result: str) -> str:
    """Build the deep exploration instruction prompt."""
    return f"""You are a domain-aware web explorer. Analyze search results to find relevant information for the given query and intent. Prefer clicking URLs for deeper information when available.

**Guidelines:**
1. Analyze initial search results for facts relevant to the search query.
2. If insufficient, click URLs for deeper information or issue follow-up searches.
3. Return only information relevant to the search query and intent.

**Input:**
- **Search query:** {search_query}
- **Search intent:** {search_intent}
- **Search results:** {search_result}
"""


def get_summarization_prompt(url: str, raw_content: str, max_chars: int = 8000) -> str:
    """Build the web content summarization prompt."""
    trimmed = raw_content[:max_chars]
    return f"""You are an information extraction assistant. Read the web page content and extract key facts relevant to the search query.

Requirements:
1. Only keep information highly relevant to the original search
2. Output in bullet points or short paragraphs, under 500 words
3. Ignore ads, navigation, and other irrelevant content
4. Include potentially useful URL links

**URL:** {url}
**Page content:** {trimmed}
"""


def get_answer_generation_prompt(question: str, search_evidence: str) -> str:
    """Build the final answer generation prompt."""
    return f"""Based on the following search evidence, provide a precise answer to the question.

### Question ###
{question}

### Search Evidence ###
{search_evidence}

### Instructions ###
Provide a complete and accurate answer. If the evidence is insufficient, state what is missing.
End your answer with: Final answer: <your concise answer>
"""
