"""Deep Web Explorer Sub-Agent (Section 3.3, Step 3).

Interleaves retrieval D and reasoning R to produce answers
following the reasoning-with-experiences paradigm of Eq. (1).
"""

import re
from typing import Any, Callable, Dict, List, Optional, Set


class DeepWebExplorer:
    """Nested deep web explorer with iterative search and click.

    Given a search query and intent, performs multi-step exploration:
    1. Analyze initial search results
    2. Click relevant URLs for deeper information
    3. Optionally issue follow-up searches
    4. Return consolidated information
    """

    BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
    END_SEARCH_QUERY = "<|end_search_query|>"
    BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
    END_SEARCH_RESULT = "<|end_search_result|>"
    BEGIN_CLICK_LINK = "<|begin_click_link|>"
    END_CLICK_LINK = "<|end_click_link|>"
    BEGIN_CLICK_RESULT = "<|begin_click_result|>"
    END_CLICK_RESULT = "<|end_click_result|>"

    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        search_fn: Optional[Callable] = None,
        click_fn: Optional[Callable] = None,
        summarize_fn: Optional[Callable] = None,
        max_interactions: int = 10,
        top_k: int = 10,
    ):
        self.llm_fn = llm_fn
        self.search_fn = search_fn
        self.click_fn = click_fn
        self.summarize_fn = summarize_fn
        self.max_interactions = max_interactions
        self.top_k = top_k

    def build_exploration_prompt(
        self, search_query: str, search_intent: str, search_result: str,
    ) -> str:
        """Build the deep exploration instruction prompt."""
        return f"""You are a domain-aware web explorer. Analyze search results to find relevant information for the given query and intent. Prefer clicking URLs for deeper information when available.

**Guidelines:**

1. **Analyze initial search results:** Review each result for facts relevant to the search query and intent.
2. **Get more information:** If insufficient, use:
   - Click: {self.BEGIN_CLICK_LINK}target URL{self.END_CLICK_LINK}
   - Search: {self.BEGIN_SEARCH_QUERY}new query{self.END_SEARCH_QUERY}
3. **Extract relevant information:** Return information related to the search query and intent.
4. **Output format:**
   If further exploration is needed:
   **Click URL**
   {self.BEGIN_CLICK_LINK}URL{self.END_CLICK_LINK}
   OR
   **Search**
   {self.BEGIN_SEARCH_QUERY}new query{self.END_SEARCH_QUERY}
   If information is sufficient:
   **Final Information**
   [relevant information]

**Input:**
- **Search query:** {search_query}
- **Search intent:** {search_intent}
- **Search results:** {search_result}

Now analyze and extract information relevant to "{search_query}" and its intent.
"""

    def explore(
        self,
        search_query: str,
        search_intent: str,
        search_result: str,
        search_cache: Optional[Dict] = None,
        url_cache: Optional[Dict] = None,
        global_executed_queries: Optional[Set[str]] = None,
    ) -> str:
        """Run the deep web exploration loop.

        Returns the accumulated output text from the sub-agent.
        """
        if self.llm_fn is None:
            raise RuntimeError("No LLM function provided.")

        search_cache = search_cache or {}
        url_cache = url_cache or {}
        global_executed_queries = global_executed_queries or set()

        prompt = self.build_exploration_prompt(
            search_query, search_intent, search_result
        )
        executed_queries: Set[str] = set()
        clicked_urls: Set[str] = set()
        interactions = 0
        output_acc = ""

        while True:
            response = self.llm_fn(
                prompt=prompt,
                stop=[self.END_SEARCH_QUERY, self.END_CLICK_LINK],
                echo_stream=False,
            )
            output_acc += response
            prompt += response

            if response.rstrip().endswith(self.END_SEARCH_QUERY):
                if interactions >= self.max_interactions:
                    prompt += (
                        f"\n{self.BEGIN_SEARCH_RESULT}"
                        f"Deep search limit reached"
                        f"{self.END_SEARCH_RESULT}\n"
                    )
                    continue
                new_q = self._extract_between(
                    response, self.BEGIN_SEARCH_QUERY, self.END_SEARCH_QUERY
                )
                if (
                    not new_q
                    or new_q in executed_queries
                    or new_q in global_executed_queries
                ):
                    prompt += (
                        f"\n{self.BEGIN_SEARCH_RESULT}"
                        f"Already searched '{new_q}'"
                        f"{self.END_SEARCH_RESULT}\n"
                    )
                    continue
                executed_queries.add(new_q)
                global_executed_queries.add(new_q)
                interactions += 1
                if self.search_fn and new_q not in search_cache:
                    search_cache[new_q] = self.search_fn(new_q)

            elif response.rstrip().endswith(self.END_CLICK_LINK):
                if self.click_fn is None:
                    prompt += (
                        f"\n{self.BEGIN_CLICK_RESULT}"
                        f"Click tool unavailable"
                        f"{self.END_CLICK_RESULT}\n"
                    )
                    continue
                url = self._extract_between(
                    response, self.BEGIN_CLICK_LINK, self.END_CLICK_LINK
                )
                if not url or url in clicked_urls:
                    prompt += (
                        f"\n{self.BEGIN_CLICK_RESULT}"
                        f"Already clicked '{url}'"
                        f"{self.END_CLICK_RESULT}\n"
                    )
                    continue
                clicked_urls.add(url)
                interactions += 1

            else:
                break

        return output_acc

    @staticmethod
    def _extract_between(text: str, start: str, end: str) -> str:
        pattern = re.escape(start) + r"(.*?)" + re.escape(end)
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else ""
