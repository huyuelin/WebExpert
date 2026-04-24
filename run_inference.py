#!/usr/bin/env python3
"""WebExpert Inference Pipeline.

Runs the full WebExpert pipeline:
  1. Experience retrieval for the input question
  2. Facet induction and experience gate
  3. Domain-grounded query planning
  4. Deep web exploration
  5. Answer generation

Usage:
  python run_inference.py --question "What is the GDP growth rate of China in 2024?"
  python run_inference.py --input questions.jsonl --output results.json
"""

import argparse
import json
import time
from typing import Dict, List, Optional

from webexpert.experience.retriever import ExperienceRetriever
from webexpert.experience.facet_inducer import FacetInducer
from webexpert.experience.gate import ExperienceGate
from webexpert.planning.domain_planner import DomainPlanner
from webexpert.browsing.deep_explorer import DeepWebExplorer


def run_webexpert(
    question: str,
    retriever: ExperienceRetriever,
    facet_inducer: FacetInducer,
    gate: ExperienceGate,
    planner: DomainPlanner,
    explorer: DeepWebExplorer,
    search_fn=None,
    top_k: int = 5,
) -> Dict:
    """Run the full WebExpert inference pipeline on a single question.

    Steps (Section 3.3):
      1. Experience retrieval: E^{(k)} = Top-k { s(f(q), f(r)) : r in E }
      2. Experience gate: bias toward active facets or fallback
      3. Domain-grounded query generation
      4. Deep browsing: feed z to search-and-browse controller
    """
    start_time = time.time()

    # Step 1: Experience retrieval
    experiences, scores = retriever.retrieve(question, top_k=top_k)

    # Step 2: Facet induction and experience gate
    active_facets = facet_inducer.tag_text(question)
    gate_result = gate.gate(question, experiences, scores, active_facets)

    # Step 3: Domain-grounded query planning
    experience_context = gate.format_experience_context(gate_result)
    facet_bias = gate_result.get("facet_keywords", {}) if gate_result["mode"] == "facet_biased" else None
    query_plan = planner.plan(question, experience_context, facet_bias)

    # Step 4: Deep browsing
    search_cache: Dict = {}
    url_cache: Dict = {}
    executed_queries = set()
    all_search_results = []

    for target in query_plan:
        query = target["query"]
        intent = target["intent"]
        if query in executed_queries:
            continue
        executed_queries.add(query)

        if search_fn:
            result = search_cache.get(query) or search_fn(query)
            search_cache[query] = result
            all_search_results.append({
                "query": query,
                "intent": intent,
                "domain": target.get("domain", ""),
            })

    result = {
        "question": question,
        "retrieved_experiences": len(experiences),
        "gate_mode": gate_result["mode"],
        "confidence": gate_result["confidence"],
        "active_facets": active_facets,
        "query_plan": query_plan,
        "search_results": all_search_results,
        "processing_time": time.time() - start_time,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="WebExpert Inference Pipeline")
    parser.add_argument("--question", type=str, help="Single question to process")
    parser.add_argument("--input", type=str, help="Input JSONL file with questions")
    parser.add_argument("--output", type=str, default="output/results.json", help="Output file")
    parser.add_argument("--experience-base", type=str, required=True, help="Path to experience base JSONL")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k experiences to retrieve")
    parser.add_argument("--gate-threshold", type=float, default=0.3, help="Experience gate confidence threshold")
    args = parser.parse_args()

    # Initialize components
    retriever = ExperienceRetriever(experience_path=args.experience_base)
    facet_inducer = FacetInducer()
    gate = ExperienceGate(confidence_threshold=args.gate_threshold, top_k=args.top_k)
    planner = DomainPlanner(max_queries=3)
    explorer = DeepWebExplorer(max_interactions=10, top_k=args.top_k)

    results = []

    if args.question:
        result = run_webexpert(
            question=args.question,
            retriever=retriever,
            facet_inducer=facet_inducer,
            gate=gate,
            planner=planner,
            explorer=explorer,
            top_k=args.top_k,
        )
        results.append(result)
        print(f"Gate mode: {result['gate_mode']}, Confidence: {result['confidence']:.4f}")
        print(f"Query plan: {json.dumps(result['query_plan'], ensure_ascii=False, indent=2)}")

    elif args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                question = item.get("question", item.get("Question", ""))
                if not question:
                    continue
                result = run_webexpert(
                    question=question,
                    retriever=retriever,
                    facet_inducer=facet_inducer,
                    gate=gate,
                    planner=planner,
                    explorer=explorer,
                    top_k=args.top_k,
                )
                results.append(result)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
