#!/usr/bin/env python3
"""Experiment Runner for GAIA, GPQA, HLE, and WebWalkerQA.

Usage:
  python run_experiments.py --datasets gaia,gpqa,hle,webwalkerqa --output-dir experiments/
  python run_experiments.py --datasets gaia --max-samples 50 --resume
"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from webexpert.experience.retriever import ExperienceRetriever
from webexpert.experience.facet_inducer import FacetInducer
from webexpert.experience.gate import ExperienceGate
from webexpert.planning.domain_planner import DomainPlanner
from webexpert.browsing.deep_explorer import DeepWebExplorer
from webexpert.evaluation.metrics import match_prediction, exact_match, f1_score


DATASET_CONFIGS = {
    "gaia": {"path": "dataset/GAIA/", "question_key": "Question", "answer_key": "Final answer"},
    "gpqa": {"path": "dataset/GPQA/", "question_key": "question", "answer_key": "answer"},
    "hle": {"path": "dataset/HLE/", "question_key": "question", "answer_key": "answer"},
    "webwalkerqa": {"path": "dataset/WebWalkerQA/", "question_key": "question", "answer_key": "answer"},
}


def load_dataset(dataset_name: str, max_samples: int = None, override_path: str = None) -> List[Dict]:
    """Load dataset from JSON/JSONL files."""
    config = DATASET_CONFIGS.get(dataset_name, {})
    data_path = override_path or config.get("path", "")
    items = []
    p = Path(data_path)
    if not p.exists():
        print(f"[Warning] Dataset path not found: {data_path}")
        return items
    for fp in sorted(p.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix == ".jsonl":
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
        elif fp.suffix == ".json":
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    items.extend(data)
                else:
                    items.append(data)
    if max_samples:
        items = items[:max_samples]
    return items


def run_experiments(
    datasets: List[str],
    experience_base: str = "expert_outputs/experience_base.jsonl",
    max_samples: Optional[int] = None,
    resume: bool = True,
    output_dir: str = "experiments",
    top_k: int = 5,
    gate_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Run experiments on specified datasets."""
    retriever = ExperienceRetriever(experience_path=experience_base)
    facet_inducer = FacetInducer()
    gate = ExperienceGate(confidence_threshold=gate_threshold, top_k=top_k)
    planner = DomainPlanner(max_queries=3)
    explorer = DeepWebExplorer(max_interactions=10)

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for ds in datasets:
        print(f"\n===== Experiment: {ds.upper()} =====")
        dataset = load_dataset(ds, max_samples=max_samples)
        if not dataset:
            continue

        config = DATASET_CONFIGS.get(ds, {})
        q_key = config.get("question_key", "question")
        a_key = config.get("answer_key", "answer")
        output_file = os.path.join(output_dir, f"{ds}_results.json")

        existing = {}
        if resume and os.path.exists(output_file):
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    arr = json.load(f)
                for it in arr:
                    key = it.get("sample_key")
                    if key:
                        existing[key] = it
            except Exception:
                pass

        ds_results = list(existing.values())

        for i, item in enumerate(dataset):
            question = item.get(q_key, "")
            ground_truth = item.get(a_key, "")
            if not question:
                continue
            sample_key = hashlib.sha1(question.encode()).hexdigest()[:12]
            if sample_key in existing:
                continue

            print(f"  [{ds}] Sample {i+1}/{len(dataset)}: {question[:80]}...")

            result = {
                "question": question,
                "ground_truth": ground_truth,
                "dataset": ds,
                "sample_key": sample_key,
                "sample_id": i + 1,
            }

            # Run WebExpert pipeline
            start = time.time()
            experiences, scores = retriever.retrieve(question, top_k=top_k)
            active_facets = facet_inducer.tag_text(question)
            gate_result = gate.gate(question, experiences, scores, active_facets)
            result["gate_mode"] = gate_result["mode"]
            result["confidence"] = gate_result["confidence"]
            result["active_facets"] = active_facets
            result["processing_time"] = time.time() - start

            ds_results.append(result)

            # Incremental save
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(ds_results, f, ensure_ascii=False, indent=2)

        all_results[ds] = ds_results
        print(f"[{ds}] Saved {len(ds_results)} results to {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="WebExpert Experiment Runner")
    parser.add_argument("--datasets", default="gaia,gpqa,hle,webwalkerqa")
    parser.add_argument("--experience-base", default="expert_outputs/experience_base.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--gate-threshold", type=float, default=0.3)
    args = parser.parse_args()

    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    run_experiments(
        datasets=ds_list,
        experience_base=args.experience_base,
        max_samples=args.max_samples,
        resume=args.resume,
        output_dir=args.output_dir,
        top_k=args.top_k,
        gate_threshold=args.gate_threshold,
    )


if __name__ == "__main__":
    main()
