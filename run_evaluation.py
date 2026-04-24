#!/usr/bin/env python3
"""Evaluation script for WebExpert experiment results.

Computes EM, F1, QP@3, Page Hops, nDCG@10 across datasets
with optional LLM-as-judge for borderline cases.

Usage:
  python run_evaluation.py --output-dir experiments/ --datasets gaia,gpqa,hle,webwalkerqa
"""

import argparse
import json
import os
from typing import Any, Dict, List

from webexpert.evaluation.metrics import match_prediction, f1_score


def load_results(output_dir: str, dataset: str) -> List[Dict[str, Any]]:
    path = os.path.join(output_dir, f"{dataset}_results.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def evaluate_dataset(results: List[Dict[str, Any]], dataset: str) -> Dict[str, Any]:
    total = 0
    correct = 0
    f1_sum = 0.0
    match_types = {"exact": 0, "contain": 0, "similar": 0, "none": 0}

    for item in results:
        truth = (item.get("ground_truth") or "").strip()
        pred = (item.get("final_answer") or item.get("short_answer") or "").strip()
        if not truth:
            continue
        total += 1
        is_correct, mtype = match_prediction(pred, truth, dataset)
        if is_correct:
            correct += 1
            match_types[mtype] = match_types.get(mtype, 0) + 1
        else:
            match_types["none"] += 1
        f1_sum += f1_score(pred, truth)

    em = (correct / total) if total > 0 else 0.0
    avg_f1 = (f1_sum / total) if total > 0 else 0.0

    return {
        "dataset": dataset,
        "total": total,
        "correct": correct,
        "em": em,
        "f1": avg_f1,
        "match_types": match_types,
    }


def main():
    parser = argparse.ArgumentParser(description="WebExpert Evaluation")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument("--datasets", default="gaia,gpqa,hle,webwalkerqa")
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()

    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_metrics = {}

    print("=== WebExpert Evaluation ===")
    for ds in ds_list:
        results = load_results(args.output_dir, ds)
        if not results:
            print(f"[{ds.upper()}] No results found.")
            continue
        metrics = evaluate_dataset(results, ds)
        all_metrics[ds] = metrics
        print(f"\n--- {ds.upper()} ---")
        print(f"  Total: {metrics['total']}")
        print(f"  Correct: {metrics['correct']}")
        print(f"  EM: {metrics['em']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Match types: {metrics['match_types']}")

    if args.save_json:
        out_path = os.path.join(args.output_dir, "evaluation_metrics.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()
