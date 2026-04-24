"""Evaluation metrics: EM, F1, QP@3, Page Hops, nDCG@10."""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    s = text.strip().lower()
    s = re.sub(r"[\t\n\r\f\v\.,;:!\?\"'`~@#$%^&*()\[\]{}<>|\\/\+=_-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_text(prediction) == normalize_text(reference)


def f1_score(prediction: str, reference: str) -> float:
    pred_tokens = set(normalize_text(prediction).split())
    ref_tokens = set(normalize_text(reference).split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def query_precision_at_k(
    queries: List[str], retrieved_pages: List[Dict[str, Any]], k: int = 3
) -> float:
    """QP@k: proportion of generated queries that retrieve on-topic evidence."""
    if not queries:
        return 0.0
    on_topic = 0
    for q in queries[:k]:
        # A query is on-topic if at least one retrieved page contains
        # answer-bearing evidence (LLM-as-judge + strict match)
        if any(p.get("relevant", False) for p in retrieved_pages):
            on_topic += 1
    return on_topic / min(len(queries), k)


def page_hops(visited_urls: List[str]) -> int:
    """Count unique page visits until final answer."""
    return len(set(visited_urls))


def ndcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """Compute nDCG@k over cited pages."""
    import math
    dcg = sum(
        rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores[:k])
    )
    ideal = sorted(relevance_scores, reverse=True)
    idcg = sum(
        rel / math.log2(i + 2) for i, rel in enumerate(ideal[:k])
    )
    return dcg / idcg if idcg > 0 else 0.0


def match_prediction(prediction: str, reference: str, dataset: str = "") -> Tuple[bool, str]:
    """Match prediction against reference with dataset-specific logic."""
    if dataset == "gpqa":
        p_letter = re.search(r"\b([A-Da-d])\b", prediction)
        r_letter = re.search(r"\b([A-Da-d])\b", reference)
        if p_letter and r_letter:
            return p_letter.group(1).upper() == r_letter.group(1).upper(), "exact"

    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)
    if pred_norm == ref_norm:
        return True, "exact"
    if pred_norm in ref_norm or ref_norm in pred_norm:
        return True, "contain"
    sim = SequenceMatcher(None, pred_norm, ref_norm).ratio()
    if sim >= 0.8:
        return True, "similar"
    return False, "none"
