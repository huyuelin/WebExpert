import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

INPUT_FILE = "/workspace2/linwen/dpo_dataset_prepare/expert_search_queries_dpo_dataset.jsonl"
OUTPUT_DIR = "/workspace2/linwen/dpo_dataset_prepare/radar_charts"

DIMENSIONS = [
    "意图准确性",
    "领域相关性",
    "内容冗余程度",
    "时效性",
    "查询语言清晰度",
]

CATEGORY_SCORE = {
    "完全准确": 100,
    "高相关": 100,
    "无冗余": 100,
    "拥有": 100,
    "语言清晰明了": 100,

    "部分准确": 60,
    "中等相关": 60,
    "轻微冗余": 60,
    "不拥有": 0,
    "语言稍显复杂": 60,

    "不准确": 0,
    "低相关": 0,
    "严重冗余": 0,
    "语言不清晰": 0,
}


def calc_redundancy(parsed: List[Dict]) -> str:
    seen_queries = set()
    for item in parsed:
        q = item.get("query", "")
        if q in seen_queries:
            return "严重冗余"
        seen_queries.add(q)
    return "无冗余" if len(seen_queries) == len(parsed) else "轻微冗余"


def has_recent_year(parsed: List[Dict]) -> str:
    for item in parsed:
        q = item.get("query", "")
        if "2025" in q or "2024" in q:
            return "拥有"
    return "不拥有"


def language_clarity(query: str) -> str:
    # 简易判定：字段长度和是否存在过多空格/标点等
    if len(query) > 50:
        return "语言稍显复杂"
    return "语言清晰明了"


def judge_intent_accuracy(parsed: List[Dict], risk_keywords: List[str]) -> str:
    # 若 intent 中包含任何关键风险词则视为相关
    hit = 0
    for item in parsed:
        intent = item.get("intent", "")
        if any(k in intent for k in risk_keywords):
            hit += 1
    if hit == len(parsed):
        return "完全准确"
    if hit >= len(parsed) // 2:
        return "部分准确"
    return "不准确"


def judge_domain_coverage(parsed: List[Dict]) -> str:
    domains = {item.get("domain", "") for item in parsed}
    if len(domains) >= 5:
        return "高相关"
    if len(domains) >= 3:
        return "中等相关"
    return "低相关"


def evaluate_candidate(parsed: List[Dict]) -> Dict[str, int]:
    # 简易风险关键词集合，可自行扩充
    risk_keywords = ["经济周期", "行业景气度", "区域", "风险", "突发"]

    scores = {}
    scores["意图准确性"] = CATEGORY_SCORE[judge_intent_accuracy(parsed, risk_keywords)]
    scores["领域相关性"] = CATEGORY_SCORE[judge_domain_coverage(parsed)]
    scores["内容冗余程度"] = CATEGORY_SCORE[calc_redundancy(parsed)]
    scores["时效性"] = CATEGORY_SCORE[has_recent_year(parsed)]
    # 语言清晰度：取所有 query 的平均
    clarity_scores = []
    for item in parsed:
        clarity_scores.append(CATEGORY_SCORE[language_clarity(item.get("query", ""))])
    scores["查询语言清晰度"] = int(sum(clarity_scores) / len(clarity_scores)) if clarity_scores else 0
    return scores


def plot_radar(scores: Dict[str, int], title: str, save_path: str):
    values = [scores[d] for d in DIMENSIONS]
    values += values[:1]  # 闭合
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS) + 1, endpoint=True)

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIMENSIONS, fontproperties="SimHei")
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0", "50", "100"])
    ax.set_title(title, fontproperties="SimHei")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            apply_seqno = obj.get("apply_seqno")
            for idx in (1, 2, 3):
                parsed = obj.get(f"generation_{idx}_parsed")
                if not parsed:
                    continue
                scores = evaluate_candidate(parsed)
                out_path = os.path.join(OUTPUT_DIR, f"{apply_seqno}_gen{idx}.png")
                plot_radar(scores, f"{apply_seqno}-候选{idx}", out_path)
                print(f"已保存雷达图: {out_path}")


if __name__ == "__main__":
    main() 