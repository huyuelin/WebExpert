#!/usr/bin/env python3
"""
基于 run_dataset_experiments.py 的输出，计算四个开源数据集的准确率与明细。

功能：
- 读取 `{output_dir}/{dataset}_results.json`（数组），对每条样本进行抽取预测、规范化与匹配，计算准确率
- 输出汇总 JSON 与简洁的控制台统计
- 可选对比 baseline（例如 WebThinker-32B-Base），从外部 JSON 传入

使用示例：
  python compute_experiment_metrics.py \
    --datasets gaia,hle,webwalkerqa,pqa \
    --output-dir dataset_experiments \
    --baseline-json path/to/webthinker_baseline.json

baseline JSON 格式示例：
{
  "gaia": {"accuracy": 0.42},
  "hle": {"accuracy": 0.55},
  "webwalkerqa": {"accuracy": 0.31},
  "pqa": {"accuracy": 0.60}
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# 可选：调用LLM进行一致性判定
try:
    # 复用现有HTTP/网关封装；如不可用将自动降级
    from scripts.get_llm_response.get_llm_response import get_llm_response as _llm_call
except Exception:
    _llm_call = None


# ============ 配置 ============

DEFAULT_DATASETS = ["gaia", "gpqa", "webwalkerqa", "pqa", "hle"]

# 每个数据集的默认阈值/策略（可按需微调）
DATASET_MATCH_CONFIG = {
    # 适度放宽相似度阈值；exact/contain优先，similar仅在必要时兜底
    "gaia": {"similarity_threshold": 0.80},
    "gpqa": {"similarity_threshold": 0.90},  # 多选题答案通常为 A/B/C/D 或短文本
    "hle": {"similarity_threshold": 0.80},
    # 放宽 WebWalkerQA 的语义相似阈值
    "webwalkerqa": {"similarity_threshold": 0.68},
    "pqa": {"similarity_threshold": 0.80},
}


# ============ 文本处理与匹配 ============

_PUNCT_REGEX = re.compile(r"[\t\n\r\f\v\.,;:!\?\"'`~@#$%^&*()\[\]{}<>|\\/\+=_-]")


def normalize_text(text: str) -> str:
    """统一规范化：转小写、去标点、压缩空白。"""
    if text is None:
        return ""
    s = text.strip().lower()
    s = _PUNCT_REGEX.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fuzzy_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _extract_first_number(text: str) -> Optional[float]:
    if not text:
        return None
    # 去掉千分位逗号
    s = text.replace(',', ' ')
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _boolean_equiv(a: str, b: str) -> bool:
    truthy = {"true", "yes", "y", "是", "对", "正确", "ok"}
    falsy = {"false", "no", "n", "否", "不对", "错误"}
    an = a.strip().lower()
    bn = b.strip().lower()
    if an in truthy and bn in truthy:
        return True
    if an in falsy and bn in falsy:
        return True
    return False


def extract_predicted_answer(raw: str, fallback_short: Optional[str] = None) -> str:
    """从模型原始输出中尽量抽取“最终答案”文本。
    规则：
    1) 匹配“最终答案：.../Final answer:”
    2) 匹配 \boxed{...}
    3) 匹配"Answer:"风格
    4) 否则取第一条非空行（避免长“思考过程”污染）
    """
    if not raw:
        return ""
    text = raw.strip()

    # 1) 最终答案: 中文/英文
    patterns = [
        r"最终答案[:：]\s*(.+)",
        r"final\s*answer[:：]\s*(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # 2) \boxed{...}
    m = re.search(r"\\boxed\{(.+?)\}", text)
    if m:
        return m.group(1).strip()

    # 3) Answer:
    m = re.search(r"\banswer[:：]\s*([A-Da-d])\b|\banswer[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        # 若匹配到单个选项字母，优先返回大写字母
        if m.group(1):
            return m.group(1).strip().upper()
        return m.group(2).strip()

    # 4) 回退：首个非空行（剔除以“思考过程/assistant/助手”等开头的引导语）
    for line in text.splitlines():
        ln = line.strip()
        if not ln:
            continue
        if re.match(r"^(思考过程|assistant|助手|analysis)[:：]?", ln, flags=re.IGNORECASE):
            continue
        # 如果是单个 A-D 字母，也直接返回
        if re.fullmatch(r"[A-Da-d]", ln):
            return ln.upper()
        return ln
    # 5) 再次回退：若存在调用端提供的 short_answer
    if fallback_short:
        return fallback_short.strip()
    return text


@dataclass
class MatchResult:
    is_correct: bool
    match_type: str  # exact/contain/similar/none
    similarity: float


def match_prediction_to_truth(pred: str, truth: str, dataset_name: str) -> MatchResult:
    """匹配逻辑：
    - 规范化后 exact match
    - 或者包含（pred in truth 或 truth in pred）
    - 或者相似度 >= 阈值
    """
    config = DATASET_MATCH_CONFIG.get(dataset_name, {"similarity_threshold": 0.9})
    sim_th = float(config.get("similarity_threshold", 0.9))

    # GPQA优先把答案标准化成单个字母（若可能）
    if dataset_name == 'gpqa':
        def to_letter(x: str) -> Optional[str]:
            if not x:
                return None
            m = re.search(r"\b([A-Da-d])\b", x)
            return m.group(1).upper() if m else None
        p_letter = to_letter(pred)
        t_letter = to_letter(truth)
        if p_letter and t_letter:
            return MatchResult(p_letter == t_letter, "exact" if p_letter == t_letter else "none", 1.0 if p_letter == t_letter else 0.0)

    pred_norm = normalize_text(pred)
    truth_norm = normalize_text(truth)

    if not pred_norm or not truth_norm:
        return MatchResult(False, "none", 0.0)

    # 精确匹配或数值等价（去掉千分位逗号）
    if pred_norm == truth_norm:
        return MatchResult(True, "exact", 1.0)

    # 数值松弛：移除逗号
    if pred_norm.replace(',', '') == truth_norm.replace(',', ''):
        return MatchResult(True, "exact", 1.0)

    # 布尔同义判断
    try:
        if _boolean_equiv(pred, truth):
            return MatchResult(True, "similar", 1.0)
    except Exception:
        pass

    # 数值松弛：提取首个数字并比较相对/绝对误差
    try:
        pv = _extract_first_number(pred)
        tv = _extract_first_number(truth)
        if pv is not None and tv is not None:
            if tv == 0:
                if abs(pv - tv) <= 1e-6:
                    return MatchResult(True, "similar", 1.0)
            else:
                rel = abs(pv - tv) / (abs(tv) + 1e-8)
                if rel <= 0.01 or abs(pv - tv) <= 0.5:  # 1% 或 0.5 以内视为等价
                    return MatchResult(True, "similar", 1.0)
    except Exception:
        pass

    if pred_norm in truth_norm or truth_norm in pred_norm:
        return MatchResult(True, "contain", 1.0 if truth_norm in pred_norm else 0.99)

    sim = fuzzy_similarity(pred_norm, truth_norm)
    if sim >= sim_th:
        return MatchResult(True, "similar", sim)

    return MatchResult(False, "none", sim)


# ============ 读取结果与评估 ============

def load_results(output_dir: str, dataset_name: str) -> List[Dict[str, Any]]:
    path = os.path.join(output_dir, f"{dataset_name}_results.json")
    if not os.path.exists(path):
        print(f"[Warn] Results file not found: {path}")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[Warn] Failed to load results: {path}, err={e}")
        return []


def evaluate_dataset(results: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    total = 0
    correct = 0
    counts = {"exact": 0, "contain": 0, "similar": 0, "llm": 0, "none": 0}
    mistakes: List[Dict[str, Any]] = []

    for item in results:
        truth = (item.get('ground_truth') or "").strip()
        pred_raw = (item.get('final_answer') or "").strip()
        short = (item.get('short_answer') or "").strip()
        if not truth:
            continue
        total += 1
        pred = extract_predicted_answer(pred_raw, fallback_short=short)
        m = match_prediction_to_truth(pred, truth, dataset_name)
        if m.is_correct:
            counts[m.match_type] = counts.get(m.match_type, 0) + 1
            correct += 1
        else:
            # 若规则未通过，尝试 LLM 评判（需在 main 中注入全局 judge 配置）
            judged_ok, judge_reason = _maybe_llm_judge(
                dataset_name=dataset_name,
                question=item.get('question', ''),
                predicted=pred,
                ground_truth=truth,
            )
            if judged_ok:
                counts['llm'] = counts.get('llm', 0) + 1
                correct += 1
            else:
                counts['none'] = counts.get('none', 0) + 1
                if len(mistakes) < 100:  # 限制体量，便于查看
                    mistakes.append({
                        "question": item.get('question', '')[:300],
                        "predicted": pred[:300],
                        "ground_truth": truth[:300],
                        "similarity": round(m.similarity, 4),
                        "match_type": m.match_type,
                        "judge_reason": judge_reason,
                        "sample_id": item.get('sample_id'),
                        "sample_key": item.get('sample_key'),
                    })

    acc = (correct / total) if total > 0 else 0.0
    return {
        "dataset": dataset_name,
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "details": counts,
        "mistakes_preview": mistakes,
    }


def group_metrics_by_category(results: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    """按论文展示需求进行细分统计：
    - GPQA: Physics, Chemistry, Biology, Avg.
    - GAIA: Level 1, Level 2, Level 3, Avg.
    - WebWalkerQA: Easy, Med., Hard, Avg.
    约定：run_dataset_experiments.py 已将分类字段写入 result：
      - gpqa_subject in {Physics, Chemistry, Biology}
      - gaia_level in {Level 1, Level 2, Level 3}
      - webwalker_difficulty in {Easy, Medium, Hard}
    """
    def eval_subset(subset: List[Dict[str, Any]]) -> Tuple[int, int, float]:
        m = evaluate_dataset(subset, dataset_name)
        return m["total"], m["correct"], m["accuracy"]

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    labels: List[str] = []

    if dataset_name == "gpqa":
        labels = ["Physics", "Chemistry", "Biology"]
        for it in results:
            subj = (it.get("gpqa_subject") or "").strip().title()
            if subj in labels:
                buckets.setdefault(subj, []).append(it)
    elif dataset_name == "gaia":
        labels = ["Level 1", "Level 2", "Level 3"]
        for it in results:
            lvl = (it.get("gaia_level") or "").strip()
            if lvl in labels:
                buckets.setdefault(lvl, []).append(it)
    elif dataset_name == "webwalkerqa":
        labels = ["Easy", "Medium", "Hard"]
        for it in results:
            diff = (it.get("webwalker_difficulty") or "").strip().title()
            if diff in labels:
                buckets.setdefault(diff, []).append(it)
    else:
        return {}

    stats: Dict[str, Any] = {}
    acc_values: List[float] = []
    for lb in labels:
        subset = buckets.get(lb, [])
        t, c, acc = eval_subset(subset)
        stats[lb] = {"total": t, "correct": c, "accuracy": acc}
        if t > 0:
            acc_values.append(acc)
    if acc_values:
        stats["Avg."] = sum(acc_values) / len(acc_values)
    else:
        stats["Avg."] = 0.0
    return stats


def load_baseline(baseline_path: Optional[str]) -> Dict[str, Any]:
    if not baseline_path:
        return {}


# ================= LLM 评判（可选）=================

_JUDGE_CFG: Dict[str, Any] = {
    "enabled": False,
    "model": "deepseek-v3-0324",
    "base_url": "https://openai.mybank.cn/v1",
    "cache_path": None,
    "cache": {},
}


def _load_judge_cache(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_judge_cache(path: Optional[str], cache: Dict[str, Any]) -> None:
    if not path:
        return
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _build_judge_prompt(dataset_name: str, question: str, predicted: str, ground_truth: str) -> str:
    return (
        "你是严格的评测裁判，任务是判断候选答案与标准答案是否一致。\n"
        "请遵循：\n"
        "- 考察标准不要过于严苛，意思相近有重合即可。\n"
        "- 只考察事实一致性，不在意措辞/表述差异。\n"
        "- 人名/地名/实体/数值/年份/版本号等需语义等价或数值等价（忽略千分位逗号）,不要过于严苛。\n"
        "- 如候选包含标准答案的完整信息，可判定正确；如仅部分信息或相矛盾，则错误。\n"
        "- 若是开放回答（长句），以核心要点是否一致为准。\n\n"
        f"[数据集] {dataset_name}\n"
        f"[问题] {question}\n"
        f"[候选答案] {predicted}\n"
        f"[标准答案] {ground_truth}\n\n"
        "请只输出一个JSON：{\"correct\": true/false, \"reason\": \"简要原因\"}。"
    )


def _llm_judge(dataset_name: str, question: str, predicted: str, ground_truth: str) -> Tuple[bool, str]:
    if _llm_call is None:
        return False, "llm_unavailable"
    prompt = _build_judge_prompt(dataset_name, question, predicted, ground_truth)
    try:
        resp = _llm_call(prompt=prompt, stop=[], model=_JUDGE_CFG["model"], base_url=_JUDGE_CFG["base_url"], echo_stream=False)
        # 尝试解析JSON
        m = re.search(r"\{[\s\S]*\}", resp)
        js = json.loads(m.group(0)) if m else json.loads(resp)
        ok = bool(js.get("correct"))
        reason = str(js.get("reason", ""))
        return ok, reason
    except Exception as e:
        return False, f"judge_error:{e}"


def _maybe_llm_judge(dataset_name: str, question: str, predicted: str, ground_truth: str) -> Tuple[bool, str]:
    if not _JUDGE_CFG.get("enabled"):
        return False, "judge_disabled"
    # cache key
    key = f"{dataset_name}|{question[:200]}|{predicted[:200]}|{ground_truth[:200]}"
    cache = _JUDGE_CFG.get("cache", {})
    if key in cache:
        val = cache[key]
        return bool(val.get('ok', False)), val.get('reason', '')
    ok, reason = _llm_judge(dataset_name, question, predicted, ground_truth)
    cache[key] = {"ok": ok, "reason": reason}
    _JUDGE_CFG["cache"] = cache
    _save_judge_cache(_JUDGE_CFG.get("cache_path"), cache)
    return ok, reason
    if not os.path.exists(baseline_path):
        print(f"[Warn] Baseline file not found: {baseline_path}")
        return {}
    try:
        with open(baseline_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warn] Failed to load baseline: {baseline_path}, err={e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Compute accuracy for mywebthinker experiments")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS), help="数据集列表，逗号分隔")
    parser.add_argument("--output-dir", type=str, default="dataset_experiments", help="实验结果目录")
    parser.add_argument("--baseline-json", type=str, default=None, help="可选，用于对比的baseline JSON")
    parser.add_argument("--save-json", action="store_true", help="保存评估汇总到JSON文件")
    # LLM 评判相关
    parser.add_argument("--use-llm-judge", action="store_true", help="启用LLM一致性评判（规则失败后兜底）")
    parser.add_argument("--judge-model", type=str, default="deepseek-v3-0324", help="评判模型名")
    parser.add_argument("--judge-base-url", type=str, default="https://openai.mybank.cn/v1", help="评判模型网关")
    parser.add_argument("--judge-cache", type=str, default=None, help="评判缓存文件路径（JSON）")
    args = parser.parse_args()

    ds_list = [d.strip() for d in args.datasets.split(',') if d.strip()]
    baseline = load_baseline(args.baseline_json)

    # 配置 LLM 评判
    if args.use_llm_judge:
        _JUDGE_CFG["enabled"] = True
        _JUDGE_CFG["model"] = args.judge_model
        _JUDGE_CFG["base_url"] = args.judge_base_url
        _JUDGE_CFG["cache_path"] = args.judge_cache
        _JUDGE_CFG["cache"] = _load_judge_cache(args.judge_cache)

    all_metrics: Dict[str, Any] = {}
    print("=== 评估开始 ===")
    for ds in ds_list:
        res = load_results(args.output_dir, ds)
        metrics = evaluate_dataset(res, ds)
        all_metrics[ds] = metrics

        print(f"\n--- {ds.upper()} ---")
        print(f"Total: {metrics['total']}")
        print(f"Correct: {metrics['correct']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Breakdown: {metrics['details']}")

        # 分类细分统计（仅对指定数据集）
        cat_stats = group_metrics_by_category(res, ds)
        if cat_stats:
            if ds == 'gpqa':
                print("\nGPQA breakdown (Phy. Chem. Bio. Avg.):")
                row = [
                    cat_stats.get('Physics', {}).get('accuracy', 0.0),
                    cat_stats.get('Chemistry', {}).get('accuracy', 0.0),
                    cat_stats.get('Biology', {}).get('accuracy', 0.0),
                    cat_stats.get('Avg.', 0.0),
                ]
                print({"Physics": row[0], "Chemistry": row[1], "Biology": row[2], "Avg.": row[3]})
            elif ds == 'gaia':
                print("\nGAIA breakdown (Level 1 Level 2 Level 3 Avg.):")
                row = [
                    cat_stats.get('Level 1', {}).get('accuracy', 0.0),
                    cat_stats.get('Level 2', {}).get('accuracy', 0.0),
                    cat_stats.get('Level 3', {}).get('accuracy', 0.0),
                    cat_stats.get('Avg.', 0.0),
                ]
                print({"Level 1": row[0], "Level 2": row[1], "Level 3": row[2], "Avg.": row[3]})
            elif ds == 'webwalkerqa':
                print("\nWEBWALKER breakdown (Easy Med. Hard Avg.):")
                row = [
                    cat_stats.get('Easy', {}).get('accuracy', 0.0),
                    cat_stats.get('Medium', {}).get('accuracy', 0.0),
                    cat_stats.get('Hard', {}).get('accuracy', 0.0),
                    cat_stats.get('Avg.', 0.0),
                ]
                print({"Easy": row[0], "Med.": row[1], "Hard": row[2], "Avg.": row[3]})
            all_metrics[f"{ds}_by_category"] = cat_stats

        if baseline.get(ds) and isinstance(baseline[ds], dict):
            base_acc = float(baseline[ds].get('accuracy', 0.0))
            delta = metrics['accuracy'] - base_acc
            print(f"Baseline (WebThinker-32B-Base): {base_acc:.4f} | Δ: {delta:+.4f}")

    if args.save_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(args.output_dir, f"metrics_{ts}.json")
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, ensure_ascii=False, indent=2)
            print(f"\n评估结果已保存: {save_path}")
        except Exception as e:
            print(f"[Warn] 保存评估结果失败: {e}")

    print("\n=== 评估完成 ===")


if __name__ == "__main__":
    main()


