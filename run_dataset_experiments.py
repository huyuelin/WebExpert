#!/usr/bin/env python3
"""
四个数据集实验脚本 - 使用新的数据集专家经验
实验数据集: PQA, GAIA, WebWalkerQA, HLE
"""

import json
import time
import os
import re
from typing import Dict, List, Set, Any
from datetime import datetime

# 导入现有的接口和工具
import sys
sys.path.append('scripts')

from scripts.prompt.prompt import (
    get_dataset_expert_experience,  # 使用新的数据集专家经验
)

# 使用真实本地模型与实际搜索引擎
from scripts.get_llm_response.get_llm_response import get_llm_response
from scripts.search.search_engine_mybank import search_engine_with_rag

# 点击工具（可选）
try:
    # 与 run_mywebthinker 的导入保持一致
    from scripts.click_url_and_return_md.click_url_and_return_md import click_url_and_return_md
except Exception as _e:
    print(f"[Warn] Click 模块导入失败：{_e}，已禁用点击功能。")
    click_url_and_return_md = None

# ===== 实验配置 =====
MAX_SEARCH_LIMIT = 8  # 每条样例最多搜索次数
TOP_K = 8  # 搜索结果条数
MODEL_NAME = "deepseek-v3-0324"
AUX_MODEL_NAME = "deepseek-v3-0324"
BASE_URL = "https://openai.mybank.cn/v1"  # 本地模型实现会忽略该参数
MAX_TOKEN_LIMIT = 81920
MAX_DEEP_INTERACTIONS = 6

# ===== 数据集配置（可被命令行参数覆盖）=====
DATASETS_CONFIG = {
    "gaia": {
        # 可改为完整数据路径或目录，例如："dataset/GAIA/"（将自动扫描 json/jsonl）
        "path": "dataset/GAIA/",
        "format": "auto",
        "question_key": "Question",
        "answer_key": "Final answer"
    },
    "gpqa": {
        # 建议使用下载脚本保存为规范jsonl，字段：question, answer, subject
        "path": "dataset/GPQA/",
        "format": "auto",
        "question_key": "question",
        "answer_key": "answer"
    },
    "hle": {
        # 完整集通常为jsonl文件或目录
        "path": "dataset/HLE/",
        "format": "auto",
        "question_key": "question",
        "answer_key": "answer"
    },
    "webwalkerqa": {
        # 自动扫描WebWalkerQA目录下所有包含question/answer键的jsonl
        "path": "dataset/WebWalkerQA/",
        "format": "auto", 
        "question_key": "question",
        "answer_key": "answer"
    },
    "pqa": {
        # PQA原为视觉任务，默认仍用text模式（若指定路径为json/jsonl则自动解析）
        "path": "dataset/PQA/",
        "format": "auto",
        "question_key": "description",
        "answer_key": "explanation"
    }
}

# ===== 特殊标记 =====
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
BEGIN_CLICK_LINK = "<|begin_click_link|>"
END_CLICK_LINK = "<|end_click_link|>"
BEGIN_CLICK_RESULT = "<|begin_click_result|>"
END_CLICK_RESULT = "<|end_click_result|>"

THINK_OPEN = "<think>\n"
THINK_CLOSE = "</think>\n"

# ===== 兜底与容错 =====
from typing import Optional
import argparse
import hashlib
from pathlib import Path

def safe_get_llm_response(prompt: str, stop: Optional[list] = None) -> str:
    """调用LLM；若失败则返回可被 parse_expert_targets 解析的兜底文本。"""
    try:
        return get_llm_response(
            prompt=add_think_tag(prompt),
            stop=stop or [],
            model=MODEL_NAME,
            base_url=BASE_URL,
            echo_stream=False,
        )
    except Exception as e:
        print(f"[Warn] get_llm_response 调用失败，使用兜底输出。err={e}")
        return (
            "思考过程：根据问题解析生成搜索查询和意图。\n\n"
            "最终输出：\n"
            "**第一个搜索查询及意图：** 领域：知识检索-基础概念查询\n\n"
            "搜索关键词：定义 关键术语 核心概念\n"
            "意图：先查清概念定义与基础框架。\n\n"
            "**第二个搜索查询及意图：** 领域：知识检索-技术实现详解\n\n"
            "搜索关键词：实现 方法 步骤 示例\n"
            "意图：查找可操作步骤与示例，便于解题。\n"
        )

def safe_search_engine_with_rag(query: str) -> dict:
    """调用检索；失败时返回单条模拟候选，保证流程可运行。"""
    try:
        res = search_engine_with_rag(query)
        if not isinstance(res, dict):
            raise ValueError("检索返回非dict")
        return res
    except Exception as e:
        print(f"[Warn] 搜索引擎调用失败，使用兜底搜索结果。err={e}")
        return {
            "data": {
                "candidated_texts": [
                    {
                        "webTitle": f"搜索兜底：{query}",
                        "webUrl": f"https://example.com/search?q={query}",
                        "chunkContent": f"兜底内容：与'{query}'相关的信息示例。",
                        "webPublishTime": "2025-01-01",
                    }
                ]
            }
        }

def add_think_tag(text: str) -> str:
    """若 prompt 中无 <think>，则在末尾追加"""
    return text if THINK_OPEN in text else f"{text.rstrip()}\n{THINK_OPEN}"

def strip_think(text: str) -> str:
    """去除 </think> 收尾标签"""
    return text.replace(THINK_CLOSE, "")

def extract_between(text: str, start_marker: str, end_marker: str) -> str:
    """提取两标记之间的内容"""
    pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def format_search_results(candidates: List[Dict], top_k: int = 5) -> str:
    """格式化搜索结果"""
    if not candidates:
        return "No search results."
    documents = []
    for i, item in enumerate(candidates[:min(top_k, len(candidates))]):
        doc_str = {
            "index": i + 1,
            "title": item.get("webTitle", ""),
            "url": item.get("webUrl", item.get("url", "")),
            "content": item.get("chunkContent", ""),
            "time": item.get("webPublishTime", ""),
        }
        documents.append(doc_str)
    return json.dumps(documents, ensure_ascii=False, indent=2)

# ===== 数据集专家经验版本的领域分析 =====
def get_dataset_expert_domain_analysis_prompt(question: str, dataset_name: str):
    """使用数据集专家经验生成领域分析提示"""
    prompt = f"""
    你是一个具备深度网页搜索能力的AI助手，专门为{dataset_name.upper()}数据集中的问题提供准确的信息检索和分析。
    
    根据以下专家经验逻辑，分析该问题需要搜索哪些相关领域的网页信息，生成2-3个高质量的搜索查询。
    
    ### 专家经验逻辑：###
    {get_dataset_expert_experience()}
    ###
    
    ### 问题：###
    {question}
    ###
    
    请分析问题的关键要素，确定需要检索的信息类型，然后生成具体的搜索查询和意图。
    
    ### 输出格式：###
    **第一个搜索查询及意图：** 领域：xxx
    
    搜索关键词：xxxx
    意图：xxxx
    
    **第二个搜索查询及意图：** 领域：xxx
    
    搜索关键词：xxxx  
    意图：xxxx
    
    **第三个搜索查询及意图：** 领域：xxx
    
    搜索关键词：xxxx
    意图：xxxx
    ###
    """
    return prompt

def parse_expert_targets(raw_response: str) -> list[dict]:
    """解析专家规划的搜索targets"""
    text = strip_think(raw_response).strip()
    targets = []
    
    # 正则匹配搜索查询
    pattern = re.compile(
        r'\*\*第[^*]*?搜索查询及意图：\*\*\s*领域[:：]\s*([^\n]+?)\s*\n\s*(?:\*\*)?搜索关键[词字](?:\*\*)?[:：]\s*([^\n]+?)\s*\n\s*(?:\*\*)?意图(?:\*\*)?[:：]\s*([^\n]+)',
        re.VERBOSE
    )
    
    for match in pattern.finditer(text):
        domain = match.group(1).strip()
        query = match.group(2).strip()
        intent = match.group(3).strip()
        if query:
            targets.append({"domain": domain, "query": query, "intent": intent})
    
    return targets

def run_dataset_expert_stage(question: str, dataset_name: str) -> list[dict]:
    """使用数据集专家经验运行专家阶段"""
    expert_prompt = get_dataset_expert_domain_analysis_prompt(question, dataset_name)
    
    raw = safe_get_llm_response(expert_prompt, stop=[])
    
    targets = parse_expert_targets(raw)
    if not targets:
        print(f"[{dataset_name}] Expert Stage 解析失败，未提取到任何 targets。")
    return targets

# ===== 数据加载器 =====
def _iter_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items

def _iter_json(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw]
    return []

def _scan_dir_for_data(dir_path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    p = Path(dir_path)
    if not p.exists():
        return data
    for file in sorted(p.rglob('*')):
        if not file.is_file():
            continue
        lower = file.name.lower()
        try:
            if lower.endswith('.jsonl'):
                data.extend(_iter_jsonl(str(file)))
            elif lower.endswith('.json'):
                data.extend(_iter_json(str(file)))
        except Exception:
            # 忽略坏文件，继续扫描
            continue
    return data

def load_dataset(dataset_name: str, max_samples: int = None, override_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """加载指定数据集"""
    config = DATASETS_CONFIG.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    data_path = override_path or config["path"]
    data: List[Dict[str, Any]] = []

    # 自动模式：支持目录递归扫描与单文件
    if config.get("format") in ("auto", None):
        if os.path.isdir(data_path):
            data = _scan_dir_for_data(data_path)
            if not data:
                print(f"[Warning] No data files found under directory: {data_path}")
        elif os.path.isfile(data_path):
            lower = data_path.lower()
            if lower.endswith('.jsonl'):
                data = _iter_jsonl(data_path)
            elif lower.endswith('.json'):
                data = _iter_json(data_path)
            else:
                print(f"[Warning] Unsupported file format: {data_path}")
        else:
            print(f"[Warning] Dataset path not found: {data_path}")
    elif config["format"] == "json":
        data = _iter_json(data_path)
    elif config["format"] == "jsonl":
        data = _iter_jsonl(data_path)
    elif config["format"] == "text":
        # PQA是视觉任务，我们创建一些概念性问题来测试
        data = [
            {
                "description": "What is perceptual reasoning in visual tasks?",
                "explanation": "Perceptual reasoning involves understanding visual patterns and relationships"
            },
            {
                "description": "How does pattern completion work in cognitive tasks?", 
                "explanation": "Pattern completion requires identifying missing elements based on logical sequences"
            },
            {
                "description": "What are the key principles of visual analogy?",
                "explanation": "Visual analogy involves recognizing relationships between visual elements"
            }
        ]
    
    if max_samples:
        data = data[:max_samples]
    
    return data

def extract_question_answer(item: Dict[str, Any], dataset_name: str) -> tuple[str, str]:
    """从数据项中提取问题和答案"""
    config = DATASETS_CONFIG[dataset_name]
    question = item.get(config["question_key"], "")
    answer = item.get(config["answer_key"], "")
    return question, answer


# ===== 断点重连支持 =====
def compute_sample_key(item: Dict[str, Any], dataset_name: str) -> str:
    """为样本生成稳定key，默认用问题文本sha1。"""
    config = DATASETS_CONFIG[dataset_name]
    q = item.get(config["question_key"], "") or json.dumps(item, ensure_ascii=False)
    return hashlib.sha1(q.encode('utf-8')).hexdigest()

def load_existing_results(output_file: str) -> Dict[str, Any]:
    if not os.path.exists(output_file):
        return {}
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            arr = json.load(f)
        # 兼容：若是列表，转换为以sample_key为键的字典
        existing: Dict[str, Any] = {}
        for it in arr if isinstance(arr, list) else []:
            key = it.get('sample_key') or it.get('sample_id')
            if key:
                existing[str(key)] = it
        return existing
    except Exception:
        return {}

def append_result_incremental(output_file: str, result: Dict[str, Any]) -> None:
    # 先读旧，再写新（幂等覆盖同sample_key）
    existing_map = load_existing_results(output_file)
    key = result.get('sample_key')
    if key:
        existing_map[str(key)] = result
    arr = list(existing_map.values())
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)

def load_search_cache(cache_path: str) -> Dict[str, Any]:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def save_search_cache(cache_path: str, cache: Dict[str, Any]) -> None:
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ===== 简化版的处理流程 =====
def process_question_with_dataset_expert(question: str, dataset_name: str, search_cache: Dict) -> Dict:
    """使用数据集专家经验处理单个问题"""
    print(f"\n[{dataset_name}] Processing: {question[:100]}...")
    
    # 1. 专家阶段 - 使用数据集专家经验
    expert_targets = run_dataset_expert_stage(question, dataset_name)
    
    # 2. 构建专门的指令提示
    instruction = f"""你是专门处理{dataset_name.upper()}数据集问题的AI助手。请根据以下专家规划的搜索策略，进行深度网页搜索来回答问题。

### 问题: ###
{question}

### 专家搜索规划: ###
{json.dumps(expert_targets, ensure_ascii=False, indent=2)}

### 专家经验指导: ###
{get_dataset_expert_experience()}

请按照专家规划逐步搜索相关信息，并基于搜索结果提供准确的答案。

你可以使用以下工具：
- 搜索：<|begin_search_query|>查询内容<|end_search_query|>
- 点击链接：<|begin_click_link|>URL<|end_click_link|>

    最后请提供完整的答案。并且严格在最后一行以如下格式输出最终答案：
    最终答案：<你的简洁答案>
只输出这一行作为答案行，不要在该行之后追加任何其他文本。
"""
    
    # 对GPQA强约束：只输出字母 A/B/C/D 作为最终答案
    if dataset_name == 'gpqa':
        instruction += "\n\n注意：如果问题包含A-D选项，请在最终答案只输出大写字母 A/B/C/D 之一，不要输出其他字符或解释。"
    prompt = add_think_tag(instruction)
    
    # 3. 初始化处理状态
    result = {
        'question': question,
        'dataset': dataset_name,
        'expert_targets': expert_targets,
        'search_queries': [],
        'final_answer': '',
        'prompt_tokens': 0,
        'processing_time': 0
    }
    
    start_time = time.time()
    executed_queries = set()
    search_count = 0
    
    # 4. 先执行专家规划的搜索
    for target in expert_targets:
        if search_count >= MAX_SEARCH_LIMIT:
            break
            
        query = target["query"]
        if query in executed_queries:
            continue
            
        executed_queries.add(query)
        search_count += 1
        
        print(f"  [Search] {query}")
        result['search_queries'].append(query)
        
        # 执行搜索
        if query in search_cache:
            search_result = search_cache[query]
        else:
            search_result = safe_search_engine_with_rag(query)
            search_cache[query] = search_result

        # 格式化搜索结果
        candidates = search_result.get("data", {}).get("candidated_texts", []) if isinstance(search_result, dict) else []
        formatted_docs = format_search_results(candidates, top_k=TOP_K)

        # 添加到prompt
        search_block = f"""
{BEGIN_SEARCH_QUERY}{query}{END_SEARCH_QUERY}

{BEGIN_SEARCH_RESULT}
{formatted_docs}
{END_SEARCH_RESULT}
"""
        prompt += search_block

        # 这里暂不执行 deep_web_explorer，以降低依赖复杂度
    
    # 5. 最终推理生成答案
    prompt += "\n\n请基于以上搜索结果，提供该问题的完整答案："
    
    final_response = safe_get_llm_response(prompt, stop=[])
    
    result['final_answer'] = strip_think(final_response).strip()
    # 提取短答案，便于评测
    try:
        fr = result['final_answer']
        short = ""
        # 优先提取“最终答案：”或"Final answer:"模式
        for pat in [r"最终答案[:：]\s*(.+)", r"[Ff]inal\s*answer[:：]\s*(.+)"]:
            m = re.search(pat, fr)
            if m:
                short = m.group(1).strip()
                break
        if not short:
            # 匹配 \\boxed{...}
            m = re.search(r"\\boxed\{(.+?)\}", fr)
            if m:
                short = m.group(1).strip()
        if not short:
            # Answer:
            m = re.search(r"\b[Aa]nswer[:：]\s*(.+)", fr)
            if m:
                short = m.group(1).strip()
        if not short:
            # 取末尾最后一行非空作为兜底
            lines = [ln.strip() for ln in fr.strip().splitlines() if ln.strip()]
            if lines:
                short = lines[-1]
        result['short_answer'] = short
    except Exception:
        result['short_answer'] = ""
    result['processing_time'] = time.time() - start_time
    
    return result

# ===== 主实验函数 =====
def run_experiments(
    datasets: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    resume: bool = True,
    output_dir: str = "dataset_experiments",
    override_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """运行四个数据集的实验"""
    print("=== 开始数据集实验 - 使用新的数据集专家经验 ===")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 全局搜索缓存
    cache_path = os.path.join(output_dir, 'search_cache.json')
    search_cache = load_search_cache(cache_path) if resume else {}
    all_results = {}
    
    # 对每个数据集进行实验
    target_datasets = datasets or list(DATASETS_CONFIG.keys())
    for dataset_name in target_datasets:
        print(f"\n===== 实验数据集: {dataset_name.upper()} =====")
        
        # 加载数据集（限制样本数量以便测试）
        dataset_path_override = (override_paths or {}).get(dataset_name)
        dataset = load_dataset(dataset_name, max_samples=max_samples, override_path=dataset_path_override)  
        if not dataset:
            print(f"[Error] Failed to load dataset: {dataset_name}")
            continue
        
        # 断点续跑：读取已存在结果，跳过已完成样本
        output_file = os.path.join(output_dir, f"{dataset_name}_results.json")
        existing_map = load_existing_results(output_file) if resume else {}
        processed_keys = set(existing_map.keys())
        dataset_results = list(existing_map.values()) if existing_map else []
        
        for i, item in enumerate(dataset):
            print(f"\n--- {dataset_name} Sample {i+1}/{len(dataset)} ---")
            
            try:
                # 提取问题和答案
                question, ground_truth = extract_question_answer(item, dataset_name)
                if not question:
                    print(f"[Skip] Empty question in sample {i+1}")
                    continue
                sample_key = compute_sample_key(item, dataset_name)
                if sample_key in processed_keys:
                    print(f"[Skip] Already processed key={sample_key[:8]}...")
                    continue
                
                # 使用数据集专家经验处理问题
                result = process_question_with_dataset_expert(question, dataset_name, search_cache)
                result['ground_truth'] = ground_truth
                result['sample_id'] = i + 1
                result['sample_key'] = sample_key

                # 附加数据集特定的分类标签，便于后续分类评估
                try:
                    if dataset_name == 'gaia':
                        # 标准字段名可能为 'Level' 或小写 'level'
                        lvl = item.get('Level') or item.get('level')
                        if isinstance(lvl, str) and lvl:
                            # 统一成 'Level 1/2/3'
                            lvl_norm = lvl.strip()
                            if lvl_norm.lower() in ('1','2','3'):
                                lvl_norm = f"Level {lvl_norm}"
                            result['gaia_level'] = lvl_norm
                    elif dataset_name == 'webwalkerqa':
                        info = item.get('info') or item.get('Info') or {}
                        if isinstance(info, dict):
                            diff = info.get('difficulty_level') or info.get('Difficulty_Level')
                            if isinstance(diff, str) and diff:
                                result['webwalker_difficulty'] = diff.strip().capitalize()
                    elif dataset_name == 'gpqa':
                        # 我们的下载脚本会写入 subject 字段
                        subj = item.get('subject') or item.get('Subject')
                        if isinstance(subj, str) and subj:
                            # 标准化为首字母大写: Physics/Chemistry/Biology
                            subj_norm = subj.strip().lower()
                            mapping = {
                                'phy': 'Physics', 'physics': 'Physics',
                                'chem': 'Chemistry', 'chemistry': 'Chemistry',
                                'bio': 'Biology', 'biology': 'Biology'
                            }
                            result['gpqa_subject'] = mapping.get(subj_norm, subj.strip().title())
                except Exception:
                    pass
                
                dataset_results.append(result)
                # 增量写入（断点续跑关键）
                append_result_incremental(output_file, result)
                # 持久化搜索缓存
                save_search_cache(cache_path, search_cache)
                
                # 打印结果预览
                print(f"  Ground Truth: {ground_truth[:100]}...")
                print(f"  Generated Answer: {result['final_answer'][:100]}...")
                print(f"  Processing Time: {result['processing_time']:.2f}s")
                
            except Exception as e:
                print(f"[Error] Processing sample {i+1}: {e}")
                continue
        
        all_results[dataset_name] = dataset_results
        
        print(f"[{dataset_name}] Results saved to {output_file}")
    
    # 保存所有结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results_file = os.path.join(output_dir, f"all_results_{timestamp}.json")
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 实验完成 ===")
    print(f"所有结果保存到: {all_results_file}")
    
    # 简单的结果统计
    print(f"\n=== 实验统计 ===")
    for dataset_name, results in all_results.items():
        if results:
            avg_time = sum(r['processing_time'] for r in results) / len(results)
            avg_queries = sum(len(r['search_queries']) for r in results) / len(results)
            print(f"{dataset_name}: {len(results)} samples, "
                  f"平均处理时间: {avg_time:.2f}s, "
                  f"平均搜索查询数: {avg_queries:.1f}")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mywebthinker dataset experiments with resume support")
    parser.add_argument("--datasets", type=str, default="gaia,hle,webwalkerqa,pqa",
                        help="逗号分隔数据集列表，如: gaia,hle,webwalkerqa,pqa")
    parser.add_argument("--max-samples", type=int, default=None, help="每个数据集最大样本数（None为全部）")
    parser.add_argument("--resume", action="store_true", help="启用断点重连（默认启用）")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="禁用断点重连")
    parser.set_defaults(resume=True)
    parser.add_argument("--output-dir", type=str, default="dataset_experiments", help="结果输出目录")
    parser.add_argument("--gaia-path", type=str, default=None, help="覆盖GAIA路径（文件或目录）")
    parser.add_argument("--gpqa-path", type=str, default=None, help="覆盖GPQA路径（文件或目录）")
    parser.add_argument("--hle-path", type=str, default=None, help="覆盖HLE路径（文件或目录）")
    parser.add_argument("--webwalkerqa-path", type=str, default=None, help="覆盖WebWalkerQA路径（文件或目录）")
    parser.add_argument("--pqa-path", type=str, default=None, help="覆盖PQA路径（文件或目录）")
    args = parser.parse_args()

    ds_list = [d.strip() for d in args.datasets.split(',') if d.strip()]
    overrides = {}
    if args.gaia_path:
        overrides['gaia'] = args.gaia_path
    if args.gpqa_path:
        overrides['gpqa'] = args.gpqa_path
    if args.hle_path:
        overrides['hle'] = args.hle_path
    if args.webwalkerqa_path:
        overrides['webwalkerqa'] = args.webwalkerqa_path
    if args.pqa_path:
        overrides['pqa'] = args.pqa_path

    try:
        results = run_experiments(
            datasets=ds_list,
            max_samples=args.max_samples,
            resume=args.resume,
            output_dir=args.output_dir,
            override_paths=overrides or None,
        )
        print("\n实验成功完成！")
    except Exception as e:
        print(f"\n实验失败: {e}")
        import traceback
        traceback.print_exc()