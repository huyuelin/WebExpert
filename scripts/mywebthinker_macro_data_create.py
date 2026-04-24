import json
import re
import time
from typing import Dict, List, Set
import os  # 新增：用于路径创建
from openai import OpenAI  # 为安全流式函数导入
from get_llm_response.get_llm_response import get_llm_response as _orig_get_llm_response


from prompt.prompt import (
    get_first_task_instruction_prompt,
    get_search_intent_instruction,          # ← 新增
    get_deep_web_explorer_instruction,      # ← 新增
    get_expert_domain_analysis_prompt,       # ← 新增
    get_expert_experience,
)

from search.search_engine_mybank import search_engine_with_rag
# 点击工具可选使用

# ==== 点击工具（可选） ====
# 部分环境中爬虫依赖可能缺失，或运行时抛出异常。
# 这里在 import 时就进行兜底，若导入失败则后续流程会自动禁用点击功能，
# 并在 Prompt 中标注"点击工具不可用"，避免整个流程直接终止。
try:
    from click_url_and_return_md.click_url_and_return_md import click_url_and_return_md
except Exception as _e:  # noqa: BLE001  # 捕获所有异常，避免意外中断
    print(f"[Warn] Click 模块导入失败：{_e}，已禁用点击功能。")
    click_url_and_return_md = None  # type: ignore


# ===== 新增：token 统计 =====
try:
    import tiktoken
except ImportError:
    tiktoken = None

# ================== 全局配置 ==================
# ---------------- 工具函数 ----------------
MAX_DEEP_INTERACTIONS = 10        # Deep-Web Explorer 最大 search+click 交互次数
url_cache: Dict[str, str] = {}    # 点击结果缓存，避免重复抓取


# 直接修改下面变量即可调试，无需命令行参数。
"""批量模式下，此变量不再使用，保持为 None"""
SINGLE_INFO: str | None = None

# ==== 批量输入输出相关配置 ====
# 输入数据（jsonl）路径
INPUT_JSONL_FILE: str = \
    "/Users/linwen/Desktop/agent_AC/mybank_webthinker/webexpert_dataset_prepare_v2/train_stage1_all_feat_v3_prompt_v4_0519.jsonl"  
# 输出目录及文件（支持断点续跑）
OUTPUT_DIR: str = \
    "/Users/linwen/Desktop/agent_AC/mybank_webthinker/macro_data_by_mywebthinker"

# 固定输出文件名，若脚本中断可自动续写
OUTPUT_JSONL_FILE: str = "macro_data_output.jsonl"

# 原脚本中的 OUTPUT_FILE 不再使用，由运行时动态生成
MAX_SEARCH_LIMIT: int = 10  # 每条样例最多搜索次数
TOP_K: int = 10  # 搜索结果条数
#MODEL_NAME: str = "qwq-32b"  # 主模型
#MODEL_NAME: str = "qwen3-32b"
MODEL_NAME: str = "deepseek-v3-0324"
# 辅助摘要模型（用于 Deep-Web 点击内容压缩）
#AUX_MODEL_NAME: str = "qwen3-32b"
AUX_MODEL_NAME: str = "deepseek-v3-0324"
# 原始网页内容截断上限（字符数），防止辅助模型 prompt 过长
MAX_RAW_HTML_CHARS: int = 8000
BASE_URL: str = "https://openai.mybank.cn/v1"  # 私有化网关地址
ECHO_STREAM: bool = True  # 是否边收边打印模型输出
MAX_TOKEN_LIMIT: int = 81920  # Prompt token 上限，超过后禁止新增搜索/点击

# ==== 特殊标记 ====
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
BEGIN_CLICK_LINK = "<|begin_click_link|>"
END_CLICK_LINK = "<|end_click_link|>"
BEGIN_CLICK_RESULT = "<|begin_click_result|>"
END_CLICK_RESULT = "<|end_click_result|>"

# ===== <think> 相关 =====
THINK_OPEN  = "<think>\n"
THINK_CLOSE = "</think>\n"

def add_think_tag(text: str) -> str:
    """若 prompt 中无 <think>，则在末尾追加"""
    return text if THINK_OPEN in text else f"{text.rstrip()}\n{THINK_OPEN}"

def strip_think(text: str) -> str:
    """去除 </think> 收尾标签（模型推理结果用）"""
    return text.replace(THINK_CLOSE, "")

# ---------------- 工具函数 ----------------

def extract_between(text: str, start_marker: str, end_marker: str) -> str:
    """提取两标记之间的内容，若未找到返回空字符串"""
    pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def format_search_results(candidates: List[Dict], top_k: int = 5) -> str:
    """将 RAG 返回的候选结果格式化为可读字符串"""
    if not candidates:
        return "No search results."
    documents = []
    if top_k > len(candidates):
        top_k = len(candidates)
    for i, item in enumerate(candidates[:top_k]):
        title = item.get("webTitle") or ""
        url = item.get("webUrl") or item.get("url") or ""
        chunkContent = item.get("chunkContent") or ""
        webPublishTime = item.get("webPublishTime") or ""
        doc_str = {
            "index": i + 1,
            "title": title,
            "url": url,
            "chunkContent": chunkContent,
            "webPublishTime": webPublishTime,
        }
        documents.append(doc_str)
    res_str = json.dumps(documents, ensure_ascii=False, indent=2)
    return res_str


def count_tokens(text: str, model: str = MODEL_NAME) -> int:
    """统计文本 token 数；若缺少 tiktoken 库则近似估算"""
    if tiktoken:
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    # fallback：大致 4 字符≈1 token
    return len(text) // 4


# ========== 新增辅助 ==========
def init_sequence(case_txt: str, prompt: str) -> dict:
    """初始化并返回序列字典"""
    return {
        'case': case_txt,
        'prompt': prompt,
        'output': '',
        'history': [],
        'finished': False,
        'search_count': 0,
        'executed_search_queries': set(),
    }

# --------- 专家输出解析 ---------
def parse_expert_targets(raw_response: str) -> list[dict]:
    """解析 LLM 给出的专家规划，提取搜索 query 与意图。
        Markdown/自然语言：形如
       #### **① xxx** \n- **搜索关键词**：AAA \n- **意图**：BBB
    """
    text = strip_think(raw_response).strip()



    targets: list[dict] = []

    # ---- 正则解析 Markdown 段落 ----
    patterns = [
        re.compile(
            r"""
            \*\*第[^*]*?搜索查询及意图：\*\*        # 标题行：**第x个搜索查询及意图：**
            \s*领域[:：]\s*([^\n]+?)\s*\n           # domain
            \s*(?:\*\*)?搜索关键[词字](?:\*\*)?[:：]\s*([^\n]+?)\s*\n  # query
            \s*(?:\*\*)?意图(?:\*\*)?[:：]\s*([^\n]+)   # intent
            """,
            re.VERBOSE,
        )
    ]

    for pat in patterns:
        for m in pat.finditer(text):
            domain = m.group(1).strip()
            query  = m.group(2).strip()
            intent = m.group(3).strip()
            if query:
                targets.append({"domain": domain, "query": query, "intent": intent})

    return targets


# ======= 提取最终中宏观信息工具函数 =======
def extract_macro_info(output_text: str) -> str:
    """从完整输出文本中截取最终的中宏观信息摘要段落。"""
    if not output_text:
        return ""

    # 先尝试按"开始/结束"标记提取，增加更多宽容的匹配模式
    patterns = [
        r"### 中宏观信息总结开始[\s\S]*?### 中宏观信息总结结束",
        r"### 基于借款人信息的审批信贷专家经验相关的中宏观信息[\s\S]*?($|-------以下内容)",
        r"中宏观信息总结开始[\s\S]*?中宏观信息总结结束",
        r"基于借款人信息的审批信贷专家经验相关的中宏观信息[\s\S]*?($|-------)",
        r"中宏观信息[\s\S]*?总结[\s\S]*?($|---|###)",
        r"## 中宏观信息[\s\S]*?($|---|##)",
        r"# 中宏观信息[\s\S]*?($|---|#)",
        r"中宏观[\s\S]*?信息[\s\S]*?($|---|###)",
        r"总结[\s\S]*?中宏观[\s\S]*?($|---|###)",
        r"### 中宏观信息总结\n[\s\S]*?",
    ]
    
    for pat in patterns:
        m = re.search(pat, output_text, re.IGNORECASE)
        if m:
            return m.group(0).strip()

    # 若未匹配，退而求其次，截取最后约 3000 字尝试查找开头
    tail = output_text[-3000:]
    for pat in patterns:
        m = re.search(pat, tail, re.IGNORECASE)
        if m:
            return m.group(0).strip()

    # 如果以上都没有匹配到，尝试提取 </think> 后的所有内容
    think_pattern = r"</think>([\s\S]*?)$"
    think_match = re.search(think_pattern, output_text)
    if think_match:
        return think_match.group(1).strip()

    return ""  # 未找到则返回空字符串


def run_expert_stage(case_txt: str) -> list[dict]:
    """
    调用 LLM 生成搜索规划，返回 list[dict]，每个 dict 与 prompt 约定的字段一致
    """
    expert_prompt = get_expert_domain_analysis_prompt(
        pending_info=case_txt,
        approval_logic=get_expert_experience() # 可把表格内容抽成常量
    )
    raw = _orig_get_llm_response(
        prompt=add_think_tag(expert_prompt),
        stop=[],
        model=MODEL_NAME,
        base_url=BASE_URL,
        echo_stream=False,
    )
    targets = parse_expert_targets(raw)
    if not targets:
        print("Expert Stage 解析失败，未提取到任何 targets。")
    return targets

# =================================

# ---------------- 主处理逻辑 ----------------

def process_single_case(case_info: str, search_cache: Dict) -> dict:
    """处理单条样例，出现异常时返回包含错误信息的 seq。"""

    # 先初始化空 prompt，确保异常情况下也能拿到 seq['prompt']
    seq = init_sequence(case_info, '')

    try:
        # 0) Expert Stage 先行
        expert_targets = run_expert_stage(case_info)   # list[dict]

        # ---------- 初始化主 prompt ----------
        instruction = get_first_task_instruction_prompt(
            MAX_SEARCH_LIMIT,
            case_info,
            json.dumps(expert_targets, ensure_ascii=False, indent=2),
        )

        prompt = add_think_tag(instruction)

        # ====================================
        seq['prompt'] = prompt
        seq["planned_queries"] = expert_targets  # ← 保存专家阶段规划

        for item in expert_targets:
            q = item["query"]
            intent = item["intent"]
            domain = item["domain"]
            print(f"\n[领域] {domain} [搜索] {q} [意图] {intent} ")

        print("\n ########################专家模块判断相关领域和生成搜索查询和搜索意图完毕，开始进行网页深度搜索浏览：########################")

        # ---------- 主循环之前，先批量跑一遍专家规划的搜索 ----------
        for item in expert_targets:
            q      = item["query"]
            intent = item["intent"]
            if q in seq["executed_search_queries"]:
                continue
            seq["executed_search_queries"].add(q)
            seq["search_count"] += 1
            print(f"\n[搜索] {q} [意图] {intent}")
            # 执行搜索
            res = search_cache.get(q) or search_engine_with_rag(q)
            search_cache[q] = res
            formatted_docs = format_search_results(res.get("data", {}).get("candidated_texts", []), top_k=TOP_K)
            block = (
                f"\n{BEGIN_SEARCH_QUERY}{q}{END_SEARCH_QUERY}"
                f"\n{BEGIN_SEARCH_RESULT}\n{formatted_docs}\n{END_SEARCH_RESULT}\n"
            )
            seq['prompt'] += block
            seq["output"] += block
            seq["history"].append(block)

            # 交给 deep_web_explorer 做深挖
            deep_info = deep_web_explorer(
                search_query=q,
                search_intent=intent,
                search_result=formatted_docs,
                search_cache=search_cache,
                url_cache=url_cache,
                global_executed_queries=seq["executed_search_queries"],
            )
            if deep_info:
                deep_block = (
                    f"\n{BEGIN_SEARCH_RESULT}\n【深度网页探索信息】\n{deep_info}\n{END_SEARCH_RESULT}\n"
                )
                seq['prompt'] += deep_block
                seq["output"] += deep_block
                seq["history"].append(deep_block)

        # ========== 进入原有 while True 主循环  结束了每一个领域的查询信息，开始汇总处理==========
        while True:
            # 1) LLM 推理
            raw_buf = _orig_get_llm_response(
                prompt=seq['prompt'],
                stop=[END_SEARCH_QUERY, END_CLICK_LINK],
                model=MODEL_NAME,
                base_url=BASE_URL,
                echo_stream=ECHO_STREAM,
            )
            buffer = strip_think(raw_buf)

            # 2) 写入序列
            seq['output']   += buffer
            seq['history'].append(buffer)
            seq['prompt']   += buffer

            # 本轮循环的拼接缓存，先置空，后续各分支按需追加
            msg = ""

            # 3) Token 上限处理
            if count_tokens(seq['prompt']) > MAX_TOKEN_LIMIT:
                if buffer.rstrip().endswith(END_SEARCH_QUERY):
                    msg = f"\n{BEGIN_SEARCH_RESULT}已达到 Token 上限，停止搜索。{END_SEARCH_RESULT}\n"
                elif buffer.rstrip().endswith(END_CLICK_LINK):
                    msg = f"\n{BEGIN_CLICK_RESULT}已达到 Token 上限，停止点击。{END_CLICK_RESULT}\n"
                else:
                    seq['finished'] = True
                    break
                seq['prompt'] += msg
                seq['output'] += msg
                seq['history'].append(msg)
                continue

            # 4-A) 搜索分支（其余逻辑原样，但把 prompt/output 的 += 改为 seq['prompt'] / seq['output']）
            if buffer.rstrip().endswith(END_SEARCH_QUERY):
                if seq['search_count'] >= MAX_SEARCH_LIMIT:
                    msg = f"\n{BEGIN_SEARCH_RESULT}已达到搜索上限，停止搜索。{END_SEARCH_RESULT}\n"
                else:
                    search_query = extract_between(buffer, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
                    if not search_query or len(search_query) < 2:
                        # 非法查询，忽略
                        msg = f"\n{BEGIN_SEARCH_RESULT}无效搜索查询。{END_SEARCH_RESULT}\n"
                    else:
                        if search_query in seq['executed_search_queries']:
                            msg = f"\n{BEGIN_SEARCH_RESULT}重复查询，你已经搜索过'{search_query}'了，请参考之前结果，不要再搜索'{search_query}'了。{END_SEARCH_RESULT}\n"
                        else:
                            seq['executed_search_queries'].add(search_query)
                            seq['search_count'] += 1
                            print(f"\n[搜索] {search_query}")

                            # 查询缓存
                            if search_query in search_cache:
                                res = search_cache[search_query]
                            else:
                                res = search_engine_with_rag(search_query)
                                search_cache[search_query] = res

                            candidates = []
                            # ---------- 生成初步搜索结果 ----------
                            candidates = res.get("data", {}).get("candidated_texts", []) if isinstance(res, dict) else []
                            formatted_docs = format_search_results(candidates, top_k=TOP_K)

                            # 1) 让 LLM 先阐释"搜索意图"
                            raw_intent = _orig_get_llm_response(
                                prompt=add_think_tag(get_search_intent_instruction(seq['prompt'])),
                                stop=[],
                                model=MODEL_NAME,
                                base_url=BASE_URL,
                                echo_stream=False,
                            )
                            search_intent = strip_think(raw_intent)

                            # 2) 进行深度网页探索
                            deep_info = deep_web_explorer(
                                search_query=search_query,
                                search_intent=search_intent,
                                search_result=formatted_docs,
                                search_cache=search_cache,
                                url_cache=url_cache,
                                global_executed_queries=seq['executed_search_queries'],
                            )

                            # 3) 将初步 & 深度结果一并写回主 Prompt
                            search_result_block = (
                                f"\n{BEGIN_SEARCH_RESULT}\n"
                                f"【初步搜索结果】\n{formatted_docs}\n\n"
                                f"【深度网页探索信息】\n{deep_info}\n"
                                f"{END_SEARCH_RESULT}\n"
                            )
                            msg += search_result_block
                            seq['output'] += search_result_block   # ← 新增
                seq['prompt'] += msg
                continue

            # 4-B) 点击分支（同上）
            elif buffer.rstrip().endswith(END_CLICK_LINK):
                if click_url_and_return_md is None:
                    msg = f"\n{BEGIN_CLICK_RESULT}点击工具不可用。{END_CLICK_RESULT}\n"
                else:
                    url = extract_between(buffer, BEGIN_CLICK_LINK, END_CLICK_LINK)
                    if not url:
                        msg = f"\n{BEGIN_CLICK_RESULT}未识别到有效链接。{END_CLICK_RESULT}\n"
                    else:
                        print(f"\n[点击] {url}")
                        try:
                            # 1) 抓取原始网页内容（Markdown/HTML）
                            raw_md = url_cache.get(url)
                            if raw_md is None and callable(click_url_and_return_md):
                                # 若缓存无数据且工具可用，则尝试实际抓取
                                raw_md = click_url_and_return_md(url)

                            # 若抓取结果仍为空或空字符串，则视为失败
                            if not raw_md or not str(raw_md).strip():
                                raise ValueError("click_url_and_return_md 返回内容为空或无法解析")

                            # 缓存原始内容，避免重复抓取
                            url_cache[url] = raw_md

                            # 2) 使用辅助模型摘要网页内容
                            summary_md = summarize_web_content(url, raw_md)

                            # 3) 构造点击结果区块，并写入 msg/output，后续统一追加到 prompt
                            click_block = (
                                f"\n{BEGIN_CLICK_RESULT}\n"
                                f"{summary_md}\n"
                                f"{END_CLICK_RESULT}\n"
                            )
                            msg += click_block
                            seq['output'] += click_block
                        except Exception as e:
                            err_block = (
                                f"\n{BEGIN_CLICK_RESULT}\n抓取失败：{e}\n{END_CLICK_RESULT}\n"
                            )
                            msg += err_block
                            seq['output'] += err_block
                seq['prompt'] += msg
                continue

            # 5) 结束
            else:
                seq['finished'] = True
                # set → list，便于 JSON 持久化
                seq['executed_search_queries'] = list(seq['executed_search_queries'])
                print("-------以下内容是结合所有网页搜索信息得到的最终结果：-------")
                return seq

    except Exception as e:
        # 捕获异常并记录到 seq，确保上层能够写入输出文件
        seq['error'] = str(e)
        print(f"[Error] 处理样例时发生异常：{e}")
        return seq


def deep_web_explorer(
    search_query: str,
    search_intent: str,
    search_result: str,
    search_cache: Dict,
    url_cache: Dict,
    global_executed_queries: Set[str],
) -> str:
    """
    LLM 嵌套式深度网页探索：
    1. 以搜索结果为起点，可继续 <|begin_search_query|>…<|end_search_query|> 或点击链接
    2. 多轮交互直到 LLM 不再请求搜索/点击，返回 **最终信息** 段
    """
    # 初始化子-agent prompt，并附加 THINK_OPEN
    prompt = add_think_tag(
        get_deep_web_explorer_instruction(
            search_query=search_query,
            search_intent=search_intent,
            search_result=search_result,
        )
    )

    executed_queries, clicked_urls = set(), set()
    interactions = 0
    output_acc = ""

    while True:
        raw_buf = _orig_get_llm_response(
            prompt=prompt,
            stop=[END_SEARCH_QUERY, END_CLICK_LINK],
            model=MODEL_NAME,
            base_url=BASE_URL,
            echo_stream=False,
        )
        buf = strip_think(raw_buf)    # ← 去掉 </think>
        output_acc += buf
        prompt     += buf

        # ---------- 处理二次搜索 ----------
        if buf.rstrip().endswith(END_SEARCH_QUERY):
            if interactions >= MAX_DEEP_INTERACTIONS:
                prompt += f"\n{BEGIN_SEARCH_RESULT}已达 Deep-Web 搜索上限{END_SEARCH_RESULT}\n"
                continue

            new_q = extract_between(buf, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
            if not new_q or new_q in executed_queries or new_q in global_executed_queries:
                prompt += f"\n{BEGIN_SEARCH_RESULT}你已经搜索过'{new_q}'了，不允许再搜索'{new_q}'，请使用之前检索到的信息{END_SEARCH_RESULT}\n"
                continue

            executed_queries.add(new_q)
            global_executed_queries.add(new_q)
            interactions += 1
            print(f"  [Deep-Search] {new_q}")

            if new_q in search_cache:
                res = search_cache[new_q]
            else:
                res = search_engine_with_rag(new_q)
                search_cache[new_q] = res

            docs = res.get("data", {}).get("candidated_texts", []) if isinstance(res, dict) else []
            formatted = format_search_results(docs, top_k=TOP_K)
            prompt += f"\n{BEGIN_SEARCH_RESULT}\n{formatted}\n{END_SEARCH_RESULT}\n"
            continue

        # ---------- 处理点击 ----------
        elif buf.rstrip().endswith(END_CLICK_LINK):
            if click_url_and_return_md is None:
                prompt += f"\n{BEGIN_CLICK_RESULT}点击工具不可用{END_CLICK_RESULT}\n"
                continue

            url = extract_between(buf, BEGIN_CLICK_LINK, END_CLICK_LINK)
            if not url or url in clicked_urls:
                prompt += f"\n{BEGIN_CLICK_RESULT}你已经搜索过'{url}'了，不允许再搜索'{url}'，请使用之前点击得到的信息{END_SEARCH_RESULT}{END_CLICK_RESULT}\n"
                continue

            clicked_urls.add(url)
            interactions += 1
            print(f"  [Deep-Click] {url}")

            try:
                # 1) 抓取原始网页内容（Markdown/HTML）
                raw_md = url_cache.get(url)
                if raw_md is None and callable(click_url_and_return_md):
                    raw_md = click_url_and_return_md(url)

                if not raw_md or not str(raw_md).strip():
                    raise ValueError("click_url_and_return_md 返回内容为空或无法解析")

                url_cache[url] = raw_md  # 缓存原始内容

                # 2) 使用辅助模型摘要网页内容
                summary_md = summarize_web_content(url, raw_md)

                # 3) 将摘要插入到主 prompt
                prompt += f"\n{BEGIN_CLICK_RESULT}\n{summary_md}\n{END_CLICK_RESULT}\n"
            except Exception as e:
                err_msg = f"抓取失败：{e}"
                prompt += f"\n{BEGIN_CLICK_RESULT}\n{err_msg}\n{END_CLICK_RESULT}\n"
            continue

        # ---------- 推理结束 ----------
        else:
            break

    return output_acc


def summarize_web_content(url: str, raw_content: str) -> str:
    """使用辅助模型对抓取的网页原始内容进行摘要，仅保留与点击意图相关的关键信息。"""
    # 若原始内容为空，直接返回占位符，避免后续截断或 LLM 调用异常
    if not raw_content or not str(raw_content).strip():
        return "<**网页内容无法被读取**>"

    # 截断原始内容，避免 prompt 过长
    raw_trimmed = raw_content[:MAX_RAW_HTML_CHARS]

    summary_prompt = f"""你是一名信息提炼助手，将协助另一位主模型完成复杂任务。网页信息markdown可能是正常的，也可能会显示无法被正常显示或需要验证的，如果是正常的网页内容就正常进行网页内容总结；如果无法正常显示或需要验证，请输出"<**网页内容无法被读取**>"。

请根据给定的 "检索/点击意图"，阅读网页内容，提取最相关的事实性信息并用中文简洁总结。

要求：
1) 只保留与搜索query高度相关的信息；
2) 以条目或段落形式输出，不超过 500 字；
3) 忽略广告、导航等无关内容；
4) 如果没有仍和相关信息，只输出"<**网页内容无法被读取**>"；
5) 如果有相关信息，输出相关信息并附带潜在有用的 url 链接。


【搜索query】
{url}

【网页内容】
{raw_trimmed}

### 示例输出格式1：网页可以被正常读取的情况###

【当前网页内容摘要】
xxxx

【潜在有用的 url1 链接】
https://www.example.com

【对于url1的摘要】
xxxx

【潜在有用的 url2 链接】
https://www.example.com

【对于url2的摘要】
xxxx

###

### 示例输出格式2：网页无法被正常读取的情况###

【当前网页内容摘要】
<**网页内容无法被读取**>

###
"""
        
    # 调用辅助模型进行摘要，关闭流式输出
    summary = _orig_get_llm_response(
        prompt=add_think_tag(summary_prompt),
        stop=[],
        model=AUX_MODEL_NAME,
        base_url=BASE_URL,
        echo_stream=False,
    )

    # 辅助模型输出可能包含 <think> 标签，统一清洗
    return strip_think(summary).strip()


# ======= 安全版 get_llm_response（容错流式） =======

def safe_get_llm_response(
    prompt: str,
    stop: List[str],
    model: str = MODEL_NAME,
    base_url: str = BASE_URL,
    echo_stream: bool = True,
    timeout: int = 3600,
) -> str:
    """替代原始 get_llm_response，遇到异常时返回已收集 buffer 并继续流程。"""
    buffer = ""
    try:
        client = OpenAI(api_key="sk-placeholder", base_url=base_url)
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
            timeout=timeout,
            max_tokens=3000,
        )

        for chunk in stream:
            try:
                delta = getattr(chunk.choices[0].delta, "content", "") or ""
            except Exception:
                # 若 delta 解析失败，跳过本次
                continue

            buffer += delta
            if echo_stream:
                print(delta, end="", flush=True)

            if any(tag in buffer for tag in stop):
                break
    except Exception as e:
        print(f"[safe_get_llm_response] 捕获异常：{e}，返回已生成内容继续流程。")

    return buffer

# 使用安全函数覆盖原 get_llm_response 引用
get_llm_response = safe_get_llm_response

def main():
    """批量读取指定 jsonl 文件，生成中宏观信息并输出到目标目录。"""

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_JSONL_FILE)

    # -------- 断点重连：收集已处理 apply_seqno --------
    processed_seqnos: Set[str] = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as ck_f:
            for ck_line in ck_f:
                ck_line = ck_line.strip()
                if not ck_line:
                    continue
                try:
                    ck_obj = json.loads(ck_line)
                    seq_no = ck_obj.get("apply_seqno")
                    if seq_no:
                        processed_seqnos.add(str(seq_no))
                except Exception:
                    # 如果历史行损坏，忽略
                    continue
        print(f"[断点续跑] 已检测到 {len(processed_seqnos)} 条已完成记录，将跳过。")

    cache: Dict[str, dict] = {}
    start_time = time.time()

    with open(INPUT_JSONL_FILE, "r", encoding="utf-8") as f_in, \
         open(output_path, "a", encoding="utf-8") as f_out:

        for line_no, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[Warn] 第{line_no}行 JSON 解析失败：{e}")
                continue

            # ---------- 构造借款人信息文本 ----------
            parts: List[str] = []
            for key in ["text_1", "industry", "mainproduct", "text_2", "主营企业"]:
                if key in obj and obj[key]:
                    parts.append(str(obj[key]))
            case_info = "\n".join(parts)

            apply_seqno = str(obj.get("apply_seqno", f"unknown_{line_no}"))

            # 如果已处理，跳过
            if apply_seqno in processed_seqnos:
                print(f"[Skip] apply_seqno {apply_seqno} 已处理，跳过。")
                continue

            print(f"\n===== 处理 apply_seqno: {apply_seqno} (行 {line_no}) =====")

            seq = process_single_case(case_info, cache)

            macro_info = extract_macro_info(seq.get("output", "")) if not seq.get("error") else ""

            result_record = {
                "apply_seqno": apply_seqno,
                "borrower_info": case_info,
                "process_prompt": seq.get("prompt", ""),
                "macro_info": macro_info,
                "status": "failed" if seq.get("error") else "success",
                "error": seq.get("error", ""),
            }

            f_out.write(json.dumps(result_record, ensure_ascii=False) + "\n")

    print(
        f"\n全部完成，总耗时 {time.time() - start_time:.1f}s，结果已保存至 {output_path}"
    )


if __name__ == "__main__":
    main()





