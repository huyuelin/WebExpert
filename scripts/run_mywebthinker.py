import json
import re
import time
from typing import Dict, List, Set


from prompt.prompt import (
    get_first_task_instruction_prompt,
    get_search_intent_instruction,          # ← 新增
    get_deep_web_explorer_instruction,      # ← 新增
    get_expert_domain_analysis_prompt,       # ← 新增
    get_expert_experience,
)

from search.search_engine_mybank import search_engine_with_rag
from scripts.get_llm_response.online_get_llm_response import get_llm_response
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
SINGLE_INFO: str | None = (
    # """
    # 授信金额：60万元，拦截金额：50万元，已电联且配合。
    # 主营企业杭州余杭区五常街道张争光餐饮店，2018年2月成立，个体工商户，借款人33岁已婚。
    # 1、经营情况：无任何经营GMV数据，偿债三期经营GMV为1.93万元/月，主要来自饿了么；口述22年、23年销售额分别为70万元、60万元，24年1-10月销售额为50万元。
    # 2、征信及负债情况：当前无经营贷合作机构，近1个月他行申贷2家，未见获贷；当前总贷款规模为97.33万元，无经营性贷款；我行授信规模为月均偿债规模的31倍，存在过度授信风险。
    # 3、征信及还款表现：征信还款表现正常，暂无重大外部负面信息。

    # """
    # """
    # #基本信息：企业成立时间:5063天，法定代表人占股50.00%，江苏省苏州市 区域客户。 
    # #经营情况：所属行业:五金零售，核实电商数据:927.57万元，月均77.3万，近12月销售合计同比增长9%，近6月销售合计同比增长6%，近3月销售合计同比下滑7%。 
    # #负债情况及征信情况：整体负债:0.00元，征信还款无明显异常，剔除网商、抵押：法定代表人外部金融机构单家最高授信额0.01万，企业单家最高授信2.5万。 
    # #网商还款表现及外面负面情况：网商贷历史累计支用4次，历史逾期次数0次，无重大外部负面风险。 
    # #网商授信及支用情况：网商1001对客利率8.8%，网商1001授信160.4万，无支用。
    # """

    """
    text_1:	### 借款人信息\n年龄：40\n户籍地：河北\n学历：初中及以下\n婚姻状态：已婚\n经营城市数量：1.0\n最长经营月份数：121\n经营省份数量：1.0\n### 主营企业\n企业名称：任丘市鼎盛采暖设备有限公司\n企业类型：有限责任公司（自然人投资或控股）\n注册资本：500.0万元\n持股比例：50%\n成立时间：20210608\n成为法人时间：20210608\n法人变更时间：None\n\n\n### 关联企业信息\n作为股东在营（开业）企业数量: 4\n公司名称:任丘市乐迅玛科技有限公司\n公司类型:有限责任公司（自然人投资或控股）\n持股比例(%):63.65\n注册资本(万元):2105.26\n公司名称:任丘市费加罗德商贸有限公司\n公司类型:有限责任公司（自然人独资）\n持股比例(%):100\n注册资本(万元):300.0\n公司名称:保定市环鼎电子商务有限公司\n公司类型:有限责任公司（自然人投资或控股）\n持股比例(%):50\n注册资本(万元):100.0\n\n### 电核信息\n法人是否为控制人:SAME_PERSON\n业务模式:PRODUCTION\n企业名称:任丘市费加罗德商贸有限公司\n注册日期:2021-06-08\n是否有企业注册:True\n主要营业地址:河北&amp;middot;沧州&amp;middot;任丘市\n主要企业名称:任丘市鼎盛采暖设备有限公司\n联系状态:接通\n企业类型:INDUSTRY_INTEGRATION\n行业代码名称:建筑装饰及水暖管道零件制造\n业务区域:河北省沧州市任丘市\n主要营业地址详情:河北省沧州市任丘市于村乡西于村492号\n主要产品:生产销售：常压民用锅炉、常压民用采暖炉、民用暖气片\n申请日期:2025-02-15\n是否面谈:N\n是否多企业经营:True\n法人实际关系:同一人\n法人开始日期:2021-06-08\n法人姓名:吕海涛\n注册地址:河北\n\n电核备注：1、主营公司：任丘市鼎盛采暖设备有限公司（非申贷企业）\n2、工商：法人实际持股比例：50%、工商近两年重大变更：无、行政处罚：无、经营状态异常提示：无；\n3、收入：销售确认（目前：100万左右、去年：客户口述不清楚、前年：客户口述不清楚），口述开票比例：差不多100%，客户口述基本都开，口述今年销售同比增幅：下滑；\n上游：各行业，结算方式：账期现结，月结，季度结都有；\n下游：各行业，结算方式：账期现结，月结，季度结都有，承兑：无；\n库存50-100万，备货30天，毛利率客户口述不清楚；\n进出口客户表示刚做，进出口比例及国家区域还未统计；\n4、负债规模：无调整项；\n5、涉诉：无诉讼；\n6、下游企业：北京京东世纪贸易有限公司占比销售的70.33%，排查诉讼和工商情况：在营状态，未涉及失信限高，涉及刑事一审8笔案件，刑事二审2笔案件。\n## 经营标签\n1688-诚信通商家:是\n1688-主营一级类目:数码家电\n1688-主营二级类目:生活电器\n1688-品牌:环鼎\n
    industry	:	建筑装饰及水暖管道零件制造
    mainproduct	:	生产销售：常压民用锅炉、常压民用采暖炉、民用暖气片
    text_2	:	### 经营稳定性\n行业：锅炉及辅助设备制造\n经营资质：环境管理体系认证, 质量管理体系认证（ISO9001）\n电商用户，主要采信电商、1688或淘系数据\n主电商平台：京东商城,非内部平台\n经营模式：PRODUCTION\n销售渠道：客户口述不清楚，不是自己在处理\n网商贷资金用途：TEMPORARY_BACKUP\n借款人与控制人是同一人\n新鲜度-数据截止日期距离调查日期：0月\n近一年GMV合计: 9838.48万元\n口述gmv：缺失\n从24个月占比最高的数据源分析： \n近12月销售同比:上升23.56%\n近6月销售同比:上升7.31%\n近3月销售同比:上升7.37%\n最大2月占近12月的销售比例:46.00%\n近6个月交易为0的月份数:0\n交易起量时间:满足要求\n近3年的每季度GMV分布：\n2023年Q1: 2927.66万元, 当年占比: 36.83%\n2023年Q2: 683.71万元, 当年占比: 8.60%\n2023年Q3: 534.28万元, 当年占比: 6.72%\n2023年Q4: 3802.82万元, 当年占比: 47.84%\n2024年Q1: 3523.68万元, 当年占比: 39.96%\n2024年Q2: 1609.27万元, 当年占比: 18.25%\n2024年Q3: 331.10万元, 当年占比: 3.75%\n2024年Q4: 3353.76万元, 当年占比: 38.03%\n2025年Q1: 2818.25万元, 当年占比: 100.00%\n\n### 营运能力\n流动资产周转率:3.65\n### 回款能力-原告诉讼纠纷\n法研院融合近1年作为原告的民事诉讼案件金额: 0.0元\n法研院融合近1年作为原告的民事诉讼案件笔数:0.0\n### 上下游信息\n链主合作年限：缺失\n#### 任丘市费加罗德商贸有限公司：\n主要上游：\n嘉兴市世环电器有限公司 41.0万元 占比46.04%，合作年份：3 年；\n南京苏宁易购电子商务有限公司 29.0万元 占比32.69%，合作年份：2 年；\n主要下游：\n福州卓凡酒店用品贸易有限公司 72.0万元 占比84.72%，合作年份：1 年；\n\n#### 任丘市鼎盛采暖设备有限公司：\n主要上游：\n嘉兴市世环电器有限公司 230.0万元 占比26.72%，合作年份：2 年；\n宁波一鑫电子科技有限公司 104.0万元 占比12.12%，合作年份：1 年；\n主要下游：\n北京京东世纪贸易有限公司 1130.0万元 占比94.22%，合作年份：5 年；\n### 近1年中标数据\n没有符合条件的中标记录。\n
    主营企业	:	['任丘市鼎盛采暖设备有限公司', '任丘市费加罗德商贸有限公司']
    
    
    """
)
# INPUT_FILE = "./data/pending_cases.txt"  # 若要批量处理，将 SINGLE_INFO 设为 None，并指定文件路径
INPUT_FILE: str | None = None
OUTPUT_FILE: str = "output/output.json"

MAX_SEARCH_LIMIT: int = 10  # 每条样例最多搜索次数
TOP_K: int = 10  # 搜索结果条数
#MODEL_NAME: str = "qwq-32b"  # 主模型
MODEL_NAME: str = "qwen3-32b"
# 辅助摘要模型（用于 Deep-Web 点击内容压缩）
AUX_MODEL_NAME: str = "qwen3-32b"
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

def run_expert_stage(case_txt: str) -> list[dict]:
    """
    调用 LLM 生成搜索规划，返回 list[dict]，每个 dict 与 prompt 约定的字段一致
    """
    expert_prompt = get_expert_domain_analysis_prompt(
        pending_info=case_txt,
        approval_logic=get_expert_experience() # 可把表格内容抽成常量
    )
    raw = get_llm_response(
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
    # 0) Expert Stage 先行
    expert_targets = run_expert_stage(case_info)   # list[dict]

    # ---------- 初始化主 prompt ----------
    instruction = get_first_task_instruction_prompt(
        MAX_SEARCH_LIMIT,
        case_info,
        json.dumps(expert_targets, ensure_ascii=False, indent=2),
    )

    prompt = add_think_tag(instruction)

    # # 把 Expert 结果作为"前置背景"写入 prompt，方便主模型参考
    # if expert_targets:
    #     prompt += (
    #         "\n【专家规划】以下是需要优先检索的 L2 领域及搜索意图，请依次执行：\n"
    #         + json.dumps(expert_targets, ensure_ascii=False, indent=2)
    #         + "\n"
    #     )

    # ====================================
    seq = init_sequence(case_info, prompt)
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
        raw_buf = get_llm_response(
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
                        raw_intent = get_llm_response(
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
        raw_buf = get_llm_response(
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
    summary = get_llm_response(
        prompt=add_think_tag(summary_prompt),
        stop=[],
        model=AUX_MODEL_NAME,
        base_url=BASE_URL,
        echo_stream=False,
    )

    # 辅助模型输出可能包含 <think> 标签，统一清洗
    return strip_think(summary).strip()


def main():
    if not SINGLE_INFO and not INPUT_FILE:
        raise ValueError("请设置 SINGLE_INFO 或 INPUT_FILE 之一")
    if SINGLE_INFO:
        cases = [SINGLE_INFO]
    else:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            # 每行一个 JSON 或纯文本
            cases = [line.strip() for line in f if line.strip()]
    
    print(f"处理审批人案例信息: {cases}")

    outputs = []
    cache: Dict[str, dict] = {}
    start_time = time.time()
    for idx, case in enumerate(cases, 1):
        print(f"\n===== 处理第 {idx}/{len(cases)} 条样例 =====")
        seq = process_single_case(case, cache)
        outputs.append(seq)          # 直接保存序列字典

    # 序列字典中 set 已转为 list，可直接 json.dump
    from datetime import datetime
    now = datetime.now()
    output_file = f"output/output_{now.strftime('%m%d_%H%M')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    print(f"\n全部完成，总耗时 {time.time() - start_time:.1f}s，结果已保存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()


