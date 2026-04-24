from openai import OpenAI
from typing import List

def get_llm_response(
    prompt: str,
    stop: List[str],
    model: str = "deepseek-v3-0324",
    #model: str = "qwen3-32b",
    #model: str = "qwq-32b",
    #api_key: str = "sk-ymuxmwq2zdmtnjkyny00ymu2ltlmymitzju3njjmmzc2zdc5",
    api_key: str = "sk-n2y1nzfkm2mtzja3zi00ztq2lwixngytmjlkzty5yjm5mtvm",
    base_url: str = "https://openai.mybank.cn/v1",
    timeout: int = 3600,
    echo_stream: bool = True,   # 是否边收边打印
) -> str:
    """
    连续流式调用大模型，检测到 stop_tag 即停止并返回累计的 buffer。

    Parameters
    ----------
    prompt : str
        整个用户提示词。
    stop : List[str]
        停止标记列表；一旦 buffer 中出现任意标记即立刻终止流。
    model : str, optional
        所调用的模型名称。
    api_key : str, optional
        OpenAI API Key（可改为从环境变量读取）。
    base_url : str, optional
        私有部署网关地址。
    timeout : int, optional
        请求超时（秒）。
    echo_stream : bool, optional
        若为 True，则边收到 delta 边打印到控制台。

    Returns
    -------
    str
        累计至停止标记前（含标记本身或可自行裁剪）的完整输出。
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    buffer = ""
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ],
        # extra_body={
        #     "enable_thinking": True,
        #     'thinking_budget': 1500,

        # },
        stream=True,
        timeout=timeout,
        max_tokens=6000,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        buffer += delta
        if echo_stream:
            print(delta, end="", flush=True)

        # 检测停止标记（写法灵活，可按需改为只检测 buffer 末尾）
        if any(tag in buffer for tag in stop):
            break

    return buffer


# ------------------ 用    例 ------------------
if __name__ == "__main__":
    END_SEARCH_QUERY = "<|end_search_query|>"
    my_prompt = """You are a reasoning assistant with the ability to perform web searches to help you answer the user\'s question accurately. You have special tools:\n\n- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\nThen, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\nYou can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to 20.\n\nOnce you have all the information you need, continue your reasoning.\n\nExample:\nQuestion: "Alice David is the voice of Lara Croft in a video game developed by which company?"\nAssistant thinking steps:\n- I need to find out who voices Lara Croft in the video game.\n- Then, I need to determine which company developed that video game.\n\nAssistant:\n<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n(System returns processed information from relevant web pages)\n\nAssistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\nAssistant:\n<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n(System returns processed information from relevant web pages)\n\nAssistant continues reasoning with the new information...\n\nRemember:\n- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n- When done searching, continue your reasoning.\n\nPlease answer the following question. You should think step by step to solve it.\n\nProvide your final answer in the format \\boxed{YOUR_ANSWER}.\n\nQuestion:\nWhat is OpenAI Deep Research?\n\n"""
    #my_prompt = """你是一个信贷审核专家,以下是用户信息、风险评估标准和依据风险评估标准对该用户的风险分析，需要你检查这段分析中存在的错误或者遗漏的点。已知申请时间为20240122 # 用户信息 ## 履约行为 无 ### 明显负向 无 ## 涉诉情况 ### 原告诉讼纠纷 法研院融合近1年作为原告的民事诉讼案件金额: 0.0元 法研院融合近1年作为原告的民事诉讼案件笔数:0.0 无 # 风险评估标准 - 外部征信履约记录: 通过借款人个人征信及企业征信当前及历史的还款表现，判断客户还款意愿及还款能力，违约次数与还款意愿呈负相关, 违约记录越多表明还款意愿越低，履约能力差，重点关注逾期、欠款等异常行为，需根据具体情况判断轻微/一般/严重负向。如果没有逾期行为，表示履约能力正常，为中性。 - 内部履约行为: 网商贷的历史支用及还款表现行为，分析客户的还款习惯和意愿，任何形式的违约行为均表明用户还款意愿存在问题，违约行为越多，客户的还款意愿越差，需根据具体情况判断轻微/一般/严重负向。如果没有逾期行为，表示履约能力正常，为中性。 a. 客户在我行的生命周期: 若客户与我行合作周期长，且支用还款行为正常，风险较低。若为新户（未使用网商贷）或是流失老户（近2年未支用网商贷），一般负向； b. 在贷余额笔数: 用笔数过多且满额支用网商贷不符合企业一般用款习惯，可能存在拆单还网商贷的情况（比如大于30笔），说明客户资金较为紧张，为负向。 c. 近12个月平均余额存留期（含未结清）& 近12个月支用加权定价: 近12个月平均余额存留期表示近12月平均用款时长，若长期高定价支用，即近12个月平均余额存留期>=270天且近12个月支用加权定价>=14.4%，说明用户长期难以获取低息贷款，外部融资能力较弱，为负向。若短期(近12个月平均余额存留期<270天)高定价支用或长期低定价(近12个月支用加权定价<14.4%)支用，为中性。 d. 月账单还款账户数：若月账单还款账户数较多，客户可能存在资金不足，需要通过多个渠道才能凑齐还款金额，月账单还款账户数>3为一般负向。 e. 还款日到期前支用次数: 还款日到期前支用次数多，说明客户资金较为紧张，支用次数>5为负向。 - 涉诉情况: 客户当前或是历史被起诉及履约行为，判断客户的履约意愿及履约能力。 a. 若存在大额案件或涉案总金额相比月均GMV较大或法研院融合近1年作为原告的民事诉讼案件笔数>10或法研院融合近1年作为原告的民事诉讼案件金额>月GMV，需根据具体情况判断轻微/一般/严重负向。若涉案总金额相比月流水较小或案件次数较少，说明涉诉对企业影响较小，为中性。 b. 重点关注合同纠纷、金融、借贷纠纷等金融类性质的案件，从涉案情况来判断企业履约能力和意愿、资金流是否健康及涉案是否会影响企业经营，其他类型的案件尤其是小额案件对企业履约能力影响较小，可降低关注度。 c. 若案件已结案，优先关注案件的结案金额，相比未结案，已结案的案件风险更小。 d. 若存在被告执行案件且案由涉及财产保全，或存在被告破产案件，说明企业存在严重经营问题，判定为负向。 e. 重点关注借款人涉诉情况及后续履约表现，司法纠纷及违约记录直接反映借款人的履约能力和意愿的缺失，判定为负向。如果没有逾期行为，表示履约能力正常，风险中性。 # 分析 正向** - 无逾期/涉诉记录，中性。 **负向** - 无 # 以下是典型容易出错的点，除这些点外如果有其他错误的点也可以抓出来。 - 优先关注案件的结案金额，若结案金额相比月GMV较小，不是负向，只是中性偏负。如果判断错误需要提示“标准错误” - 如果有未结案件且没有输出，需要提示“风险缺失”。 - 除了合同纠纷、金融、借贷纠纷等金融类性质之外的其他类型案件，影响较小，属于中性偏负，如果分析中提到影响较大需要提示“标准错误” - 近12个月平均余额存留期（含未结清）& 近12个月支用加权定价: 近12个月平均余额存留期表示近12月平均用款时长，若长期高定价支用，即近12个月平均余额存留期>=270天且近12个月支用加权定价>=14.4%，说明用户长期难以获取低息贷款，外部融资能力较弱，为一般负向。若短期(近12个月平均余额存留期<270天)高定价支用为轻微负向。若表述中描述错误需要提示“标准错误”。 - 若月账单还款账户数>3, 客户可能存在资金不足，需要通过多个渠道才能凑齐还款金额，为一般负向。若用户信息中出现但未抓出来该负向需要提示“风险缺失”。 - 还款日到期前支用次数: 还款日到期前支用次数>5，说明客户资金较为紧张，为严重负向。若用户信息中出现但未没抓出来该负向需要提示“风险缺失”。 - 输出的信息是否来自用户信息，如果不是从用户信息中来，提示"信息错误" - 是否存在明显的逻辑问题，如前后矛盾或逻辑混乱，提示"逻辑错误" 请你检查给出的分析是否存在错误，若存在则指出问题及错误类型，错误类型包括以下4类 (1) 信息错误：分析中提到的用户信息与给定的用户信息不一致，重点关注数值、日期及是否为用户信息中未提到的部分 (2) 标准错误：分析的结论与不符合风险评估标准不一致，如正/中/负向的判定及程度（轻微/一般/严重）判定不恰当 (3) 风险缺失：用户存在风险但分析中未提及，注意中性或者正向可以不提及，不算风险缺失，只有负向未提及才算风险缺失。 (4) 逻辑错误：分析存在数值大小比较错误、前后矛盾或逻辑混乱 请对风险分析中的每个点逐项检查，如不存在错误，则输出无错误。错误类型放在最后，先分析错误原因，再给出错误类型。如果有风险缺失，单独列一项**风险缺失**来记录缺失内容。注意没有在**风险评估标准**中提到的风险不算风险缺失，不要提及。请只输出正向、负向、风险缺失，不要输出其他分析内容。 """
    result = get_llm_response(my_prompt, stop=[END_SEARCH_QUERY])
    print("\n\n=== 最终 buffer ===\n", result)
