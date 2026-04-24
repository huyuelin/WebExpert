import pandas as pd
import re
# VLLM调用大模型
import sys
import time
sys.path.append('/workspace2/chiwu/works/Risk_ToT')
import json
from infer import ToTGenerator

# 1. 读取Excel文件所有sheet到字典
input_path = '/workspace2/chiwu/works/critc_extraction/bertopic_by_topic.xlsx'
sheets = pd.read_excel(input_path, sheet_name=None)  # None则读全部sheet，结果为字典

prompt_input = """以下是一些主题词参考：电核配合度
    经营地异常
    中标数据
    年龄风险
    GMV风险
    授信风险
    销售波动
    负债压力
    企业成立时间
    流动资产周转率
    外部合作
    上游合作
    非银机构
    销售下滑
    经营数据
    销售趋势
    同业认可度
    偿债压力
    电商平台
    法人信息
    交易数据
    信用卡使用
    网商贷风险
    融资风险
    下游集中度
    负债结构
    上下游合作
    企业稳定性
    经营时间
    库存压力
    纳税等级
    注册资本
    关联企业
    授信额度
    交易数据质量
    行业风险
    特殊交易
    区域风险
    征信记录
    企业资质
    还款/逾期行为
    涉诉记录
    请你对这些批评内容归纳出一个主题词，该主题词表达了这些批评内容的分析方面，可参考上面给出的主题词，也可以归纳出一个新的类似格式的主题词。
    输出格式为\\topic{}"""


import nest_asyncio
nest_asyncio.apply()
import math
from tqdm import tqdm  # 进度条核心

# 2. 输出 xlsx 的路径
output_xlsx = "/workspace2/chiwu/works/critc_extraction/sentences_with_topic_new.xlsx"


rows = []

start_topic_idx = 0  # 如果需要从某个序号起，可改此参数
sheet_items = list(sheets.items())

with ToTGenerator(
    model_path="",
    model_name="deepseek-r1-npu",
    tp_size=0,
    api_key="sk-n2y1nzfkm2mtzja3zi00ztq2lwixngytmjlkzty5yjm5mtvm",
    url="http://industrial.models.antcloud.mybank-inc.cn/v1/",
) as generator:
    sheet_items = list(sheets.items())
    # tqdm包裹外层topics循环
    for topic_name, df_topic in tqdm(sheet_items[start_topic_idx:], desc="主题Topic处理"):
        contents = df_topic['text'].tolist()
        batch_size = 100
        num_batches = math.ceil(len(contents) / batch_size)

        themes = set()
        # tqdm包裹内层batch处理
        for i in tqdm(range(num_batches), 
                      desc=f"{topic_name}分批", 
                      leave=False, 
                      position=1):
            batch_contents = contents[i * batch_size : (i + 1) * batch_size]
            prompt_text = '\n'.join(batch_contents)
            prompt = f"""你是一个信贷审核专家，以下是若干条根据风险点分析结论的批评结果：
{prompt_text}
{prompt_input}"""

            while True:
              try:
                  result = generator.generate_thoughts(prompt, n=1, temperature=0, max_concurrent=1)
                  # 如果这里你还想验证内容是否合理，比如result为空或不符合要求，可以在这里加判断
                  # 例如: if not result or result不合理: continue
                  break  # 成功就跳出循环
              except Exception as e:
                  print(f"出错了：{e}，正在重试……")
                  time.sleep(0.5)
            # 提取出主题词
            # 匹配 \topic{...}
            match = re.search(r'\\topic\{(.*?)\}', result[0]['content'])
            if match:
                topic = match.group(1)
                print("主题词为：", topic)
            else:
                print("未找到主题词")
            themes.add(topic)
        final_key = ','.join(sorted(themes))              # themes 字符串
        joined_sent = '\n'.join(contents)                 # 句子合并

        rows.append({
            "sheet": topic_name,
            "themes": final_key,
            "sentences": joined_sent
        })

pd.DataFrame(rows).to_excel(output_xlsx, index=False)