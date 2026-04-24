import json, re, time, sys
from typing import List
import pandas as pd
from tqdm import tqdm

sys.path.append("/workspace2/chiwu/works/Risk_ToT")
from infer import ToTGenerator

# ========== 路径 ==========
input_xlsx   = "/workspace2/chiwu/works/critc_extraction/sentences_after_reclass——0703.xlsx"
output_jsonl = "/workspace2/chiwu/works/critc_extraction/critic_output_iter1.jsonl"

batch_size = 20  # 每批次多少条

# ========== 原 prompt 函数（保持不变） ==========
def get_critic_prompt(topic: str, value_list: List[str]):
    value = "\n".join(value_list)
    return f"""
  以下是关于「{topic}」的多条错误经验，输出合并同类错误经验后的所有经验：
## 注意
- 将表述内容高度重复、参数范围或数值不同但思路一致的内容合并为一条更通用的经验。
- 对于仅仅因为判断边界差异（如10%、20%、30%等）而表述多条的情况，用“区间”或“范围”表达一条综合性规则，避免冗余。
- 请只保留表达思路或判定角度有实质差异的条目，其余归纳在一起。
- 不要机械列举每个变化参数的情况。
- 合并同一情况下的处置方式，比如 1. 户籍地与经营地不一致应为中性，不应列为负向。2.户籍地与经营地不一致分析中正确标注为中性，但归类在负向中，应调整归类。这两条应该合并为：户籍地与经营地不一致应为中性。
- 每个类目下最多10条经验；
- 用尽量简洁的语言表述；
## 经验
{value}
## 输出格式（仅输出json格式，不要输出其他信息）
{{"{topic}":["经验1","经验2"...]}}
  """

# ========== 读取 xlsx ==========
df = pd.read_excel(input_xlsx)           # 需有 themes、sentences 两列
df["sentences"] = df["sentences"].fillna("")

# ========== 调模型 ==========
with ToTGenerator(
        model_path="",
        model_name="deepseek-r1-npu",
        tp_size=0,
        api_key="sk-n2y1nzfkm2mtzja3zi00ztq2lwixngytmjlkzty5yjm5mtvm",
        url="http://industrial.models.antcloud.mybank-inc.cn/v1/",
) as generator, open(output_jsonl, "w", encoding="utf-8") as fout:

    for _, row in tqdm(df.iterrows(), total=len(df), desc="主题进度"):
        topic_key = str(row["themes"]).strip()
        value_list = [s.strip() for s in str(row["sentences"]).split("\n") if s.strip()]
        if not value_list:
            continue

        risk_critic_list = []

        # 这里推荐修正为 range(0, len(value_list), batch_size)
        for i in tqdm(range(0, len(value_list), batch_size), desc=f"[{topic_key}]分批", leave=False):
            group = value_list[i:i + batch_size]
            input_prompt = get_critic_prompt(topic_key, group)

            while True:
                try:
                    result = generator.generate_thoughts(
                        input_prompt, n=1, temperature=0, max_concurrent=1
                    )
                    match = re.search(r"(\{[\s\S]*\})", result[0]["content"])
                    if not match:
                        raise ValueError("未找到JSON结构体")
                    data = json.loads(match.group(1))
                    risk_critic_list.extend(data.get(topic_key, []))
                    break
                except Exception as e:
                    print(f"[{topic_key}] 解析失败：{e}，重试中…")
                    time.sleep(0.5)

        risk_critic_list = list(dict.fromkeys(risk_critic_list))
        fout.write(json.dumps({topic_key: risk_critic_list}, ensure_ascii=False) + "\n")
        print(f"✓ 主题 {topic_key} 写入 {len(risk_critic_list)} 条经验")

print(f"全部完成，结果保存到 {output_jsonl}")
