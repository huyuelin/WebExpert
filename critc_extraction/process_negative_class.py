import pandas as pd
from collections import defaultdict
import re
import time
import sys
from tqdm import tqdm

sys.path.append("/workspace2/chiwu/works/Risk_ToT")
import json
from infer import ToTGenerator

###############################################################################
# 1. 读取原始表
###############################################################################
in_path = "/workspace2/chiwu/works/critc_extraction/sentences_with_topic_merged.xlsx"
out_path = "/workspace2/chiwu/works/critc_extraction/sentences_after_reclass——0703.xlsx"

df = pd.read_excel(in_path)

###############################################################################
# 2. 取出 “负类” 行，并拆分句子
###############################################################################
neg_row = df[df["themes"] == "负类"]
if neg_row.empty:
    print("⚠️  表中不存在 '负类' 行，直接复制保存")
    df.to_excel(out_path, index=False)
    raise SystemExit

sent_list = neg_row.iloc[0]["sentences"].split("\n")
print(f"需重新分类的负类句子：{len(sent_list)} 条")

###############################################################################
# 分类prompt
###############################################################################


###############################################################################
# 4. 逐句分类并归并
###############################################################################
# 建一个 dict {theme: [句子, ...]}
append_dict = defaultdict(list)

with ToTGenerator(
    model_path="",
    model_name="deepseek-r1-npu",
    tp_size=0,
    api_key="sk-n2y1nzfkm2mtzja3zi00ztq2lwixngytmjlkzty5yjm5mtvm",
    url="http://industrial.models.antcloud.mybank-inc.cn/v1/",
) as generator:
    for s in tqdm(sent_list, desc="重新分类负类句子", total=len(sent_list)):
        # 进行分类
        prompt = f"""
    你是一个信贷风险批评经验分析专家，以下是一条错误经验，请输出该经验的所属类别，必须属于下面给出的任何类别之一。
    ## 类别（按照逗号分隔）
    同业认可度,销售趋势,偿债压力,上下游合作标准,涉诉记录,非银机构,行业风险,企业稳定性,经营数据,负债趋势,区域风险,负债结构,信用卡使用,流动资产周转率,交易数据,外部合作机构,关联企业跨行业,交易连续性,负债压力,房贷授信信息,交易起量时间,企业类型,审核标准与信息准确性,电核配合度,GMV差异,征信记录,征信白户风险,年龄与经营经验,还款/逾期行为,授信负债比,风险判定阈值,企业资质,新户风险,电核信息,网商贷风险,婚姻状态,还款行为,支用定价与存留期,上下游同一企业,销售同比数据缺失,网商贷生命周期风险
    ## 错误经验
    {s}
    ## 输出格式
    \\category{{类别名称}}
    """
        while True:
            try:
                result = generator.generate_thoughts(
                    prompt, n=1, temperature=0, max_concurrent=1
                )
                m = re.search(r"\\category\{(.*?)\}", result[0]["content"])
                if m:
                    new_theme = m.group(1).strip()
                    print(f"{s}:提取到类别 →", new_theme)     # 销售趋势
                    break  # 成功就跳出循环
                else:
                    print("未找到 \\category{} 格式")
            except Exception as e:
                print(f"出错了：{e}，正在重试……")
                time.sleep(0.5)
        append_dict[new_theme].append(s)

# 打印分配情况
for k, v in append_dict.items():
    print(f" → 主题 <{k}> 获得 {len(v)} 条")

###############################################################################
# 5. 把句子追加到 df 中相应的行；没有则新增
###############################################################################
for theme, new_sents in append_dict.items():
    if (df["themes"] == theme).any():                 # 已存在该主题
        idx = df.index[df["themes"] == theme][0]
        orig_sents = df.at[idx, "sentences"]
        df.at[idx, "sentences"] = orig_sents + "\n" + "\n".join(new_sents)
    else:                                             # 新主题行
        # 生成 topic_idx：取当前最大 idx + 1 ；也可按需解析
        next_idx = df["topic_idx"].max() + 1
        df = pd.concat([
            df,
            pd.DataFrame([{
                "topic_idx": next_idx,
                "themes": theme,
                "sentences": "\n".join(new_sents)
            }])
        ], ignore_index=True)

###############################################################################
# 6. 删除原“负类”行，并重新排序
###############################################################################
df = df[df["themes"] != "负类"].copy()
df = df.sort_values("topic_idx").reset_index(drop=True)

###############################################################################
# 7. 写出：Sheet1 主表；Sheet2 给出所有主题串
###############################################################################
all_themes_str = ",".join(df["themes"].tolist())

with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
    df.to_excel(writer, sheet_name="merged", index=False)
    pd.DataFrame({"all_themes":[all_themes_str]}).to_excel(
        writer, sheet_name="all_themes", index=False
    )

print(f"✅ 处理完成，保存到 {out_path}")
