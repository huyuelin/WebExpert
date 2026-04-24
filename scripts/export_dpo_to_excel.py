import os
import json
import pandas as pd

# INPUT_FILE = "/workspace2/linwen/dpo_dataset_prepare/expert_search_queries_dpo_dataset.jsonl"
# OUTPUT_XLSX = "/workspace2/linwen/dpo_dataset_prepare/expert_search_queries_dpo_dataset.xlsx"

INPUT_FILE = "/Users/linwen/Desktop/agent_AC/mybank_webthinker/dpo_dataset_prepare/expert_search_queries_dpo_dataset.jsonl"
OUTPUT_XLSX = "/Users/linwen/Desktop/agent_AC/mybank_webthinker/dpo_dataset_prepare/expert_search_queries_dpo_dataset.xlsx"


def build_borrower_info(row: dict) -> str:
    """组合借款人信息到一个单元格，多行展示"""
    parts = [
        f"apply_seqno: {row.get('apply_seqno', '')}",
        f"industry: {row.get('industry', '')}",
        f"mainproduct: {row.get('mainproduct', '')}",
        "--- text_1 ---",
        row.get("text_1", ""),
        "--- text_2 ---",
        row.get("text_2", ""),
    ]
    return "\n".join(parts)


def list_to_str(lst) -> str:
    if not lst:
        return ""
    # Pretty JSON
    return json.dumps(lst, ensure_ascii=False, indent=2)


def main():
    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(f"文件不存在: {INPUT_FILE}")

    rows = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            row = {
                "借款人信息": build_borrower_info(obj),
                "候选1": list_to_str(obj.get("generation_1_parsed")),
                "候选2": list_to_str(obj.get("generation_2_parsed")),
                "候选3": list_to_str(obj.get("generation_3_parsed")),
                "最佳(填1/2/3)": "",
                "次佳": "",
                "最差": "",
                "标注理由": "",
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    # 自动换行显示需设置Excel列宽/样式，简单导出即可
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"已导出 {len(rows)} 行至 {OUTPUT_XLSX}")


if __name__ == "__main__":
    main() 