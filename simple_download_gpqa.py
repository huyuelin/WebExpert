#!/usr/bin/env python3
"""
简单的 GPQA 数据集下载与标准化脚本
- 来源：HuggingFace Idavidrein/gpqa （含 gold/extended/diamond 三个难度子集）
- 目标：统一输出到 dataset/GPQA/ 下的 jsonl 文件，字段：
  question, answer, subject（Physics/Chemistry/Biology），source_subset（gold/extended/diamond）

使用：
  export HF_TOKEN=xxxxxxxx   # 如需鉴权
  python simple_download_gpqa.py
"""

import os
import json
from pathlib import Path
from typing import Dict
import random


def _process_sample(example: Dict, subset_name: str) -> Dict:
    # 字段名兼容
    q_stem = example.get("Question") or example.get("question") or ""
    correct = example.get("Correct Answer") or example.get("answer") or ""
    subj = example.get("Subject") or example.get("subject") or ""

    # 收集四个选项（正确 + 3个错误），并随机打乱，记录正确答案字母
    wrongs = [
        example.get("Incorrect Answer 1"),
        example.get("Incorrect Answer 2"),
        example.get("Incorrect Answer 3"),
    ]
    choices = [c for c in [correct] + wrongs if isinstance(c, str) and c.strip()]
    if len(choices) < 4:
        # 兜底：若数据缺项则不构造MCQ，只保留原问答
        letter_answer = ""
        question_text = q_stem
    else:
        random.shuffle(choices)
        idx = choices.index(correct)
        letter_answer = "ABCD"[idx]
        options_block = "\n".join(
            [f"A) {choices[0]}", f"B) {choices[1]}", f"C) {choices[2]}", f"D) {choices[3]}"]
        )
        question_text = f"{q_stem}\n\n{options_block}"

    # 标准化学科
    subj_norm = (str(subj).strip().lower())
    mapping = {
        'phy': 'Physics', 'physics': 'Physics',
        'chem': 'Chemistry', 'chemistry': 'Chemistry',
        'bio': 'Biology', 'biology': 'Biology'
    }
    subj_final = mapping.get(subj_norm, (str(subj).strip().title() if subj else ""))

    return {
        "question": question_text,
        # 若构造了选项，则将标准答案统一为字母；否则回退到原文本
        "answer": letter_answer or correct,
        "subject": subj_final,
        "source_subset": subset_name,
    }


def main():
    try:
        from datasets import load_dataset
    except Exception:
        os.system("pip install datasets -q")
        from datasets import load_dataset

    save_dir = Path("dataset/GPQA")
    save_dir.mkdir(parents=True, exist_ok=True)

    subsets = [
        ("gpqa_main", "gold"),        # 主集 gold
        ("gpqa_extended", "extended"),
        ("gpqa_diamond", "diamond"),
    ]

    for subset, name in subsets:
        print(f"🔄 下载 GPQA 子集: {subset} ...")
        ds = load_dataset("Idavidrein/gpqa", subset, split="train")
        out_file = save_dir / f"{name}.jsonl"
        with open(out_file, 'w', encoding='utf-8') as f:
            for ex in ds:
                rec = _process_sample(ex, name)
                if rec["question"] and rec["answer"]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"✅ 保存 {name}: {len(ds)} 条 -> {out_file}")

    # 写入数据说明
    meta = {
        "source": "Idavidrein/gpqa",
        "files": ["gold.jsonl", "extended.jsonl", "diamond.jsonl"],
        "schema": {"question": "str", "answer": "str", "subject": "Physics/Chemistry/Biology", "source_subset": "str"}
    }
    with open(save_dir / "_meta.json", 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("🎉 GPQA 下载完成！")


if __name__ == "__main__":
    main()


