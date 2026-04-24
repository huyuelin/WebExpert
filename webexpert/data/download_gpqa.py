"""GPQA dataset downloader.

Usage:
  python -m webexpert.data.download_gpqa --output-dir dataset/GPQA
"""

import argparse
import json
import os
import random


def download_gpqa(
    output_dir: str = "dataset/GPQA",
    subsets: tuple = ("gpqa_extended",),
) -> None:
    """Download GPQA from HuggingFace.

    Constructs MCQ format with shuffled options.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading GPQA to {output_dir}...")

    for subset in subsets:
        ds = load_dataset("Idavidrein/GPQA", subset, trust_remote_code=True)
        for split in ds:
            items = []
            for row in ds[split]:
                options = [
                    row.get("Correct Answer", ""),
                    row.get("Incorrect Answer 1", ""),
                    row.get("Incorrect Answer 2", ""),
                    row.get("Incorrect Answer 3", ""),
                ]
                random.shuffle(options)
                correct = row.get("Correct Answer", "")
                answer_letter = chr(65 + options.index(correct)) if correct in options else "A"
                items.append({
                    "question": row.get("Question", ""),
                    "answer": answer_letter,
                    "subject": row.get("Subdomain", row.get("Domain", "")),
                    "options": {chr(65 + i): opt for i, opt in enumerate(options)},
                })
            out_path = os.path.join(output_dir, f"{subset}_{split}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Saved {len(items)} items to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dataset/GPQA")
    args = parser.parse_args()
    download_gpqa(args.output_dir)
