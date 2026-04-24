#!/usr/bin/env python3
"""Experience Extraction CLI.

Run the critic-guided expert experience extraction pipeline
on domain-specific QA corpora.

Usage:
  python run_extraction.py --dataset GAIA --data-path dataset/GAIA/
  python run_extraction.py --all
"""

import argparse
import json
from pathlib import Path

from webexpert.experience.extractor import CriticGuidedExtractor


DATASET_CONFIGS = {
    "GAIA": {
        "text_fields": ["Question", "Final answer", "Annotator Metadata"],
        "domain_terms": ["CFA Institute", "FDA", "SEC", "GAAP", "IFRS"],
    },
    "GPQA": {
        "text_fields": ["question", "answer", "explanation"],
        "domain_terms": ["CFA", "FDA", "NIH", "NSF", "IEEE"],
    },
    "HLE": {
        "text_fields": ["question", "answer", "explanation"],
        "domain_terms": [],
    },
    "WebWalkerQA": {
        "text_fields": ["question", "answer", "trajectory", "reasoning"],
        "domain_terms": [],
    },
}


def load_qa_tuples(data_path: str) -> list:
    """Load QA tuples from a directory of JSON/JSONL files."""
    path = Path(data_path)
    items = []
    for fp in sorted(path.rglob("*")):
        if not fp.is_file():
            continue
        if fp.suffix not in (".json", ".jsonl"):
            continue
        with open(fp, "r", encoding="utf-8") as f:
            if fp.suffix == ".jsonl":
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))
            else:
                data = json.load(f)
                if isinstance(data, list):
                    items.extend(data)
                else:
                    items.append(data)
    return items


def run_extraction(
    dataset: str,
    data_path: str,
    output_dir: str = "expert_outputs",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    """Run the extraction pipeline for a single dataset."""
    print(f"Extracting expert experiences for {dataset} from {data_path}")

    qa_tuples = load_qa_tuples(data_path)
    if not qa_tuples:
        print(f"No data found for {dataset} at {data_path}")
        return

    config = DATASET_CONFIGS.get(dataset, {})
    domain_terms = config.get("domain_terms", [])

    extractor = CriticGuidedExtractor(embedding_model=embedding_model)
    rules = extractor.run(qa_tuples, domain_terms=domain_terms)

    out_path = Path(output_dir) / dataset.lower() / "experience_base.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    extractor.save(str(out_path))
    print(f"Saved {len(rules)} expert rules to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="WebExpert Experience Extraction")
    parser.add_argument("--dataset", type=str, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="expert_outputs")
    parser.add_argument("--embedding-model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--all", action="store_true",
                        help="Run extraction for all datasets under dataset/")
    args = parser.parse_args()

    if args.all:
        base = Path(args.data_path)
        for ds in DATASET_CONFIGS:
            ds_path = base / ds
            if ds_path.exists():
                run_extraction(ds, str(ds_path), args.output_dir, args.embedding_model)
            else:
                print(f"Skipping {ds}: {ds_path} not found")
    else:
        assert args.dataset, "Specify --dataset or use --all"
        run_extraction(args.dataset, args.data_path, args.output_dir, args.embedding_model)


if __name__ == "__main__":
    main()
