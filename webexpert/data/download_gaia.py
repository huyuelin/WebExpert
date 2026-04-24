"""GAIA dataset downloader.

Usage:
  python -m webexpert.data.download_gaia --output-dir dataset/GAIA
"""

import argparse
import json
import os
from pathlib import Path


def download_gaia(output_dir: str = "dataset/GAIA", snapshot: str = "gaia_benchmark/gaia") -> None:
    """Download GAIA benchmark from HuggingFace.

    Requires HF authentication for the gaia_benchmark dataset.
    Set HF_TOKEN environment variable or login via huggingface-cli.
    """
    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("pip install datasets huggingface_hub")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading GAIA to {output_dir}...")

    try:
        snapshot_path = snapshot_download(
            repo_id=snapshot,
            repo_type="dataset",
            local_dir=output_dir,
        )
        print(f"GAIA downloaded to {snapshot_path}")
    except Exception as e:
        print(f"Snapshot download failed ({e}), trying load_dataset...")
        ds = load_dataset(snapshot, trust_remote_code=True)
        for split in ds:
            split_path = os.path.join(output_dir, f"{split}.jsonl")
            with open(split_path, "w", encoding="utf-8") as f:
                for item in ds[split]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Saved {len(ds[split])} items to {split_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="dataset/GAIA")
    args = parser.parse_args()
    download_gaia(args.output_dir)
