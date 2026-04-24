#!/usr/bin/env python3
"""
GAIA 数据集下载（snapshot_download 版）
- 规避 datasets 老式脚本限制（错误: "Dataset scripts are no longer supported, but found GAIA.py"）
- 直接快照克隆 HF 仓库后，遍历 JSON/JSONL，将符合 GAIA 格式的样本抽取并标准化保存

输出：dataset/GAIA/official/
  - gaia_all.jsonl         # 汇总
  - gaia_level_1.jsonl     # 可选
  - gaia_level_2.jsonl
  - gaia_level_3.jsonl
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Iterable, Optional
from huggingface_hub import login, snapshot_download


REPO_ID = "gaia-benchmark/GAIA"


def _iter_jsonl(p: Path) -> Iterable[Dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _iter_json(p: Path) -> Iterable[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for it in data:
                if isinstance(it, dict):
                    yield it
        elif isinstance(data, dict):
            yield data
    except Exception:
        return


def _normalize_gaia_item(it: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # 兼容大小写与不同键名
    q = it.get("Question") or it.get("question")
    ans = it.get("Final answer") or it.get("final_answer") or it.get("Final Answer")
    lvl = it.get("Level") or it.get("level")
    if not (q and ans and lvl):
        return None
    lvl_s = str(lvl).strip()
    if lvl_s and lvl_s[0].isdigit():
        lvl_s = f"Level {lvl_s}"
    elif lvl_s.lower().startswith("level"):
        # 规范化空格
        parts = lvl_s.split()
        if len(parts) == 2 and parts[0].lower() == "level" and parts[1].isdigit():
            lvl_s = f"Level {parts[1]}"
    return {
        "Question": str(q),
        "Final answer": str(ans),
        "Level": lvl_s,
        "Annotator Metadata": it.get("Annotator Metadata") or it.get("annotator_metadata") or {},
    }


def download_gaia_dataset() -> bool:
    print("🚀 开始下载GAIA数据集(快照)...")

    # 登录（若已通过 huggingface-cli 登录，也可跳过）
    token = os.getenv("HF_TOKEN")
    if token:
        try:
            login(token=token)
            print("✅ HuggingFace认证成功")
        except Exception as e:
            print(f"⚠️  认证失败但继续尝试快照下载: {e}")
    else:
        print("ℹ️ 未检测到 HF_TOKEN，将尝试在已有认证上下文中下载。建议 export HF_TOKEN=... 以提高成功率。")

    try:
        repo_dir = snapshot_download(repo_id=REPO_ID, repo_type="dataset")
        print(f"✅ 快照下载完成: {repo_dir}")
    except Exception as e:
        print(f"❌ 快照下载失败: {e}")
        print("请确保已申请访问并设置 HF_TOKEN: https://huggingface.co/datasets/gaia-benchmark/GAIA")
        return False

    repo_path = Path(repo_dir)
    candidates = list(repo_path.rglob("*.json")) + list(repo_path.rglob("*.jsonl"))
    if not candidates:
        print("❌ 在快照中未找到 JSON/JSONL 文件")
        return False

    output_dir = Path("dataset/GAIA/official")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_file = output_dir / "gaia_all.jsonl"
    lvl_files = {
        "Level 1": output_dir / "gaia_level_1.jsonl",
        "Level 2": output_dir / "gaia_level_2.jsonl",
        "Level 3": output_dir / "gaia_level_3.jsonl",
    }

    total, kept = 0, 0
    with all_file.open("w", encoding="utf-8") as fa:
        writers = {}
        for k, p in lvl_files.items():
            writers[k] = p.open("w", encoding="utf-8")
        try:
            for fp in candidates:
                iterator = _iter_jsonl(fp) if fp.suffix == ".jsonl" else _iter_json(fp)
                for raw in iterator:
                    total += 1
                    norm = _normalize_gaia_item(raw)
                    if not norm:
                        continue
                    kept += 1
                    fa.write(json.dumps(norm, ensure_ascii=False) + "\n")
                    w = writers.get(norm.get("Level"))
                    if w:
                        w.write(json.dumps(norm, ensure_ascii=False) + "\n")
        finally:
            for w in writers.values():
                try:
                    w.close()
                except Exception:
                    pass

    print(f"📊 共扫描 {total} 条，保留 {kept} 条 GAIA 样本 -> {all_file}")
    for k, p in lvl_files.items():
        if p.exists():
            # 统计条数
            try:
                cnt = sum(1 for _ in _iter_jsonl(p))
            except Exception:
                cnt = 0
            print(f"  - {k}: {cnt} 条 -> {p}")

    return kept > 0


if __name__ == "__main__":
    ok = download_gaia_dataset()
    if not ok:
        raise SystemExit(1)