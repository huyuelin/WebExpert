#!/usr/bin/env python3
"""
数据集诊断和修复工具
Dataset Diagnostic and Fix Tool
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any

def diagnose_datasets():
    """诊断所有数据集的状态"""
    
    dataset_dir = Path("/Users/linwen/Desktop/agent_AC/mybank_webthinker/dataset")
    results = {}
    
    print("🔍 数据集诊断报告")
    print("="*50)
    
    # 检查GAIA
    gaia_path = dataset_dir / "GAIA"
    gaia_status = check_gaia_dataset(gaia_path)
    results["GAIA"] = gaia_status
    
    # 检查HLE  
    hle_path = dataset_dir / "HLE"
    hle_status = check_hle_dataset(hle_path)
    results["HLE"] = hle_status
    
    # 检查PQA
    pqa_path = dataset_dir / "PQA"
    pqa_status = check_pqa_dataset(pqa_path)
    results["PQA"] = pqa_status
    
    # 检查WebWalkerQA
    webwalker_path = dataset_dir / "WebWalkerQA"
    webwalker_status = check_webwalker_dataset(webwalker_path)
    results["WebWalkerQA"] = webwalker_status
    
    # 打印汇总
    print("\n📊 汇总:")
    for dataset, status in results.items():
        status_icon = "✅" if status["usable"] else "❌"
        print(f"{status_icon} {dataset}: {status['summary']}")
    
    return results

def check_gaia_dataset(path: Path) -> Dict:
    """检查GAIA数据集"""
    print(f"\n📂 检查 GAIA 数据集: {path}")
    
    if not path.exists():
        return {"usable": False, "summary": "目录不存在"}
    
    # 检查是否有实际数据文件
    json_files = list(path.glob("**/*.json"))
    jsonl_files = list(path.glob("**/*.jsonl"))
    
    if not json_files and not jsonl_files:
        print("   ❌ 没有找到数据文件，只有下载说明")
        return {
            "usable": False, 
            "summary": "需要从HuggingFace下载",
            "fix": "运行GAIA下载脚本"
        }
    
    # 检查数据文件内容
    data_count = 0
    for file_path in json_files + jsonl_files:
        try:
            if file_path.suffix == '.jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data_count += 1
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        data_count += len(data)
                    else:
                        data_count += 1
        except Exception as e:
            continue
    
    print(f"   📊 找到 {data_count} 个数据项")
    
    return {
        "usable": data_count > 0,
        "summary": f"包含 {data_count} 个数据项" if data_count > 0 else "数据文件为空",
        "data_count": data_count
    }

def check_hle_dataset(path: Path) -> Dict:
    """检查HLE数据集"""
    print(f"\n📂 检查 HLE 数据集: {path}")
    
    if not path.exists():
        return {"usable": False, "summary": "目录不存在"}
    
    # 查找JSON/JSONL文件
    json_files = list(path.glob("**/*.json"))
    jsonl_files = list(path.glob("**/*.jsonl"))
    
    if not json_files and not jsonl_files:
        print("   ❌ 没有找到JSON/JSONL数据文件")
        # 检查是否有其他格式的数据文件
        csv_files = list(path.glob("**/*.csv"))
        txt_files = list(path.glob("**/*.txt"))
        
        if csv_files:
            print(f"   📄 找到 {len(csv_files)} 个CSV文件")
        if txt_files:
            print(f"   📄 找到 {len(txt_files)} 个TXT文件")
            
        return {
            "usable": False,
            "summary": "没有JSON格式数据，需要下载或转换",
            "fix": "需要获取HLE数据集的JSON版本"
        }
    
    print(f"   ✅ 找到 {len(json_files + jsonl_files)} 个数据文件")
    
    return {
        "usable": True,
        "summary": f"包含 {len(json_files + jsonl_files)} 个数据文件"
    }

def check_pqa_dataset(path: Path) -> Dict:
    """检查PQA数据集"""
    print(f"\n📂 检查 PQA 数据集: {path}")
    
    if not path.exists():
        return {"usable": False, "summary": "目录不存在"}
    
    template_file = path / "docs" / "template.json"
    
    if template_file.exists():
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查数据格式
            if isinstance(data, dict) and "train" in data:
                train_data = data["train"][0] if data["train"] else {}
                if "input" in train_data and isinstance(train_data["input"], list):
                    print("   ⚠️  这是数字矩阵数据（视觉推理），不适合文本经验提取")
                    return {
                        "usable": False,
                        "summary": "数据格式不兼容（数字矩阵）",
                        "fix": "PQA是视觉推理数据集，不适合文本经验提取"
                    }
        except Exception as e:
            print(f"   ❌ 读取数据文件出错: {e}")
    
    return {
        "usable": False,
        "summary": "没有适合的文本数据",
        "fix": "需要找到PQA数据集的文本描述版本"
    }

def check_webwalker_dataset(path: Path) -> Dict:
    """检查WebWalkerQA数据集"""
    print(f"\n📂 检查 WebWalkerQA 数据集: {path}")
    
    if not path.exists():
        return {"usable": False, "summary": "目录不存在"}
    
    jsonl_files = list(path.glob("**/*.jsonl"))
    
    if not jsonl_files:
        return {"usable": False, "summary": "没有找到JSONL文件"}
    
    total_items = 0
    valid_files = []
    
    for file_path in jsonl_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                count = sum(1 for line in f if line.strip())
                if count > 0:
                    total_items += count
                    valid_files.append((file_path.name, count))
        except Exception as e:
            continue
    
    print(f"   ✅ 找到 {len(valid_files)} 个有效文件，共 {total_items} 个数据项")
    for file_name, count in valid_files:
        print(f"     - {file_name}: {count} 项")
    
    return {
        "usable": True,
        "summary": f"{len(valid_files)} 个文件，{total_items} 个数据项",
        "data_count": total_items
    }

def create_gaia_downloader():
    """创建GAIA数据集下载脚本"""
    
    download_script = '''#!/usr/bin/env python3
"""
GAIA数据集下载脚本
GAIA Dataset Downloader
"""

import os
import sys
from pathlib import Path

try:
    from datasets import load_dataset
    import json
    
    print("🔄 正在下载GAIA数据集...")
    
    # 下载数据集
    dataset = load_dataset("gaia-benchmark/GAIA")
    
    # 保存路径
    save_path = Path("/Users/linwen/Desktop/agent_AC/mybank_webthinker/dataset/GAIA/data")
    save_path.mkdir(exist_ok=True, parents=True)
    
    # 保存为JSONL格式
    for split in dataset.keys():
        output_file = save_path / f"{split}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset[split]:
                f.write(json.dumps(item, ensure_ascii=False) + "\\n")
        
        print(f"✅ 保存 {split} 数据到: {output_file} ({len(dataset[split])} 项)")
    
    print("🎉 GAIA数据集下载完成!")
    
except ImportError:
    print("❌ 请先安装datasets库: pip install datasets")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ 下载失败: {e}")
    print("请确保:")
    print("1. 已申请GAIA数据集访问权限: https://huggingface.co/datasets/gaia-benchmark/GAIA")
    print("2. 已设置HuggingFace token: huggingface-cli login")
    sys.exit(1)
'''
    
    script_path = Path("/Users/linwen/Desktop/agent_AC/mybank_webthinker/download_gaia.py")
    
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(download_script)
    
    # 设置执行权限
    os.chmod(script_path, 0o755)
    
    print(f"✅ 创建GAIA下载脚本: {script_path}")
    print("使用方法: python download_gaia.py")

def create_dataset_fixes():
    """创建数据集修复建议"""
    
    fixes = {
        "GAIA": {
            "问题": "数据集未下载",
            "解决方案": [
                "1. 访问 https://huggingface.co/datasets/gaia-benchmark/GAIA 申请访问权限",
                "2. 设置HuggingFace token: huggingface-cli login", 
                "3. 运行: python download_gaia.py"
            ]
        },
        
        "HLE": {
            "问题": "没有找到JSON格式数据文件",
            "解决方案": [
                "1. 检查HLE官方仓库获取数据: https://github.com/CAIS-HLEGAI/HLE_benchmark",
                "2. 或者创建示例数据用于测试"
            ]
        },
        
        "PQA": {
            "问题": "数据格式不兼容（数字矩阵视觉推理）",
            "解决方案": [
                "1. PQA主要用于视觉推理，不适合文本经验提取",
                "2. 可以跳过此数据集，或寻找包含文本描述的版本"
            ]
        }
    }
    
    print("\n🔧 修复建议:")
    print("="*50)
    
    for dataset, info in fixes.items():
        print(f"\n📊 {dataset}:")
        print(f"问题: {info['问题']}")
        print("解决方案:")
        for solution in info['解决方案']:
            print(f"  {solution}")

def main():
    """主函数"""
    
    # 诊断数据集
    results = diagnose_datasets()
    
    # 创建修复工具
    create_gaia_downloader()
    create_dataset_fixes()
    
    # 给出具体行动建议
    print("\n🎯 建议的行动步骤:")
    print("="*30)
    
    if not results["GAIA"]["usable"]:
        print("1. 下载GAIA数据集: python download_gaia.py")
    
    if not results["HLE"]["usable"]:
        print("2. 获取HLE数据集或创建示例数据")
    
    if results["WebWalkerQA"]["usable"]:
        print("3. WebWalkerQA可正常使用，先用它测试流水线")
    
    print("\n💡 快速测试建议:")
    print("先运行: python run_extraction.py --dataset WebWalkerQA")

if __name__ == "__main__":
    main()