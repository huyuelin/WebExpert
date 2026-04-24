#!/usr/bin/env python3
"""
数据集专家经验演示脚本 - 展示新专家经验系统的完整工作流程
"""

import json
import os
from datetime import datetime

# 导入专家经验函数
import sys
sys.path.append('scripts')

from prompt.prompt import (
    get_expert_experience,
    get_dataset_expert_experience,
)

def demonstrate_expert_experience():
    """演示数据集专家经验的应用效果"""
    print("=== 数据集专家经验系统演示 ===")
    
    # 1. 展示原始信贷专家经验 vs 新的数据集专家经验
    print("\n1. 专家经验对比:")
    print("\n--- 原始信贷专家经验 (用于信贷审批) ---")
    original_experience = get_expert_experience()
    print(original_experience[:500] + "...")
    
    print("\n--- 新的数据集专家经验 (用于四个评测数据集) ---")
    dataset_experience = get_dataset_expert_experience()
    print(dataset_experience[:500] + "...")
    
    # 2. 针对不同数据集的问题示例，展示专家经验如何指导搜索策略
    test_cases = [
        {
            "dataset": "GAIA",
            "question": "什么是机器学习中的过拟合现象？如何防止过拟合？",
            "expected_domains": ["知识检索-基础概念查询", "知识检索-技术实现详解"],
            "expected_queries": ["机器学习 过拟合 定义", "防止过拟合 方法 技术"]
        },
        {
            "dataset": "WebWalkerQA", 
            "question": "Who is the granddaughter of Mufti Mohammad Sayeed?",
            "expected_domains": ["事实核查-具体事实验证", "复杂查询-关联信息挖掘"],
            "expected_queries": ["Mufti Mohammad Sayeed family granddaughter", "Mohammad Sayeed political family tree"]
        },
        {
            "dataset": "HLE",
            "question": "设计一个端到端的推荐系统架构",
            "expected_domains": ["专业领域-技术开发实践", "复杂查询-多步骤分解"],
            "expected_queries": ["推荐系统架构设计", "端到端推荐系统实现"]
        },
        {
            "dataset": "PQA",
            "question": "What is perceptual reasoning in visual tasks?",
            "expected_domains": ["知识检索-基础概念查询", "专业领域-学术理论研究"],
            "expected_queries": ["perceptual reasoning definition", "visual tasks cognitive psychology"]
        }
    ]
    
    print("\n2. 基于新专家经验的搜索策略生成示例:")
    
    results = []
    
    for case in test_cases:
        print(f"\n--- {case['dataset']} 数据集问题 ---")
        print(f"问题: {case['question']}")
        print(f"预期领域: {', '.join(case['expected_domains'])}")
        print(f"预期查询: {', '.join(case['expected_queries'])}")
        
        # 根据专家经验分析问题特征
        analysis = analyze_question_with_expert_experience(case['question'], case['dataset'])
        print(f"专家分析: {analysis}")
        
        # 保存结果
        result = {
            "dataset": case['dataset'],
            "question": case['question'],
            "expected_domains": case['expected_domains'],
            "expected_queries": case['expected_queries'],
            "expert_analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
    
    # 3. 展示四个数据集的差异化搜索策略
    print("\n3. 四个数据集的差异化搜索策略:")
    
    dataset_strategies = {
        "GAIA": {
            "主要领域": ["知识检索", "事实核查", "推理分析"],
            "搜索特点": "注重概念准确性和多源验证",
            "关键策略": "使用精确学术术语，优先权威来源"
        },
        "WebWalkerQA": {
            "主要领域": ["事实核查", "复杂查询", "网页搜索"],
            "搜索特点": "需要多步骤推理和关联挖掘",
            "关键策略": "分解复杂问题，挖掘隐含关系"
        },
        "HLE": {
            "主要领域": ["专业领域", "技术开发", "推理分析"],
            "搜索特点": "技术性强，需要实践结合理论",
            "关键策略": "技术文档优先，注重实现细节"
        },
        "PQA": {
            "主要领域": ["知识检索", "专业领域", "学术理论"],
            "搜索特点": "认知科学和视觉推理理论",
            "关键策略": "学术文献为主，理论框架清晰"
        }
    }
    
    for dataset, strategy in dataset_strategies.items():
        print(f"\n{dataset}:")
        for key, value in strategy.items():
            if isinstance(value, list):
                print(f"  {key}: {', '.join(value)}")
            else:
                print(f"  {key}: {value}")
    
    # 4. 保存演示结果
    os.makedirs("dataset_experiments", exist_ok=True)
    demo_file = f"dataset_experiments/expert_experience_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    demo_data = {
        "demonstration_info": {
            "original_expert_experience_preview": original_experience[:500],
            "dataset_expert_experience_preview": dataset_experience[:500],
            "timestamp": datetime.now().isoformat()
        },
        "test_cases": results,
        "dataset_strategies": dataset_strategies
    }
    
    with open(demo_file, 'w', encoding='utf-8') as f:
        json.dump(demo_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n演示结果已保存到: {demo_file}")
    
    return demo_data

def analyze_question_with_expert_experience(question: str, dataset_name: str) -> str:
    """基于数据集专家经验分析问题特征"""
    
    # 获取专家经验表
    experience_table = get_dataset_expert_experience()
    
    # 简单的问题特征分析（实际应用中会使用LLM进行更复杂的分析）
    question_lower = question.lower()
    
    analysis_points = []
    
    # 分析问题类型
    if any(word in question_lower for word in ["what is", "define", "definition", "什么是"]):
        analysis_points.append("识别为基础概念查询类问题，应使用精确学术术语搜索")
        
    if any(word in question_lower for word in ["how to", "implement", "design", "如何", "实现", "设计"]):
        analysis_points.append("识别为技术实现类问题，需要理论与实践结合")
        
    if any(word in question_lower for word in ["who is", "when", "where", "which", "谁是", "何时", "哪里"]):
        analysis_points.append("识别为事实核查类问题，需要精确验证和多源确认")
        
    if len(question.split()) > 15 or "," in question:
        analysis_points.append("识别为复杂推理问题，需要分步骤分解搜索")
    
    # 根据数据集特点补充分析
    if dataset_name == "GAIA":
        analysis_points.append("GAIA数据集特点：注重知识准确性，建议使用学术来源")
    elif dataset_name == "WebWalkerQA":
        analysis_points.append("WebWalkerQA数据集特点：需要网页搜索和信息整合")
    elif dataset_name == "HLE":
        analysis_points.append("HLE数据集特点：技术性问题，需要实践指导")
    elif dataset_name == "PQA":
        analysis_points.append("PQA数据集特点：视觉认知问题，需要学术理论支撑")
    
    return " | ".join(analysis_points) if analysis_points else "需要进一步分析问题特征"

def compare_search_strategies():
    """对比原始专家经验与数据集专家经验的搜索策略差异"""
    print("\n=== 搜索策略对比分析 ===")
    
    # 示例问题
    example_question = "什么是深度学习中的注意力机制？"
    
    print(f"示例问题: {example_question}")
    
    print("\n--- 原始信贷专家经验指导下的搜索策略 ---")
    print("会倾向于搜索:")
    print("- 注意力机制在金融风控中的应用")
    print("- 深度学习模型在信贷审批中的使用")
    print("- 相关技术的监管政策和合规要求")
    
    print("\n--- 新数据集专家经验指导下的搜索策略 ---")
    print("会倾向于搜索:")
    print("- 注意力机制的学术定义和理论基础")
    print("- 深度学习中注意力机制的技术实现")
    print("- 相关的权威学术文献和技术文档")
    print("- 注意力机制在不同应用场景中的表现")
    
    print("\n关键差异:")
    print("1. 信贷专家经验：面向业务应用，关注风险和合规")
    print("2. 数据集专家经验：面向知识准确性，关注学术权威性")
    print("3. 搜索范围：从垂直领域扩展到通用知识领域")
    print("4. 信息源选择：从行业报告转向学术文献")

if __name__ == "__main__":
    try:
        # 运行演示
        demo_results = demonstrate_expert_experience()
        
        # 运行对比分析
        compare_search_strategies()
        
        print("\n=== 总结 ===")
        print("✅ 成功创建了基于四个数据集分析的新专家经验")
        print("✅ 新专家经验涵盖了知识检索、事实核查、推理分析等关键领域")
        print("✅ 为不同类型的问题提供了差异化的搜索策略指导")
        print("✅ 相比原始信贷专家经验，更适用于通用知识问答任务")
        
        print(f"\n实验框架已就绪，可以接入真实的LLM和搜索引擎进行完整测试！")
        
    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()