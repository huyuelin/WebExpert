# 专家经验提取流水线配置

"""
通用专家经验提取流水线配置文件
Expert Experience Extraction Pipeline Configuration
"""

import os
from pathlib import Path

# 基础配置
BASE_DIR = Path("/Users/linwen/Desktop/agent_AC/mybank_webthinker")
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "expert_outputs"

# 模型配置
EMBEDDING_MODELS = {
    'multilingual': 'sentence-transformers/all-MiniLM-L6-v2',
    'chinese': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'large': 'sentence-transformers/all-mpnet-base-v2',
    # 如果有本地模型
    'qwen_local': "/workspace2/chiwu/models/Qwen3-Embedding-8B"
}

# 数据集特定配置
DATASET_CONFIGS = {
    'GAIA': {
        'description': 'General AI Assistant benchmark',
        'text_fields': ['Question', 'Final answer', 'Annotator Metadata'],
        'label_field': 'Level',
        'categories': ['Level 1', 'Level 2', 'Level 3'],
        'expected_files': ['*.jsonl', '*.json'],
        'min_text_length': 20,
        'clustering': {
            'min_cluster_size': 10,
            'umap_n_neighbors': 10,
            'umap_n_components': 8
        }
    },
    
    'HLE': {
        'description': 'Human-Like Evaluation dataset', 
        'text_fields': ['question', 'answer', 'explanation', 'reasoning'],
        'label_field': 'difficulty',
        'categories': ['Easy', 'Medium', 'Hard'],
        'expected_files': ['*.jsonl', '*.json'],
        'min_text_length': 15,
        'clustering': {
            'min_cluster_size': 8,
            'umap_n_neighbors': 12,
            'umap_n_components': 10
        }
    },
    
    'PQA': {
        'description': 'Progressive Question Answering dataset',
        'text_fields': ['question', 'answer', 'reasoning_steps', 'description'],
        'label_field': 'task_type', 
        'categories': [
            'Closure Filling', 'Continuity Connection', 
            'Proximity Identification', 'Reflection Symmetry',
            'Rotation Symmetry', 'Shape Reconstruction',
            'Shape&Pattern Similarity'
        ],
        'expected_files': ['*.json'],
        'min_text_length': 10,
        'clustering': {
            'min_cluster_size': 15,
            'umap_n_neighbors': 15,
            'umap_n_components': 12
        }
    },
    
    'WebWalkerQA': {
        'description': 'Web navigation and question answering dataset',
        'text_fields': ['question', 'answer', 'trajectory', 'reasoning', 'steps'],
        'label_field': 'domain',
        'categories': ['General', 'Academic', 'Professional', 'Shopping', 'Navigation'],
        'expected_files': ['*.jsonl'],
        'min_text_length': 25,
        'clustering': {
            'min_cluster_size': 20,
            'umap_n_neighbors': 20,
            'umap_n_components': 15
        }
    }
}

# 主题生成配置
TOPIC_GENERATION_CONFIG = {
    'reference_topics': [
        "问题解决策略", "推理方法", "知识应用", "技能要求",
        "错误模式", "成功模式", "领域特征", "复杂度分析",
        "评估标准", "改进建议", "最佳实践", "注意事项"
    ],
    
    'expert_prompt_template': """
    以下是关于「{topic}」的多条经验内容，请输出该领域的专家经验：
    
    ## 注意
    - 将表述内容高度重复、参数范围或数值不同但思路一致的内容合并为一条更通用的经验
    - 对于仅仅因为判断边界差异而表述多条的情况，用"区间"或"范围"表达一条综合性规则
    - 请只保留表达思路或判定角度有实质差异的条目，其余归纳在一起
    - 合并同一情况下的处置方式
    - 每个类目下最多10条经验
    - 用尽量简洁的语言表述
    
    ## 内容
    {content}
    
    ## 输出格式（仅输出json格式，不要输出其他信息）
    {{"{topic}":["经验1","经验2"...]}}
    """
}

# 文本处理配置
TEXT_PROCESSING_CONFIG = {
    'sentence_split_patterns': [
        r'[.!?;。！？；]\s+',
        r'\n\n+',
        r'[\r\n]+(?=\d+\.)',  # 编号列表
    ],
    
    'noise_patterns': [
        r'^[0-9]+\.$',        # 纯数字
        r'^[a-zA-Z\s]+$',     # 纯英文
        r'^[^\w\u4e00-\u9fff]+$',  # 纯符号（排除中文）
        r'^.{1,5}$',          # 太短的文本
    ],
    
    'min_sentence_length': 10,
    'max_sentence_length': 1000,
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'tsne': {
        'n_components': 2,
        'perplexity': 50,
        'n_iter': 1000,
        'random_state': 42
    },
    
    'plot_style': {
        'figure_size': (12, 8),
        'dpi': 120,
        'style': 'seaborn-v0_8-whitegrid'
    }
}

# 输出配置
OUTPUT_CONFIG = {
    'save_embeddings': True,
    'save_clusters': True,
    'save_visualizations': True,
    'save_intermediate_results': True,
    
    'file_formats': {
        'experiences': 'jsonl',
        'sentences': 'jsonl', 
        'clusters': 'xlsx',
        'embeddings': 'npy'
    }
}

# 并行处理配置
PROCESSING_CONFIG = {
    'batch_size': 32,
    'max_workers': 4,
    'chunk_size': 1000,
    'use_gpu': True if 'CUDA_VISIBLE_DEVICES' in os.environ else False
}

# LLM配置（如果使用外部LLM服务）
LLM_CONFIG = {
    'provider': 'deepseek',  # 或 'openai', 'qwen', 'local'
    'model_name': 'deepseek-r1-npu',
    'api_key': None,  # 从环境变量读取
    'url': "http://industrial.models.antcloud.mybank-inc.cn/v1/",
    'temperature': 0,
    'max_concurrent': 1,
    'retry_attempts': 3,
    'timeout': 60
}