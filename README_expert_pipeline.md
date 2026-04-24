# 通用专家经验提取流水线

## 概述

本流水线基于 `critic_extraction` 方法论，可以从通用数据集中提取领域特定的专家经验。支持 GAIA、HLE、PQA、WebWalkerQA 等多种数据集格式。

## 核心方法论

流水线基于以下步骤：

1. **数据预处理**: 从原始数据中提取和清理文本内容
2. **句子级别提取**: 将文本分解为句子级别的内容 
3. **嵌入生成**: 使用预训练模型生成文本嵌入
4. **聚类分析**: 使用 BERTopic + HDBSCAN 进行主题聚类
5. **主题标注**: 为每个聚类生成有意义的主题标签
6. **主题合并**: 合并相似的主题避免冗余
7. **经验生成**: 从聚类内容中提取专家经验规则
8. **结果保存**: 生成结构化的专家经验输出

## 文件结构

```
mybank_webthinker/
├── expert_experience_pipeline.py    # 核心流水线代码
├── pipeline_config.py              # 配置文件
├── run_extraction.py               # 使用示例和命令行工具
├── dataset/                        # 数据集目录
│   ├── GAIA/
│   ├── HLE/
│   ├── PQA/
│   └── WebWalkerQA/
└── expert_outputs/                 # 输出结果目录
    ├── gaia/
    ├── hle/
    ├── pqa/
    └── webwalkerqa/
```

## 安装依赖

```bash
pip install -r requirements.txt
```

requirements.txt:
```
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
bertopic>=0.15.0
hdbscan>=0.8.29
umap-learn>=0.5.3
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
jieba>=0.42.1
nest-asyncio>=1.5.6
openpyxl>=3.1.0
```

## 快速开始

### 1. 分析数据集结构

```bash
python run_extraction.py --analyze
```

这会显示各个数据集的基本信息和文件结构。

### 2. 提取单个数据集

```bash
# 提取GAIA数据集的专家经验
python run_extraction.py --dataset GAIA

# 提取HLE数据集的专家经验  
python run_extraction.py --dataset HLE
```

### 3. 批量提取所有数据集

```bash
python run_extraction.py --all
```

### 4. 使用自定义配置

```bash
python run_extraction.py --custom
```

## 编程接口使用

```python
from expert_experience_pipeline import UniversalExpertExtractor

# 创建提取器
extractor = UniversalExpertExtractor(
    dataset_name="GAIA",
    data_path="/path/to/GAIA/dataset",
    output_dir="my_outputs",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# 运行流水线
expert_experiences = extractor.run_pipeline()

# 查看结果
for topic, experiences in expert_experiences.items():
    print(f"主题: {topic}")
    for exp in experiences:
        print(f"  - {exp}")
```

## 支持的数据集

### 1. GAIA (General AI Assistant)
- **描述**: 通用AI助手评测数据集
- **文件格式**: JSON/JSONL
- **主要字段**: Question, Final answer, Level, Annotator Metadata

### 2. HLE (Human-Like Evaluation)
- **描述**: 类人评估数据集
- **文件格式**: JSON/JSONL  
- **主要字段**: question, answer, explanation, difficulty

### 3. PQA (Progressive Question Answering)
- **描述**: 渐进式问答数据集，主要用于视觉推理
- **文件格式**: JSON
- **主要字段**: question, answer, task_type, reasoning_steps

### 4. WebWalkerQA
- **描述**: 网页导航和问答数据集
- **文件格式**: JSONL
- **主要字段**: question, answer, trajectory, domain

## 输出格式

### 1. 专家经验文件 (`{dataset}_expert_experiences.jsonl`)

```json
{"问题解决策略": ["在复杂问题场景下，需要分步骤进行分析", "遇到多步推理时应建立清晰的逻辑链条"]}
{"推理方法": ["使用归纳推理时要确保样本的代表性", "演绎推理需要验证前提的准确性"]}
```

### 2. 专家经验函数 (`get_{dataset}_expert_experience.py`)

```python
def get_gaia_expert_experience():
    """GAIA 领域专家经验"""
    expert_experience = f"""
    | 领域分类    | 子分类             | 专家经验                                                          |
    |------------|-------------------|----------------------------------------------------------------|
    | 问题解决    | 分析方法          | 在复杂问题场景下，需要分步骤进行分析                                   |
    | 推理策略    | 逻辑链条          | 遇到多步推理时应建立清晰的逻辑链条                                   |
    """
    return expert_experience
```

### 3. 句子级别数据 (`{dataset}_sentences.jsonl`)

保存所有提取的句子级别内容，用于后续分析。

## 配置说明

### 数据集配置 (`pipeline_config.py`)

可以为每个数据集指定：
- `text_fields`: 要提取的文本字段
- `label_field`: 标签字段  
- `categories`: 预期类别
- `min_text_length`: 最小文本长度
- `clustering`: 聚类参数

### 聚类配置

- `min_cluster_size`: 最小聚类大小
- `umap_n_neighbors`: UMAP邻居数
- `umap_n_components`: UMAP降维维度

### 模型配置

- `embedding_model`: 嵌入模型名称
- `batch_size`: 批处理大小

## 自定义数据集

要添加新的数据集支持，需要：

1. 在 `pipeline_config.py` 中添加数据集配置
2. 在 `UniversalExpertExtractor` 中实现对应的加载函数
3. 实现数据项处理函数

例如：

```python
def _load_my_dataset(self) -> List[Dict]:
    """加载自定义数据集"""
    # 实现数据加载逻辑
    pass

def _process_my_item(self, data: Dict) -> Dict:
    """处理自定义数据项"""
    # 实现数据处理逻辑
    pass
```

## 输出目录结构

```
expert_outputs/
├── gaia/
│   ├── GAIA_expert_experiences.jsonl
│   ├── GAIA_sentences.jsonl
│   └── get_gaia_expert_experience.py
├── hle/
│   ├── HLE_expert_experiences.jsonl
│   ├── HLE_sentences.jsonl  
│   └── get_hle_expert_experience.py
└── all_datasets_summary.json
```

## 性能优化

1. **GPU加速**: 设置 `use_gpu=True` 启用GPU加速嵌入生成
2. **批处理**: 调整 `batch_size` 平衡内存使用和速度
3. **并行处理**: 设置 `max_workers` 启用多进程处理
4. **内存优化**: 对于大数据集，可以分批处理避免内存溢出

## 注意事项

1. **数据质量**: 确保输入数据格式正确且内容完整
2. **文本长度**: 过短的文本可能无法提取有效经验
3. **聚类参数**: 根据数据集特点调整聚类参数
4. **嵌入模型**: 选择适合数据集语言和领域的嵌入模型

## 故障排除

### 常见问题

1. **数据加载失败**
   - 检查数据集路径和文件格式
   - 确认文件编码为UTF-8

2. **聚类结果不理想**
   - 调整 `min_cluster_size` 参数
   - 尝试不同的嵌入模型

3. **内存不足**
   - 减小 `batch_size`
   - 分批处理大数据集

4. **提取经验质量差**
   - 检查文本预处理质量
   - 调整噪声过滤规则

## 扩展功能

1. **集成LLM**: 使用大语言模型改进经验生成质量
2. **可视化**: 添加聚类结果的可视化展示
3. **评估指标**: 加入专家经验质量评估机制
4. **增量更新**: 支持增量添加新数据

## 参考

本流水线基于以下工作：
- BERTopic: https://github.com/MaartenGr/BERTopic
- Sentence Transformers: https://www.sbert.net/
- HDBSCAN: https://hdbscan.readthedocs.io/

## 联系

如有问题或建议，请提交 Issue 或联系开发团队。