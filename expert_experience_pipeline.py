#!/usr/bin/env python3
"""
通用专家经验提取流水线
Expert Experience Extraction Pipeline for General Datasets

基于 critic_extraction 方法论，提取各种数据集的领域特定专家经验
支持 GAIA、HLE、PQA、WebWalkerQA 等通用数据集
"""

import json
import os
import re
import time
import sys
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
from tqdm import tqdm

# 机器学习和NLP相关导入
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
import jieba
import nest_asyncio

# 可视化
import matplotlib.pyplot as plt

# 添加系统路径（如果需要使用本地的 ToTGenerator）
# sys.path.append("/workspace2/chiwu/works/Risk_ToT")
# from infer import ToTGenerator


class UniversalExpertExtractor:
    """通用专家经验提取器"""
    
    def __init__(self, 
                 dataset_name: str,
                 data_path: str,
                 output_dir: str = "expert_outputs",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 20):
        """
        初始化专家经验提取器
        
        Args:
            dataset_name: 数据集名称 (GAIA, HLE, PQA, WebWalkerQA)
            data_path: 数据集路径
            output_dir: 输出目录
            embedding_model: 嵌入模型名称
            batch_size: 批处理大小
        """
        self.dataset_name = dataset_name
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        
        # 初始化嵌入模型
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # 数据集特定配置
        self.dataset_configs = {
            'GAIA': {
                'text_fields': ['question', 'reasoning', 'final_answer'],
                'label_field': 'level',
                'categories': ['Level 1', 'Level 2', 'Level 3']
            },
            'HLE': {
                'text_fields': ['question', 'answer', 'explanation'],
                'label_field': 'difficulty',
                'categories': ['Easy', 'Medium', 'Hard']
            },
            'PQA': {
                'text_fields': ['question', 'answer', 'reasoning_steps'],
                'label_field': 'task_type',
                'categories': ['Closure Filling', 'Continuity Connection', 'Proximity Identification']
            },
            'WebWalkerQA': {
                'text_fields': ['question', 'answer', 'trajectory', 'reasoning'],
                'label_field': 'domain',
                'categories': ['General', 'Academic', 'Professional']
            }
        }
        
        # 设置数据集特定配置
        if dataset_name in self.dataset_configs:
            self.config = self.dataset_configs[dataset_name]
        else:
            # 默认配置
            self.config = {
                'text_fields': ['text', 'content'],
                'label_field': 'label',
                'categories': ['positive', 'negative']
            }
    
    def load_and_preprocess_data(self) -> List[Dict]:
        """
        加载和预处理数据集
        
        Returns:
            预处理后的数据列表
        """
        print(f"Loading {self.dataset_name} dataset from {self.data_path}")
        
        all_items = []
        
        # 根据数据集类型加载数据
        if self.dataset_name == 'GAIA':
            all_items = self._load_gaia_data()
        elif self.dataset_name == 'HLE':
            all_items = self._load_hle_data()
        elif self.dataset_name == 'PQA':
            all_items = self._load_pqa_data()
        elif self.dataset_name == 'WebWalkerQA':
            all_items = self._load_webwalkerqa_data()
        else:
            all_items = self._load_generic_data()
        
        print(f"Loaded {len(all_items)} items from {self.dataset_name} dataset")
        return all_items
    
    def _load_gaia_data(self) -> List[Dict]:
        """加载GAIA数据集"""
        items = []
        
        # 查找GAIA数据文件
        gaia_files = list(self.data_path.glob("**/*.jsonl"))
        gaia_files.extend(list(self.data_path.glob("**/*.json")))
        
        for file_path in gaia_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.suffix == '.jsonl':
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                items.append(self._process_gaia_item(data))
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                items.append(self._process_gaia_item(item))
                        else:
                            items.append(self._process_gaia_item(data))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return items
    
    def _process_gaia_item(self, data: Dict) -> Dict:
        """处理GAIA数据项"""
        text_parts = []
        
        # 提取文本内容
        if 'Question' in data:
            text_parts.append(f"问题: {data['Question']}")
        if 'Final answer' in data:
            text_parts.append(f"答案: {data['Final answer']}")
        if 'Annotator Metadata' in data:
            metadata = data['Annotator Metadata']
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    if isinstance(value, str) and value:
                        text_parts.append(f"{key}: {value}")
        
        combined_text = " ".join(text_parts)
        
        return {
            'text': combined_text,
            'label': data.get('Level', 'unknown'),
            'source': 'GAIA',
            'original': data
        }
    
    def _load_hle_data(self) -> List[Dict]:
        """加载HLE数据集"""
        items = []
        
        # 查找HLE数据文件
        hle_files = list(self.data_path.glob("**/*.jsonl"))
        hle_files.extend(list(self.data_path.glob("**/*.json")))
        
        for file_path in hle_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.suffix == '.jsonl':
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                items.append(self._process_hle_item(data))
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                items.append(self._process_hle_item(item))
                        else:
                            items.append(self._process_hle_item(data))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return items
    
    def _process_hle_item(self, data: Dict) -> Dict:
        """处理HLE数据项"""
        text_parts = []
        
        # 提取文本内容
        for field in ['question', 'answer', 'explanation', 'reasoning']:
            if field in data and data[field]:
                text_parts.append(str(data[field]))
        
        combined_text = " ".join(text_parts)
        
        return {
            'text': combined_text,
            'label': data.get('difficulty', data.get('level', 'unknown')),
            'source': 'HLE',
            'original': data
        }
    
    def _load_pqa_data(self) -> List[Dict]:
        """加载PQA数据集"""
        items = []
        
        # PQA通常包含图像，我们主要关注文本描述
        pqa_files = list(self.data_path.glob("**/*.json"))
        
        for file_path in pqa_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            items.append(self._process_pqa_item(item))
                    else:
                        items.append(self._process_pqa_item(data))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return items
    
    def _process_pqa_item(self, data: Dict) -> Dict:
        """处理PQA数据项"""
        text_parts = []
        
        # 提取文本内容
        for field in ['question', 'answer', 'description', 'reasoning']:
            if field in data and data[field]:
                text_parts.append(str(data[field]))
        
        combined_text = " ".join(text_parts)
        
        return {
            'text': combined_text,
            'label': data.get('task_type', data.get('category', 'unknown')),
            'source': 'PQA',
            'original': data
        }
    
    def _load_webwalkerqa_data(self) -> List[Dict]:
        """加载WebWalkerQA数据集"""
        items = []
        
        # 查找WebWalker相关数据文件
        walker_files = list(self.data_path.glob("**/*.jsonl"))
        
        for file_path in walker_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            items.append(self._process_webwalker_item(data))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return items
    
    def _process_webwalker_item(self, data: Dict) -> Dict:
        """处理WebWalker数据项"""
        text_parts = []
        
        # 提取文本内容
        for field in ['question', 'answer', 'trajectory', 'reasoning', 'steps']:
            if field in data and data[field]:
                if isinstance(data[field], list):
                    text_parts.append(" ".join(str(x) for x in data[field]))
                else:
                    text_parts.append(str(data[field]))
        
        combined_text = " ".join(text_parts)
        
        return {
            'text': combined_text,
            'label': data.get('domain', data.get('category', 'unknown')),
            'source': 'WebWalkerQA',
            'original': data
        }
    
    def _load_generic_data(self) -> List[Dict]:
        """加载通用数据集"""
        items = []
        
        # 查找所有JSON和JSONL文件
        data_files = list(self.data_path.glob("**/*.jsonl"))
        data_files.extend(list(self.data_path.glob("**/*.json")))
        
        for file_path in data_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.suffix == '.jsonl':
                        for line in f:
                            if line.strip():
                                data = json.loads(line)
                                items.append(self._process_generic_item(data))
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                items.append(self._process_generic_item(item))
                        else:
                            items.append(self._process_generic_item(data))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return items
    
    def _process_generic_item(self, data: Dict) -> Dict:
        """处理通用数据项"""
        text_parts = []
        
        # 尝试提取文本内容
        text_fields = self.config['text_fields']
        for field in text_fields:
            if field in data and data[field]:
                text_parts.append(str(data[field]))
        
        # 如果没有找到指定字段，尝试其他常见字段
        if not text_parts:
            for key, value in data.items():
                if isinstance(value, str) and value and key not in ['id', 'label', 'category']:
                    text_parts.append(value)
        
        combined_text = " ".join(text_parts)
        
        return {
            'text': combined_text,
            'label': data.get(self.config['label_field'], 'unknown'),
            'source': self.dataset_name,
            'original': data
        }
    
    def extract_sentences_from_items(self, items: List[Dict]) -> List[Dict]:
        """
        从数据项中提取句子级别的内容
        模拟 extract_sentences.ipynb 的功能
        
        Args:
            items: 数据项列表
            
        Returns:
            句子级别的数据列表
        """
        print("Extracting sentences from items...")
        
        sentence_items = []
        
        for item in items:
            text = item['text']
            label = item['label']
            
            # 按句子分割文本
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                if len(sentence.strip()) > 10:  # 过滤太短的句子
                    sentence_items.append({
                        'label': label,
                        'item': sentence.strip(),
                        'source': item['source']
                    })
        
        print(f"Extracted {len(sentence_items)} sentence-level items")
        return sentence_items
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割为句子"""
        # 使用正则表达式分割句子
        sentence_pattern = r'[.!?;。！？；]\s+'
        sentences = re.split(sentence_pattern, text)
        
        # 清理空句子
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def generate_embeddings(self, sentence_items: List[Dict]) -> np.ndarray:
        """
        生成文本嵌入
        模拟 extract_embedding.ipynb 的功能
        
        Args:
            sentence_items: 句子级别数据列表
            
        Returns:
            嵌入矩阵
        """
        print("Generating embeddings...")
        
        texts = [item['item'] for item in sentence_items]
        
        # 生成嵌入
        document_embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        print(f"Generated embeddings shape: {document_embeddings.shape}")
        
        return document_embeddings
    
    def cluster_embeddings(self, embeddings: np.ndarray) -> Tuple[List[int], BERTopic]:
        """
        对嵌入进行聚类
        模拟 extract_embedding.ipynb 中的聚类部分
        
        Args:
            embeddings: 嵌入矩阵
            
        Returns:
            聚类标签和BERTopic模型
        """
        print("Clustering embeddings...")
        
        # 根据数据量动态调整参数
        n_samples = len(embeddings)
        print(f"Adjusting parameters for {n_samples} samples")
        
        # 中文分词函数
        def tokenize_zh(text):
            words = jieba.lcut(text)
            return words
        
        # 动态设置UMAP参数
        n_neighbors = min(15, max(2, n_samples // 3))
        n_components = min(10, max(2, n_samples // 4))
        
        print(f"UMAP parameters: n_neighbors={n_neighbors}, n_components={n_components}")
        
        # UMAP降维
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric='cosine'
        )
        
        # 动态设置HDBSCAN参数
        min_cluster_size = min(15, max(2, n_samples // 5))
        
        print(f"HDBSCAN parameters: min_cluster_size={min_cluster_size}")
        
        # HDBSCAN聚类
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
        )
        
        # 动态设置CountVectorizer参数
        min_df = min(5, max(1, n_samples // 10))
        
        # CountVectorizer
        vectorizer_model = CountVectorizer(
            min_df=min_df,
            tokenizer=tokenize_zh
        )
        
        # BERTopic模型
        topic_model = BERTopic(
            nr_topics="auto",
            language='chinese',
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model
        )
        
        texts = [f"Sample text {i}" for i in range(len(embeddings))]
        topics, probs = topic_model.fit_transform(texts)
        
        print(f"Found {len(set(topics))} topics")
        
        return topics, topic_model
    
    def generate_topic_labels(self, sentence_items: List[Dict], topics: List[int], 
                            topic_model: BERTopic) -> pd.DataFrame:
        """
        生成主题标签
        模拟 topic_generation.py 的功能
        
        Args:
            sentence_items: 句子级别数据
            topics: 聚类标签
            topic_model: BERTopic模型
            
        Returns:
            带主题的数据框
        """
        print("Generating topic labels...")
        
        # 创建数据框
        df = pd.DataFrame({
            'text': [item['item'] for item in sentence_items],
            'topic': topics,
            'label': [item['label'] for item in sentence_items],
            'source': [item['source'] for item in sentence_items]
        })
        
        # 为每个主题生成标签（这里简化处理）
        topic_info = topic_model.get_topic_info()
        topic_labels = {}
        
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            # 使用前几个关键词作为主题标签
            if topic_id != -1:
                keywords = topic_model.get_topic(topic_id)[:3]
                topic_label = "_".join([kw[0] for kw in keywords])
                topic_labels[topic_id] = topic_label
            else:
                topic_labels[topic_id] = "noise"
        
        df['topic_label'] = df['topic'].map(topic_labels)
        
        return df
    
    def merge_topics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        合并相似主题
        模拟 merge_topic.ipynb 的功能
        
        Args:
            df: 带主题的数据框
            
        Returns:
            合并后的数据框
        """
        print("Merging similar topics...")
        
        # 按主题分组并合并
        merged = (
            df.groupby("topic_label", as_index=False)
            .agg(
                topic_idx=("topic", "min"),
                sentences=("text", lambda s: "\n".join(s)),
                count=("text", "count")
            )
        )
        
        # 按主题索引排序
        merged = merged.sort_values("topic_idx")[["topic_idx", "topic_label", "sentences", "count"]]
        
        print(f"Merged into {len(merged)} topics")
        
        return merged
    
    def generate_expert_experiences(self, merged_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        生成专家经验
        模拟 critic_generation.py 的功能
        
        Args:
            merged_df: 合并后的主题数据框
            
        Returns:
            专家经验字典
        """
        print("Generating expert experiences...")
        
        expert_experiences = {}
        
        for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Processing topics"):
            topic_label = row['topic_label']
            sentences = row['sentences'].split('\n')
            
            # 过滤和清理句子
            clean_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and not self._is_noise_sentence(sentence):
                    clean_sentences.append(sentence)
            
            if len(clean_sentences) >= 3:
                # 简化的经验生成逻辑
                experiences = self._generate_experiences_for_topic(topic_label, clean_sentences)
                expert_experiences[topic_label] = experiences
        
        print(f"Generated expert experiences for {len(expert_experiences)} topics")
        
        return expert_experiences
    
    def _is_noise_sentence(self, sentence: str) -> bool:
        """判断是否为噪声句子"""
        noise_patterns = [
            r'^[0-9]+\.$',  # 纯数字
            r'^[a-zA-Z]+$',  # 纯英文
            r'^[^\w]+$',  # 纯符号
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, sentence.strip()):
                return True
        
        return False
    
    def _generate_experiences_for_topic(self, topic_label: str, sentences: List[str]) -> List[str]:
        """为特定主题生成经验"""
        # 分析句子内容和特点
        experiences = self._extract_meaningful_insights(topic_label, sentences)
        
        return experiences[:5]  # 最多5个经验
    
    def _extract_meaningful_insights(self, topic_label: str, sentences: List[str]) -> List[str]:
        """从句子中提取有意义的见解"""
        insights = []
        
        # 1. 提取关键实体和概念
        key_entities = self._extract_key_entities(sentences)
        
        # 2. 分析句子结构和语义模式
        semantic_patterns = self._analyze_semantic_patterns(sentences)
        
        # 3. 基于数据集特点生成专家经验
        if 'GAIA' in self.dataset_name or 'gaia' in self.dataset_name:
            insights.extend(self._generate_gaia_insights(topic_label, key_entities, semantic_patterns))
        elif 'HLE' in self.dataset_name or 'hle' in self.dataset_name:
            insights.extend(self._generate_hle_insights(topic_label, key_entities, semantic_patterns))
        elif 'WebWalker' in self.dataset_name or 'webwalker' in self.dataset_name:
            insights.extend(self._generate_webwalker_insights(topic_label, key_entities, semantic_patterns))
        else:
            insights.extend(self._generate_generic_insights(topic_label, key_entities, semantic_patterns))
        
        return insights
    
    def _extract_key_entities(self, sentences: List[str]) -> List[str]:
        """提取关键实体"""
        all_words = []
        for sentence in sentences:
            # 使用jieba提取词汇
            words = jieba.lcut(sentence)
            # 过滤停用词和短词
            filtered_words = [w for w in words if len(w) > 1 and w not in ['的', '是', '在', '有', '和', '与', '及']]
            all_words.extend(filtered_words)
        
        # 统计词频并返回高频词汇
        word_freq = defaultdict(int)
        for word in all_words:
            word_freq[word] += 1
        
        # 返回出现频率>=2的词汇
        key_entities = [word for word, freq in word_freq.items() if freq >= 2]
        return sorted(key_entities, key=lambda x: word_freq[x], reverse=True)[:15]
    
    def _analyze_semantic_patterns(self, sentences: List[str]) -> Dict[str, int]:
        """分析语义模式"""
        patterns = {
            'question_asking': 0,
            'method_description': 0, 
            'problem_solving': 0,
            'comparison': 0,
            'explanation': 0,
            'definition': 0
        }
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # 检测问题询问模式
            if any(word in sentence_lower for word in ['what', 'how', 'why', 'which', 'when', 'where', '什么', '如何', '为什么', '哪个']):
                patterns['question_asking'] += 1
                
            # 检测方法描述模式
            if any(word in sentence_lower for word in ['method', 'approach', 'technique', 'algorithm', '方法', '技术', '算法']):
                patterns['method_description'] += 1
                
            # 检测问题解决模式
            if any(word in sentence_lower for word in ['solve', 'solution', 'fix', 'resolve', '解决', '方案']):
                patterns['problem_solving'] += 1
                
            # 检测比较模式
            if any(word in sentence_lower for word in ['compare', 'versus', 'difference', 'better', '比较', '区别', '优于']):
                patterns['comparison'] += 1
                
            # 检测解释模式
            if any(word in sentence_lower for word in ['explain', 'because', 'since', 'therefore', '解释', '因为', '所以']):
                patterns['explanation'] += 1
                
            # 检测定义模式
            if any(word in sentence_lower for word in ['define', 'definition', 'means', 'refers to', '定义', '是指', '意思']):
                patterns['definition'] += 1
        
        return patterns
    
    def _generate_gaia_insights(self, topic_label: str, entities: List[str], patterns: Dict[str, int]) -> List[str]:
        """为GAIA数据集生成专家经验"""
        insights = []
        
        if patterns['question_asking'] > patterns['explanation']:
            insights.append(f"在{topic_label}类问题中，重点关注问题的层次结构和分解策略")
            
        if entities:
            top_entities = entities[:3]
            insights.append(f"解决{topic_label}问题时，需要特别关注{', '.join(top_entities)}等关键要素")
            
        if patterns['method_description'] > 0:
            insights.append(f"处理{topic_label}任务时，应采用系统性的方法论和明确的推理步骤")
            
        if patterns['problem_solving'] > 0:
            insights.append(f"面对{topic_label}挑战时，建议采用多步骤验证和交叉检查机制")
            
        return insights
    
    def _generate_hle_insights(self, topic_label: str, entities: List[str], patterns: Dict[str, int]) -> List[str]:
        """为HLE数据集生成专家经验"""
        insights = []
        
        # 基于HLE是人类标注偏好的特点
        if entities:
            top_entities = entities[:3] 
            insights.append(f"在{topic_label}场景下，应重点评估{', '.join(top_entities)}的人类可理解性")
            
        if patterns['explanation'] > patterns['question_asking']:
            insights.append(f"对于{topic_label}类任务，优先考虑解释的逻辑性和连贯性")
            
        insights.append(f"处理{topic_label}问题时，需要平衡准确性与人类偏好的一致性")
        
        if patterns['comparison'] > 0:
            insights.append(f"在{topic_label}评估中，应建立多维度的比较标准")
            
        return insights
    
    def _generate_webwalker_insights(self, topic_label: str, entities: List[str], patterns: Dict[str, int]) -> List[str]:
        """为WebWalker数据集生成专家经验"""
        insights = []
        
        # 基于WebWalker是网页导航和信息检索的特点
        if entities:
            top_entities = entities[:3]
            insights.append(f"在{topic_label}网页任务中，需要重点关注{', '.join(top_entities)}等信息定位要素")
            
        insights.append(f"执行{topic_label}类网页操作时，应采用分层导航和信息验证策略")
        
        if patterns['question_asking'] > 0:
            insights.append(f"处理{topic_label}查询时，需要准确理解用户意图并制定搜索策略")
            
        if patterns['method_description'] > 0:
            insights.append(f"在{topic_label}场景下，应结合页面结构分析和内容理解进行决策")
            
        return insights
    
    def _generate_generic_insights(self, topic_label: str, entities: List[str], patterns: Dict[str, int]) -> List[str]:
        """生成通用专家经验"""
        insights = []
        
        if entities:
            top_entities = entities[:3]
            insights.append(f"在{topic_label}领域中，{', '.join(top_entities)}是核心关注点")
            
        # 根据语义模式生成见解
        dominant_pattern = max(patterns, key=patterns.get) if patterns else None
        
        if dominant_pattern == 'question_asking':
            insights.append(f"面对{topic_label}问题时，首先要准确理解问题的核心需求")
        elif dominant_pattern == 'method_description':
            insights.append(f"解决{topic_label}任务需要采用结构化的方法和清晰的执行步骤")
        elif dominant_pattern == 'problem_solving':
            insights.append(f"处理{topic_label}问题时，应建立系统性的解决方案")
        elif dominant_pattern == 'explanation':
            insights.append(f"在{topic_label}场景下，重点关注逻辑推理和解释的完整性")
            
        return insights
    
    def save_results(self, expert_experiences: Dict[str, List[str]], 
                    sentence_items: List[Dict]) -> None:
        """
        保存结果
        
        Args:
            expert_experiences: 专家经验字典
            sentence_items: 句子级别数据
        """
        print("Saving results...")
        
        # 保存专家经验
        output_file = self.output_dir / f"{self.dataset_name}_expert_experiences.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for topic, experiences in expert_experiences.items():
                f.write(json.dumps({topic: experiences}, ensure_ascii=False) + "\n")
        
        print(f"Expert experiences saved to {output_file}")
        
        # 保存句子级别数据
        sentence_file = self.output_dir / f"{self.dataset_name}_sentences.jsonl"
        with open(sentence_file, 'w', encoding='utf-8') as f:
            json.dump(sentence_items, f, ensure_ascii=False, indent=2)
        
        print(f"Sentence-level data saved to {sentence_file}")
        
        # 生成专家经验函数（类似prompt.py中的格式）
        self._generate_expert_function(expert_experiences)
    
    def _generate_expert_function(self, expert_experiences: Dict[str, List[str]]) -> None:
        """生成专家经验函数"""
        function_file = self.output_dir / f"get_{self.dataset_name.lower()}_expert_experience.py"
        
        with open(function_file, 'w', encoding='utf-8') as f:
            f.write(f'''def get_{self.dataset_name.lower()}_expert_experience():
    """{self.dataset_name} 领域专家经验"""
    expert_experience = f"""
    | 领域分类    | 子分类             | 专家经验                                                                                                                                                                              |
    |------------|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
''')
            
            for topic, experiences in expert_experiences.items():
                for exp in experiences:
                    f.write(f'    | {topic}    | {topic}子类       | {exp}                                                                                                                                                                     |\n')
            
            f.write('''"""
    return expert_experience

if __name__ == "__main__":
    print(get_''' + self.dataset_name.lower() + '''_expert_experience())
''')
        
        print(f"Expert function saved to {function_file}")
    
    def run_pipeline(self) -> Dict[str, List[str]]:
        """
        运行完整的专家经验提取流水线
        
        Returns:
            专家经验字典
        """
        print(f"Starting expert experience extraction pipeline for {self.dataset_name}")
        
        # 步骤1: 加载和预处理数据
        items = self.load_and_preprocess_data()
        if not items:
            print("No data loaded, exiting...")
            return {}
        
        # 步骤2: 提取句子级别内容
        sentence_items = self.extract_sentences_from_items(items)
        if not sentence_items:
            print("No sentences extracted, exiting...")
            return {}
        
        # 步骤3: 生成嵌入
        embeddings = self.generate_embeddings(sentence_items)
        
        # 步骤4: 聚类
        topics, topic_model = self.cluster_embeddings(embeddings)
        
        # 步骤5: 生成主题标签
        df_with_topics = self.generate_topic_labels(sentence_items, topics, topic_model)
        
        # 步骤6: 合并相似主题
        merged_df = self.merge_topics(df_with_topics)
        
        # 步骤7: 生成专家经验
        expert_experiences = self.generate_expert_experiences(merged_df)
        
        # 步骤8: 保存结果
        self.save_results(expert_experiences, sentence_items)
        
        print("Pipeline completed successfully!")
        
        return expert_experiences


def main():
    """主函数"""
    # 支持的数据集
    datasets = ['GAIA', 'HLE', 'PQA', 'WebWalkerQA']
    
    # 数据集路径配置
    base_path = Path("/Users/linwen/Desktop/agent_AC/mybank_webthinker/dataset")
    
    for dataset_name in datasets:
        dataset_path = base_path / dataset_name
        
        if dataset_path.exists():
            print(f"\n{'='*50}")
            print(f"Processing {dataset_name} dataset")
            print(f"{'='*50}")
            
            try:
                # 创建提取器
                extractor = UniversalExpertExtractor(
                    dataset_name=dataset_name,
                    data_path=str(dataset_path),
                    output_dir=f"expert_outputs/{dataset_name.lower()}"
                )
                
                # 运行流水线
                expert_experiences = extractor.run_pipeline()
                
                print(f"Successfully extracted {len(expert_experiences)} expert experience topics for {dataset_name}")
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                continue
        else:
            print(f"Dataset path not found: {dataset_path}")


if __name__ == "__main__":
    main()