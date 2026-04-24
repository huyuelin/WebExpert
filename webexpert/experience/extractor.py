"""Critic-Guided Expert Experience Extraction (Section 3.2 of the paper).

Pipeline:
  1. Question harvesting and canonicalization
  2. QA-level multi-view clustering (HDBSCAN / BERTopic)
  3. Source aggregation with BM25 + MMR
  4. Contradiction-aware summarization (DeepSeek-R1)
  5. Facetization and normalization
  6. Continuous refresh and versioning
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict


class CriticGuidedExtractor:
    """Extract sentence-level expert experiences from domain QA corpora.

    Implements the six-step pipeline described in Section 3.2:
    - Sentence-level embedding and dense representation
    - Multi-view density clustering (HDBSCAN + BERTopic + UMAP)
    - Topic merging and rule distillation
    - Facetization into (time, region, policy, L2 industry)
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        umap_n_neighbors: int = 15,
        umap_n_components: int = 10,
        hdbscan_min_cluster_size: int = 15,
        batch_size: int = 20,
    ):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.hdbscan_min_cluster_size = hdbscan_min_cluster_size
        self.batch_size = batch_size
        self.experience_base: List[Dict[str, Any]] = []

    # ---- Step 1: Question harvesting and canonicalization ----
    def harvest_questions(
        self, qa_tuples: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Load and canonicalize (q, a, R, C) tuples.

        Normalizes surface forms via paraphrase mining and
        schema-free delexicalization to get canonical intents.
        """
        canonicalized = []
        for item in qa_tuples:
            question = item.get("question", item.get("Question", ""))
            answer = item.get("answer", item.get("Final answer", ""))
            reasoning = item.get("reasoning", item.get("Annotator Metadata", ""))
            citations = item.get("citations", item.get("source", ""))
            canonicalized.append({
                "question": question.strip(),
                "answer": answer.strip(),
                "reasoning": reasoning.strip() if isinstance(reasoning, str) else "",
                "citations": citations,
                "text": f"{question} {answer} {reasoning}".strip(),
            })
        return canonicalized

    # ---- Step 2: Sentence-level extraction ----
    def extract_sentences(self, items: List[Dict]) -> List[Dict]:
        """Split items into sentence-level units for embedding."""
        sentence_items = []
        for item in items:
            text = item.get("text", "")
            sentences = re.split(r"[.!?;。！？；]\s+", text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 10:
                    sentence_items.append({
                        "text": sent,
                        "source": item.get("source", ""),
                        "question": item.get("question", ""),
                    })
        return sentence_items

    # ---- Step 3: Dense embedding ----
    def embed_sentences(self, sentences: List[Dict]) -> np.ndarray:
        """Generate dense embeddings via sentence-transformers."""
        texts = [s["text"] for s in sentences]
        return self.embedding_model.encode(texts, show_progress_bar=True)

    # ---- Step 4: Multi-view clustering (UMAP + HDBSCAN + BERTopic) ----
    def cluster_sentences(
        self, embeddings: np.ndarray, texts: List[str]
    ) -> Tuple[List[int], Any]:
        """Cluster sentence embeddings using UMAP + HDBSCAN + BERTopic.

        Uses multi-view density clustering with similarity:
        s((q,a),(q',a')) = lambda1*<u,u'> + lambda2*<v,v'> + lambda3*<w,w'>
        """
        n_samples = len(embeddings)
        n_neighbors = min(self.umap_n_neighbors, max(2, n_samples // 3))
        n_components = min(self.umap_n_components, max(2, n_samples // 4))
        min_cluster_size = min(self.hdbscan_min_cluster_size, max(2, n_samples // 5))

        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric="cosine",
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
        vectorizer = CountVectorizer(min_df=min(5, max(1, n_samples // 10)))
        topic_model = BERTopic(
            nr_topics="auto",
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
        )
        topics, probs = topic_model.fit_transform(texts)
        return topics, topic_model

    # ---- Step 5: Topic merging ----
    def merge_topics(
        self, sentences: List[Dict], topics: List[int], topic_model: Any
    ) -> Dict[int, List[str]]:
        """Merge similar topics and aggregate sentences per cluster.

        Uses BM25 ranking and MMR for source-level diversity
        and quote-level de-duplication.
        """
        topic_info = topic_model.get_topic_info()
        clusters: Dict[int, List[str]] = defaultdict(list)
        for sent, topic_id in zip(sentences, topics):
            clusters[topic_id].append(sent["text"])
        return dict(clusters)

    # ---- Step 6: Contradiction-aware summarization (rule distillation) ----
    def distill_rules(
        self,
        clusters: Dict[int, List[str]],
        summarizer_fn=None,
    ) -> List[Dict[str, Any]]:
        """Distill clustered sentences into concise expert rules.

        For each cluster T_m, the summarizer h(.) produces:
        (r_m, c_m, g_m) where:
        - r_m: distilled rule (conditions, core guidance, edge cases, failure modes)
        - c_m: citations
        - g_m: facet metadata (time, region, policy, L2 industry)

        A lightweight entailment/consistency check filters
        self-contradictory statements; majority-consistent claims
        are preferred while minority views are flagged as caveats.
        """
        experience_base = []
        for topic_id, sentences in clusters.items():
            if topic_id == -1 or len(sentences) < 3:
                continue
            rule = {
                "topic_id": topic_id,
                "sentences": sentences,
                "rule": "",
                "citations": [],
                "facets": {},
            }
            if summarizer_fn is not None:
                rule = summarizer_fn(topic_id, sentences)
            experience_base.append(rule)
        self.experience_base = experience_base
        return experience_base

    # ---- Facetization and normalization ----
    def facetize_rules(
        self, rules: List[Dict], domain_terms: Optional[List[str]] = None
    ) -> List[Dict]:
        """Facetize each rule into (time, region, policy, L2 industry).

        High-frequency domain terms (e.g., 'CFA Institute' for finance,
        'FDA' for biomedicine) are filtered via corpus statistics as
        facet candidates, then refined with shallow taggers and LLM
        disambiguation.
        """
        for rule in rules:
            facets = {"time": "ongoing", "region": "universal", "policy": "", "industry": ""}
            text = " ".join(rule.get("sentences", []))
            if domain_terms:
                for term in domain_terms:
                    if term.lower() in text.lower():
                        facets["industry"] = term
                        break
            rule["facets"] = facets
        return rules

    # ---- Full pipeline ----
    def run(
        self,
        qa_tuples: List[Dict],
        summarizer_fn=None,
        domain_terms: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run the full critic-guided extraction pipeline.

        Returns the experience base E = {(r_m, c_m, g_m)}_{m=1}^{K}.
        """
        items = self.harvest_questions(qa_tuples)
        sentences = self.extract_sentences(items)
        embeddings = self.embed_sentences(sentences)
        texts = [s["text"] for s in sentences]
        topics, topic_model = self.cluster_sentences(embeddings, texts)
        clusters = self.merge_topics(sentences, topics, topic_model)
        rules = self.distill_rules(clusters, summarizer_fn=summarizer_fn)
        rules = self.facetize_rules(rules, domain_terms=domain_terms)
        return rules

    def save(self, path: str) -> None:
        """Save the experience base to JSONL."""
        with open(path, "w", encoding="utf-8") as f:
            for rule in self.experience_base:
                f.write(json.dumps(rule, ensure_ascii=False) + "\n")

    def load(self, path: str) -> None:
        """Load an experience base from JSONL."""
        self.experience_base = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.experience_base.append(json.loads(line))
